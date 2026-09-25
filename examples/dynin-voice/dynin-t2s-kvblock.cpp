// dynin-t2s-kvblock.cpp - text -> Dynin-Omni t2s speech units, with an
// optional block-local (windowed) KV-cache decode, and a --compare mode
// that checks the windowed decode against the dense reference on the same
// seed.
//
// The masked-diffusion decode below is a direct, unabridged port of the
// t2s_generate_mmu_like recipe in the Dynin-Omni reference implementation
// (AIDAS Lab, Seoul National University; arXiv:2604.00007;
// https://github.com/AIDASLab/Dynin-Omni): per-block confidence-based token
// transfer and classifier-free guidance against a once-only unconditional
// canvas; plus one engine-level optimisation: instead of re-decoding the
// full canvas from position 0 on every single diffusion step, each block's
// FIRST step (step 0) does the dense full-canvas decode (after clearing the
// KV cache), and every later step within that block decodes only the
// half-open window [block_start - margin, block_end + margin), clamped to
// the canvas, reading every position outside that window from the KV cache
// as it was computed at the block's step 0. The margin is the --kv-margin
// flag (M). Setting it to 0 disables windowing entirely and every step
// becomes a dense decode -- the "exact" reference path.
//
// This relies on engine changes made in the same patch, all behind
// llama_context_params::diffusion_prefix_cache: the LLaDA graph gets a
// real KV cache, re-decoding a (seq, pos) that is already in the cache
// overwrites that cell instead of duplicating it, and
// llama_batch_allocr::init(..., allow_pos_rewrite=true) skips the batch
// sequence-position consistency checks (the M-RoPE branch and the
// consecutive-position branch) so a window batch whose positions are
// already in the cache is accepted. No position is rewritten.
//
// Usage:
//   dynin-t2s-kvblock -m <dynin.gguf> --u2s <u2s.gguf> --text "<sentence>"
//       [--steps N] [--block B] [--kv-margin M] [--cfg C] [--seed S]
//       --out <dir> [--wav] [--compare]
//       [--whisper-cli <exe> --whisper-model <bin>]
//
// -m / --u2s   the diffusion checkpoint and the EMOVA U2S vocoder GGUF.
// --text       the sentence to speak (one per run; quote it).
// --steps      total diffusion steps across the whole speech region.
//              Reference default 383; also used as the token length (the
//              number of speech-unit positions generated), matching the
//              reference recipe's default where both figures are equal.
// --block      block length for the block-local diffusion schedule.
//              Reference default 128.
// --kv-margin  window margin M for the windowed decode. 0 = dense/exact.
//              If omitted, defaults to 32 when --steps==48 and to 0
//              (dense) otherwise.
// --cfg        classifier-free-guidance scale. Reference default 2.5.
// --seed       RNG seed for the sampling step. Default 1234.
// --out        output directory; the last path component is created if
//              missing (its parent must already exist).
// --wav        also render the resulting unit sequence(s) to wav through
//              the U2S vocoder.
// --compare    run TWO passes on the SAME text and seed: "exact"
//              (kv-margin forced to 0) and "margin" (the --kv-margin
//              value, or its computed default). Writes match.json
//              (unit-level agreement between the two passes) and
//              timing.json (per-block forward-pass ms, total ms for the
//              block loop -- which excludes the one-off unconditional
//              decode -- and forward-pass counts) into --out. Without --compare, only the
//              single run at --kv-margin is performed.
// --whisper-cli / --whisper-model
//              optional scoring oracle: transcribes the --wav output(s)
//              and reports word overlap against --text. No default path;
//              omit either flag to skip transcription entirely.
#define NOMINMAX
#include "dynin_ears.h"
#include "dynin_vocab.h"
#include "dynin_vocoder.h"

#include "llama.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace {

// prompting_utils.py reserved_token_mapping (verbatim ids for this checkpoint).
constexpr llama_token TOK_T2S  = 126099;
constexpr llama_token TOK_SOA  = 126097;
constexpr llama_token TOK_EOA  = 126098;
constexpr llama_token TOK_IPAD = 126093;
constexpr int32_t AUDIO_CODEBOOK_SIZE = 4096;
constexpr int32_t REL_EOA = AUDIO_CODEBOOK_SIZE;
constexpr int32_t REL_EOS = AUDIO_CODEBOOK_SIZE + 1;

// training/data.py T2S_INSTRUCTION, first entry (reproducible choice: the
// reference implementation draws one of six at random, a fixed choice does
// not change the <|t2s|>/<|soa|> structure or the text content being spoken).
constexpr const char * kT2SInstruction = "Generate speech for the given text.";

// config.json's u2s_style2idx entry for
// "gender-female_emotion-neutral_speed-normal_pitch-normal".
constexpr int kDefaultStyleIndex = 41;

void mkdir_p(const std::string & path) {
#if defined(_WIN32)
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

std::string join_path(const std::string & dir, const std::string & name) {
    if (dir.empty()) return name;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + name;
    return dir + "/" + name;
}

std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> v((size_t) steps);
    const int32_t base = mask_count / steps, rem = mask_count % steps;
    for (int32_t i = 0; i < steps; i++) v[(size_t) i] = base + (i < rem ? 1 : 0);
    return v;
}

std::vector<llama_token> tokenize_special(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> toks(text.size() + 16);
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(),
                                /*add_special*/ false, /*parse_special*/ true);
    if (n < 0) {
        toks.resize((size_t) (-n));
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
    }
    toks.resize((size_t) std::max(0, n));
    return toks;
}

std::vector<std::string> words_lower(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            cur.push_back((char) std::tolower(c));
        } else if (!cur.empty()) {
            out.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

float word_overlap_fraction(const std::string & reference, const std::string & hyp) {
    auto rw = words_lower(reference);
    auto hw = words_lower(hyp);
    std::set<std::string> ref_unique(rw.begin(), rw.end());
    std::set<std::string> hyp_set(hw.begin(), hw.end());
    if (ref_unique.empty()) return 0.0f;
    int overlap = 0;
    for (const auto & w : ref_unique) if (hyp_set.count(w)) overlap++;
    return (float) overlap / (float) ref_unique.size();
}

void wav_stats(const std::vector<int16_t> & pcm, float & peak, double & rms) {
    peak = 0.0f;
    double sumsq = 0.0;
    for (int16_t s : pcm) {
        float f = (float) s / 32768.0f;
        peak = std::max(peak, std::fabs(f));
        sumsq += (double) f * (double) f;
    }
    rms = pcm.empty() ? 0.0 : std::sqrt(sumsq / pcm.size());
}

struct Args {
    std::string dynin_path;
    std::string u2s_path;
    std::string text;
    int32_t steps      = 383;
    int32_t block      = 128;
    int32_t kv_margin   = -1;  // -1 = "not given", resolved after parsing
    float   cfg        = 2.5f;
    uint64_t seed       = 1234ULL;
    std::string out_dir;
    bool wav            = false;
    bool compare        = false;
    std::string whisper_cli;
    std::string whisper_model;
};

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m <dynin.gguf> --u2s <u2s.gguf> --text \"<sentence>\" --out <dir>\n"
        "       [--steps N] [--block B] [--kv-margin M] [--cfg C] [--seed S]\n"
        "       [--wav] [--compare] [--whisper-cli <exe> --whisper-model <bin>]\n",
        argv0);
}

bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s requires a value\n", flag); return nullptr; }
            return argv[++i];
        };
        if (arg == "-m")                 { const char * v = next("-m");                if (!v) return false; a.dynin_path = v; }
        else if (arg == "--u2s")         { const char * v = next("--u2s");              if (!v) return false; a.u2s_path = v; }
        else if (arg == "--text")        { const char * v = next("--text");            if (!v) return false; a.text = v; }
        else if (arg == "--steps")       { const char * v = next("--steps");           if (!v) return false; a.steps = std::atoi(v); }
        else if (arg == "--block")       { const char * v = next("--block");           if (!v) return false; a.block = std::atoi(v); }
        else if (arg == "--kv-margin")   { const char * v = next("--kv-margin");       if (!v) return false; a.kv_margin = std::atoi(v); }
        else if (arg == "--cfg")         { const char * v = next("--cfg");             if (!v) return false; a.cfg = std::strtof(v, nullptr); }
        else if (arg == "--seed")        { const char * v = next("--seed");            if (!v) return false; a.seed = (uint64_t) std::strtoull(v, nullptr, 10); }
        else if (arg == "--out")         { const char * v = next("--out");             if (!v) return false; a.out_dir = v; }
        else if (arg == "--wav")         { a.wav = true; }
        else if (arg == "--compare")     { a.compare = true; }
        else if (arg == "--whisper-cli")   { const char * v = next("--whisper-cli");   if (!v) return false; a.whisper_cli = v; }
        else if (arg == "--whisper-model") { const char * v = next("--whisper-model"); if (!v) return false; a.whisper_model = v; }
        else if (arg == "-h" || arg == "--help") { return false; }
        else { fprintf(stderr, "error: unrecognized argument '%s'\n", arg.c_str()); return false; }
    }
    if (a.kv_margin < 0) {
        // Default: margin 32 at steps==48; every other step count defaults to
        // the dense path (0). Pass --kv-margin explicitly to override.
        a.kv_margin = (a.steps == 48) ? 32 : 0;
    }
    return !a.dynin_path.empty() && !a.u2s_path.empty() && !a.text.empty() && !a.out_dir.empty();
}

// ---------------------------------------------------------------- session

struct Session {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    int32_t             n_vocab = 0;
    dynin_vocab::Map    vmap;
    llama_token         mask_id = LLAMA_TOKEN_NULL;
    llama_token         tok_bos = LLAMA_TOKEN_NULL;
    llama_token         tok_eos = LLAMA_TOKEN_NULL;
    bool                shift_logits = false;
};

struct RunResult {
    bool                  ok = false;
    std::vector<int32_t>  unit_ids;
    std::vector<double>   block_ms;
    double                total_ms = 0.0;
    int                   n_forward_passes = 0;
};

// One full text -> units generation. kv_margin == 0 is the dense/exact
// path (every step re-decodes the whole canvas after clearing the KV
// cache); kv_margin > 0 windows every step after the block's first.
RunResult run_t2s(Session & s, const std::string & text, int32_t token_length, int32_t steps,
                   int32_t block_length, float cfg_scale, float temperature, int32_t max_text_len,
                   int32_t kv_margin, uint64_t seed) {
    RunResult r;

    const std::string header = "<|start_header_id|>user<|end_header_id|>\n" + std::string(kT2SInstruction) +
                                "\n" + text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";

    std::vector<llama_token> text_ids = tokenize_special(s.vocab, header);
    if (text_ids.empty() || text_ids.front() != s.tok_bos) {
        text_ids.insert(text_ids.begin(), s.tok_bos);
    }
    std::vector<llama_token> temp_ids;
    temp_ids.reserve(text_ids.size() + 2);
    temp_ids.push_back(TOK_T2S);
    temp_ids.insert(temp_ids.end(), text_ids.begin(), text_ids.end());
    temp_ids.push_back(s.tok_eos);

    std::vector<llama_token> text_block;
    if (max_text_len >= (int32_t) temp_ids.size()) {
        text_block.assign((size_t) (max_text_len - (int32_t) temp_ids.size()), TOK_IPAD);
        text_block.insert(text_block.end(), temp_ids.begin(), temp_ids.end());
    } else {
        temp_ids.resize((size_t) max_text_len - 1);
        temp_ids.push_back(s.tok_eos);
        text_block = temp_ids;
    }

    std::vector<llama_token> prefix = text_block;
    prefix.push_back(TOK_SOA);
    const int32_t speech_region_start = (int32_t) prefix.size();
    const int32_t max_length          = speech_region_start + token_length;

    if ((uint32_t) max_length > llama_n_ctx(s.ctx)) {
        fprintf(stderr, "[DYNIN-T2S-ERROR text='%s' reason=prompt_too_long max_length=%d n_ctx=%u]\n",
                text.c_str(), max_length, llama_n_ctx(s.ctx));
        return r;
    }

    std::vector<llama_token> work((size_t) max_length);
    std::copy(prefix.begin(), prefix.end(), work.begin());
    std::fill(work.begin() + speech_region_start, work.end(), s.mask_id);

    llama_batch batch = llama_batch_init(max_length, 0, 2);

    // Every position of the current canvas, logits enabled everywhere, after
    // clearing the KV cache first. `clear_after` additionally clears once the
    // decode is read out, giving full isolation from whatever comes next --
    // used for the exact/reference path (every step independent) and for the
    // one-off uncond decode. The margin path's own per-block step-0 seed
    // decode uses clear_after=false instead, so the cache it just built stays
    // in place for that block's later windowed steps to read from.
    auto decode_dense = [&](const std::vector<llama_token> & canvas, bool clear_after) -> std::vector<float> {
        llama_memory_seq_rm(llama_get_memory(s.ctx), -1, 0, -1);
        batch.n_tokens = max_length;
        for (int32_t i = 0; i < max_length; i++) {
            batch.token[i]     = canvas[(size_t) i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = (canvas[(size_t) i] == TOK_IPAD) ? 1 : 0;
            batch.logits[i]    = 1;
        }
        std::vector<float> out;
        if (llama_decode(s.ctx, batch) != 0) return out;
        r.n_forward_passes++;
        const float * logits = llama_get_logits(s.ctx);
        out.assign(logits, logits + (size_t) max_length * (size_t) s.n_vocab);
        if (clear_after) {
            llama_memory_seq_rm(llama_get_memory(s.ctx), -1, 0, -1);
        }
        return out;
    };

    // uncond is bit-identical on every step (the whole speech region is
    // re-masked fresh each time it would be read) -- decoded once and reused.
    std::vector<float> uncond_logits;
    if (cfg_scale > 0.0f) {
        std::vector<llama_token> uncond_canvas = work;
        std::fill(uncond_canvas.begin() + speech_region_start, uncond_canvas.end(), s.mask_id);
        uncond_logits = decode_dense(uncond_canvas, /*clear_after=*/ true);
        if (uncond_logits.empty()) {
            fprintf(stderr, "[DYNIN-T2S-ERROR text='%s' reason=decode_failed_uncond]\n", text.c_str());
            llama_batch_free(batch);
            return r;
        }
    }

    // Row selector for a logits buffer produced with every input position's
    // logits enabled. When the checkpoint's own diffusion.shift_logits GGUF
    // metadata is true, row 0's logits are read as-is and every later row
    // is read one position back (a next-token-prediction convention some
    // checkpoints use); this checkpoint's own metadata reads false, making
    // this a no-op, but the read is honoured rather than assumed.
    auto get_logits_at = [&](const std::vector<float> & buf, int32_t row) -> const float * {
        if (s.shift_logits) row = std::max(0, row - 1);
        return buf.data() + (size_t) row * (size_t) s.n_vocab;
    };

    const int32_t num_blocks  = (token_length + block_length - 1) / block_length;
    const int32_t inner_steps = std::max(1, steps / num_blocks);

    r.block_ms.assign((size_t) num_blocks, 0.0);

    std::mt19937_64 rng(seed);
    std::vector<float> probs((size_t) (AUDIO_CODEBOOK_SIZE + 2));
    bool decode_failed = false;
    const auto t_start = std::chrono::high_resolution_clock::now();

    for (int32_t block = 0; block < num_blocks && !decode_failed; block++) {
        const int32_t block_start = speech_region_start + block * block_length;
        const int32_t block_end   = std::min(block_start + block_length, max_length);

        int32_t block_mask_count = 0;
        for (int32_t i = block_start; i < block_end; i++) if (work[(size_t) i] == s.mask_id) block_mask_count++;
        std::vector<int32_t> schedule = get_num_transfer_tokens(block_mask_count, inner_steps);

        for (int32_t step = 0; step < inner_steps; step++) {
            std::vector<int32_t> mask_positions;
            for (int32_t i = block_start; i < block_end; i++) if (work[(size_t) i] == s.mask_id) mask_positions.push_back(i);
            if (mask_positions.empty()) break;

            const auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<float> cond_logits;
            int32_t row_offset = 0; // position of row 0 of cond_logits within the canvas

            if (kv_margin <= 0) {
                // Exact/reference path: every step is a fully isolated dense
                // decode, clearing the KV cache both before and after.
                cond_logits = decode_dense(work, /*clear_after=*/ true);
                row_offset  = 0;
            } else if (step == 0) {
                // Margin path, block seed: clear before, decode the full
                // canvas, but leave the resulting cache in place (clear_after
                // = false) so this block's later windowed steps can read it.
                cond_logits = decode_dense(work, /*clear_after=*/ false);
                row_offset  = 0;
            } else {
                const int32_t win_start = std::max(0, block_start - kv_margin);
                const int32_t win_end   = std::min(max_length, block_end + kv_margin);
                const int32_t n_step_tokens = win_end - win_start;
                batch.n_tokens = n_step_tokens;
                for (int32_t i = 0; i < n_step_tokens; i++) {
                    const int32_t pos  = win_start + i;
                    batch.token[i]     = work[(size_t) pos];
                    batch.pos[i]       = pos;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = (work[(size_t) pos] == TOK_IPAD) ? 1 : 0;
                    batch.logits[i]    = 1;
                }
                if (llama_decode(s.ctx, batch) != 0) {
                    decode_failed = true;
                    break;
                }
                r.n_forward_passes++;
                const float * logits = llama_get_logits(s.ctx);
                cond_logits.assign(logits, logits + (size_t) n_step_tokens * (size_t) s.n_vocab);
                row_offset = win_start;
            }
            if (cond_logits.empty()) { decode_failed = true; break; }

            const auto t1 = std::chrono::high_resolution_clock::now();
            r.block_ms[(size_t) block] += std::chrono::duration<double, std::milli>(t1 - t0).count();

            std::vector<std::pair<float, int32_t>> confidences;
            std::vector<llama_token>               sampled(mask_positions.size());
            confidences.reserve(mask_positions.size());

            for (size_t mi = 0; mi < mask_positions.size(); mi++) {
                const int32_t pos = mask_positions[mi];
                const float * cl  = get_logits_at(cond_logits, pos - row_offset);
                const float * ul  = cfg_scale > 0.0f ? get_logits_at(uncond_logits, pos) : nullptr;

                for (int32_t k = 0; k < AUDIO_CODEBOOK_SIZE; k++) {
                    float c = cl[s.vmap.speech_base + k];
                    if (ul) { float u = ul[s.vmap.speech_base + k]; c = u + (cfg_scale + 1.0f) * (c - u); }
                    probs[(size_t) k] = c;
                }
                { float c = cl[TOK_EOA];   if (ul) { float u = ul[TOK_EOA];   c = u + (cfg_scale + 1.0f) * (c - u); } probs[(size_t) REL_EOA] = c; }
                { float c = cl[s.tok_eos]; if (ul) { float u = ul[s.tok_eos]; c = u + (cfg_scale + 1.0f) * (c - u); } probs[(size_t) REL_EOS] = c; }

                if (temperature > 0.0f) {
                    const float t = std::max(temperature, 1e-5f);
                    for (float & v : probs) v /= t;
                }
                float mx = probs[0];
                for (float v : probs) mx = std::max(mx, v);
                std::vector<double> ex(probs.size());
                double sum = 0.0;
                for (size_t k = 0; k < probs.size(); k++) { ex[k] = std::exp((double) (probs[k] - mx)); sum += ex[k]; }
                for (double & v : ex) v /= sum;

                int r_idx = 0;
                if (temperature <= 0.0f) {
                    double best_p = ex[0];
                    for (size_t k = 1; k < ex.size(); k++) if (ex[k] > best_p) { best_p = ex[k]; r_idx = (int) k; }
                } else {
                    std::discrete_distribution<int> dist(ex.begin(), ex.end());
                    r_idx = dist(rng);
                }
                sampled[mi] = (llama_token) (r_idx < AUDIO_CODEBOOK_SIZE ? s.vmap.speech_base + r_idx
                                                                          : (r_idx == REL_EOA ? TOK_EOA : s.tok_eos));
                confidences.emplace_back((float) ex[(size_t) r_idx], (int32_t) mi);
            }

            int32_t transfer = (step < (int32_t) schedule.size()) ? schedule[(size_t) step] : (int32_t) mask_positions.size();
            transfer         = std::min(transfer, (int32_t) confidences.size());
            if (transfer > 0) {
                std::partial_sort(confidences.begin(), confidences.begin() + transfer, confidences.end(),
                    [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second < b.second;
                    });
                for (int32_t k = 0; k < transfer; k++) {
                    const int32_t mi = confidences[(size_t) k].second;
                    work[(size_t) mask_positions[(size_t) mi]] = sampled[(size_t) mi];
                }
            }
        }
        printf("[DYNIN-T2S-KVBLOCK margin=%d block=%d ms=%.2f]\n", kv_margin, block, r.block_ms[(size_t) block]);
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    r.total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (kv_margin > 0 && llama_get_memory(s.ctx)) {
        llama_memory_seq_rm(llama_get_memory(s.ctx), -1, 0, -1);
    }

    llama_batch_free(batch);

    if (decode_failed) {
        fprintf(stderr, "[DYNIN-T2S-ERROR text='%s' reason=decode_failed]\n", text.c_str());
        return r;
    }

    for (int32_t i = speech_region_start; i < max_length; i++) {
        llama_token t = work[(size_t) i];
        if (t == TOK_EOA || t == s.tok_eos) break;
        r.unit_ids.push_back((int32_t) (t - s.vmap.speech_base));
    }
    r.ok = true;
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    Args a;
    if (!parse_args(argc, argv, a)) {
        usage(argv[0]);
        return 1;
    }
    mkdir_p(a.out_dir);

    ggml_time_init();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = 999;

    Session s;
    s.model = llama_model_load_from_file(a.dynin_path.c_str(), model_params);
    if (!s.model) {
        fprintf(stderr, "error: failed to load model '%s'\n", a.dynin_path.c_str());
        return 1;
    }
    if (!llama_model_is_diffusion(s.model)) {
        fprintf(stderr, "error: model is not a diffusion architecture\n");
        llama_model_free(s.model);
        return 1;
    }

    s.vocab   = llama_model_get_vocab(s.model);
    s.n_vocab = llama_vocab_n_tokens(s.vocab);
    s.vmap    = dynin_vocab::read(s.vocab);
    dynin_vocab::print(s.vmap, "DYNIN-T2S-VOCAB");
    if (!s.vmap.ok) {
        fprintf(stderr, "error: vocabulary map refused: %s\n", s.vmap.why);
        llama_model_free(s.model);
        return 1;
    }

    const int32_t max_text_len = 1024;
    const int32_t speech_region_start_upper = max_text_len + 1; // text_block + <|soa|>
    const int32_t max_length_upper = speech_region_start_upper + a.steps;

    llama_context_params ctx_params   = llama_context_default_params();
    ctx_params.n_ctx                  = (uint32_t) std::max(4096, max_length_upper + 64);
    ctx_params.n_batch                = ctx_params.n_ctx;
    ctx_params.n_ubatch               = ctx_params.n_ctx;
    ctx_params.n_seq_max              = 2; // seq 0 = real tokens, seq 1 = the [iPAD] wall
    ctx_params.diffusion_prefix_cache = true;

    s.ctx = llama_init_from_model(s.model, ctx_params);
    if (!s.ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(s.model);
        return 1;
    }
    llama_set_attn_pad_token(s.ctx, TOK_IPAD);

    s.mask_id = llama_vocab_mask(s.vocab);
    if (s.mask_id == LLAMA_TOKEN_NULL) {
        fprintf(stderr, "error: model has no mask token\n");
        llama_free(s.ctx);
        llama_model_free(s.model);
        return 1;
    }
    s.tok_bos = llama_vocab_bos(s.vocab);
    s.tok_eos = llama_vocab_eos(s.vocab);

    char sl_buf[8];
    s.shift_logits = false;
    if (llama_model_meta_val_str(s.model, "diffusion.shift_logits", sl_buf, sizeof(sl_buf)) >= 0) {
        s.shift_logits = (strcmp(sl_buf, "true") == 0);
    }
    printf("[DYNIN-T2S-SCHEDULE steps=%d block=%d cfg=%.2f kv_margin=%d seed=%llu compare=%d]\n",
           a.steps, a.block, (double) a.cfg, a.kv_margin, (unsigned long long) a.seed, a.compare ? 1 : 0);

    dynin_vocoder_model * voice_model = nullptr;
    if (a.wav) {
        std::string verr;
        voice_model = dynin_vocoder_load(a.u2s_path.c_str(), verr);
        if (!voice_model) {
            fprintf(stderr, "error (u2s load): %s\n", verr.c_str());
            llama_free(s.ctx);
            llama_model_free(s.model);
            return 1;
        }
    }

    auto render_wav = [&](const std::vector<int32_t> & unit_ids, const std::string & path, uint64_t seed) -> bool {
        if (!voice_model) return false;
        dynin_vocoder_audio audio;
        int t_y = 0;
        std::string verr;
        if (!dynin_vocoder_synthesize_ex(voice_model, unit_ids, kDefaultStyleIndex, audio, verr, seed, &t_y)) {
            fprintf(stderr, "[DYNIN-U2S-ERROR %s]\n", verr.c_str());
            return false;
        }
        if (!dynin_vocoder_write_wav(path, audio, verr)) {
            fprintf(stderr, "[DYNIN-U2S-ERROR writing '%s': %s]\n", path.c_str(), verr.c_str());
            return false;
        }
        float peak; double rms;
        wav_stats(audio.pcm16, peak, rms);
        const double seconds = (double) audio.pcm16.size() / (double) audio.sample_rate;
        printf("[DYNIN-U2S-OUT wav=%s seconds=%.3f samples=%zu peak=%.5f rms=%.5f]\n",
               path.c_str(), seconds, audio.pcm16.size(), peak, rms);
        return true;
    };

    dynin_ears::DyninEars ears;
    ears.exe   = a.whisper_cli;
    ears.model = a.whisper_model;
    const bool have_whisper = !a.whisper_cli.empty() && !a.whisper_model.empty();

    auto transcribe = [&](const std::string & wav_path) -> std::pair<std::string, float> {
        if (!have_whisper) return {"", -1.0f};
        int ms = 0;
        std::string transcript = ears.transcribe(wav_path, &ms);
        float overlap = word_overlap_fraction(a.text, transcript);
        return {transcript, overlap};
    };

    if (!a.compare) {
        RunResult run = run_t2s(s, a.text, a.steps, a.steps, a.block, a.cfg, 1.0f, max_text_len, a.kv_margin, a.seed);
        if (!run.ok) {
            fprintf(stderr, "error: generation failed\n");
            if (voice_model) dynin_vocoder_free(voice_model);
            llama_free(s.ctx);
            llama_model_free(s.model);
            return 1;
        }
        printf("[DYNIN-T2S text='%s' n_units=%zu total_ms=%.1f n_forward_passes=%d]\n",
               a.text.c_str(), run.unit_ids.size(), run.total_ms, run.n_forward_passes);

        if (a.wav) {
            const std::string wav_path = join_path(a.out_dir, "t2s.wav");
            if (render_wav(run.unit_ids, wav_path, a.seed) && have_whisper) {
                auto [transcript, overlap] = transcribe(wav_path);
                printf("[DYNIN-T2S-CHECK text='%s' whisper='%s' word_overlap=%.4f]\n",
                       a.text.c_str(), transcript.c_str(), overlap);
            }
        }
    } else {
        RunResult exact  = run_t2s(s, a.text, a.steps, a.steps, a.block, a.cfg, 1.0f, max_text_len, 0, a.seed);
        RunResult margin = run_t2s(s, a.text, a.steps, a.steps, a.block, a.cfg, 1.0f, max_text_len, a.kv_margin, a.seed);

        if (!exact.ok || !margin.ok) {
            fprintf(stderr, "error: --compare generation failed (exact_ok=%d margin_ok=%d)\n", exact.ok, margin.ok);
            if (voice_model) dynin_vocoder_free(voice_model);
            llama_free(s.ctx);
            llama_model_free(s.model);
            return 1;
        }

        const size_t n_common = std::min(exact.unit_ids.size(), margin.unit_ids.size());
        size_t n_match = 0;
        int64_t first_divergence = -1;
        for (size_t i = 0; i < n_common; i++) {
            if (exact.unit_ids[i] == margin.unit_ids[i]) {
                n_match++;
            } else if (first_divergence < 0) {
                first_divergence = (int64_t) i;
            }
        }
        if (first_divergence < 0 && exact.unit_ids.size() != margin.unit_ids.size()) {
            first_divergence = (int64_t) n_common;
        }
        const double match_fraction = n_common > 0 ? (double) n_match / (double) n_common : 0.0;

        std::string transcript_exact, transcript_margin;
        float overlap_exact = -1.0f, overlap_margin = -1.0f;
        std::string wav_exact_path, wav_margin_path;
        if (a.wav) {
            wav_exact_path  = join_path(a.out_dir, "t2s_exact.wav");
            wav_margin_path = join_path(a.out_dir, "t2s_margin.wav");
            render_wav(exact.unit_ids, wav_exact_path, a.seed);
            render_wav(margin.unit_ids, wav_margin_path, a.seed);
            if (have_whisper) {
                std::tie(transcript_exact, overlap_exact)   = transcribe(wav_exact_path);
                std::tie(transcript_margin, overlap_margin) = transcribe(wav_margin_path);
                printf("[DYNIN-T2S-CHECK run=exact  text='%s' whisper='%s' word_overlap=%.4f]\n",
                       a.text.c_str(), transcript_exact.c_str(), overlap_exact);
                printf("[DYNIN-T2S-CHECK run=margin text='%s' whisper='%s' word_overlap=%.4f]\n",
                       a.text.c_str(), transcript_margin.c_str(), overlap_margin);
            }
        }

        // match.json
        {
            const std::string path = join_path(a.out_dir, "match.json");
            FILE * fp = fopen(path.c_str(), "w");
            if (fp) {
                fprintf(fp, "{\n");
                fprintf(fp, "  \"text\": \"%s\",\n", a.text.c_str());
                fprintf(fp, "  \"kv_margin\": %d,\n", a.kv_margin);
                fprintf(fp, "  \"seed\": %llu,\n", (unsigned long long) a.seed);
                fprintf(fp, "  \"lengths\": { \"exact\": %zu, \"margin\": %zu },\n",
                        exact.unit_ids.size(), margin.unit_ids.size());
                fprintf(fp, "  \"exact_match_fraction\": %.6f,\n", match_fraction);
                fprintf(fp, "  \"first_divergence_index\": %lld\n", (long long) first_divergence);
                fprintf(fp, "}\n");
                fclose(fp);
            }
            printf("[DYNIN-T2S-MATCH lengths=(%zu,%zu) exact_match_fraction=%.4f first_divergence_index=%lld]\n",
                   exact.unit_ids.size(), margin.unit_ids.size(), match_fraction, (long long) first_divergence);
        }

        // timing.json
        {
            const std::string path = join_path(a.out_dir, "timing.json");
            FILE * fp = fopen(path.c_str(), "w");
            if (fp) {
                fprintf(fp, "{\n");
                auto write_run = [&](const char * name, const RunResult & r, bool last) {
                    fprintf(fp, "  \"%s\": {\n", name);
                    fprintf(fp, "    \"ms_per_block\": [");
                    for (size_t i = 0; i < r.block_ms.size(); i++) {
                        fprintf(fp, "%s%.3f", i ? ", " : "", r.block_ms[i]);
                    }
                    fprintf(fp, "],\n");
                    fprintf(fp, "    \"total_ms\": %.3f,\n", r.total_ms);
                    fprintf(fp, "    \"n_forward_passes\": %d\n", r.n_forward_passes);
                    fprintf(fp, "  }%s\n", last ? "" : ",");
                };
                write_run("exact", exact, false);
                write_run("margin", margin, true);
                fprintf(fp, "}\n");
                fclose(fp);
            }
            printf("[DYNIN-T2S-TIMING exact_total_ms=%.1f exact_forward_passes=%d margin_total_ms=%.1f margin_forward_passes=%d]\n",
                   exact.total_ms, exact.n_forward_passes, margin.total_ms, margin.n_forward_passes);
        }
    }

    if (voice_model) dynin_vocoder_free(voice_model);
    llama_free(s.ctx);
    llama_model_free(s.model);
    llama_backend_free();
    return 0;
}
