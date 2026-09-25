// dynin-u2s.cpp - standalone units-to-speech CLI.
//
// Takes a sequence of EMOVA content-unit ids (the output of a text-to-units
// diffusion decode, or of dynin_units_encode_wav()'s own speech-to-unit
// encoder) and renders them to a wav file through dynin_vocoder's HiFi-GAN
// decoder graph.
//
// Usage:
//   dynin-u2s -m <u2s.gguf> -i <units_file> -o <out.wav>
//             [--style <index>] [--seed <n>] [--sr <hz>] [--cpu]
//
// -m   path to the EMOVA U2S GGUF (see tools/convert_emova_u2s_to_gguf.py).
// -i   units file, see "Units file format" below.
// -o   output wav path.
// --style   style/speaker index in [0, n_styles); default 41, which is the
//           checkpoint's "gender-female_emotion-neutral_speed-normal_pitch-normal"
//           entry (config.json's own u2s_style2idx table). The tool prints
//           n_styles on load so any other index can be validated by range.
// --seed    seeds every random draw the decoder makes; default 1234, same
//           input + same seed always reproduces the same output.
// --sr      overrides the sample rate value WRITTEN INTO the wav header.
//           This does not resample the audio -- the decoder's native rate
//           (dynin_vocoder_sample_rate(), normally 22050 Hz) is what the
//           samples actually are. Only use this if a downstream tool needs
//           a specific declared rate; the tool prints a warning if the
//           override does not match the native rate.
// --cpu     forces the CPU path of the generator graph instead of the
//           default CUDA/CPU auto-pick (sets DYNIN_U2S_BACKEND=cpu before
//           loading the model; the auto-pick otherwise times both backends
//           once and keeps whichever is both correct and faster).
//
// Units file format (-i):
//   * binary ".i32": a flat array of little-endian int32 content-unit ids,
//     each in [0, 4095], no header. This is the same layout
//     dynin_units_encode_wav()'s own diagnostic dump writes for its
//     "unit_ids" stage, and the same layout dynin-t2s-kvblock can write.
//   * anything else is read as text: decimal integers separated by any mix
//     of whitespace, commas, or newlines, e.g. "12, 88 401\n7". A file that
//     is not valid UTF-8 text and does not end in ".i32" is rejected with
//     an error rather than silently reinterpreted.
#include "dynin_vocoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <cstdlib>
static void set_env(const char * name, const char * value) { _putenv_s(name, value); }
#else
#include <cstdlib>
static void set_env(const char * name, const char * value) { setenv(name, value, 1); }
#endif

namespace {

struct Args {
    std::string gguf_path;
    std::string units_path;
    std::string out_path;
    int         style   = 41;
    uint64_t    seed    = 1234ULL;
    int         sr_override = 0; // 0 = not set
    bool        force_cpu = false;
};

void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m <u2s.gguf> -i <units_file> -o <out.wav> "
            "[--style <index>] [--seed <n>] [--sr <hz>] [--cpu]\n",
            argv0);
}

bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "-m") {
            const char * v = next("-m");
            if (!v) return false;
            a.gguf_path = v;
        } else if (arg == "-i") {
            const char * v = next("-i");
            if (!v) return false;
            a.units_path = v;
        } else if (arg == "-o") {
            const char * v = next("-o");
            if (!v) return false;
            a.out_path = v;
        } else if (arg == "--style") {
            const char * v = next("--style");
            if (!v) return false;
            a.style = std::atoi(v);
        } else if (arg == "--seed") {
            const char * v = next("--seed");
            if (!v) return false;
            a.seed = (uint64_t) std::strtoull(v, nullptr, 10);
        } else if (arg == "--sr") {
            const char * v = next("--sr");
            if (!v) return false;
            a.sr_override = std::atoi(v);
        } else if (arg == "--cpu") {
            a.force_cpu = true;
        } else if (arg == "-h" || arg == "--help") {
            return false;
        } else {
            fprintf(stderr, "error: unrecognized argument '%s'\n", arg.c_str());
            return false;
        }
    }
    return !a.gguf_path.empty() && !a.units_path.empty() && !a.out_path.empty();
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Reads a flat little-endian int32 array with no header.
bool read_units_binary(const std::string & path, std::vector<int32_t> & out, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open '" + path + "'"; return false; }
    f.seekg(0, std::ios::end);
    const std::streamoff n_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n_bytes < 0 || (n_bytes % 4) != 0) {
        err = "file size " + std::to_string((long long) n_bytes) + " is not a multiple of 4 bytes";
        return false;
    }
    out.resize((size_t) n_bytes / 4);
    if (!out.empty() && !f.read(reinterpret_cast<char *>(out.data()), n_bytes)) {
        err = "short read on '" + path + "'";
        return false;
    }
    return true;
}

// Decimal integers separated by whitespace and/or commas.
bool read_units_text(const std::string & path, std::vector<int32_t> & out, std::string & err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open '" + path + "'"; return false; }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char & c : content) {
        if (c == ',') c = ' ';
    }
    std::istringstream iss(content);
    int32_t v;
    while (iss >> v) out.push_back(v);
    if (out.empty()) { err = "no integers found in '" + path + "'"; return false; }
    return true;
}

bool read_units(const std::string & path, std::vector<int32_t> & out, std::string & err) {
    if (ends_with(path, ".i32")) {
        return read_units_binary(path, out, err);
    }
    // Try text first (the common case for a hand-edited or JSON-adjacent
    // file); fall back to binary only if the extension gives no hint and
    // text parsing finds nothing.
    std::string text_err;
    if (read_units_text(path, out, text_err)) return true;
    out.clear();
    return read_units_binary(path, out, err);
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

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);

    Args a;
    if (!parse_args(argc, argv, a)) {
        usage(argv[0]);
        return 1;
    }

    std::vector<int32_t> unit_ids;
    std::string err;
    if (!read_units(a.units_path, unit_ids, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    for (int32_t u : unit_ids) {
        if (u < 0 || u > 4095) {
            fprintf(stderr, "error: unit id %d out of range [0,4095] in '%s'\n", (int) u, a.units_path.c_str());
            return 1;
        }
    }
    printf("[DYNIN-U2S-IN file=%s n_units=%zu]\n", a.units_path.c_str(), unit_ids.size());

    if (a.force_cpu) {
        set_env("DYNIN_U2S_BACKEND", "cpu");
    }

    dynin_vocoder_model * model = dynin_vocoder_load(a.gguf_path.c_str(), err);
    if (!model) {
        fprintf(stderr, "error (load): %s\n", err.c_str());
        return 1;
    }

    const int n_styles = dynin_vocoder_n_styles(model);
    if (a.style < 0 || a.style >= n_styles) {
        fprintf(stderr, "error: --style %d out of range [0,%d)\n", a.style, n_styles);
        dynin_vocoder_free(model);
        return 1;
    }
    printf("[DYNIN-U2S-MODEL n_styles=%d n_speakers=%d style_dim=%d inter_channels=%d sample_rate=%d style=%d seed=%llu]\n",
           n_styles, dynin_vocoder_n_speakers(model), dynin_vocoder_style_dim(model),
           dynin_vocoder_inter_channels(model), dynin_vocoder_sample_rate(model), a.style,
           (unsigned long long) a.seed);

    dynin_vocoder_audio audio;
    int t_y = 0;
    if (!dynin_vocoder_synthesize_ex(model, unit_ids, a.style, audio, err, a.seed, &t_y)) {
        fprintf(stderr, "error (synthesize): %s\n", err.c_str());
        dynin_vocoder_free(model);
        return 1;
    }

    if (a.sr_override > 0 && a.sr_override != audio.sample_rate) {
        fprintf(stderr,
                "[DYNIN-U2S-WARN requested_sr=%d native_sr=%d note=header_only_no_resample]\n",
                a.sr_override, audio.sample_rate);
        audio.sample_rate = a.sr_override;
    }

    if (!dynin_vocoder_write_wav(a.out_path, audio, err)) {
        fprintf(stderr, "error (write wav): %s\n", err.c_str());
        dynin_vocoder_free(model);
        return 1;
    }

    float peak; double rms;
    wav_stats(audio.pcm16, peak, rms);
    const double seconds = (double) audio.pcm16.size() / (double) audio.sample_rate;
    printf("[DYNIN-U2S-OUT wav=%s samples=%zu seconds=%.3f sample_rate=%d peak=%.5f rms=%.5f t_y=%d]\n",
           a.out_path.c_str(), audio.pcm16.size(), seconds, audio.sample_rate, peak, rms, t_y);

    dynin_vocoder_free(model);
    return 0;
}
