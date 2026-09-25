// dynin_units.cpp - EMOVA S2U encoder on ggml, with an optional CUDA path
// via ggml_backend_sched. See dynin_units.h for the interface.
//
// Attribution: a C++/ggml translation of the S2U speech tokenizer in the
// EMOVA speech tokenizer (https://github.com/emova-ollm/EMOVA_speech_tokenizer,
// the EMOVA authors, arXiv:2409.18042; speech_tokenization/SPIRAL_L2_BN_FSQ_CTC/),
// licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). That code builds on SPIRAL
// (Huawei, arXiv:2201.10207) and NVIDIA NeMo, both Apache-2.0; the FSQ
// quantizer below follows its my_scripts/fsq.py ("Copyright 2023 Google
// LLC", Apache-2.0; Mentzer et al., arXiv:2309.15505). Changes relative to
// that source: the PyTorch modules were re-expressed in C++ and ggml and
// the weights are read from a GGUF; the computation is meant to be the
// same. Full attribution is in dynin_units.h and in the NOTICE file of the
// repository this example ships with.
#include "dynin_units.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ============================================================ WAV loading
namespace {

struct WavAudio {
    std::vector<float> samples; // mono, roughly [-1,1]
    int                 sample_rate = 0;
};

constexpr double kPi = 3.14159265358979323846;

uint32_t rd_u32(const uint8_t * p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
uint16_t rd_u16(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

bool read_wav_pcm(const std::string & path, WavAudio & out, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { err = "cannot open wav: " + path; return false; }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() < 44 || std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file: " + path;
        return false;
    }

    uint16_t audio_format = 0, n_channels = 1, bits_per_sample = 16;
    uint32_t sample_rate = 16000;
    const uint8_t * data_ptr = nullptr;
    size_t          data_len = 0;

    size_t pos = 12;
    while (pos + 8 <= buf.size()) {
        char     id[5]  = { 0 };
        std::memcpy(id, &buf[pos], 4);
        uint32_t sz = rd_u32(&buf[pos + 4]);
        size_t   body = pos + 8;
        if (body + sz > buf.size()) sz = (uint32_t) (buf.size() - body); // tolerate truncated trailing chunk
        if (std::memcmp(id, "fmt ", 4) == 0 && sz >= 16) {
            audio_format    = rd_u16(&buf[body + 0]);
            n_channels      = rd_u16(&buf[body + 2]);
            sample_rate     = rd_u32(&buf[body + 4]);
            bits_per_sample = rd_u16(&buf[body + 14]);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_ptr = &buf[body];
            data_len = sz;
        }
        pos = body + sz + (sz & 1); // chunks are word-aligned
    }
    if (!data_ptr) { err = "no data chunk in wav: " + path; return false; }
    if (audio_format != 1 && audio_format != 0xFFFE) { err = "only PCM wav supported, got format " + std::to_string(audio_format); return false; }
    if (bits_per_sample != 16) { err = "only 16-bit PCM supported, got " + std::to_string(bits_per_sample) + "-bit"; return false; }
    if (n_channels < 1) { err = "invalid channel count"; return false; }

    size_t n_frames = data_len / (size_t) (2 * n_channels);
    out.samples.resize(n_frames);
    const int16_t * s16 = reinterpret_cast<const int16_t *>(data_ptr);
    for (size_t i = 0; i < n_frames; ++i) {
        int32_t acc = 0;
        for (int c = 0; c < n_channels; ++c) acc += s16[i * n_channels + c];
        out.samples[i] = (float) acc / (float) n_channels / 32768.0f;
    }
    out.sample_rate = (int) sample_rate;
    return true;
}

void resample_to_16k(std::vector<float> & x, int sr_in) {
    const int sr_out = 16000;
    if (sr_in == sr_out || x.empty()) return;
    if (sr_in % sr_out == 0) {
        const int    ratio     = sr_in / sr_out;
        const int    half_taps = ratio * 64;       // 192 taps each side for ratio=3 -> 385-tap filter
        const int    n_taps    = 2 * half_taps + 1;
        const double fc        = 0.5 / ratio;

        std::vector<double> h(n_taps);
        double              sum = 0.0;
        for (int i = 0; i < n_taps; ++i) {
            int    n    = i - half_taps;
            double sinc = (n == 0) ? (2.0 * fc) : (std::sin(2.0 * kPi * fc * n) / (kPi * n));
            double w    = 0.42 - 0.5 * std::cos(2.0 * kPi * i / (n_taps - 1))
                                + 0.08 * std::cos(4.0 * kPi * i / (n_taps - 1)); // Blackman window
            h[i] = sinc * w;
            sum += h[i];
        }
        for (double & v : h) v /= sum; // unity DC gain

        const int           n_out = (int) x.size() / ratio;
        std::vector<float>  y((size_t) std::max(0, n_out));
        const int           n_in = (int) x.size();
        for (int o = 0; o < n_out; ++o) {
            const int center = o * ratio;
            double    acc    = 0.0;
            for (int i = 0; i < n_taps; ++i) {
                int idx = center + (i - half_taps);
                if (idx < 0) idx = 0;
                if (idx >= n_in) idx = n_in - 1;
                acc += h[i] * (double) x[idx];
            }
            y[o] = (float) acc;
        }
        x.swap(y);
        return;
    }
    double              step = (double) sr_in / (double) sr_out;
    size_t              n_out = (size_t) ((double) x.size() / step);
    std::vector<float>  y(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        double  pos = i * step;
        size_t  i0  = (size_t) pos;
        size_t  i1  = std::min(i0 + 1, x.size() - 1);
        float   frac = (float) (pos - (double) i0);
        y[i] = x[i0] * (1.0f - frac) + x[i1] * frac;
    }
    x.swap(y);
}

// ============================================================ FFT (radix-2)
void fft_pow2(std::vector<std::complex<float>> & a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * kPi / (double) len;
        std::complex<float> wlen((float) std::cos(ang), (float) std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                std::complex<float> u = a[i + k];
                std::complex<float> v = a[i + k + len / 2] * w;
                a[i + k]           = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// ============================================================ mel frontend
struct MelHparams {
    int   n_mels = 128, n_fft = 512, hop = 160, win = 320;
    float preemph = 0.97f, log_guard = 5.960464477539063e-08f;
};

std::vector<float> compute_log_mel(const std::vector<float> & wav_in, const MelHparams & p,
                                    const std::vector<float> & fb, const std::vector<float> & window,
                                    int & T_out, int & seq_len_out) {
    std::vector<float> x = wav_in;
    float peak = 0.0f;
    for (float v : x) peak = std::max(peak, std::fabs(v));
    float inv = 1.0f / (peak + 1e-5f);
    for (float & v : x) v *= inv;

    std::vector<float> xp(x.size());
    if (!x.empty()) {
        xp[0] = x[0];
        for (size_t i = 1; i < x.size(); ++i) xp[i] = x[i] - p.preemph * x[i - 1];
    }

    const int n_samples = (int) xp.size();
    const int seq_len    = (n_samples + p.hop - 1) / p.hop;

    const int pad = p.n_fft / 2;
    std::vector<float> padded((size_t) n_samples + 2 * pad, 0.0f);
    for (int i = 0; i < pad; ++i) {
        int src = pad - i;
        src = std::min(std::max(src, 0), n_samples > 0 ? n_samples - 1 : 0);
        padded[i] = n_samples > 0 ? xp[src] : 0.0f;
    }
    for (int i = 0; i < n_samples; ++i) padded[pad + i] = xp[i];
    for (int i = 0; i < pad; ++i) {
        int src = n_samples - 2 - i;
        src = std::min(std::max(src, 0), n_samples > 0 ? n_samples - 1 : 0);
        padded[(size_t) pad + n_samples + i] = n_samples > 0 ? xp[src] : 0.0f;
    }

    const int F = 1 + n_samples / p.hop;

    std::vector<float> eff_win((size_t) p.n_fft, 0.0f);
    const int win_off = (p.n_fft - p.win) / 2;
    for (int i = 0; i < p.win && i < (int) window.size(); ++i) eff_win[win_off + i] = window[i];

    const int n_freq = p.n_fft / 2 + 1;
    std::vector<float> logmel((size_t) p.n_mels * F, 0.0f);
    std::vector<std::complex<float>> buf((size_t) p.n_fft);
    std::vector<float> power((size_t) n_freq);

    for (int t = 0; t < F; ++t) {
        int start = t * p.hop;
        for (int i = 0; i < p.n_fft; ++i) {
            size_t idx = (size_t) start + i;
            float  s   = idx < padded.size() ? padded[idx] : 0.0f;
            buf[i] = std::complex<float>(s * eff_win[i], 0.0f);
        }
        fft_pow2(buf);
        for (int f = 0; f < n_freq; ++f) {
            power[f] = buf[f].real() * buf[f].real() + buf[f].imag() * buf[f].imag();
        }
        for (int m = 0; m < p.n_mels; ++m) {
            const float * row = &fb[(size_t) m * n_freq];
            double        acc = 0.0;
            for (int f = 0; f < n_freq; ++f) acc += (double) row[f] * (double) power[f];
            logmel[(size_t) m * F + t] = (float) std::log(acc + (double) p.log_guard);
        }
    }

    for (int m = 0; m < p.n_mels; ++m) {
        float * row    = &logmel[(size_t) m * F];
        int     nvalid = std::min(seq_len, F);
        double  mean   = 0.0;
        for (int t = 0; t < nvalid; ++t) mean += row[t];
        mean /= std::max(1, nvalid);
        double var = 0.0;
        for (int t = 0; t < nvalid; ++t) { double d = row[t] - mean; var += d * d; }
        double sd = (nvalid > 1) ? std::sqrt(var / (nvalid - 1)) : 0.0;
        sd += 1e-5;
        for (int t = 0; t < F; ++t) row[t] = (float) ((row[t] - mean) / sd);
    }

    for (int m = 0; m < p.n_mels; ++m) {
        float * row = &logmel[(size_t) m * F];
        for (int t = seq_len; t < F; ++t) row[t] = 0.0f;
    }
    const int Tp = ((F + 15) / 16) * 16;
    std::vector<float> out((size_t) p.n_mels * Tp, 0.0f);
    for (int m = 0; m < p.n_mels; ++m) {
        std::memcpy(&out[(size_t) m * Tp], &logmel[(size_t) m * F], sizeof(float) * (size_t) F);
    }
    T_out       = Tp;
    seq_len_out = seq_len;
    return out;
}

// ============================================================ FSQ (fsq.py, levels=[8,8,8,8])
void dump_stage_f32(const std::string & dump_dir, FILE * dump_index, const char * name, const ggml_tensor * t) {
    std::string path = dump_dir + "/" + name + ".f32";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "dynin_units: dump: cannot open %s\n", path.c_str()); return; }
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const char * base = (const char *) t->data;
    for (int64_t i3 = 0; i3 < ne3; ++i3)
        for (int64_t i2 = 0; i2 < ne2; ++i2)
            for (int64_t i1 = 0; i1 < ne1; ++i1)
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const float * p = (const float *) (base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
                    std::fwrite(p, sizeof(float), 1, f);
                }
    std::fclose(f);
    fprintf(dump_index, "%s %lld %lld %lld %lld\n", name, (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3);
}

void dump_stage_f32_raw(const std::string & dump_dir, FILE * dump_index, const char * name, const int64_t ne[4], const float * data) {
    std::string path = dump_dir + "/" + name + ".f32";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "dynin_units: dump: cannot open %s\n", path.c_str()); return; }
    size_t count = (size_t) (ne[0] * ne[1] * ne[2] * ne[3]);
    std::fwrite(data, sizeof(float), count, f);
    std::fclose(f);
    fprintf(dump_index, "%s %lld %lld %lld %lld\n", name, (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3]);
}

void dump_stage_i32(const std::string & dump_dir, FILE * dump_index, const char * name, const std::vector<int32_t> & v) {
    std::string path = dump_dir + "/" + name + ".i32";
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "dynin_units: dump: cannot open %s\n", path.c_str()); return; }
    if (!v.empty()) std::fwrite(v.data(), sizeof(int32_t), v.size(), f);
    std::fclose(f);
    fprintf(dump_index, "%s %lld 1 1 1\n", name, (long long) v.size());
}

int32_t fsq_quantize_to_unit_id(const float z[4]) {
    static const int levels[4] = { 8, 8, 8, 8 };
    static const int basis[4]  = { 1, 8, 64, 512 };
    const float       eps      = 1e-3f;
    int64_t           idx      = 0;
    for (int d = 0; d < 4; ++d) {
        float L        = (float) levels[d];
        float half_l   = (L - 1.0f) * (1.0f - eps) / 2.0f;
        float offset   = (levels[d] % 2 == 1) ? 0.0f : 0.5f;
        float shift    = std::atan(offset / half_l);
        float bound    = std::tanh(z[d] + shift) * half_l - offset;
        float quantized = std::round(bound);
        float half_width = (float) (levels[d] / 2);
        int64_t scaled = (int64_t) std::llround((double) quantized + (double) half_width);
        idx += scaled * basis[d];
    }
    return (int32_t) idx;
}

bool s2u_cuda_enabled() {
    const char * e = std::getenv("DYNIN_S2U_BACKEND");
    if (e && *e) {
        std::string s(e);
        if (s == "cpu" || s == "CPU" || s == "0") return false;
    }
    return true; // default cuda
}

} // namespace

// ============================================================ model
struct dynin_units_model {
    struct gguf_context * gguf = nullptr;
    struct ggml_context * wctx = nullptr; // owns all loaded weight tensor data

    int   n_mels = 128, n_fft = 512, hop = 160, win = 320;
    float preemph = 0.97f, log_guard = 5.960464477539063e-08f, ln_eps = 1e-5f;

    int s1_dim = 512, s1_layer = 2, s1_head = 8, s1_ffn = 2048, s1_pck = 128, s1_pcg = 16;
    int s2_dim = 768, s2_layer = 10, s2_head = 12, s2_ffn = 3072, s2_pck = 128, s2_pcg = 16;

    std::vector<float> mel_fb;     // [n_mels * (n_fft/2+1)]
    std::vector<float> mel_window; // [win]

    // CUDA backend and weight map
    ggml_backend_t                                 cuda_backend         = nullptr;
    ggml_backend_t                                 cuda_cpu_backend     = nullptr;
    ggml_backend_sched_t                           cuda_sched           = nullptr;
    ggml_context                                 * cuda_wctx            = nullptr;
    std::unordered_map<std::string, ggml_tensor *> cuda_weight_map;
    bool                                           cuda_init_attempted  = false;
    bool                                           cuda_init_ok         = false;

    ~dynin_units_model() {
        if (gguf) gguf_free(gguf);
        if (wctx) ggml_free(wctx);
        if (cuda_wctx) ggml_free(cuda_wctx);
        if (cuda_sched) ggml_backend_sched_free(cuda_sched);
        if (cuda_backend) ggml_backend_free(cuda_backend);
        if (cuda_cpu_backend) ggml_backend_free(cuda_cpu_backend);
    }
};

static bool ensure_cuda_units_weights(dynin_units_model * m, std::string & err) {
    if (m->cuda_init_attempted) return m->cuda_init_ok;
    m->cuda_init_attempted = true;

#ifndef GGML_USE_CUDA
    // Non-CUDA build: ggml_backend_cuda_init() has no implementation linked
    // in, so it cannot be called at all. Callers already branch on this
    // function's bool return and fall back to the CPU path when it is
    // false, so this is a normal (not exceptional) outcome here.
    err = "built without GGML_USE_CUDA";
    return false;
#else
    m->cuda_backend = ggml_backend_cuda_init(0);
    if (!m->cuda_backend) { err = "ggml_backend_cuda_init(0) returned null"; return false; }

    m->cuda_cpu_backend = ggml_backend_cpu_init();
    if (!m->cuda_cpu_backend) { err = "ggml_backend_cpu_init() returned null"; return false; }

    ggml_backend_t sched_backends[2] = { m->cuda_backend, m->cuda_cpu_backend };
    m->cuda_sched = ggml_backend_sched_new(sched_backends, nullptr, 2, 8192, false, false);
    if (!m->cuda_sched) { err = "ggml_backend_sched_new() returned null"; return false; }

    int n_tensors = gguf_get_n_tensors(m->gguf);
    ggml_init_params ip = { (size_t) n_tensors * ggml_tensor_overhead() + 65536, nullptr, /*no_alloc=*/ true };
    m->cuda_wctx = ggml_init(ip);
    if (!m->cuda_wctx) { err = "ggml_init(cuda_wctx) failed"; return false; }

    std::vector<ggml_tensor *> host_tensors;
    host_tensors.reserve(n_tensors);
    std::vector<std::string> tensor_names;
    tensor_names.reserve(n_tensors);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(m->gguf, i);
        ggml_tensor * host_t = ggml_get_tensor(m->wctx, name);
        if (!host_t) continue;
        ggml_tensor * gpu_t = ggml_new_tensor(m->cuda_wctx, host_t->type, GGML_MAX_DIMS, host_t->ne);
        ggml_set_name(gpu_t, name);
        m->cuda_weight_map[name] = gpu_t;
        host_tensors.push_back(host_t);
        tensor_names.push_back(name);
    }

    if (!ggml_backend_alloc_ctx_tensors(m->cuda_wctx, m->cuda_backend)) {
        err = "ggml_backend_alloc_ctx_tensors(cuda_wctx) failed";
        return false;
    }

    for (size_t i = 0; i < tensor_names.size(); ++i) {
        ggml_tensor * gpu_t  = m->cuda_weight_map[tensor_names[i]];
        ggml_tensor * host_t = host_tensors[i];
        ggml_backend_tensor_set(gpu_t, host_t->data, 0, ggml_nbytes(host_t));
    }

    m->cuda_init_ok = true;
    return true;
#endif  // GGML_USE_CUDA
}

static ggml_tensor * W(dynin_units_model * m, const std::string & name, bool use_cuda = false) {
    if (use_cuda) {
        auto it = m->cuda_weight_map.find(name);
        if (it != m->cuda_weight_map.end()) return it->second;
    }
    ggml_tensor * t = ggml_get_tensor(m->wctx, name.c_str());
    if (!t) {
        fprintf(stderr, "dynin_units: missing tensor '%s' in GGUF\n", name.c_str());
        std::abort();
    }
    return t;
}

dynin_units_model * dynin_units_load(const char * gguf_path, std::string & err) {
    auto * m = new dynin_units_model();

    struct gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &m->wctx };
    m->gguf = gguf_init_from_file(gguf_path, params);
    if (!m->gguf) {
        err = std::string("gguf_init_from_file failed for ") + gguf_path;
        delete m;
        return nullptr;
    }

    auto get_u32 = [&](const char * key, int defv) -> int {
        int64_t i = gguf_find_key(m->gguf, key);
        return i < 0 ? defv : (int) gguf_get_val_u32(m->gguf, i);
    };
    auto get_f32 = [&](const char * key, float defv) -> float {
        int64_t i = gguf_find_key(m->gguf, key);
        return i < 0 ? defv : gguf_get_val_f32(m->gguf, i);
    };

    m->n_mels   = get_u32("emova.n_mels", 128);
    m->n_fft    = get_u32("emova.n_fft", 512);
    m->hop      = get_u32("emova.hop_length", 160);
    m->win      = get_u32("emova.win_length", 320);
    m->preemph  = get_f32("emova.preemph", 0.97f);
    m->log_guard = get_f32("emova.log_zero_guard", 5.960464477539063e-08f);
    m->ln_eps   = get_f32("emova.ln_eps", 1e-5f);

    m->s1_dim = get_u32("emova.s1.dim", 512);
    m->s1_layer = get_u32("emova.s1.n_layer", 2);
    m->s1_head = get_u32("emova.s1.n_head", 8);
    m->s1_ffn  = get_u32("emova.s1.ffn", 2048);
    m->s1_pck  = get_u32("emova.s1.pos_conv_kernel", 128);
    m->s1_pcg  = get_u32("emova.s1.pos_conv_groups", 16);

    m->s2_dim = get_u32("emova.s2.dim", 768);
    m->s2_layer = get_u32("emova.s2.n_layer", 10);
    m->s2_head = get_u32("emova.s2.n_head", 12);
    m->s2_ffn  = get_u32("emova.s2.ffn", 3072);
    m->s2_pck  = get_u32("emova.s2.pos_conv_kernel", 128);
    m->s2_pcg  = get_u32("emova.s2.pos_conv_groups", 16);

    ggml_tensor * fb = W(m, "mel.fb");     // ne = [n_freq, n_mels]
    ggml_tensor * wi = W(m, "mel.window"); // ne = [win]
    int n_freq = m->n_fft / 2 + 1;
    m->mel_fb.assign((float *) fb->data, (float *) fb->data + (size_t) n_freq * m->n_mels);
    m->mel_window.assign((float *) wi->data, (float *) wi->data + m->win);

    fprintf(stderr, "dynin_units: loaded %s (n_mels=%d n_fft=%d hop=%d win=%d, s1 dim=%d layers=%d, s2 dim=%d layers=%d)\n",
            gguf_path, m->n_mels, m->n_fft, m->hop, m->win, m->s1_dim, m->s1_layer, m->s2_dim, m->s2_layer);

    if (s2u_cuda_enabled()) {
        std::string cerr;
        if (!ensure_cuda_units_weights(m, cerr)) {
            fprintf(stderr, "dynin_units: CUDA init failed (%s), will use CPU\n", cerr.c_str());
        }
    }

    return m;
}

void dynin_units_free(dynin_units_model * m) { delete m; }

// ============================================================ ggml graph pieces

static ggml_tensor * mask_time_ne0(ggml_context * ctx, ggml_tensor * x, int64_t valid_len) {
    const int64_t T = x->ne[0];
    if (valid_len >= T) return x;
    if (valid_len <= 0) return ggml_scale(ctx, x, 0.0f);
    ggml_tensor * keep = ggml_cont(ctx, ggml_view_3d(ctx, x, valid_len, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0));
    ggml_tensor * pad  = ggml_cont(ctx, ggml_view_3d(ctx, x, T - valid_len, x->ne[1], x->ne[2], x->nb[1], x->nb[2],
                                                      (size_t) valid_len * x->nb[0]));
    pad = ggml_scale(ctx, pad, 0.0f);
    return ggml_concat(ctx, keep, pad, 0);
}

static ggml_tensor * mask_time_ne1(ggml_context * ctx, ggml_tensor * x, int64_t valid_len) {
    const int64_t D = x->ne[0];
    const int64_t T = x->ne[1];
    if (valid_len >= T) return x;
    if (valid_len <= 0) return ggml_scale(ctx, x, 0.0f);
    ggml_tensor * keep = ggml_cont(ctx, ggml_view_3d(ctx, x, D, valid_len, x->ne[2], x->nb[1], x->nb[2], 0));
    ggml_tensor * pad  = ggml_cont(ctx, ggml_view_3d(ctx, x, D, T - valid_len, x->ne[2], x->nb[1], x->nb[2],
                                                      (size_t) valid_len * x->nb[1]));
    pad = ggml_scale(ctx, pad, 0.0f);
    return ggml_concat(ctx, keep, pad, 1);
}

static ggml_tensor * make_key_pad_mask(ggml_context * ctx, int64_t T, int64_t valid_len, std::vector<float> & mask_data) {
    mask_data.resize((size_t) (T * T));
    for (int64_t q = 0; q < T; ++q) {
        for (int64_t k = 0; k < T; ++k) {
            mask_data[(size_t) (q * T + k)] = (k < valid_len) ? 0.0f : -INFINITY;
        }
    }
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    return mask;
}

static ggml_tensor * conv_1d_act(ggml_context * ctx, ggml_tensor * w16, ggml_tensor * xp, int stride, bool use_cuda) {
    if (use_cuda) {
        ggml_tensor * im2col = ggml_im2col(ctx, w16, xp, stride, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
        ggml_tensor * mm = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
                                         ggml_reshape_2d(ctx, w16, w16->ne[0] * w16->ne[1], w16->ne[2]));
        ggml_mul_mat_set_prec(mm, GGML_PREC_F32);
        return ggml_reshape_3d(ctx, mm, im2col->ne[1], w16->ne[2], im2col->ne[2]);
    } else {
        return ggml_conv_1d(ctx, w16, xp, stride, 0, 1);
    }
}

static ggml_tensor * conv_norm_act(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                                   ggml_tensor * ln_w, ggml_tensor * ln_b, int stride, int pad_l, int pad_r,
                                   bool relu, float ln_eps, bool use_cuda) {
    ggml_tensor * xp = x;
    if (pad_l != 0 || pad_r != 0) xp = ggml_pad_ext(ctx, x, pad_l, pad_r, 0, 0, 0, 0, 0, 0);
    ggml_tensor * w16 = ggml_cast(ctx, w, GGML_TYPE_F16);
    ggml_tensor * c  = conv_1d_act(ctx, w16, xp, stride, use_cuda); // [T_out, Cout, 1]
    ggml_tensor * bb = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
    c = ggml_add(ctx, c, bb);

    ggml_tensor * ct = ggml_cont(ctx, ggml_transpose(ctx, c)); // [Cout, T_out, 1]
    ct = ggml_norm(ctx, ct, ln_eps);
    ct = ggml_add(ctx, ggml_mul(ctx, ct, ln_w), ln_b);
    ggml_tensor * co = ggml_cont(ctx, ggml_transpose(ctx, ct)); // [T_out, Cout, 1]
    if (relu) co = ggml_relu(ctx, co);
    return co;
}

static ggml_tensor * pos_conv_residual(ggml_context * ctx, ggml_tensor * x_ct /* [C, T] */,
                                       ggml_tensor * w /* [K, Cin_g, Cout] */, ggml_tensor * b /* [Cout] */,
                                       int groups, int kernel, bool use_cuda) {
    const int64_t C  = x_ct->ne[0];
    const int64_t T  = x_ct->ne[1];
    const int64_t Cg = C / groups;
    const int     pad = kernel / 2;

    ggml_tensor * x_cm = ggml_cont(ctx, ggml_transpose(ctx, x_ct)); // [T, C]
    x_cm = ggml_reshape_3d(ctx, x_cm, T, C, 1);
    ggml_tensor * xp = ggml_pad_ext(ctx, x_cm, pad, pad, 0, 0, 0, 0, 0, 0); // [T+2*pad, C, 1]

    ggml_tensor * acc = nullptr;
    for (int64_t g = 0; g < groups; ++g) {
        ggml_tensor * wg = ggml_cont(ctx, ggml_view_3d(ctx, w, w->ne[0], w->ne[1], Cg,
                                                        w->nb[1], w->nb[2], (size_t) g * Cg * w->nb[2]));
        wg = ggml_cast(ctx, wg, GGML_TYPE_F16);
        ggml_tensor * xg = ggml_cont(ctx, ggml_view_3d(ctx, xp, xp->ne[0], Cg, 1,
                                                        xp->nb[1], xp->nb[2], (size_t) g * Cg * xp->nb[1]));
        ggml_tensor * og = conv_1d_act(ctx, wg, xg, 1, use_cuda); // [T+1, Cg, 1]
        acc = acc ? ggml_concat(ctx, acc, og, 1) : og;
    }
    ggml_tensor * bb = ggml_reshape_3d(ctx, b, 1, C, 1);
    acc = ggml_add(ctx, acc, bb); // [T+1, C, 1]

    ggml_tensor * trimmed = ggml_cont(ctx, ggml_view_3d(ctx, acc, T, C, 1, acc->nb[1], acc->nb[2], 0));
    ggml_tensor * gel = ggml_gelu_erf(ctx, trimmed); // [T, C, 1]
    ggml_tensor * g_ct = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, gel, T, C))); // [C, T]
    return ggml_add(ctx, x_ct, g_ct);
}

static ggml_tensor * transformer_stage(ggml_context * ctx, dynin_units_model * m, ggml_tensor * x_ct,
                                       const std::string & prefix, int n_layer, int n_head, int ffn,
                                       int pos_kernel, int pos_groups, ggml_tensor * kpad_mask, bool use_cuda) {
    const int64_t D  = x_ct->ne[0];
    const int64_t T  = x_ct->ne[1];
    const int64_t hd = D / n_head;
    const float   scale = 1.0f / std::sqrt((float) hd);

    ggml_tensor * cur = pos_conv_residual(ctx, x_ct, W(m, prefix + ".pos_conv.weight", use_cuda),
                                          W(m, prefix + ".pos_conv.bias", use_cuda),
                                          pos_groups, pos_kernel, use_cuda);

    for (int li = 0; li < n_layer; ++li) {
        std::string lp = prefix + ".layer" + std::to_string(li);

        ggml_tensor * residual = cur;
        ggml_tensor * h = ggml_norm(ctx, cur, m->ln_eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, W(m, lp + ".attn_ln.weight", use_cuda)), W(m, lp + ".attn_ln.bias", use_cuda));

        ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat(ctx, W(m, lp + ".q.weight", use_cuda), h), W(m, lp + ".q.bias", use_cuda));
        ggml_tensor * K = ggml_add(ctx, ggml_mul_mat(ctx, W(m, lp + ".k.weight", use_cuda), h), W(m, lp + ".k.bias", use_cuda));
        ggml_tensor * V = ggml_add(ctx, ggml_mul_mat(ctx, W(m, lp + ".v.weight", use_cuda), h), W(m, lp + ".v.bias", use_cuda));

        ggml_tensor * Qr = ggml_permute(ctx, ggml_reshape_3d(ctx, Q, hd, n_head, T), 0, 2, 1, 3); // [hd,T,nh]
        ggml_tensor * Kr = ggml_permute(ctx, ggml_reshape_3d(ctx, K, hd, n_head, T), 0, 2, 1, 3); // [hd,T,nh]
        ggml_tensor * KQ = ggml_mul_mat(ctx, Kr, Qr);                                             // [T,T,nh]
        ggml_tensor * KQsm = ggml_soft_max_ext(ctx, KQ, kpad_mask, scale, 0.0f);
        ggml_tensor * Vr = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, V, hd, n_head, T), 1, 2, 0, 3)); // [T,hd,nh]
        ggml_tensor * KQV = ggml_mul_mat(ctx, Vr, KQsm);                                          // [hd,T,nh]
        ggml_tensor * merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);                                 // [hd,nh,T]
        ggml_tensor * attn = ggml_cont_2d(ctx, merged, D, T);                                      // [D,T]
        attn = ggml_add(ctx, ggml_mul_mat(ctx, W(m, lp + ".out.weight", use_cuda), attn), W(m, lp + ".out.bias", use_cuda));

        cur = ggml_add(ctx, residual, attn);

        ggml_tensor * residual2 = cur;
        ggml_tensor * h2 = ggml_norm(ctx, cur, m->ln_eps);
        h2 = ggml_add(ctx, ggml_mul(ctx, h2, W(m, lp + ".final_ln.weight", use_cuda)), W(m, lp + ".final_ln.bias", use_cuda));
        ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, W(m, lp + ".fc1.weight", use_cuda), h2), W(m, lp + ".fc1.bias", use_cuda));
        ff = ggml_gelu_erf(ctx, ff);
        ff = ggml_add(ctx, ggml_mul_mat(ctx, W(m, lp + ".fc2.weight", use_cuda), ff), W(m, lp + ".fc2.bias", use_cuda));
        cur = ggml_add(ctx, residual2, ff);
    }

    ggml_tensor * out = ggml_norm(ctx, cur, m->ln_eps);
    out = ggml_add(ctx, ggml_mul(ctx, out, W(m, prefix + ".final_ln.weight", use_cuda)), W(m, prefix + ".final_ln.bias", use_cuda));
    return out; // [D, T]
}

// ============================================================ top-level encode

bool dynin_units_encode_wav(dynin_units_model * m, const std::string & wav_path, dynin_units_result & out,
                              std::string & err, const char * dump_dir) {
    WavAudio wav;
    if (!read_wav_pcm(wav_path, wav, err)) return false;
    resample_to_16k(wav.samples, wav.sample_rate);
    const float seconds = wav.samples.empty() ? 0.0f : (float) wav.samples.size() / 16000.0f;

    MelHparams mp;
    mp.n_mels = m->n_mels; mp.n_fft = m->n_fft; mp.hop = m->hop; mp.win = m->win;
    mp.preemph = m->preemph; mp.log_guard = m->log_guard;

    int T0 = 0, seq_len = 0;
    std::vector<float> logmel = compute_log_mel(wav.samples, mp, m->mel_fb, m->mel_window, T0, seq_len);
    if (T0 <= 0) { err = "empty mel output for " + wav_path; return false; }

    bool use_cuda = s2u_cuda_enabled();
    if (use_cuda) {
        std::string cerr;
        if (!ensure_cuda_units_weights(m, cerr)) {
            fprintf(stderr, "dynin_units: CUDA init failed (%s), falling back to CPU\n", cerr.c_str());
            use_cuda = false;
        }
    }

    std::vector<uint8_t> mem;
    ggml_context * ctx = nullptr;
    if (use_cuda) {
        struct ggml_init_params gip = { /*mem_size=*/ 64u * 1024u * 1024u, /*mem_buffer=*/ nullptr, /*no_alloc=*/ true };
        ctx = ggml_init(gip);
    } else {
        const size_t buf_size = (size_t) 1280 * 1024 * 1024;
        mem.resize(buf_size);
        struct ggml_init_params gip = { buf_size, mem.data(), /*no_alloc=*/ false };
        ctx = ggml_init(gip);
    }
    if (!ctx) { err = "ggml_init failed"; return false; }

    ggml_tensor * mel_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T0, m->n_mels, 1);
    ggml_set_name(mel_in, "mel_in");
    if (use_cuda) {
        ggml_set_input(mel_in);
    } else {
        std::memcpy(mel_in->data, logmel.data(), sizeof(float) * logmel.size());
    }

    const int64_t valid_a = (seq_len + 1) / 2;
    const int64_t valid_b = (valid_a + 1) / 2;

    // ---- stage 1: 3 conv layers --------------------------------------------
    ggml_tensor * c = conv_norm_act(ctx, mel_in, W(m, "s1.conv0.weight", use_cuda), W(m, "s1.conv0.bias", use_cuda),
                                    W(m, "s1.norm0.weight", use_cuda), W(m, "s1.norm0.bias", use_cuda), 2, 1, 2, true, m->ln_eps, use_cuda);
    c = mask_time_ne0(ctx, c, valid_a);
    c = conv_norm_act(ctx, c, W(m, "s1.conv1.weight", use_cuda), W(m, "s1.conv1.bias", use_cuda),
                      W(m, "s1.norm1.weight", use_cuda), W(m, "s1.norm1.bias", use_cuda), 2, 1, 2, true, m->ln_eps, use_cuda);
    c = conv_norm_act(ctx, c, W(m, "s1.conv2.weight", use_cuda), W(m, "s1.conv2.bias", use_cuda),
                      W(m, "s1.norm2.weight", use_cuda), W(m, "s1.norm2.bias", use_cuda), 1, 0, 0, false, m->ln_eps, use_cuda);
    ggml_tensor * s1_conv_out = c;

    std::vector<float> mask_data;
    ggml_tensor * kpad = make_key_pad_mask(ctx, s1_conv_out->ne[0], valid_b, mask_data);
    ggml_set_name(kpad, "kpad_mask");
    if (use_cuda) {
        ggml_set_input(kpad);
    } else {
        std::memcpy(kpad->data, mask_data.data(), sizeof(float) * mask_data.size());
    }

    // ---- stage 1 transformer -----------------------------------------------
    ggml_tensor * x1 = ggml_cont(ctx, ggml_transpose(ctx, c)); // [512, T1]
    x1 = transformer_stage(ctx, m, x1, "s1", m->s1_layer, m->s1_head, m->s1_ffn, m->s1_pck, m->s1_pcg, kpad, use_cuda);
    ggml_tensor * x1_masked = mask_time_ne1(ctx, x1, valid_b);

    // ---- stage 2: 2 conv layers --------------------------------------------
    const int64_t T1 = x1->ne[1];
    ggml_tensor * c2in = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, x1_masked)), T1, m->s1_dim, 1);
    ggml_tensor * c2 = conv_norm_act(ctx, c2in, W(m, "s2.conv0.weight", use_cuda), W(m, "s2.conv0.bias", use_cuda),
                                     W(m, "s2.norm0.weight", use_cuda), W(m, "s2.norm0.bias", use_cuda), 1, 2, 2, true, m->ln_eps, use_cuda);
    c2 = conv_norm_act(ctx, c2, W(m, "s2.conv1.weight", use_cuda), W(m, "s2.conv1.bias", use_cuda),
                       W(m, "s2.norm1.weight", use_cuda), W(m, "s2.norm1.bias", use_cuda), 1, 0, 0, false, m->ln_eps, use_cuda);
    ggml_tensor * s2_conv_out = c2;

    // ---- stage 2 transformer -----------------------------------------------
    ggml_tensor * x2 = ggml_cont(ctx, ggml_transpose(ctx, c2)); // [768, T1]
    x2 = transformer_stage(ctx, m, x2, "s2", m->s2_layer, m->s2_head, m->s2_ffn, m->s2_pck, m->s2_pcg, kpad, use_cuda);

    // ---- pre_quant: Linear(768 -> 4) ---------------------------------------
    ggml_tensor * pq = ggml_add(ctx, ggml_mul_mat(ctx, W(m, "pre_quant.weight", use_cuda), x2), W(m, "pre_quant.bias", use_cuda));
    ggml_set_name(pq, "pre_quant");
    ggml_set_output(pq);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, pq);

    const auto t_enc_start = std::chrono::high_resolution_clock::now();

    if (use_cuda) {
        ggml_backend_sched_reset(m->cuda_sched);
        if (!ggml_backend_sched_alloc_graph(m->cuda_sched, gf)) {
            err = "ggml_backend_sched_alloc_graph failed";
            ggml_free(ctx);
            return false;
        }

        ggml_backend_tensor_set(mel_in, logmel.data(), 0, logmel.size() * sizeof(float));
        ggml_backend_tensor_set(kpad, mask_data.data(), 0, mask_data.size() * sizeof(float));

        enum ggml_status st = ggml_backend_sched_graph_compute(m->cuda_sched, gf);
        if (st != GGML_STATUS_SUCCESS) {
            err = "ggml_backend_sched_graph_compute failed, status=" + std::to_string((int) st);
            ggml_free(ctx);
            return false;
        }
    } else {
        int n_threads = (int) std::max(1u, std::thread::hardware_concurrency());
        enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        if (st != GGML_STATUS_SUCCESS) {
            err = "ggml_graph_compute_with_ctx failed, status=" + std::to_string((int) st);
            ggml_free(ctx);
            return false;
        }
    }

    const double enc_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_enc_start).count();
    fprintf(stderr, "[DYNIN-S2U backend=%s enc_ms=%.1f]\n", use_cuda ? "cuda" : "cpu", enc_ms);
    fflush(stderr);

    const int64_t Tf = pq->ne[1];
    const int64_t valid_out = std::min(valid_b, Tf);
    std::vector<float> pq_host;
    const float * pqd = nullptr;
    if (use_cuda) {
        pq_host.resize((size_t) ggml_nelements(pq));
        ggml_backend_tensor_get(pq, pq_host.data(), 0, pq_host.size() * sizeof(float));
        pqd = pq_host.data();
    } else {
        pqd = (const float *) pq->data;
    }

    out.unit_ids.resize((size_t) valid_out);
    for (int64_t t = 0; t < valid_out; ++t) {
        float z[4] = { pqd[t * 4 + 0], pqd[t * 4 + 1], pqd[t * 4 + 2], pqd[t * 4 + 3] };
        out.unit_ids[(size_t) t] = fsq_quantize_to_unit_id(z);
    }
    out.seconds = seconds;
    out.rate_hz = seconds > 0.0f ? (float) valid_out / seconds : 0.0f;

    if (dump_dir) {
        std::string dump_index_path = std::string(dump_dir) + "/dump_index.txt";
        FILE * dump_index = std::fopen(dump_index_path.c_str(), "wb");
        if (!dump_index) {
            fprintf(stderr, "dynin_units: dump: cannot open %s\n", dump_index_path.c_str());
        } else {
            auto dump_tensor = [&](const char * name, ggml_tensor * t) {
                if (use_cuda) {
                    std::vector<float> h((size_t) ggml_nelements(t));
                    ggml_backend_tensor_get(t, h.data(), 0, h.size() * sizeof(float));
                    dump_stage_f32_raw(dump_dir, dump_index, name, t->ne, h.data());
                } else {
                    dump_stage_f32(dump_dir, dump_index, name, t);
                }
            };
            dump_tensor("mel_in", mel_in);
            dump_tensor("s1_conv", s1_conv_out);
            dump_tensor("s1_tf", x1);
            dump_tensor("s2_conv", s2_conv_out);
            dump_tensor("s2_tf", x2);
            dump_tensor("pre_quant", pq);
            dump_stage_i32(dump_dir, dump_index, "unit_ids", out.unit_ids);
            std::fclose(dump_index);
        }
    }

    ggml_free(ctx);
    return true;
}
