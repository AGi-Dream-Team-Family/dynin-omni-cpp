// dynin_units.h - EMOVA S2U (speech-to-unit) encoder ported to ggml.
//
// Attribution: this file and dynin_units.cpp are a C++/ggml translation of
// the S2U speech tokenizer in the EMOVA speech tokenizer
// (https://github.com/emova-ollm/EMOVA_speech_tokenizer, the EMOVA authors,
// arXiv:2409.18042), under speech_tokenization/SPIRAL_L2_BN_FSQ_CTC/. That
// code is licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). It builds on SPIRAL
// (Huang et al., arXiv:2201.10207; source files marked "Copyright (C)
// 2021/2022 Huawei Technologies Co., Ltd.", Apache-2.0) inside a copy of
// NVIDIA NeMo (Apache-2.0), and its my_scripts/fsq.py is marked "Copyright
// 2023 Google LLC", Apache-2.0, adapted from the google-research FSQ
// notebook (Finite Scalar Quantization, Mentzer et al., arXiv:2309.15505).
// Changes relative to that source: the PyTorch mel frontend, convolution
// and transformer stages and the FSQ quantizer were re-expressed in C++ and
// ggml, and the weights are read from a GGUF; the computation is meant to
// be the same.
//
// Neither CLI in this directory calls this encoder (see CMakeLists.txt).
// The S2U GGUF it loads is not part of this release, and neither is the
// script that converts the S2U checkpoint to GGUF.
//
// CPU backend (ggml_graph_compute_with_ctx), with an optional CUDA path
// through ggml_backend_sched (see dynin_units.cpp).
//
// GELU note: hparams.yaml of the reference S2U model sets
// `activation_fn: gelu` (both stages), which resolves to PyTorch's `F.gelu`, whose
// default is the EXACT erf-based GELU, and `nn.GELU()` (pos_conv) defaults
// the same way -- neither is the tanh approximation. This port previously
// called `ggml_gelu` (this fork's ggml, ggml-cpu/vec.h: tanh-approx,
// `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`) at both GELU call sites
// (pos_conv_residual, transformer_stage's FFN). Switched to `ggml_gelu_erf`
// (present in this fork, ggml-cpu/ops.cpp: `0.5*x*(1+erf(x/sqrt(2)))`),
// which matches `F.gelu`/`nn.GELU()` exactly. The per-call error between the
// two forms is small (a few parts in 1e3), but it is systematic (not
// random rounding) and compounds through 12 residual transformer layers
// (2+10) feeding an FSQ quantizer with only 8 levels/dim -- exactly the
// kind of small, directional bias that can flip values across a
// quantization bin boundary while leaving rate/range/determinism checks
// (the structural checks this port already passed) untouched.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct dynin_units_model;

// Loads emova_s2u.gguf. Returns nullptr and fills `err` on failure.
dynin_units_model * dynin_units_load(const char * gguf_path, std::string & err);
void                 dynin_units_free(dynin_units_model * m);

struct dynin_units_result {
    std::vector<int32_t> unit_ids;   // one per ~40ms frame, each in [0, 4095]
    float                seconds  = 0.0f;  // input audio duration
    float                rate_hz  = 0.0f;  // units per second (should be ~25)
};

// wav_path: RIFF/WAVE, 16-bit PCM, any sample rate / channel count (resampled
// to 16 kHz mono in C++ if needed). Returns false and fills `err` on failure.
//
// dump_dir (optional diagnostic, used when comparing stage outputs with the
// reference implementation; no such tool ships here): when non-null, writes each major stage tensor as
// raw float32 (ggml ne[] order, i0 fastest) to "<dump_dir>/<stage>.f32", the
// unit ids as raw int32 to "<dump_dir>/unit_ids.i32", and a plain-text
// "<dump_dir>/dump_index.txt" (one line per stage: "name ne0 ne1 ne2 ne3").
// Stage names: mel_in, s1_conv, s1_tf, s2_conv, s2_tf, pre_quant, unit_ids.
bool dynin_units_encode_wav(dynin_units_model * m, const std::string & wav_path,
                              dynin_units_result & out, std::string & err,
                              const char * dump_dir = nullptr);
