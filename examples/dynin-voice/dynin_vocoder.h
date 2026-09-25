// dynin_vocoder.h - EMOVA U2S (unit-to-speech) VITS/HiFi-GAN decoder ported
// to ggml.
//
// Attribution: this header and dynin_vocoder.cpp are a C++/ggml translation
// of the UVITS decoder in the EMOVA speech tokenizer
// (https://github.com/emova-ollm/EMOVA_speech_tokenizer, the EMOVA authors,
// arXiv:2409.18042), which is licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Changes relative to that
// source: the Python/PyTorch modules were re-expressed in C++ and ggml; the
// computation is meant to be the same. The underlying design is VITS (Kim et
// al., arXiv:2106.06103, MIT) with a HiFi-GAN generator (Kong et al.,
// arXiv:2010.05646, MIT). Full attribution is in the NOTICE file of the
// repository this example ships with.
//
// The GGUF this loads is produced by tools/convert_emova_u2s_to_gguf.py, a
// one-off Python conversion script (not used at runtime).
//
// C++ only at runtime. Architecture: the HiFi-GAN generator (dec) is a real
// ggml graph (ggml_graph_compute_with_ctx) -- conv_pre/cond/4 upsample
// stages/12 MRF resblocks/conv_post/tanh, matching "all ops exist in the
// fork's ggml" (conv_1d, conv_transpose_1d, leaky_relu, tanh). enc_p, the
// stochastic duration predictor and the residual coupling flow are plain
// host C++ float math (a direct, line-by-line port of models.py/modules.py/
// attentions.py/transforms.py) rather than ggml graphs: their per-utterance
// sequence lengths are tiny (tens to a few hundred steps) and their control
// flow is either data-dependent (the rational-quadratic-spline search in the
// duration predictor, the predicted-duration-driven prior expansion) or
// involves the relative-position index gymnastics of attentions.py's
// MultiHeadAttention (_get_relative_embeddings et al) -- exactly the kind of
// thing ggml's static graph model and this fork's own ggml_conv_1d_dw
// ("very likely wrong for some cases", per its own header comment) make
// riskier to get silently wrong than a direct transliteration checked against
// the Python reference source (speech_tokenization/UVITS/ in the EMOVA
// repository).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct dynin_vocoder_model;

// Loads emova_u2s.gguf. Returns nullptr and fills `err` on failure.
dynin_vocoder_model * dynin_vocoder_load(const char * gguf_path, std::string & err);
void                 dynin_vocoder_free(dynin_vocoder_model * m);

int dynin_vocoder_n_styles(const dynin_vocoder_model * m);      // 126
int dynin_vocoder_n_speakers(const dynin_vocoder_model * m);    // 1344
int dynin_vocoder_style_dim(const dynin_vocoder_model * m);     // 256
int dynin_vocoder_inter_channels(const dynin_vocoder_model * m); // 192
int dynin_vocoder_sample_rate(const dynin_vocoder_model * m);   // 22050

struct dynin_vocoder_audio {
    std::vector<int16_t> pcm16;
    int                  sample_rate = 22050;
};

// Step 2 (generator-only smoke test): runs dec(z, g=style_embedding[style_index])
// directly on a caller-supplied latent z, ggml-graph only (conv_pre + cond +
// 4 upsamples + MRF + conv_post + tanh). z must have T*inter_channels()
// elements, ggml ne[] order (channel fastest changing per frame is NOT the
// layout -- z is [inter_channels, T] row-major, i.e. z[c*T + t]).
// style_index in [0, n_styles), or -1 for no style conditioning at all.
bool dynin_vocoder_generate_from_latent(dynin_vocoder_model * m, const std::vector<float> & z, int T,
                                        int style_index, dynin_vocoder_audio & out, std::string & err);

// Step 3 (full pipeline): enc_p -> stochastic duration predictor (reverse) ->
// prior expansion -> flow (reverse) -> dec. unit_ids are raw content-unit
// ids in [0, n_units=4096), one per ~40ms input frame (dynin_units_result's
// own unit_ids are directly usable here). style_index in [0, n_styles).
// seed drives every random draw (dp's flow noise, the prior's re-parameterised
// noise) so the same inputs always produce the same output.
bool dynin_vocoder_synthesize(dynin_vocoder_model * m, const std::vector<int32_t> & unit_ids, int style_index,
                              dynin_vocoder_audio & out, std::string & err, uint64_t seed = 1234);

// n_units_out receives the number of output audio frames T_y actually used
// (for the caller's own seconds/rate sanity check); may be null.
bool dynin_vocoder_synthesize_ex(dynin_vocoder_model * m, const std::vector<int32_t> & unit_ids, int style_index,
                                 dynin_vocoder_audio & out, std::string & err, uint64_t seed, int * t_y_out);

bool dynin_vocoder_write_wav(const std::string & path, const dynin_vocoder_audio & audio, std::string & err);
