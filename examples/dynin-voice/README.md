# examples/dynin-voice

Two standalone command-line tools built on this fork's masked-diffusion
speech support, plus the library files they share:

| File | What it is |
|---|---|
| `dynin_vocoder.h` / `.cpp` | EMOVA U2S (unit-to-speech) VITS/HiFi-GAN decoder: a ggml graph for the generator (`dec`), host C++ for the stochastic duration predictor, and host C++ with an auto-selected ggml CUDA graph for `enc_p` and the residual coupling flow. CPU and CUDA backends. A C++/ggml translation of the EMOVA speech tokenizer's UVITS decoder (Apache-2.0); see the attribution in the file header. |
| `dynin_units.h` / `.cpp` | EMOVA S2U (speech-to-unit) encoder: wav in, content-unit ids out. Not called by either CLI below (see the note in `CMakeLists.txt`); included for completeness and compiled as part of the `dynin-u2s` target so it is build- and link-checked. A C++/ggml translation of the EMOVA speech tokenizer's SPIRAL/FSQ S2U code (Apache-2.0); see the attribution in `dynin_units.h`. The S2U GGUF and its converter are not part of this release. |
| `dynin_vocab.h` | Finds the model's text/image/speech token-id boundary in the loaded GGUF vocabulary (the first token id whose piece is exactly `[PAD<id>]`) instead of reading `config.json`, whose padded figure is off by 92 for this checkpoint. Refuses to run unless the boundary equals the compiled-in expected value, 126372. |
| `dynin_ears.h` | A thin scoring-oracle wrapper: starts an external `whisper-cli` process (no whisper/ggml linkage; `CreateProcessW` on Windows, `popen()` on POSIX) to transcribe a wav file, purely for checking synthesised speech against the text it came from. Optional at run time for `dynin-t2s-kvblock` (omit the whisper flags); `dynin-u2s` does not use it. |
| `dynin-u2s.cpp` | CLI: content-unit ids in, wav out. |
| `dynin-t2s-kvblock.cpp` | CLI: text in, speech units out (rendered to wav with `--wav`), via a masked-diffusion decode with an optional windowed decode that reads the rest of the canvas from a KV cache. |

## Build

From the repository root, after configuring as in the top-level build
instructions (`cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release`):

```
cmake --build build-cuda --target dynin-u2s dynin-t2s-kvblock --config Release
```

Both targets also build as part of a full `examples` build. CUDA is
required for the GPU path: the vocoder's CUDA code is compiled only when
`GGML_USE_CUDA` is defined (ggml sets it when `GGML_CUDA=ON`), and without
it every vocoder stage runs on the CPU. Only the Windows / MSVC 2022 /
CUDA 13.1 build has been compiled and run.

## dynin-u2s

Renders a sequence of EMOVA content-unit ids to a wav file.

```
dynin-u2s -m <u2s.gguf> -i <units_file> -o <out.wav>
          [--style <index>] [--seed <n>] [--sr <hz>] [--cpu]
```

- `-m` the EMOVA U2S GGUF (see `tools/convert_emova_u2s_to_gguf.py` and its
  own README for how this is produced).
- `-i` the units file. Two formats are accepted:
  - **binary `.i32`**: a flat array of little-endian `int32` content-unit
    ids, no header. This is the same layout `dynin_units_encode_wav()`'s
    own diagnostic dump writes for its `unit_ids` stage (see
    `dump_index.txt` below).
  - **text** (any other extension): decimal integers separated by
    whitespace and/or commas, e.g. `12, 88 401\n7`. Parsing stops at the
    first non-integer token, and if no integer is found the file is re-read
    as raw `int32`.

  Every id must be in `[0, 4095]`.
- `-o` output wav path.
- `--style` style/speaker index in `[0, n_styles)`. Default 41, which is
  the `gender-female_emotion-neutral_speed-normal_pitch-normal` entry in
  the EMOVA speech tokenizer's `config.json` (`u2s_style2idx`). The tool
  prints `n_styles` on load so any other index can be checked against the
  range.
- `--seed` seeds every random draw the decoder makes. Default 1234. The
  same input and seed reproduce the same output for a given backend
  selection; set `DYNIN_U2S_BACKEND`, `DYNIN_U2S_CUDA_ENCP` and
  `DYNIN_U2S_CUDA_FLOW` to `cpu` or `cuda` to pin it.
- `--sr` overrides the sample-rate value **written into the wav header
  only**; it does not resample the audio. The decoder's native rate
  (usually 22050 Hz) is what the samples actually are; the tool prints a
  warning if the override does not match it.
- `--cpu` forces the CPU path of the generator graph (it sets
  `DYNIN_U2S_BACKEND=cpu`). `enc_p` and the flow still auto-select; also
  set `DYNIN_U2S_CUDA_ENCP=cpu` and `DYNIN_U2S_CUDA_FLOW=cpu` for an
  all-CPU run.

Backend auto-pick, when nothing is forced. On the first call in a process:

- `enc_p` and the flow each run on both backends; CUDA is kept if the
  cosine similarity to the host result is at least 0.9999 and CUDA was
  faster.
- The generator runs on both backends; CUDA is kept if the
  signal-to-error ratio of its output against the CPU output is at least
  40 dB and CUDA was faster. The decision and the measured ratio are
  printed as `[DYNIN-U2S-CUDA-CHECK ...]` and `[DYNIN-U2S-DEFAULT ...]`.
  This is a CPU-vs-CUDA self-consistency check, not a comparison with any
  reference decoder. In the smoke run for this release it measured
  57.15 dB and the generator ran on CUDA.

Printed numbers, `[DYNIN-U2S-OUT ...]`:
- `seconds` / `samples` / `sample_rate`: length of the rendered audio.
- `peak`: maximum absolute sample value, normalised to `[0, 1]`.
- `rms`: root-mean-square level over the whole clip, same normalisation.
- `t_y`: number of latent frames the duration predictor produced (each
  frame is 256 output samples, so `samples = t_y x 256`).

## dynin-t2s-kvblock

Text in, speech units out (rendered to wav with `--wav`), through a
masked-diffusion decode of Dynin-Omni-based speech-generation checkpoints.
It needs the `diffusion_prefix_cache` context flag and
`llama_set_attn_pad_token()` from the patch; see the top-level README for
what those engine changes do.

```
dynin-t2s-kvblock -m <dynin.gguf> --u2s <u2s.gguf> --text "<sentence>"
                  --out <dir>
                  [--steps N] [--block B] [--kv-margin M] [--cfg C] [--seed S]
                  [--wav] [--compare]
                  [--whisper-cli <exe> --whisper-model <bin>]
```

- `-m` the diffusion checkpoint GGUF. `--u2s` the EMOVA U2S vocoder GGUF
  (required on the command line, but only loaded if `--wav` is given).
- `--text` the sentence to speak.
- `--steps` the step budget, split as `floor(steps / n_blocks)` steps per
  block (383 at block 128 runs 3 x 127 = 381), and also the number of
  speech-unit positions generated. Default 383.
- `--block` block length for the block-local diffusion schedule. Default
  128.
- `--kv-margin` window margin `M` for the windowed decode. On each block's
  first step (step 0) the tool clears the KV cache and decodes the full
  canvas densely, leaving every position's K/V in the cache. Every later
  step in that block decodes only `[block_start - M, block_end + M)`,
  clamped to the canvas; positions outside the window are read from the
  cache as they were at the block's step 0. `M = 0` disables windowing:
  every step becomes a dense decode with the cache cleared before and
  after, the exact/reference path. If omitted, defaults to 32 when
  `--steps 48` and to 0 (dense) for every other step count.
- `--cfg` classifier-free-guidance scale. Default 2.5. The unconditional
  forward pass is decoded once per run and reused for every step (it is
  invariant: the whole speech region is re-masked before every read of
  it), not recomputed per step.
- `--seed` seeds the per-step sampling, and the vocoder when `--wav` is
  given. Default 1234.
- `--out` output directory; the last path component is created if missing
  (its parent must exist).
- `--wav` also renders the resulting unit sequence(s) to wav (`t2s.wav`, or
  `t2s_exact.wav` and `t2s_margin.wav` under `--compare`).
- `--compare` runs **two** passes on the same text and seed: `exact`
  (`--kv-margin` forced to 0) and `margin` (the `--kv-margin` value or its
  computed default), and writes `match.json` and `timing.json` into
  `--out` (see below). Without `--compare`, only the single run at
  `--kv-margin` happens.
- `--whisper-cli` / `--whisper-model` an optional scoring oracle: with
  `--wav`, transcribes the resulting wav file(s) (English only, `-l en`)
  and prints word overlap against `--text` (the fraction of the sentence's
  unique words, lower-cased and split on non-alphanumeric characters, that
  appear in the transcript). No default path is built in; omit either flag
  to skip transcription.

The prompt occupies a 1,024-position text block, left-padded with
`[iPAD]`. Pad positions are placed in sequence 1 and all other positions
in sequence 0, so the KV cache's per-sequence mask keeps pad keys out of
the real tokens' attention.

### match.json (only with `--compare`)

A real example (block 128, 48 steps, seed 42):

```json
{
  "text": "He hung his coat by the door and sat down.",
  "kv_margin": 32,
  "seed": 42,
  "lengths": { "exact": 47, "margin": 47 },
  "exact_match_fraction": 0.893617,
  "first_divergence_index": 9
}
```

- `text` is written without JSON escaping, so a `"` in `--text` produces
  invalid JSON.
- `lengths`: number of speech-unit ids each run produced (a run can stop
  early on an `<|eoa|>`/EOS token).
- `exact_match_fraction`: fraction of the overlapping positions (the
  first `min(lengths)` unit ids) where the two runs picked the identical
  unit id.
- `first_divergence_index`: index of the first unit id where the two
  runs disagree, or where one run's output ends before the other's; `-1`
  if the two sequences agree everywhere they overlap and are the same
  length.

### timing.json (only with `--compare`)

A real example (block 128, 96 steps, seed 42, one block; sentence 1 of
the fidelity bench, `receipts/raw/fidelity_2026-09-22/block128_steps96_margin32/s1/timing.json`
in the repository this example ships with):

```json
{
  "exact":  { "ms_per_block": [65615.295], "total_ms": 67548.813, "n_forward_passes": 97 },
  "margin": { "ms_per_block": [8419.499],  "total_ms": 8827.996,  "n_forward_passes": 97 }
}
```

- `ms_per_block`: milliseconds spent in that block's forward passes
  (decode plus logits copy, not sampling), one entry per block, in block
  order.
- `total_ms`: milliseconds for the whole block loop, including sampling.
  It excludes model load, the one-off unconditional decode and rendering.
- `n_forward_passes`: number of `llama_decode()` calls the run made,
  including the one-off unconditional decode when `--cfg > 0`.

## Formats

- **`.i32` units file**: flat little-endian `int32` array, no header, one
  content-unit id per element, each in `[0, 4095]`.
- **`dump_index.txt`** (written by `dynin_units_encode_wav()`'s optional
  `dump_dir` diagnostic): plain text, one line per intermediate stage,
  `"<stage_name> ne0 ne1 ne2 ne3"` (ggml tensor shape order), plus a final
  `unit_ids` line. Stage float32/int32 data itself is written alongside as
  `<dump_dir>/<stage>.f32` / `<dump_dir>/unit_ids.i32`.

## U2S weights

The EMOVA U2S GGUF is produced by `tools/convert_emova_u2s_to_gguf.py`
from the upstream `Emova-ollm/emova_speech_tokenizer_hf` checkpoint
(Apache-2.0 licence; paper arXiv:2409.18042). See that tool's own README
for the conversion details and provenance.
