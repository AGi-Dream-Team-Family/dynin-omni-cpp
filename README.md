# dynin-omni-cpp: EMOVA U2S vocoder port and block-local KV diffusion decode for Dynin-Omni in llama.cpp

Two standalone command-line tools, one weight converter, and two patches that
add speech output support for Dynin-Omni checkpoints to a llama.cpp fork
that already carries masked-diffusion (LLaDA-style) text decoding.
Licensed MIT (see `LICENSE`); third-party attributions are in `NOTICE`
and under "Credits" below.

- **`dynin-u2s`**: renders a sequence of EMOVA content-unit ids to a wav
  file through a from-scratch ggml port of the EMOVA U2S (unit-to-speech)
  VITS/HiFi-GAN decoder.
- **`dynin-t2s-kvblock`**: text in; speech units are generated in memory
  by a masked-diffusion decode of a Dynin-Omni checkpoint and, with
  `--wav`, rendered to `t2s.wav` (or `t2s_exact.wav` / `t2s_margin.wav`
  under `--compare`). The unit ids themselves are not written out. It has
  an optional block-local (windowed) decode that reads the rest of the
  canvas from a KV cache, and a `--compare` mode that measures it against
  the dense (exact) path on the same seed.
- **`tools/convert_emova_u2s_to_gguf.py`**: one-off converter from the
  upstream EMOVA U2S `model.safetensors` checkpoint to the GGUF the C++
  vocoder loads. It is not part of the patch.
- **`patches/0001-*.patch`**: a `git format-patch` of `examples/dynin-voice/`
  plus the `include/`, `src/` and `examples/CMakeLists.txt` hunks listed
  under Build, against the base commit below. `examples/dynin-voice/` in
  this repository holds byte-identical copies of the files the patch adds,
  for reading without applying it.
- **`patches/0002-*.patch`**: a three-line change to llama.cpp's Python
  converter that registers the Dynin-Omni architecture names, so that
  `convert_hf_to_gguf.py` can make the GGUF `dynin-t2s-kvblock` loads (see
  "Getting a Dynin-Omni GGUF").

This is a standalone repository, not a pull request to llama.cpp.

## Base commit

Both patches were cut against:

- Repository: `https://github.com/PrismML-Eng/llama.cpp.git`
- Commit: `7529fdaaf99ffdc5ca71ace9c7409a56b27ad92f`
- Subject: "force qwen35 to treat IGPU like a GPU so DSpark does not fall
  back to CPU (#88)" (2026-07-19)

## Build

The patch files and their SHA-256:

- `patches/0001-dynin-voice-EMOVA-U2S-vocoder-port-and-block-local-K.patch`:
  277,884 bytes, SHA-256
  `d2a4d8d03d3a309e5a1fa328196b6a04ea0b0a63b1a847f0a29c2e5cfbde2842`
- `patches/0002-convert-register-Dynin-Omni-architectures-as-LLaDA.patch`:
  1,716 bytes, SHA-256
  `21e71efb2f550defe55f64cb317a418c343ecc503a30a9749f431ab9d6e093f8`

Get this repository and a checkout of the base commit. The base commit is
not on the fork's default branch, so either fetch it by hash as below or
use a full clone of the fork (it is on `prism-v5` and other branches):

```
git clone https://github.com/AGi-Dream-Team-Family/dynin-omni-cpp.git

mkdir llama.cpp && cd llama.cpp
git init
git remote add origin https://github.com/PrismML-Eng/llama.cpp
git fetch --depth 1 origin 7529fdaaf99ffdc5ca71ace9c7409a56b27ad92f
git checkout FETCH_HEAD
```

Then apply the patches. `git am` creates one commit per patch, so it
needs a git identity; if you have none configured, pass one for this
command only (replace `<name>` and `<email>` with any name and address):

```
git -c user.name="<name>" -c user.email="<email>" \
    am ../dynin-omni-cpp/patches/0001-dynin-voice-EMOVA-U2S-vocoder-port-and-block-local-K.patch \
       ../dynin-omni-cpp/patches/0002-convert-register-Dynin-Omni-architectures-as-LLaDA.patch
```

`git apply <patch>` also works and applies the same changes without
creating a commit. The two patches touch different files: `0002` changes
only `conversion/llada.py` and `conversion/__init__.py`, and applies to
the base commit with or without `0001`. `0002` is needed only to convert
the checkpoint, not to build or run the tools.

Or copy `examples/dynin-voice/` into an existing checkout's `examples/`
directory, add `add_subdirectory(dynin-voice)` to `examples/CMakeLists.txt`,
and apply the engine hunks from the patch by hand. The patch changes
`include/llama.h`, `src/llama-batch.{h,cpp}`, `src/llama-context.{h,cpp}`,
`src/llama-cparams.h`, `src/llama-graph.cpp`, `src/llama-model.cpp`,
`src/models/llada.cpp` and `src/models/models.h`. What those hunks do:

- **`llama_context_params::diffusion_prefix_cache`** (default `false`).
  When set, `create_memory()` gives the `LLADA`, `LLADA_MOE` and `RND1`
  architectures a standard `llama_kv_cache` (without the flag they keep no
  memory, as before), and the `llada` graph reads attention input from
  that cache (`build_attn_inp_kv()`) instead of the no-cache path. Only
  the plain `llada` graph was changed: the `llada-moe` and `rnd1` graphs
  still use the no-cache path, so with the flag set those two
  architectures get a cache their graphs never read. Dynin-Omni loads as
  plain `llada`.
- **Overwrite on re-decode.** When the flag is set and the context has a
  KV cache, `llama_decode()` first removes any existing cell at the same
  `(seq, pos)`, so re-decoding a position overwrites it rather than
  duplicating it.
- **`allow_pos_rewrite`** (driven by the same flag): skips
  `llama_batch_allocr::init()`'s sequence-position consistency checks
  (both the M-RoPE branch and the `Y = X + 1` consecutive-position
  branch), so a window batch whose positions are already in the cache is
  accepted. Despite the name, no position is rewritten.
- **`llama_set_attn_pad_token()`**: hides every key position holding the
  given token id from every other query row. The hook lives in the
  no-cache attention input only; see "Known gaps" for how the tools keep
  pad positions out of attention when the KV cache is in use.

Then configure and build:

```
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target dynin-u2s dynin-t2s-kvblock --config Release
```

The release build was configured on Windows from an x64 MSVC developer
environment (`vcvars64.bat`) with `-G Ninja` added to the first command;
the cold build of the published patches, with its full logs, is in
`receipts/smoke_2026-09-25.md`.

CUDA is required for the GPU path. The vocoder's CUDA helpers are compiled
only when `GGML_USE_CUDA` is defined (ggml sets it when `GGML_CUDA=ON`);
in a build without it they report "built without GGML_USE_CUDA" and every
vocoder stage (`enc_p`, the flow and the generator) runs on the CPU, and
the diffusion model runs on whatever backend the rest of the build
provides. The only configuration compiled and run for this release is
Windows, MSVC 2022, CUDA 13.1, `GGML_CUDA=ON`, Ninja generator, on one
NVIDIA RTX 5070 Laptop GPU (8 GB).

The Whisper scoring oracle in `dynin_ears.h` starts `whisper-cli` with
`CreateProcessW` on Windows and with `popen()` / `pclose()` on POSIX. The
POSIX branch has not been compiled or run here. Neither branch has a
timeout: a hung `whisper-cli` blocks the call.

Both tools include only public headers: `llama.h`, and `ggml.h`,
`ggml-backend.h`, `ggml-alloc.h`, `ggml-cpu.h`, `ggml-cuda.h` and `gguf.h`
from `ggml/include`. Nothing is included from `src/` or `common/`.
`dynin-t2s-kvblock` also needs the new API the patch adds to `llama.h`
(`diffusion_prefix_cache`, `llama_set_attn_pad_token`).

## Getting the U2S weights

The EMOVA U2S GGUF is not committed to this repository (see `.gitignore`).
Two ways to get it:

1. **Convert it yourself** from the upstream checkpoint. The converter
   lives in `tools/` and is not in the patch. It needs `numpy` and the
   `gguf` Python package: either `pip install gguf`, or a llama.cpp
   checkout's `gguf-py` directory on `PYTHONPATH` (the script also looks
   in `../gguf-py` relative to itself, so copying it into a llama.cpp
   checkout's `tools/` works too). See `tools/README.md` for the details;
   usage is:

   ```
   python tools/convert_emova_u2s_to_gguf.py \
       --src /path/to/model.safetensors \
       --dst /path/to/emova_u2s.gguf \
       --symbols-py /path/to/emova_speech_tokenizer/speech_tokenization/UVITS/text/symbols.py
   ```

   The source checkpoint is `Emova-ollm/emova_speech_tokenizer_hf` on
   Hugging Face (Apache-2.0, arXiv:2409.18042).

2. **Download the release asset** built the same way:

   ```
   https://github.com/AGi-Dream-Team-Family/dynin-omni-cpp/releases/download/v0.1.0/emova_u2s.gguf
   ```

   - Size: 167,497,280 bytes
   - SHA-256: `983FF5C1D89A867636A9542E3C0E3B0DA451A72B4C075DF460DC614CFA97DC84`

## Getting a Dynin-Omni GGUF

`dynin-t2s-kvblock -m` takes a GGUF of the released Dynin-Omni checkpoint,
https://huggingface.co/snu-aidas/Dynin-Omni. No GGUF of it is published,
and the base commit's `convert_hf_to_gguf.py` does not recognise the
checkpoint: its `config.json` declares the architectures
`DyninOmniForConditionalGeneration` and `DyninOmniModelLM`, while the
converter registers only `LLaDAModelLM` for the `llada` graph. `0002` maps
both names to the existing LLaDA converter class. In a llama.cpp checkout
with the patches applied:

```
pip install -r requirements/requirements-convert_hf_to_gguf.txt
git clone https://huggingface.co/snu-aidas/Dynin-Omni      # git-lfs; about 16 GB of weights
python convert_hf_to_gguf.py Dynin-Omni --outtype f16 --outfile Dynin-Omni-8.1B-F16.gguf
cmake --build build-cuda --target llama-quantize --config Release
build-cuda/bin/llama-quantize Dynin-Omni-8.1B-F16.gguf Dynin-Omni-8.1B-Q4_K_M.gguf Q4_K_M
```

The converter pads the vocabulary up to the config's `vocab_size`
(138,752) with `[PAD<id>]` pieces, 12,380 of them for this checkpoint;
`dynin_vocab.h` finds the text/image boundary from the first of those
pieces (see "Vocab offsets"). It loads the tokenizer with
`trust_remote_code=True`; the checkpoint's tokenizer class is the standard
`PreTrainedTokenizerFast`.

**Where the GGUF used here came from.** Every `dynin-t2s-kvblock` number
in `receipts/` used one Q4_K_M file. It and the F16 file it was quantized
from are dated 3 Sep 2026, less than three minutes apart. The F16
file was converted from the released checkpoint with a local copy of the
fork's converter that
carried the same architecture registration as `0002`, plus one line that
forced `vocab_size = max(config value, 138752)`. That line changes nothing
for this checkpoint, whose `config.json` already says 138752, so `0002`
leaves it out. **The full-weight conversion and the quantization were not
re-run for this release**, and `llama-quantize` was not built or run for
it. What was re-run on 25 Sep: `convert_hf_to_gguf.py --vocab-only` at the
base commit with both patches applied, on the checkpoint's published
config, tokenizer and code files (each byte-identical to
`snu-aidas/Dynin-Omni` revision `ebdefada`, checked by git blob hash).
`--vocab-only` writes metadata only and reads no weights. Its output has
37 metadata keys: 36 are identical to the Q4_K_M file's, including
`general.architecture = llada`, every `llada.*` hyperparameter and the full
`tokenizer.ggml.tokens`, `token_type` and `merges` arrays; the 37th is
`general.file_type` (1, F16, against 15, Q4_K_M). The Q4_K_M file has one
more key, `general.size_label`, which vocab-only mode does not write. The
same command without `0002` stops with "Model
DyninOmniForConditionalGeneration is not supported". Details and logs:
`receipts/convert_2026-09-25.md` and `receipts/raw/convert_2026-09-25/`.

## Usage

### dynin-u2s

```
usage: dynin-u2s -m <u2s.gguf> -i <units_file> -o <out.wav> [--style <index>] [--seed <n>] [--sr <hz>] [--cpu]
```

- `-m` the EMOVA U2S GGUF.
- `-i` the units file. A `.i32` file is read as a flat little-endian
  `int32` array (no header). Any other file is parsed as text: decimal
  integers separated by whitespace and/or commas. Parsing stops at the
  first non-integer token, and if no integer is found the file is re-read
  as raw `int32`. Every id must be in `[0, 4095]`.
- `-o` output wav path.
- `--style` style/speaker index in `[0, n_styles)`. Default 41 (the
  `gender-female_emotion-neutral_speed-normal_pitch-normal` entry in the
  EMOVA speech tokenizer's `config.json`).
- `--seed` seeds every random draw the decoder makes. Default 1234.
- `--sr` overrides the sample-rate value written into the wav header only
  (it does not resample; the decoder's native rate, normally 22050 Hz, is
  what the samples actually are).
- `--cpu` forces the CPU path of the generator graph instead of the
  CUDA/CPU auto-pick. `enc_p` and the flow still choose between CPU and
  CUDA on their own; also set `DYNIN_U2S_CUDA_ENCP=cpu` and
  `DYNIN_U2S_CUDA_FLOW=cpu` for an all-CPU run.

### dynin-t2s-kvblock

```
usage: dynin-t2s-kvblock -m <dynin.gguf> --u2s <u2s.gguf> --text "<sentence>" --out <dir>
       [--steps N] [--block B] [--kv-margin M] [--cfg C] [--seed S]
       [--wav] [--compare] [--whisper-cli <exe> --whisper-model <bin>]
```

- `-m` the Dynin-Omni diffusion checkpoint as a GGUF (see "Getting a
  Dynin-Omni GGUF"). `--u2s` the EMOVA U2S vocoder
  GGUF (required on the command line, but only loaded if `--wav` is
  given).
- `--text` the sentence to speak.
- `--steps` the step budget, split as `floor(steps / n_blocks)` steps per
  block (383 at block 128 runs 3 x 127 = 381), and also the number of
  speech-unit positions generated. Default 383.
- `--block` block length for the block-local diffusion schedule. Default
  128.
- `--kv-margin` window margin `M` for the windowed decode (see below). `0`
  disables windowing (dense/exact path). If omitted, defaults to 32 when
  `--steps 48` and to 0 otherwise.
- `--cfg` classifier-free-guidance scale. Default 2.5.
- `--seed` seeds the per-step sampling, and the vocoder when `--wav` is
  given. Default 1234.
- `--out` output directory; the last path component is created if missing
  (its parent must exist).
- `--wav` also renders the resulting unit sequence(s) to wav.
- `--compare` runs **two** passes on the same text and seed, `exact`
  (`--kv-margin` forced to 0) and `margin` (the `--kv-margin` value or its
  computed default), and writes `match.json` (unit-level agreement between
  the two passes) and `timing.json` (per-block forward-pass milliseconds,
  total milliseconds for the block loop, which excludes the one-off
  unconditional decode, and forward-pass counts) into `--out`.
- `--whisper-cli` / `--whisper-model` an optional scoring oracle: with
  `--wav`, transcribes the resulting wav file(s) (English only, `-l en`)
  and prints word overlap against `--text` (the fraction of the sentence's
  unique words found in the transcript). The overlap is printed to stdout,
  not written to JSON.

See `examples/dynin-voice/README.md` for the `match.json` / `timing.json`
schemas and the units-file format in full.

## Vocab offsets

The Dynin-Omni vocabulary stacks three ranges on top of the text tokenizer.
Text tokens run `[0, text_len)`; image codes (MAGVITv2, 8192 codes) run
`[text_len, text_len + 8192)`; speech units (EMOVA, 4096 codes) run
`[text_len + 8192, text_len + 8192 + 4096)`. For this checkpoint,
`text_len = len(tokenizer) = 126372`, so image codes start at 126372 and
speech units start at 134564.

The checkpoint's `config.json` reports `llm_vocab_size = 126464`, which is
`text_len` padded up to a multiple of 128, **not** `text_len` itself.
Hardcoding 126464 (or the equivalent padded speech-base figure, 134656)
instead of the true boundary produces a systematic +92 offset on every
image code and every speech unit. `dynin_vocab.h` does not read
`config.json`: it finds the boundary in the loaded GGUF vocabulary (the
first token id whose piece is exactly `[PAD<id>]`), compares it with
126372, the `len(tokenizer)` of the released Dynin-Omni checkpoint compiled
in as the expected value, and refuses to run if the two differ.
`tools/convert_emova_u2s_to_gguf.py` similarly derives its own numeric-token
base offset from the checkpoint's own `symbols.py` rather than a hardcoded
constant.

## Block-local KV decode

The reference decode loop (`t2s_generate_mmu_like`-equivalent: per-block
confidence-based token transfer, classifier-free guidance against a
once-only unconditional canvas) re-decodes the *entire* canvas from
position 0 on every diffusion step. The canvas is a 1,024-position text
block (the prompt, left-padded with `[iPAD]`), one `<|soa|>` token, and
`--steps` speech positions: 1,073 positions at 48 steps and 1,121 at 96.
For the ten bench sentences the tokenised prompt is 29 to 33 tokens, so
991 to 995 of the 1,024 text-block positions are `[iPAD]`.

What `dynin-t2s-kvblock` does with `--kv-margin M` greater than 0:

- **Step 0 of each block:** clear the KV cache, then decode the full
  canvas densely. The K/V for every position stays in the cache.
- **Every later step in that block:** decode only the positions
  `[block_start - M, block_end + M)`, clamped to the canvas. The window's
  cells in the cache are overwritten; every position outside the window
  is read from the cache as it was computed at the block's step 0.
- **`M = 0`:** every step is a dense full-canvas decode with the cache
  cleared before and after, which is the exact reference path used as the
  comparison baseline.

The unconditional (guidance) canvas is decoded once per run, before the
timer starts, and reused. `[iPAD]` positions are placed in sequence 1 and
every other position in sequence 0, so the cache's per-sequence mask keeps
pad keys out of the real tokens' attention.

The speed-up comes from the ratio of window size to canvas size. At block
128 / 48 steps / `M = 32` the window is 80 positions against a 1,073-position
canvas, so one run decodes 1,073 + 47 x 80 = 4,833 positions instead of
48 x 1,073 = 51,504 (10.7x fewer). At block 128 / 96 steps the window is 128
positions against 1,121 (8.1x fewer per run). At block 16 / 48 steps there
are three blocks, each with its own dense step 0, and windows of 80, 64 and
48 positions (8.4x fewer). The measured wall-clock ratios below are lower
than these position ratios.

At block 128 with `M = 32` the window starts 32 positions before the speech
region, at position 993. For the seven sentences with 993 or more `[iPAD]`
positions that covers every non-pad position, so no real token is read
from a stale cache cell. For sentences 4, 5 and 10 (992, 992 and 991 pads)
the first one or two prompt tokens fall outside the window. Those three are
exactly the sentences whose unit ids differ between the two paths at block
128, at both 48 and 96 steps. It also means most of the dense path's work
at these settings is spent on `[iPAD]` positions, which cannot affect the
output.

Fidelity and speed at three schedules (block / steps / margin), measured
with `--compare` on ten short plain-register sentences
(`bench/sentences.txt`), Dynin-Omni-8.1B Q4_K_M, `--cfg 2.5`, seed 42,
n = 10 per arm. These 30 runs were made on 22 Sep 2026 with binaries built
from the source tree later committed as `145929e`, a pre-release revision
of the published patch. The
diffusion decode, the engine hunks and the t2s tool are the same code in
both; the one code difference is the vocoder's CUDA-generator gate, which
only affects wav rendering, outside every timing. The exact diff is in
`receipts/fidelity_2026-09-22.md` ("Tool versions"). The published patch's
own cold build reproduced sentence 1 of the first arm exactly: the same
unit match, transcripts and overlaps, and byte-identical wavs from the
same seeded CPU generator, so the same unit ids (the ids themselves are
not printed; `receipts/smoke_2026-09-25.md`). Unit match is the
exact-vs-windowed unit-id agreement; word
overlap is Whisper (`ggml-base.en.bin`) against each source sentence, as
printed by the tool:

| arm | mean exact ms | mean windowed ms | speedup | mean unit match | word overlap, exact | word overlap, windowed |
|---|---|---|---|---|---|---|
| block 128 / steps 48 / margin 32 | 33915.9 | 3846.2 | 8.82x | 0.8437 | 0.5796 | 0.6613 |
| block 128 / steps 96 / margin 32 | 72672.8 | 9749.6 | 7.45x | 0.9098 | 0.9375 | 0.9250 |
| block 16 / steps 48 / margin 32 | 34683.3 | 4695.1 | 7.39x | 0.6417 | 0.6984 | 0.7402 |

Speedup is mean exact ms divided by mean windowed ms. Per run it ranged
from 6.99x to 9.94x (block 128 / 48), 6.08x to 8.16x (block 128 / 96) and
7.01x to 7.78x (block 16 / 48).

Reading, plainly:

- At block 128 the windowed path produced the same unit ids as the dense
  path for 7 of 10 sentences at both 48 and 96 steps (see above for which
  three differ and why). At block 16 no sentence matched exactly (unit
  match 0.02 to 0.94): the windows of the second and third blocks start at
  positions 1,009 and 1,025, so part or all of the prompt is read from
  cache cells computed at that block's step 0.
- Step count matters more than windowing for intelligibility. Mean word
  overlap on the dense path was 0.58 at 48 steps and 0.94 at 96 steps.
- The windowed path's word overlap was not lower than the dense path's in
  two of the three arms and 0.0125 lower in the third. With ten sentences
  and a base-size Whisper model as the scorer, differences of this size
  are not evidence that either path is better.

**Machine state:** all 30 runs exited 0 on an NVIDIA RTX 5070 Laptop GPU
(8 GB). No compile job and no other process holding GPU memory was present
at any point checked during the run (spot-checked repeatedly across about
38 minutes). GPU memory was 181 to 213 MiB before and after, and about 7.7
to 8.1 GB while the tool was resident. These spot checks were read by hand
and are not in the raw logs. See `receipts/fidelity_2026-09-22.md`
for the per-sentence tables, transcripts, exact commands and tool versions.

## Known gaps

- **No text-to-image tool in this repository.** The patch carries the
  per-token attention pad mask (`llama_set_attn_pad_token`) that a
  bidirectional image canvas needs on the no-cache path: it hides the
  `[iPAD]` positions as attention keys, as the reference's
  `attention_bias` does for non-pad rows. That hook is not applied when
  `diffusion_prefix_cache` gives the context a KV cache, which is why
  `dynin-t2s-kvblock` separates pads by sequence id instead. The image
  sampler itself is not part of this release. Only text-to-speech and
  units-to-speech are covered here. One porting note for anyone adding
  classifier-free guidance on top of llama.cpp: `llama_get_logits()`
  returns the context's single output buffer, which the next
  `llama_decode()` overwrites in place, so the conditional logits must be
  copied out before the unconditional decode or the guidance silently
  collapses to the unconditional distribution. `dynin-t2s-kvblock` copies
  them.
- **No streaming ASR.** `dynin_ears.h` runs a batch `whisper-cli` process
  per wav file purely as a scoring oracle for the tools above; it is not a
  speech-input path for the model itself.
- **`llada-moe` and `rnd1` graphs** do not read the KV cache (see Build).
- **CUDA generator gate.** On the first call the vocoder runs the
  generator on both backends and keeps CUDA only if the signal-to-error
  ratio of the CUDA output against the CPU output is at least 40 dB
  (`snr_db = 10 log10(sum cpu^2 / sum (cpu - cuda)^2)`) and CUDA is
  faster. In the units-to-speech smoke run of the published patches it
  measured 57.15 dB (largest int16 difference 92, first differing sample
  4), so the generator ran on CUDA at 86.66 ms against 2,049.19 ms on the
  CPU. In the text-to-speech smoke run the gate passed (62.10 dB) but CUDA
  was not faster on that 47-unit clip, so the CPU generator was kept. This
  is a self-consistency check between two backends computing the same
  graph, not a comparison with a reference decoder; no comparison against
  the PyTorch decoder is part of this release. See
  `receipts/smoke_2026-09-25.md`. The 30 bench
  runs in `receipts/fidelity_2026-09-22.md` were rendered before this
  gate replaced an int16 max-difference rule and used the CPU generator;
  rendering is outside every timing reported there.

## How this was made

This release was prepared by Claude, Anthropic's AI models, working with the
AGi-DTF team: Claude Fable 5.1 did the engineering, the benchmarks and the
write-up; Claude Opus 5.5 did the release audit and this publication; Claude
Sonnet 5 sub-agents ran bench runs and checks.

Every number in `receipts/` was measured on the machine stated in each
receipt (one Windows laptop with an NVIDIA RTX 5070 Laptop GPU, 8 GB). The
raw logs are included under `receipts/raw/`, with only the directory part
of local absolute paths replaced by `<path>/` and Windows line endings
converted to LF. Two kinds of figure have no
raw log: the machine-state spot checks (GPU memory in use, free memory,
elapsed time), which were read by hand before and during the runs, and the
`[iPAD]` pad counts per sentence, which come from a log line of an earlier
build without the KV cache that is not included.

What was tested:

- A cold build of the published patches on a fresh fetch of the base commit
  (Windows, MSVC 2022, CUDA 13.1, `GGML_CUDA=ON`, Ninja), and one smoke run
  of each tool on that build (`receipts/smoke_2026-09-25.md`).
- The 30-run exact-vs-windowed bench (`receipts/fidelity_2026-09-22.md`),
  on the 22 Sep build of a pre-release revision whose decode code is the
  same as the published patch's.
- The vocoder's CUDA stages against its own CPU path (cosine and
  signal-to-error checks printed by the tool).
- `convert_hf_to_gguf.py --vocab-only` with `0002` on the checkpoint's
  published non-weight files, compared key by key with the GGUF used for
  every t2s number, and the same command without `0002`
  (`receipts/convert_2026-09-25.md`).

What was not tested:

- Any build other than Windows + MSVC + CUDA: no CPU-only, Linux or macOS
  build, and the POSIX branch of `dynin_ears.h` has never been compiled.
- The C++ vocoder against the PyTorch EMOVA decoder is not part of this
  release's receipts. An earlier internal comparison was made during
  development, but its logs are not included, so this release gives no
  reference-fidelity number.
- The EMOVA S2U encoder in `dynin_units.*`: it is compiled and linked
  into `dynin-u2s`, but neither tool calls it, and its GGUF and converter
  are not part of this release.
- The full-weight conversion of the checkpoint and its Q4_K_M
  quantization were not re-run for this release; the Q4_K_M file used
  throughout was made on 3 Sep 2026 (see "Getting a Dynin-Omni GGUF").
- The Dynin-Omni checkpoint at any precision other than the Q4_K_M GGUF.
- The `llada-moe` and `rnd1` graphs with the KV cache (they do not read it).
- Text-to-image: not part of this release.
- Intelligibility beyond Whisper word overlap on ten short sentences.

## Credits

This work builds directly on the projects below. Licences were checked
against each project's LICENSE file or model card.

- **llama.cpp / ggml**, the ggml authors,
  https://github.com/ggml-org/llama.cpp. MIT licence ("Copyright (c)
  2023-2026 The ggml authors"). The tools are built on ggml and the llama
  API, and the patch modifies llama.cpp source files.
- **PrismML-Eng llama.cpp fork**, https://github.com/PrismML-Eng/llama.cpp,
  the tree the patch targets (commit `7529fda`); it already carries the
  masked-diffusion (LLaDA-style) decoding this work extends. MIT licence,
  same copyright line as upstream.
- **Dynin-Omni**, AIDAS Lab, Seoul National University.
  "Dynin-Omni: Omnimodal Unified Large Diffusion Language Model",
  arXiv:2604.00007; weights at https://huggingface.co/snu-aidas/Dynin-Omni
  (MIT licence on the model card). `dynin-t2s-kvblock` ports the
  `t2s_generate_mmu_like` decode recipe from the Dynin-Omni reference
  implementation, https://github.com/AIDASLab/Dynin-Omni (MIT), and the
  vocab layout follows the released checkpoint.
- **MMaDA**, Gen-Verse, "MMaDA: Multimodal Large Diffusion Language
  Models" (Yang et al., arXiv:2505.15809), https://github.com/Gen-Verse/MMaDA
  (MIT): Dynin-Omni's base model (`Gen-Verse/MMaDA-8B-MixCoT`, the
  `general.base_model` entry in the checkpoint's GGUF metadata) and the
  source of the `mmu_generate` block-wise generation function, which
  Dynin-Omni's model code also carries and which the name of its
  `t2s_generate_mmu_like` refers to.
- **LLaDA**, "Large Language Diffusion Models" (Nie et al.,
  arXiv:2502.09992), https://github.com/ML-GSAI/LLaDA: the masked-diffusion
  language-model architecture Dynin-Omni builds on and that llama.cpp's
  `llada` graph implements. The LLaDA-8B weights on Hugging Face
  (`GSAI-ML/LLaDA-8B-Base`) are MIT; the GitHub repository has no licence
  file. No LLaDA code is included here.
- **EMOVA speech tokenizer**, the EMOVA authors,
  https://github.com/emova-ollm/EMOVA_speech_tokenizer and
  https://huggingface.co/Emova-ollm/emova_speech_tokenizer_hf, "EMOVA:
  Empowering Language Models to See, Hear and Speak with Vivid Emotions",
  arXiv:2409.18042. Apache-2.0. `dynin_vocoder.cpp` is a C++/ggml
  re-implementation of its UVITS unit-to-speech decoder, and
  `tools/convert_emova_u2s_to_gguf.py` follows its tensor layout.
  `dynin_units.h` / `dynin_units.cpp` are a C++/ggml re-implementation of
  its S2U (speech-to-unit) encoder under
  `speech_tokenization/SPIRAL_L2_BN_FSQ_CTC/`, which builds on SPIRAL
  (Huang et al., "SPIRAL: Self-supervised Perturbation-Invariant
  Representation Learning for Speech Pre-Training", arXiv:2201.10207;
  source files marked Huawei Technologies, Apache-2.0) inside a copy of
  NVIDIA NeMo (Apache-2.0), with an FSQ quantizer (`my_scripts/fsq.py`,
  marked "Copyright 2023 Google LLC", Apache-2.0; Mentzer et al., "Finite
  Scalar Quantization: VQ-VAE Made Simple", arXiv:2309.15505). All of
  these files carry an attribution header, and `NOTICE` has the full
  attribution. The converter also imports the checkpoint's
  `UVITS/text/symbols.py` at run time; that file comes from Keith Ito's
  tacotron (MIT, "Copyright (c) 2017 Keith Ito") and is not included
  here.
- **VITS**, Jaehyeon Kim, Jungil Kong and Juhee Son, "Conditional
  Variational Autoencoder with Adversarial Learning for End-to-End
  Text-to-Speech", arXiv:2106.06103, https://github.com/jaywalnut310/vits
  (MIT, "Copyright (c) 2021 Jaehyeon Kim"): the decoder design EMOVA's
  UVITS is based on.
- **HiFi-GAN**, Jungil Kong, Jaehyeon Kim and Jaekyoung Bae, "HiFi-GAN:
  Generative Adversarial Networks for Efficient and High Fidelity Speech
  Synthesis", arXiv:2010.05646, https://github.com/jik876/hifi-gan (MIT,
  "Copyright (c) 2020 Jungil Kong"): the waveform generator inside the
  VITS decoder.
- **Neural Spline Flows**, Conor Durkan, Artur Bekasov, Iain Murray and
  George Papamakarios, arXiv:1906.04032, https://github.com/bayesiains/nsf
  (MIT, "Copyright (c) 2019 Conor Durkan, Artur Bekasov, Iain Murray,
  George Papamakarios"): the rational-quadratic spline in VITS's
  `transforms.py`, which `dynin_vocoder.cpp` ports for the duration
  predictor's flow.
- **whisper.cpp**, the ggml authors, https://github.com/ggml-org/whisper.cpp
  (MIT), with the OpenAI **Whisper** `base.en` model (`ggml-base.en.bin`;
  Radford et al., "Robust Speech Recognition via Large-Scale Weak
  Supervision", arXiv:2212.04356; https://github.com/openai/whisper, MIT,
  "Copyright (c) 2022 OpenAI"). `whisper-cli` is the speech recogniser
  the fidelity bench and the smoke tests use to score intelligibility;
  `dynin_ears.h` runs it as a separate process and links none of its code.
- **Related work on KV caching for diffusion language models.** The
  block-wise cache idea the windowed decode follows is described in
  Fast-dLLM (Wu et al., "Fast-dLLM: Training-free Acceleration of Diffusion
  LLM by Enabling KV Cache and Parallel Decoding", arXiv:2505.22618) and
  dKV-Cache (Ma et al., "dKV-Cache: The Cache for Diffusion Language
  Models", arXiv:2505.15781); the patch's `create_memory()` comment names
  both. dInfer (arXiv:2510.08666) describes a "vicinity KV-cache refresh"
  that recomputes K and V for the masked tokens and their immediate
  neighbours during denoising and refreshes the full cache once a block is
  decoded, which is close to what the `--kv-margin` window does (here the
  full refresh happens at each block's step 0). No code from these projects
  is included; they are listed as related work.

## Licence

This repository's own work (the engine hunks of the patch, the converter
registration in `0002`, the parts of the two CLIs and of the converter
that are not derived from EMOVA, the bench script and the documentation)
is MIT-licensed; see `LICENSE`. The patches modify llama.cpp / ggml files, which are MIT-licensed by the ggml
authors. `dynin_vocoder.h` / `.cpp`, `dynin_units.h` / `.cpp` and the
converter re-implement code from the EMOVA speech tokenizer, which is
Apache-2.0; see `NOTICE` and `LICENSES/Apache-2.0.txt`. Both CLIs link
`dynin_vocoder.cpp`, and `dynin-u2s` also links `dynin_units.cpp`, so their
binaries contain Apache-2.0-derived code. Model weights are under their own licences and
are **not** included in the repository: Dynin-Omni under the MIT licence on
its `snu-aidas/Dynin-Omni` model card, and the EMOVA speech tokenizer under
Apache-2.0 (the `emova_u2s.gguf` release asset is a format conversion of
that checkpoint and stays under Apache-2.0).

## Author

AGi-DTF (fam@agi-dtf.life)
