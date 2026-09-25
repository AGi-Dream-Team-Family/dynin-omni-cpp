# tools/convert_emova_u2s_to_gguf.py

One-off Python conversion script. Converts the EMOVA U2S (unit-to-speech)
VITS-style decoder out of a `model.safetensors` checkpoint into an
`emova_u2s.gguf` file that `examples/dynin-voice/dynin_vocoder.h/.cpp`
loads at runtime. Python is used here only to read the checkpoint header
and rewrite weights into GGUF; nothing in this repository calls it at
runtime.

## Source model

Upstream checkpoint: `Emova-ollm/emova_speech_tokenizer_hf` on Hugging Face
(https://huggingface.co/Emova-ollm/emova_speech_tokenizer_hf). Licence:
Apache-2.0. Code: https://github.com/emova-ollm/EMOVA_speech_tokenizer
(Apache-2.0). Paper: arXiv:2409.18042. Project page: emova-ollm.github.io.
The model card notes the speech tokenizer continues as an independent
open-source project under `DiscreteSpeech`/`DSTK` on Hugging Face; worth
checking there too for updates.

The converter's tensor layout follows the EMOVA speech tokenizer's UVITS
code (`speech_tokenization/UVITS/{models,modules,attentions}.py`); see the
repository `NOTICE` for the attribution.

## Usage

```
python convert_emova_u2s_to_gguf.py \
    --src /path/to/model.safetensors \
    --dst /path/to/emova_u2s.gguf \
    --symbols-py /path/to/emova_speech_tokenizer/speech_tokenization/UVITS/text/symbols.py
```

- `--src` the checkpoint's `model.safetensors` file.
- `--dst` where to write the resulting GGUF.
- `--symbols-py` the checkpoint repository's own
  `speech_tokenization/UVITS/text/symbols.py`. The script imports this
  file directly (it has no imports of its own) purely to compute the
  numeric-token vocabulary base offset from `symbols_with_4096`, rather
  than retyping the punctuation/IPA unicode literals by hand.

Requires `numpy` and the `gguf` Python package: either `pip install gguf`,
or a llama.cpp checkout's `gguf-py` directory on `PYTHONPATH`. The script
also adds `../gguf-py` (relative to its own location) to `sys.path`, so
copying it into `tools/` of a llama.cpp checkout works as well; this
repository has no `gguf-py` of its own. No `torch`, no `transformers`.

This script is not part of `patches/0001-*.patch`.

## What it does

Reads every `decoder.*` tensor plus `style_embedding.*` directly out of
the safetensors header via a minimal hand-rolled reader (struct + json
header, numpy memmap view onto the payload with no `safetensors` package
dependency), fuses each `weight_norm` `(g, v)` pair into a single plain
weight (PyTorch's default `dim=0` convention, verified against every
`weight_g` shape in this checkpoint), and writes the whole set plus a
`emova.u2s.*` metadata block (sample rate, network dimensions, HiFi-GAN
upsample/resblock configuration, etc.) to a GGUF file.

`enc_q`, `enc_r`, and `emb_g` are carried over for checkpoint fidelity.
`enc_q` (the training-only posterior encoder) and `enc_r` (a reference-mel
style encoder, unused because inference selects a style from the fixed
`style_embedding` prototypes) are not read by the C++ runtime. `emb_g` is
the speaker embedding table, which the upstream
`synthesis_from_content_unit_style_embedding` never indexes because it
takes the style vector directly as conditioning. The CLIs do not index it
either (`--style` must be in `[0, n_styles)`); the vocoder library reads
its shape to report `n_speakers`, and reads a row only when called
directly with a style index of 1000 or more (speaker id = index - 1000).

See the script's own top-of-file comment for the two checkpoint-shape
corrections found during derivation (`enc_p.emb` row count, `decoder.emb_g`
shape) and the upstream source files (`models.py`, `modules.py`,
`attentions.py`) the tensor layout was checked against.
