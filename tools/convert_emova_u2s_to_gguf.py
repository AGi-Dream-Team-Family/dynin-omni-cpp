#!/usr/bin/env python3
# One-off dev tool: Python is used here only to inspect a checkpoint header
# and convert its weights to a GGUF file -- never at runtime. Converts the
# EMOVA U2S (unit-to-speech) VITS-style decoder out of a model.safetensors
# checkpoint into an emova_u2s.gguf file for the native C++ ggml runtime
# (examples/dynin-voice/dynin_vocoder.h/.cpp) to load.
#
# Tensor names/shapes were verified directly against model.safetensors' own
# header (not assumed from any prior summary) and against the upstream
# EMOVA speech tokenizer reference source
# (speech_tokenization/UVITS/{models,modules,attentions}.py). Two notable
# points versus a naive reading of the architecture: enc_p.emb is
# [4274,192] (the full symbols_with_4096 table: pad+punctuation+letters+IPA
# +4096 numbers), not 4096x192 -- the extra 178 rows are never addressed by
# a numeric-only input but must exist for the embedding table's row offsets
# to line up. decoder.emb_g is [1344,256] (n_speakers), not 2x256 -- unused
# at inference either way (synthesis_from_content_unit_style_embedding
# bypasses it and takes the style vector directly as g), kept only for
# checkpoint fidelity.
#
# Only reads model.safetensors + this project's own gguf-py + the
# checkpoint's own (dependency-free) text/symbols.py for the exact vocab
# offset. No torch, no transformers.
#
# Scope: copies every decoder.* tensor plus style_embedding.* (the
# generator and style/speaker embeddings), fusing weight_norm (g,v) pairs
# into a single plain weight. This includes enc_q, enc_r and emb_g, which
# the checkpoint carries but which synthesis_from_content_unit_style_embedding()
# never touches -- the C++ engine (dynin_vocoder.cpp) deliberately does not
# load them. See this directory's README.md for which tensors the runtime
# actually reads.
#
# Attribution: the tensor names, shapes and weight_norm handling mirror the
# UVITS code of the EMOVA speech tokenizer
# (https://github.com/emova-ollm/EMOVA_speech_tokenizer, the EMOVA authors,
# arXiv:2409.18042), licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). No code from that repository
# is copied here; its symbols.py is imported at run time from the user's own
# checkout via --symbols-py. The model design is VITS (Kim et al.,
# arXiv:2106.06103) with a HiFi-GAN generator (Kong et al., arXiv:2010.05646).
import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

FORK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FORK_ROOT / "gguf-py"))
import gguf  # noqa: E402


class SafetensorsFile:
    """Minimal safetensors reader: struct+json header, numpy views onto an
    mmap for the F32 payload. No safetensors/torch dependency."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.data_start = 8 + n
        self.header.pop("__metadata__", None)
        self.mmap = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name: str) -> np.ndarray:
        info = self.header[name]
        assert info["dtype"] == "F32", f"{name} is {info['dtype']}, expected F32"
        off0, off1 = info["data_offsets"]
        shape = info["shape"]
        raw = self.mmap[self.data_start + off0: self.data_start + off1]
        arr = raw.view(np.float32).reshape(shape)
        return np.ascontiguousarray(arr)

    def has(self, name: str) -> bool:
        return name in self.header


def fuse_weight_norm_dim0(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """PyTorch torch.nn.utils.weight_norm(module, name='weight') DEFAULT
    (dim=0): g has shape [d0,1,1], norm taken over dims (1,2) per d0 slice.
    weight = v * (g / norm_except_dim(v, 2, dim=0)). Confirmed against every
    weight_g shape in this checkpoint (dec.ups, dec.resblocks convs1/convs2,
    enc_q.enc.*, flow.flows.*.enc.*) -- all have g.shape[1:]==(1,1), i.e. all
    use this same default-dim convention (unlike the S2U wav2vec positional
    conv, which uses fairseq's dim=2 convention -- a different function in
    the companion S2U converter)."""
    assert weight_g.shape[1] == 1 and weight_g.shape[2] == 1, weight_g.shape
    norm = np.sqrt(np.sum(weight_v.astype(np.float64) ** 2, axis=(1, 2), keepdims=True))
    fused = weight_v.astype(np.float64) * (weight_g.astype(np.float64) / norm)
    return fused.astype(np.float32)


def load_symbol_base_offset(symbols_py: Path):
    """symbols_with_4096 = [_pad] + punctuation-chars + letters + ipa-chars +
    [str(v) for v in range(4096)]. The numeric tokens ("0".."4095") are
    appended last and in order, so _symbol_to_id[str(v)] == base_offset + v
    where base_offset = len(symbols_with_4096) - 4096. Imported directly from
    the checkpoint's own symbols.py (zero imports inside that file) rather
    than retyping the punctuation/IPA unicode literals by hand."""
    spec = importlib.util.spec_from_file_location("emova_symbols", symbols_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n_total = len(mod.symbols_with_4096)
    base_offset = n_total - 4096
    return n_total, base_offset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="path to model.safetensors")
    ap.add_argument("--dst", required=True, type=Path, help="output emova_u2s.gguf path")
    ap.add_argument(
        "--symbols-py", required=True, type=Path,
        help="path to the checkpoint's own speech_tokenization/UVITS/text/symbols.py",
    )
    args = ap.parse_args()

    st = SafetensorsFile(args.src)
    print(f"loaded header: {len(st.header)} tensors from {args.src}")

    n_vocab_total, symbol_base_offset = load_symbol_base_offset(args.symbols_py)
    emb_shape = st.get("decoder.enc_p.emb.weight").shape
    print(f"symbols_with_4096: {n_vocab_total} total, numeric-token base offset {symbol_base_offset}")
    assert emb_shape[0] == n_vocab_total, (emb_shape, n_vocab_total)

    writer = gguf.GGUFWriter(str(args.dst), "emova-u2s")
    n_tensors = 0

    def put(name: str, arr: np.ndarray):
        nonlocal n_tensors
        writer.add_tensor(name, np.ascontiguousarray(arr))
        n_tensors += 1

    def put_plain(dst_name: str, src_name: str):
        put(dst_name, st.get(src_name))

    def put_fused(dst_prefix: str, src_prefix: str, bias=True):
        g = st.get(f"{src_prefix}.weight_g")
        v = st.get(f"{src_prefix}.weight_v")
        put(f"{dst_prefix}.weight", fuse_weight_norm_dim0(g, v))
        if bias:
            put(f"{dst_prefix}.bias", st.get(f"{src_prefix}.bias"))

    # ---- metadata ------------------------------------------------------------
    # (general.architecture is already set by the GGUFWriter(..., "emova-u2s")
    # constructor call above; setting it again here would just be a duplicate.)
    writer.add_string("general.name", "EMOVA speech tokenizer U2S (VITS)")
    writer.add_uint32("emova.u2s.sampling_rate", 22050)
    writer.add_uint32("emova.u2s.n_mel", 80)
    writer.add_uint32("emova.u2s.filter_length", 1024)
    writer.add_uint32("emova.u2s.hop_length", 256)
    writer.add_uint32("emova.u2s.win_length", 1024)
    writer.add_uint32("emova.u2s.n_vocab", n_vocab_total)
    writer.add_uint32("emova.u2s.n_units", 4096)
    writer.add_uint32("emova.u2s.symbol_base_offset", symbol_base_offset)
    writer.add_uint32("emova.u2s.n_styles", 126)
    writer.add_uint32("emova.u2s.style_dim", 256)
    writer.add_uint32("emova.u2s.n_speakers", 1344)
    writer.add_uint32("emova.u2s.inter_channels", 192)
    writer.add_uint32("emova.u2s.hidden_channels", 192)
    writer.add_uint32("emova.u2s.filter_channels", 768)
    writer.add_uint32("emova.u2s.n_heads", 2)
    writer.add_uint32("emova.u2s.n_layers", 6)
    writer.add_uint32("emova.u2s.kernel_size", 3)
    writer.add_uint32("emova.u2s.window_size", 4)
    writer.add_uint32("emova.u2s.gin_channels", 256)
    # duration predictor (stochastic)
    writer.add_uint32("emova.u2s.dp.n_flows", 4)
    writer.add_uint32("emova.u2s.dp.num_bins", 10)
    writer.add_float32("emova.u2s.dp.tail_bound", 5.0)
    writer.add_float32("emova.u2s.dp.noise_scale_w", 0.8)
    # main flow
    writer.add_uint32("emova.u2s.flow.n_flows", 4)
    writer.add_uint32("emova.u2s.flow.kernel_size", 5)
    writer.add_uint32("emova.u2s.flow.dilation_rate", 1)
    writer.add_uint32("emova.u2s.flow.n_layers", 4)
    writer.add_float32("emova.u2s.flow.noise_scale", 0.667)
    writer.add_float32("emova.u2s.length_scale", 1.0)
    # generator (HiFi-GAN)
    writer.add_uint32("emova.u2s.dec.upsample_initial_channel", 512)
    writer.add_array("emova.u2s.dec.upsample_rates", [8, 8, 2, 2])
    writer.add_array("emova.u2s.dec.upsample_kernel_sizes", [16, 16, 4, 4])
    writer.add_array("emova.u2s.dec.resblock_kernel_sizes", [3, 7, 11])
    writer.add_array("emova.u2s.dec.resblock_dilations", [1, 3, 5])  # shared by all 3 kernel groups
    writer.add_float32("emova.u2s.dec.lrelu_slope", 0.1)
    writer.add_uint32("emova.u2s.samples_per_unit", 882)   # 22050 / 25
    writer.add_float32("emova.u2s.units_per_second", 25.0)

    # ==== style / speaker embeddings ========================================
    put_plain("style_embedding.weight", "style_embedding.weight")
    put_plain("emb_g.weight", "decoder.emb_g.weight")  # unused at inference, kept for fidelity

    # ==== enc_p (TextEncoder, 6-layer relative-pos transformer) =============
    put_plain("enc_p.emb.weight", "decoder.enc_p.emb.weight")
    for i in range(6):
        p = f"decoder.enc_p.encoder"
        o = "enc_p.encoder"
        for nm in ["conv_q", "conv_k", "conv_v", "conv_o"]:
            put_plain(f"{o}.attn_layers.{i}.{nm}.weight", f"{p}.attn_layers.{i}.{nm}.weight")
            put_plain(f"{o}.attn_layers.{i}.{nm}.bias", f"{p}.attn_layers.{i}.{nm}.bias")
        put_plain(f"{o}.attn_layers.{i}.emb_rel_k", f"{p}.attn_layers.{i}.emb_rel_k")
        put_plain(f"{o}.attn_layers.{i}.emb_rel_v", f"{p}.attn_layers.{i}.emb_rel_v")
        for nm in ["norm_layers_1", "norm_layers_2"]:
            put_plain(f"{o}.{nm}.{i}.GAMMA", f"{p}.{nm}.{i}.GAMMA")
            put_plain(f"{o}.{nm}.{i}.BETA", f"{p}.{nm}.{i}.BETA")
        for nm in ["conv_1", "conv_2"]:
            put_plain(f"{o}.ffn_layers.{i}.{nm}.weight", f"{p}.ffn_layers.{i}.{nm}.weight")
            put_plain(f"{o}.ffn_layers.{i}.{nm}.bias", f"{p}.ffn_layers.{i}.{nm}.bias")
    put_plain("enc_p.proj.weight", "decoder.enc_p.proj.weight")
    put_plain("enc_p.proj.bias", "decoder.enc_p.proj.bias")

    # ==== dp (StochasticDurationPredictor) ==================================
    def put_ddsconv(dst_prefix, src_prefix, n_layers=3):
        for i in range(n_layers):
            put_plain(f"{dst_prefix}.convs_sep.{i}.weight", f"{src_prefix}.convs_sep.{i}.weight")
            put_plain(f"{dst_prefix}.convs_sep.{i}.bias", f"{src_prefix}.convs_sep.{i}.bias")
            put_plain(f"{dst_prefix}.convs_1x1.{i}.weight", f"{src_prefix}.convs_1x1.{i}.weight")
            put_plain(f"{dst_prefix}.convs_1x1.{i}.bias", f"{src_prefix}.convs_1x1.{i}.bias")
            put_plain(f"{dst_prefix}.norms_1.{i}.GAMMA", f"{src_prefix}.norms_1.{i}.GAMMA")
            put_plain(f"{dst_prefix}.norms_1.{i}.BETA", f"{src_prefix}.norms_1.{i}.BETA")
            put_plain(f"{dst_prefix}.norms_2.{i}.GAMMA", f"{src_prefix}.norms_2.{i}.GAMMA")
            put_plain(f"{dst_prefix}.norms_2.{i}.BETA", f"{src_prefix}.norms_2.{i}.BETA")

    def put_flows_list(dst_prefix, src_prefix):
        # index 0 = ElementwiseAffine (m, logs); indices 1,3,5,7 = ConvFlow
        # (pre/convs/proj); indices 2,4,6,8 = Flip (no params). Preserve the
        # original PyTorch ModuleList indices verbatim -- the C++ side's
        # "skip flows[1], keep flows[7,5,3] then flows[0]" reverse-mode logic
        # depends on these exact index values matching StochasticDurationPredictor.
        put_plain(f"{dst_prefix}.0.m", f"{src_prefix}.0.m")
        put_plain(f"{dst_prefix}.0.logs", f"{src_prefix}.0.logs")
        for i in [1, 3, 5, 7]:
            put_plain(f"{dst_prefix}.{i}.pre.weight", f"{src_prefix}.{i}.pre.weight")
            put_plain(f"{dst_prefix}.{i}.pre.bias", f"{src_prefix}.{i}.pre.bias")
            put_plain(f"{dst_prefix}.{i}.proj.weight", f"{src_prefix}.{i}.proj.weight")
            put_plain(f"{dst_prefix}.{i}.proj.bias", f"{src_prefix}.{i}.proj.bias")
            put_ddsconv(f"{dst_prefix}.{i}.convs", f"{src_prefix}.{i}.convs")

    put_plain("dp.pre.weight", "decoder.dp.pre.weight")
    put_plain("dp.pre.bias", "decoder.dp.pre.bias")
    put_plain("dp.proj.weight", "decoder.dp.proj.weight")
    put_plain("dp.proj.bias", "decoder.dp.proj.bias")
    put_plain("dp.cond.weight", "decoder.dp.cond.weight")
    put_plain("dp.cond.bias", "decoder.dp.cond.bias")
    put_ddsconv("dp.convs", "decoder.dp.convs")
    put_flows_list("dp.flows", "decoder.dp.flows")
    # training-only (post_*), unused at inference, kept for checkpoint fidelity
    put_plain("dp.post_pre.weight", "decoder.dp.post_pre.weight")
    put_plain("dp.post_pre.bias", "decoder.dp.post_pre.bias")
    put_plain("dp.post_proj.weight", "decoder.dp.post_proj.weight")
    put_plain("dp.post_proj.bias", "decoder.dp.post_proj.bias")
    put_ddsconv("dp.post_convs", "decoder.dp.post_convs")
    put_flows_list("dp.post_flows", "decoder.dp.post_flows")

    # ==== flow (ResidualCouplingBlock, 4 RCL + Flip alternation) ============
    for i in [0, 2, 4, 6]:
        p, o = f"decoder.flow.flows.{i}", f"flow.flows.{i}"
        put_plain(f"{o}.pre.weight", f"{p}.pre.weight")
        put_plain(f"{o}.pre.bias", f"{p}.pre.bias")
        put_plain(f"{o}.post.weight", f"{p}.post.weight")
        put_plain(f"{o}.post.bias", f"{p}.post.bias")
        put_fused(f"{o}.enc.cond_layer", f"{p}.enc.cond_layer")
        for li in range(4):
            put_fused(f"{o}.enc.in_layers.{li}", f"{p}.enc.in_layers.{li}")
            put_fused(f"{o}.enc.res_skip_layers.{li}", f"{p}.enc.res_skip_layers.{li}")

    # ==== dec (Generator / HiFi-GAN) ========================================
    put_plain("dec.conv_pre.weight", "decoder.dec.conv_pre.weight")
    put_plain("dec.conv_pre.bias", "decoder.dec.conv_pre.bias")
    put_plain("dec.cond.weight", "decoder.dec.cond.weight")
    put_plain("dec.cond.bias", "decoder.dec.cond.bias")
    put_plain("dec.conv_post.weight", "decoder.dec.conv_post.weight")  # bias=False
    for i in range(4):
        put_fused(f"dec.ups.{i}", f"decoder.dec.ups.{i}")
    for i in range(12):
        for cv in ["convs1", "convs2"]:
            for j in range(3):
                put_fused(f"dec.resblocks.{i}.{cv}.{j}", f"decoder.dec.resblocks.{i}.{cv}.{j}")

    # ==== enc_q (PosteriorEncoder) -- training-only, unused at inference ====
    put_plain("enc_q.pre.weight", "decoder.enc_q.pre.weight")
    put_plain("enc_q.pre.bias", "decoder.enc_q.pre.bias")
    put_plain("enc_q.proj.weight", "decoder.enc_q.proj.weight")
    put_plain("enc_q.proj.bias", "decoder.enc_q.proj.bias")
    put_fused("enc_q.enc.cond_layer", "decoder.enc_q.enc.cond_layer")
    for li in range(16):
        put_fused(f"enc_q.enc.in_layers.{li}", f"decoder.enc_q.enc.in_layers.{li}")
        put_fused(f"enc_q.enc.res_skip_layers.{li}", f"decoder.enc_q.enc.res_skip_layers.{li}")

    # ==== enc_r (MelStyleEncoder) -- unused at inference (style comes from
    # the fixed style_embedding prototypes, not a reference mel) ============
    put_plain("enc_r.fc.fc.weight", "decoder.enc_r.fc.fc.weight")
    put_plain("enc_r.fc.fc.bias", "decoder.enc_r.fc.fc.bias")
    for nm in ["w_qs", "w_ks", "w_vs", "fc"]:
        put_plain(f"enc_r.slf_attn.{nm}.weight", f"decoder.enc_r.slf_attn.{nm}.weight")
        put_plain(f"enc_r.slf_attn.{nm}.bias", f"decoder.enc_r.slf_attn.{nm}.bias")
    for i in [0, 3]:  # nn.Sequential(LinearNorm, Mish, Dropout, LinearNorm, Mish, Dropout)
        put_plain(f"enc_r.spectral.{i}.fc.weight", f"decoder.enc_r.spectral.{i}.fc.weight")
        put_plain(f"enc_r.spectral.{i}.fc.bias", f"decoder.enc_r.spectral.{i}.fc.bias")
    for i in range(2):  # nn.Sequential(Conv1dGLU, Conv1dGLU), no interspersed no-param layers
        put_plain(f"enc_r.temporal.{i}.conv1.conv.weight", f"decoder.enc_r.temporal.{i}.conv1.conv.weight")
        put_plain(f"enc_r.temporal.{i}.conv1.conv.bias", f"decoder.enc_r.temporal.{i}.conv1.conv.bias")

    print(f"writing {n_tensors} tensors to {args.dst}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print("done.")
    print("size on disk:", args.dst.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
