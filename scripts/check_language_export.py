#!/usr/bin/env python3
"""Cross-check an ONNX export of a pocket-tts >= 3 language model against the OFFICIAL pocket-tts >= 3
code (run this in a pocket-tts >= 3 environment, offline, float32 on CPU):

  - voice conditioning: official _encode_audio + bos_before_voice prepend (as in
    get_state_for_audio_prompt) vs mimi_encoder.onnx;
  - text conditioning: official tokenizer.json + text conditioner vs tokenizer.model (SentencePiece,
    used by the C++ runtime) + text_conditioner.onnx.

Usage:
  python check_language_export.py --config <pocket_tts/config/<lang>.yaml> --weights <model.safetensors>
      --tokenizer-json <tokenizer.json> --onnx-dir <export dir> --audio <clip.wav> --text "<sentence>"
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer-json", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True)
    a = ap.parse_args()

    import onnxruntime as ort
    import sentencepiece as spm
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from pocket_tts.models.tts_model import TTSModel
    from pocket_tts.utils.config import load_config

    cfg = load_config(Path(a.config))
    cfg.weights_path = a.weights
    cfg.flow_lm.lookup_table.tokenizer_path = a.tokenizer_json
    model = TTSModel._from_pydantic_config_with_weights(cfg, 0.7, 1, None, -4.0)
    model.eval()
    ok = True

    # Voice conditioning.
    wav, sr = sf.read(a.audio, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 24000, "use a 24 kHz clip"
    audio = torch.from_numpy(wav)[None, None, :]
    pad = (1920 - audio.shape[-1] % 1920) % 1920
    audio = F.pad(audio, (0, pad))
    with torch.no_grad():
        prompt = model._encode_audio(audio)
        if model.flow_lm.insert_bos_before_voice:
            prompt = torch.cat([model.flow_lm.bos_before_voice, prompt], dim=1)
    enc = ort.InferenceSession(str(Path(a.onnx_dir) / "mimi_encoder.onnx"))
    onnx_cond = enc.run(None, {"audio": audio.numpy()})[0]
    py = prompt.numpy()
    print(f"voice conditioning: official {py.shape}, onnx {onnx_cond.shape}")
    if py.shape != onnx_cond.shape:
        ok = False
    else:
        d = np.abs(py - onnx_cond)
        cos = (py * onnx_cond).sum(-1) / (np.linalg.norm(py, axis=-1) * np.linalg.norm(onnx_cond, axis=-1))
        print(f"  max abs diff {d.max():.2e}, worst frame cosine {cos.min():.6f}")
        ok &= bool(cos.min() > 0.9999)

    # Text conditioning (tokenizer + embeddings).
    official_ids = model.flow_lm.conditioner.tokenizer.encode(a.text) if hasattr(
        model.flow_lm.conditioner, "tokenizer") else None
    sp = spm.SentencePieceProcessor(model_file=str(Path(a.onnx_dir) / "tokenizer.model"))
    sp_ids = sp.encode(a.text)
    print(f"text tokens: official {official_ids}\n             spm      {sp_ids}")
    if official_ids is not None:
        ok &= list(official_ids) == list(sp_ids)
    txt = ort.InferenceSession(str(Path(a.onnx_dir) / "text_conditioner.onnx"))
    in_name = txt.get_inputs()[0].name
    onnx_emb = txt.run(None, {in_name: np.array([sp_ids], dtype=np.int64)})[0]
    with torch.no_grad():
        py_emb = model.flow_lm.conditioner(torch.tensor([sp_ids])).numpy() if callable(
            getattr(model.flow_lm.conditioner, "forward", None)) else None
    if py_emb is not None and hasattr(py_emb, "shape"):
        d = np.abs(py_emb.reshape(onnx_emb.shape) - onnx_emb).max()
        print(f"text embeddings: max abs diff {d:.2e}")
        ok &= bool(d < 1e-4)

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
