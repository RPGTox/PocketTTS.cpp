#!/usr/bin/env python3
"""Convert a Pocket TTS `tokenizer.json` (HuggingFace tokenizers, Unigram + byte fallback) back into
a SentencePiece `tokenizer.model`, so the C++ runtime keeps using SentencePiece for every language.

pocket-tts >= 3 ships per-language `tokenizer.json` files, documented as "converted from the
SentencePiece model, so it produces identical token ids". The SentencePiece settings are those of
the original English `tokenizer.model` (identity normalizer, dummy prefix, no whitespace collapsing,
byte fallback, digits split); only the pieces and scores change per language.

The conversion is verified: token ids of the SentencePiece model and of the HuggingFace tokenizer
must be identical on the verification sentences (and any --check-file lines).

Usage:
  python tokenizer_json_to_spm.py --json tokenizer.json --out tokenizer.model [--lang french]
                                  [--check-file sentences.txt]
"""
import argparse
import json
import sys

import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as pb
from tokenizers import Tokenizer

SAMPLES = {
    "english": ["Hello there, traveler. The road to Baldur's Gate is long!",
                "It costs 250 gold pieces, not a copper less: 1,000 is too much.",
                "Wait... did you hear that? \"Run!\" - she said (quietly)."],
    "french": ["Bonjour, voyageur. La route est longue, et la nuit est froide.",
               "Ça coûte 250 pièces d'or, pas une de moins ; c'est déjà beaucoup.",
               "Où êtes-vous allés ? « Fuyez ! » cria-t-elle, à l'aube."],
    "german": ["Guten Tag, Reisender. Die Straße ist lang und die Nacht ist kalt.",
               "Das kostet 250 Goldstücke, keinen Heller weniger. Äpfel, Öl, Übermut!"],
    "spanish": ["Hola, viajero. El camino es largo y la noche es fría.",
                "¿Cuánto cuesta? ¡Doscientas cincuenta monedas de oro, ni una menos!"],
    "italian": ["Ciao, viaggiatore. La strada è lunga e la notte è fredda.",
                "Costa 250 monete d'oro, non una di meno. Perché?"],
    "portuguese": ["Olá, viajante. A estrada é longa e a noite está fria.",
                   "Custa 250 moedas de ouro, nem uma a menos. Não é?"],
    "dutch": ["Hallo, reiziger. De weg is lang en de nacht is koud.",
              "Het kost 250 goudstukken, geen cent minder. Één, twee!"],
}


def convert(json_path: str) -> pb.ModelProto:
    t = json.load(open(json_path, encoding="utf-8"))
    model = t["model"]
    if model.get("type") != "Unigram" or not model.get("byte_fallback"):
        raise SystemExit(f"{json_path}: expected a Unigram tokenizer with byte_fallback")
    special = {tok["content"] for tok in t.get("added_tokens", []) if tok.get("special")}
    unk_id = model.get("unk_id", 0)

    m = pb.ModelProto()
    ts = m.trainer_spec
    ts.model_type = pb.TrainerSpec.UNIGRAM
    ts.vocab_size = len(model["vocab"])
    ts.byte_fallback = True
    ts.split_by_whitespace = True
    ts.split_digits = True
    ts.treat_whitespace_as_suffix = False
    ts.unk_id, ts.bos_id, ts.eos_id, ts.pad_id = unk_id, 1, 2, 3
    ns = m.normalizer_spec
    ns.name = "identity"
    ns.precompiled_charsmap = b""
    ns.add_dummy_prefix = True
    ns.remove_extra_whitespaces = False
    ns.escape_whitespaces = True

    for i, (piece, score) in enumerate(model["vocab"]):
        p = m.pieces.add()
        p.piece, p.score = piece, float(score)
        if i == unk_id:
            p.type = pb.ModelProto.SentencePiece.UNKNOWN
        elif piece in special:
            p.type = pb.ModelProto.SentencePiece.CONTROL
        elif len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            p.type = pb.ModelProto.SentencePiece.BYTE
        else:
            p.type = pb.ModelProto.SentencePiece.NORMAL
    return m


def verify(model_bytes: bytes, json_path: str, sentences) -> int:
    sp = spm.SentencePieceProcessor(model_proto=model_bytes)
    hf = Tokenizer.from_file(json_path)
    bad = 0
    for s in sentences:
        a, b = sp.encode(s), hf.encode(s).ids
        if a != b:
            bad += 1
            print(f"  MISMATCH: {s!r}\n    spm: {a}\n    hf : {b}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="")
    ap.add_argument("--check-file", help="extra sentences, one per line (UTF-8)")
    a = ap.parse_args()

    m = convert(a.json)
    data = m.SerializeToString()
    sentences = [s for lang in SAMPLES for s in SAMPLES[lang]]  # every language: robustness
    if a.check_file:
        sentences += [l.strip() for l in open(a.check_file, encoding="utf-8") if l.strip()]
    bad = verify(data, a.json, sentences)
    if bad:
        raise SystemExit(f"{bad} of {len(sentences)} sentences tokenize differently: not written")
    with open(a.out, "wb") as f:
        f.write(data)
    print(f"{a.out}: {len(m.pieces)} pieces, {len(sentences)} sentences tokenize identically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
