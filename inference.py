"""
Inference / Demo Script for Sourashtra Translation
====================================================
Load a trained model and translate interactively or from a file.

Usage:
    python inference.py                    # Interactive mode
    python inference.py --input words.txt  # File mode
"""
import os
import sys
import argparse
import pickle
import torch

from config import Config
from model import build_model


def load_model(config):
    """Load trained model and vocabularies."""
    # Load vocabularies
    with open(os.path.join(config.CHECKPOINT_DIR, "src_vocab.pkl"), "rb") as f:
        src_vocab = pickle.load(f)
    with open(os.path.join(config.CHECKPOINT_DIR, "tgt_vocab.pkl"), "rb") as f:
        tgt_vocab = pickle.load(f)

    # Build and load model
    model = build_model(config, len(src_vocab), len(tgt_vocab))
    ckpt = torch.load(
        os.path.join(config.CHECKPOINT_DIR, "best_model.pt"),
        map_location=config.DEVICE, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, src_vocab, tgt_vocab


def translate(model, text, src_vocab, tgt_vocab, device):
    """Translate a single Sourashtra word/phrase to English."""
    src_indices, _ = src_vocab.encode(text.strip(), 80)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

    decoded_indices, attn = model.translate(
        src_tensor, tgt_vocab.sos_idx, tgt_vocab.eos_idx, max_len=120
    )
    return tgt_vocab.decode(decoded_indices)


def interactive_mode(model, src_vocab, tgt_vocab, device):
    """Interactive translation loop."""
    print("\n" + "=" * 50)
    print("  Sourashtra → English Translator")
    print("  Type a Sourashtra word in Roman script")
    print("  Type 'quit' to exit")
    print("=" * 50)

    while True:
        try:
            text = input("\nSourashtra > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not text:
                continue

            result = translate(model, text, src_vocab, tgt_vocab, device)
            print(f"English   > {result}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def file_mode(model, src_vocab, tgt_vocab, device, input_file, output_file=None):
    """Translate all words from a file."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    results = []
    for line in lines:
        translation = translate(model, line, src_vocab, tgt_vocab, device)
        results.append(f"{line}\t{translation}")
        print(f"  {line} → {translation}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(results))
        print(f"\nSaved translations to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sourashtra Translation Inference")
    parser.add_argument("--input", type=str, help="Input file with one word per line")
    parser.add_argument("--output", type=str, help="Output file for translations")
    args = parser.parse_args()

    config = Config()
    print("Loading model...")
    model, src_vocab, tgt_vocab = load_model(config)
    print(f"Model loaded on {config.DEVICE}")

    if args.input:
        file_mode(model, src_vocab, tgt_vocab, config.DEVICE, args.input, args.output)
    else:
        interactive_mode(model, src_vocab, tgt_vocab, config.DEVICE)


if __name__ == "__main__":
    main()
