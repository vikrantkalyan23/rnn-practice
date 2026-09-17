import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from app.config import MODEL_CONFIG_PATH, MODEL_PATH, TOKENIZER_PATH  # noqa: E402
from app.predictor import NextWordPredictor  # noqa: E402


def read_options():
    parser = argparse.ArgumentParser(description="Generate words with the trained model.")
    parser.add_argument("text", nargs="?", default="machine learning")
    parser.add_argument("--words", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args()


def main():
    options = read_options()
    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    predictor = NextWordPredictor(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        sequence_length=config["sequence_length"],
    )

    print(f"Input: {options.text}")
    print("\nTop next-word candidates:")
    for word, probability in predictor.predict_top_words(
        options.text,
        top_k=options.top_k,
        temperature=options.temperature,
    ):
        print(f"  {word:<15} {probability:.2%}")

    generated = predictor.generate_text(
        options.text,
        next_words=options.words,
        temperature=options.temperature,
        top_k=options.top_k,
        sample=options.sample,
    )
    print(f"\nGenerated: {generated}")


if __name__ == "__main__":
    main()
