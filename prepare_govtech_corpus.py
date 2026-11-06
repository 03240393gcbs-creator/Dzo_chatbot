"""Validate and convert an authorised Dzongkha parallel corpus for fine-tuning.

Example:
  python prepare_govtech_corpus.py --input data/authorised_corpus.csv \
      --english-column english --dzongkha-column dzongkha

This script does not download GovTech/DCDD data. Use it only with a dataset you
have permission to access and redistribute/use for model training.
"""
import argparse
from pathlib import Path
from govtech_dnlp import read_parallel_pairs, export_training_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV/TSV/JSON/JSONL/XLSX corpus file")
    parser.add_argument("--english-column", default="english")
    parser.add_argument("--dzongkha-column", default="dzongkha")
    parser.add_argument("--output", default="data/govtech_parallel.jsonl")
    args = parser.parse_args()
    pairs = read_parallel_pairs(args.input, args.english_column, args.dzongkha_column)
    if not pairs:
        raise SystemExit("No valid English/Dzongkha pairs found. Check column names and script direction.")
    count = export_training_jsonl(pairs, args.output)
    print(f"Saved {count:,} validated pairs to {args.output}")
    print(f"Next: python train_en2dz.py --jsonl {args.output} --dry-run")
    print("Training a 600M model is resource-intensive; use a suitable GPU runtime.")


if __name__ == "__main__":
    main()
