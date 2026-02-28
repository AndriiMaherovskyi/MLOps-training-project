import sys
import os
from pathlib import Path
import pandas as pd

from data_loader import load_data, merge_data


def main(raw_dir, output_dir):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    features, train_data, stores = load_data()

    # Merge
    train_df = merge_data(features, train_data, stores)

    # Save prepared dataset
    output_file = output_dir / "train_full.csv"
    train_df.to_csv(output_file, index=False)

    print(f"Prepared data saved to {output_file}")


if __name__ == "__main__":
    raw_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(raw_dir, output_dir)