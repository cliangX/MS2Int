#!/usr/bin/env python3
"""Generate SpecId TSV from filtered msms.txt."""

import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path, sep='\t')

    required_cols = ['Raw file', 'Scan number', 'Sequence', 'Charge']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df['Sequence'] = df['Sequence'].astype(str).str.replace(
        "C",
        "C[UNIMOD:4]",
        regex=False
    )

    df['SpecId'] = (
        df['Raw file'].astype(str)
        + "-" + df['Scan number'].astype(str)
        + "-" + df['Sequence'].astype(str)
        + "-" + df['Charge'].astype(str)
    )

    df[['SpecId']].to_csv(output_path, sep='\t', index=False)
    print(f"{len(df)} SpecIds -> {output_path}")

if __name__ == "__main__":
    main()
