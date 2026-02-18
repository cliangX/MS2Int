#!/usr/bin/env python3
"""Create target/decoy DataFrame from step1 TD list for downstream scoring."""

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", type=str, required=True)
    parser.add_argument("--outputfile", type=str, required=True)
    parser.add_argument("--add_dummy_score", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.inputfile)
    print(f"Input entries: {len(df)}")

    modelresult = df.rename(columns={
        "Spectrum": "Fspectrum",
        "key": "key_x",
        "exp_strip_sequence": "PEP.StrippedSequence",
    })

    required_cols = ["SourceFile", "Fspectrum", "PP.Charge", "key_x", "PEP.StrippedSequence"]
    modelresult = modelresult[required_cols].copy()
    modelresult = modelresult.drop_duplicates()
    print(f"After dedup: {len(modelresult)}")

    modelresult["Fspectrum"] = modelresult["Fspectrum"].astype(str)
    modelresult["SourceFile"] = modelresult["SourceFile"].astype(str)
    modelresult["PP.Charge"] = modelresult["PP.Charge"].astype(int)

    # Target/decoy labeling: strip modification codes from key_x
    strip_key = modelresult["key_x"].astype(str)
    for ch in ["1", "2", "3", "4"]:
        strip_key = strip_key.str.replace(ch, "", regex=False)

    modelresult["strip_key"] = strip_key
    modelresult["is_decoy"] = modelresult["strip_key"] != modelresult["PEP.StrippedSequence"]
    modelresult["TD_label"] = modelresult["is_decoy"].map({False: "target", True: "decoy"})

    if args.add_dummy_score:
        modelresult["score"] = 0.5

    modelresult.to_csv(args.outputfile, index=False)


if __name__ == "__main__":
    main()
