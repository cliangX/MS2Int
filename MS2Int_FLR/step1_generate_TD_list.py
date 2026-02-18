#!/usr/bin/env python3
"""Generate target/decoy phosphopeptide sequences (mono-phospho) from MaxQuant msms.txt."""

import pandas as pd
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfile", type=str, required=True)
    parser.add_argument("--outputfile", type=str, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    df = pd.read_table(args.inputfile, delimiter="\t")

    out = pd.DataFrame(columns=["SourceFile","Spectrum","PP.Charge","Peptide","exp_strip_sequence","key"])
    out["SourceFile"] = df["Raw file"]
    out["Spectrum"] = df["Scan number"]
    out["PP.Charge"] = df["Charge"]
    out["Peptide"] = df["Modified sequence"]

    out = out.drop_duplicates(keep="first").reset_index(drop=True)

    # Encode modifications: Phospho->1, Oxidation->2, Carbamidomethyl->3, Acetyl->4
    out["Peptide"] = out["Peptide"].str.replace("_",'',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Phospho (STY))",'1',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Phospho(Y))",'1',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Phospho(S))",'1',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Phospho(T))",'1',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Phospho (Y))",'1',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Phospho (S))",'1',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Phospho (T))",'1',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Oxidation (M))",'2',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("(Acetyl (Protein N-term))",'4',regex=False)
    out["Peptide"] = out["Peptide"].str.replace("C",'C3',regex=False)

    out = out.loc[~out["Peptide"].str.contains("4M2")]
    out = out.loc[~out["Peptide"].str.contains("4S")]
    out = out.loc[~out["Peptide"].str.contains("4T")]
    out = out.loc[~out["Peptide"].str.contains("4Y")]
    out = out.loc[~out["Peptide"].str.contains("4C")]

    out["exp_strip_sequence"] = out["Peptide"].str.replace("1",'',regex=False)
    out["exp_strip_sequence"] = out["exp_strip_sequence"].str.replace("4",'',regex=False)
    out["exp_strip_sequence"] = out["exp_strip_sequence"].str.replace("3",'',regex=False)
    out["exp_strip_sequence"] = out["exp_strip_sequence"].str.replace("2",'',regex=False)
    out["key"] = out["Peptide"].str.replace("1",'',regex=False)

    # Keep mono-phospho peptides only
    out = out.loc[out["Peptide"].str.count("1") == 1]

    sty_counts = (
        out["exp_strip_sequence"].str.count("S")
        + out["exp_strip_sequence"].str.count("T")
        + out["exp_strip_sequence"].str.count("Y")
    )

    # Need >1 S/T/Y to generate decoys
    df_multi_sty = out.loc[sty_counts > 1].reset_index(drop=True)
    df_single_sty = out.loc[sty_counts == 1].reset_index(drop=True)

    print(f"Decoy-eligible peptides: {len(df_multi_sty)}")
    print(f"Filtered (single S/T/Y): {len(df_single_sty)}")

    down = 0
    result = pd.DataFrame(
        columns=["SourceFile", "Spectrum", "PP.Charge", "exp_strip_sequence", "Peptide", "key"]
    )

    df = df_multi_sty
    for k in range(len(df)):
        count = 0
        sequence = df.loc[k,"key"]
        sequence = list(sequence)
        SourceFile = df.loc[k,"SourceFile"]
        Spectrum = df.loc[k, "Spectrum"]
        Charge = df.loc[k, "PP.Charge"]
        Peptide = df.loc[k, "Peptide"]
        exp_strip_sequence = df.loc[k, "exp_strip_sequence"]
        y = list(range(len(sequence)))
        sty_list = []

        for x in range(len(sequence)):
            if sequence[x] in ["S","T","Y"]:
                sty_list.append(x)
                sequence.insert(x+1,"1")
                sequence1 = ''.join(sequence)
                result.loc[down] = [SourceFile,Spectrum,Charge,exp_strip_sequence,Peptide,sequence1]
                down += 1
                sequence.remove("1")
                count += 1
                y.remove(x)
            if sequence[x] == "2":
                y.remove(x)
                y.remove(x-1)
            if sequence[x] == "3":
                y.remove(x)
                y.remove(x-1)
            if sequence[x] == "4":
                y.remove(x)
                y.remove(x+1)

            if x == len(sequence)-1:
                stynum = 0
                if count <= len(y):
                    b = random.sample(y, count)
                    for c in b:
                        sty = int(sty_list[stynum])
                        sequence[c],sequence[sty] = sequence[sty],sequence[c]
                        sequence.insert(c + 1, "1")
                        stynum += 1
                        sequence1 = ''.join(sequence)
                        result.loc[down] = [SourceFile,Spectrum,Charge,exp_strip_sequence,Peptide,sequence1]
                        down += 1
                        sequence.remove("1")
                        sequence[c],sequence[sty] = sequence[sty],sequence[c]
                else:
                    b = y
                    for c in b:
                        sty = int(sty_list[stynum])
                        sequence[c],sequence[sty] = sequence[sty],sequence[c]
                        sequence.insert(c + 1, "1")
                        stynum += 1
                        sequence1 = ''.join(sequence)
                        result.loc[down] = [SourceFile,Spectrum,Charge,exp_strip_sequence,Peptide,sequence1]
                        down += 1
                        sequence.remove("1")
                        sequence[c],sequence[sty] = sequence[sty],sequence[c]

    print(f"Generated target/decoy sequences: {len(result)}")
    result.to_csv(args.outputfile, index=None)


if __name__ == "__main__":
    main()
