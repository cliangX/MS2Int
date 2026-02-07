#!/usr/bin/env python3
"""
DeepFLR1: Generate Target/Decoy Phosphopeptide Sequences
======================================================
This script generates target and decoy phosphopeptide sequences for mono-phosphorylated peptides
from MaxQuant msms.txt results.

Usage:
    python DeepFLR1_generate_target_decoy.py --inputfile msms.txt --outputfile step1_target_decoy.csv

Original source: sequence_generation/step1_Targetdecoy_phosphopeptides_generation_mono.py
"""

import pandas as pd
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate target/decoy phosphopeptide sequences for mono-phosphorylated peptides")
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="Input file: MaxQuant searching result (msms.txt)"
    )
    parser.add_argument(
        "--outputfile", 
        type=str,
        required=True,
        help="Output file: Target/decoy sequence file (CSV format)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="精简输出，仅打印关键数量信息"
    )
    
    args = parser.parse_args()
    
    if not args.quiet:
        print(f"Reading input file: {args.inputfile}")
    df = pd.read_table(args.inputfile, delimiter="\t")
    
    # Initialize output dataframe
    out = pd.DataFrame(columns=["SourceFile","Spectrum","PP.Charge","Peptide","exp_strip_sequence","key"])
    out["SourceFile"] = df["Raw file"]
    out["Spectrum"] = df["Scan number"]
    out["PP.Charge"] = df["Charge"]
    out["Peptide"] = df["Modified sequence"]
    
    # Remove duplicates and reset index
    out = out.drop_duplicates(keep="first")
    out.reset_index(drop=True, inplace=True)
    
    # Process peptide modifications
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
    
    # Filter out unwanted modifications
    out = out.loc[~out["Peptide"].str.contains("4M2")]
    out = out.loc[~out["Peptide"].str.contains("4S")]
    out = out.loc[~out["Peptide"].str.contains("4T")]
    out = out.loc[~out["Peptide"].str.contains("4Y")]
    out = out.loc[~out["Peptide"].str.contains("4C")]
    
    # Create stripped sequence and key
    out["exp_strip_sequence"] = out["Peptide"].str.replace("1",'',regex=False)
    out["exp_strip_sequence"] = out["exp_strip_sequence"].str.replace("4",'',regex=False)
    out["exp_strip_sequence"] = out["exp_strip_sequence"].str.replace("3",'',regex=False)
    out["exp_strip_sequence"] = out["exp_strip_sequence"].str.replace("2",'',regex=False)
    out["key"] = out["Peptide"].str.replace("1",'',regex=False)
    
    # 仅保留“单磷酸肽段”（1 个修饰位点），这是 DeepFLR 的基础假设
    out = out.loc[out["Peptide"].str.count("1") == 1]

    # 统计每条肽中 S/T/Y 的个数
    sty_counts = (
        out["exp_strip_sequence"].str.count("S")
        + out["exp_strip_sequence"].str.count("T")
        + out["exp_strip_sequence"].str.count("Y")
    )

    # 分两类：
    # 1) sty_counts > 1: 可以生成 decoy（要求 >1 个 S/T/Y）
    # 2) sty_counts == 1: 仅 1 个 S/T/Y（本次分析中直接丢弃，不生成 target/decoy）
    df_multi_sty = out.loc[sty_counts > 1].reset_index(drop=True)
    df_single_sty = out.loc[sty_counts == 1].reset_index(drop=True)

    # 关键数量信息
    print(f"可生成 decoy 的肽段数: {len(df_multi_sty)}")
    print(f"过滤 1 个 S/T/Y 的肽段数: {len(df_single_sty)}")

    # 统一的结果 DataFrame
    down = 0
    result = pd.DataFrame(
        columns=["SourceFile", "Spectrum", "PP.Charge", "exp_strip_sequence", "Peptide", "key"]
    )

    # -------- 仅处理：可生成 decoy 的肽段（>1 个 S/T/Y） --------
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
            # Generate target sequences (phosphorylation on S/T/Y)
            if sequence[x] in ["S","T","Y"]:
                sty_list.append(x)
                sequence.insert(x+1,"1")
                sequence1 = ''.join(sequence)
                result.loc[down] = [SourceFile,Spectrum,Charge,exp_strip_sequence,Peptide,sequence1]
                down += 1
                sequence.remove("1")
                count += 1
                y.remove(x)
            # Remove positions that cannot be used for decoy generation
            if sequence[x] == "2":
                y.remove(x)
                y.remove(x-1)
            if sequence[x] == "3":
                y.remove(x)
                y.remove(x-1)
            if sequence[x] == "4":
                y.remove(x)
                y.remove(x+1)
            
            # Generate decoy sequences at the end of sequence processing
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
    print(f"生成 target/decoy 序列: {len(result)}")
    if not args.quiet:
        print(f"Saving results to: {args.outputfile}")
    result.to_csv(args.outputfile, index=None)
    if not args.quiet:
        print("✅ DeepFLR1 completed successfully!")

if __name__ == "__main__":
    main()
