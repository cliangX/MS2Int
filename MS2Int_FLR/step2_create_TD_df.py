#!/usr/bin/env python3
"""
DeepFLR2: Create Minimal ModelResult Template
=============================================
This script creates a minimal modelresult template from step1_target_decoy.csv that can be used
directly with Step4/Step5, bypassing Step2 and Step3. Users need to add their own 'score' column.

Usage:
    python DeepFLR2_create_minimal_modelresult.py --inputfile step1_target_decoy.csv --outputfile modelresult_template.csv

Required columns for Step4/5:
- SourceFile, Fspectrum, PP.Charge, key_x, PEP.StrippedSequence, score (user-provided)
"""

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create minimal modelresult template from step1_target_decoy.csv")
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="Input file: step1_target_decoy.csv from DeepFLR1"
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="Output file: minimal modelresult template (CSV format)"
    )
    parser.add_argument(
        "--add_dummy_score",
        action="store_true",
        help="Add a dummy 'score' column filled with 0.5 for testing"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="精简输出，仅打印关键数量信息"
    )
    
    args = parser.parse_args()
    
    if not args.quiet:
        print(f"Reading input file: {args.inputfile}")
    df = pd.read_csv(args.inputfile)
    
    # 关键数量信息
    print(f"输入条目: {len(df)}")
    
    # Map columns to Step4/5 requirements
    # step1_target_decoy.csv has: SourceFile, Spectrum, PP.Charge, exp_strip_sequence, Peptide, key
    # Step4/5 needs: SourceFile, Fspectrum, PP.Charge, key_x, PEP.StrippedSequence, score
    
    modelresult = df.rename(columns={
        "Spectrum": "Fspectrum",           # Spectrum -> Fspectrum
        "key": "key_x",                    # key -> key_x  
        "exp_strip_sequence": "PEP.StrippedSequence"  # exp_strip_sequence -> PEP.StrippedSequence
    })
    
    # Select基础列（后面会在此基础上增加 target/decoy 标记列）
    required_cols = ["SourceFile", "Fspectrum", "PP.Charge", "key_x", "PEP.StrippedSequence"]
    modelresult = modelresult[required_cols].copy()
    
    # Remove duplicates (should be candidate-level unique)
    if not args.quiet:
        print(f"Before deduplication: {len(modelresult)} rows")
    modelresult = modelresult.drop_duplicates()
    print(f"去重后: {len(modelresult)}")
    
    # Ensure proper data types
    modelresult["Fspectrum"] = modelresult["Fspectrum"].astype(str)
    modelresult["SourceFile"] = modelresult["SourceFile"].astype(str)
    modelresult["PP.Charge"] = modelresult["PP.Charge"].astype(int)

    # ------------------------------------------------------------------
    # 新增: 基于 DeepFLR4 同样的 strip_key 逻辑，预先给每一行打上 target/decoy 标记
    # 约定:
    #   - 去掉 key_x 中的 1/2/3/4 后得到 strip_key
    #   - strip_key == PEP.StrippedSequence 视为 target，否则为 decoy
    # 这样在 step3_modelresult.csv 中就能直接区分 target/decoy 行
    # ------------------------------------------------------------------
    strip_key = modelresult["key_x"].astype(str)
    for ch in ["1", "2", "3", "4"]:
        strip_key = strip_key.str.replace(ch, "", regex=False)

    modelresult["strip_key"] = strip_key
    modelresult["is_decoy"] = modelresult["strip_key"] != modelresult["PEP.StrippedSequence"]
    # 可选的人类可读标签
    modelresult["TD_label"] = modelresult["is_decoy"].map({False: "target", True: "decoy"})
    
    # Add dummy score if requested
    if args.add_dummy_score:
        modelresult["score"] = 0.5  # Dummy score for testing
        print("✅ Added dummy 'score' column with value 0.5")
    else:
        if not args.quiet:
            print("ℹ️  No 'score' column added. You need to merge your own scores before using with Step4/5.")
    
    if not args.quiet:
        print(f"Output columns: {list(modelresult.columns)}")
        print(f"Saving results to: {args.outputfile}")
    modelresult.to_csv(args.outputfile, index=False)
    
    if not args.quiet:
        print("✅ DeepFLR2 completed successfully!")
        print("\n📋 Next steps:")
        print("1. Merge your 'score' column to this template file")
        print("2. Ensure the merge keys are: SourceFile, Fspectrum, PP.Charge, key_x")
        print("3. Use the final file with DeepFLR4 and DeepFLR5")
        print("\n💡 Example merge code:")
        print("```python")
        print("template = pd.read_csv('modelresult_template.csv')")
        print("scores = pd.read_csv('your_scores.csv')  # must contain merge keys + 'score'")
        print("final = template.merge(scores, on=['SourceFile','Fspectrum','PP.Charge','key_x'], how='inner')")
        print("final.to_csv('modelresult_with_score.csv', index=False)")
        print("```")

if __name__ == "__main__":
    main()
