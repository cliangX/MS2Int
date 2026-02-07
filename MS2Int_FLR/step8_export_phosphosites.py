#!/usr/bin/env python3
"""
DeepFLR5: Phosphosite Export and Localization
=============================================
This script exports phosphorylation site localization results based on the selected FLR cutoff
from DeepFLR4 analysis.

Usage:
    python DeepFLR5_phosphosite_export.py --modelresultfile modelresult_with_score.csv --sequencefile step1_target_decoy.csv --inputfile1 msms.txt --inputfile2 Phospho_STY_Sites.txt --cutoff 0.02 --outputresult step5_phosphosites.csv

Original source: result_processing/step5_DeepFLR_result_processing.py
"""

import pandas as pd
import numpy as np
import argparse
import re

def ace(instance):
    """Adjust acetyl modification position"""
    if instance[1] == "4":
        instance = list(instance)
        instance.remove("4")
        instance.insert(0, "4")
        instance = ''.join(instance)
    return instance

def main():
    parser = argparse.ArgumentParser(description="Export phosphosite localization results based on FLR cutoff")
    parser.add_argument(
        "--modelresultfile",
        type=str,
        required=True,
        help="Input file: modelresult from mgfprocess.py or DeepFLR2 output with scores"
    )
    parser.add_argument(
        "--sequencefile",
        type=str,
        required=True,
        help="Input file: sequencefile from Targetdecoy_phosphopeptides_generation (step1_target_decoy.csv)"
    )
    parser.add_argument(
        "--outputresult",
        type=str,
        default="step5_phosphosites.csv",
        help="Output file: phosphosites localized by DeepFLR for target FLR"
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        required=True,
        help="Cutoff for estimated FLR (delta score threshold)"
    )
    parser.add_argument(
        "--inputfile1",
        type=str,
        required=True,
        help="Input file: MaxQuant searching result (msms.txt)"
    )
    parser.add_argument(
        "--inputfile2", 
        type=str,
        required=True,
        help="Input file: MaxQuant phosphosite result (Phospho (STY)Sites.txt)"
    )
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.modelresultfile)
    
    # Handle score column format (compatibility with old tensor format)
    if df["score"].dtype == 'object':
        df["score"] = df["score"].str.replace("tensor(",'',regex=False)
        df["score"] = df["score"].str.replace(".)",'',regex=False)
        df["score"] = df["score"].str.replace(")",'',regex=False)
    df["score"] = df["score"].astype("float")
    
    df1 = pd.read_csv(args.sequencefile)
    
    # Standardize column names for sequencefile
    df1.columns = ["SourceFile","Fspectrum","PP.Charge","exp_strip_sequence","Peptide","key_x"]
    
    df = pd.merge(df, df1, on=["SourceFile","Fspectrum","key_x","PP.Charge"], how="left")
    df = df[["SourceFile","Fspectrum","Peptide","PEP.StrippedSequence","PP.Charge","key_x","score"]]
    
    # Create stripped sequence for target/decoy classification
    df["striptrue"] = df["Peptide"].str.replace("1",'',regex=False)
    df["striptrue"] = df["striptrue"].str.replace("2",'',regex=False)
    df["striptrue"] = df["striptrue"].str.replace("3",'',regex=False)
    df["striptrue"] = df["striptrue"].str.replace("4",'',regex=False)
    df["score"] = df["score"].astype("float")
    
    # Sort by score (highest first)
    df = df.sort_values(by='score', ascending=False)
    df.reset_index(drop=True, inplace=True)
    
    # Get best scoring hit per spectrum-peptide combination
    dfmax = df.drop_duplicates(subset=['SourceFile', 'Fspectrum', 'Peptide'])
    dftruemax = dfmax.loc[dfmax["striptrue"] == dfmax["PEP.StrippedSequence"]]
    dffalsemax = dfmax.loc[dfmax["striptrue"] != dfmax["PEP.StrippedSequence"]]
    dftrue = df.loc[df["striptrue"] == df["PEP.StrippedSequence"]]
    
    # Combine for delta score calculation
    df = pd.concat([dfmax, dftrue])
    df = df.drop_duplicates()
    
    # Calculate delta scores
    f = lambda x: x.score.iloc[0] - x.score.iloc[1] if len(x) >= 2 else x.score.iloc[0]
    dfdelta = df.groupby(["Fspectrum","SourceFile","Peptide"]).apply(f)
    dfdelta = pd.DataFrame(dfdelta).reset_index(drop=False)
    dfdelta.columns = ['Fspectrum','SourceFile',"Peptide",'deltascore']
    
    # Merge with target and decoy hits
    zz = pd.merge(dfdelta, dffalsemax, on=['Fspectrum','SourceFile',"Peptide"], how='right')
    zz = zz[['Fspectrum','SourceFile',"PP.Charge","deltascore","score","key_x","striptrue","PEP.StrippedSequence"]]
    
    ww = pd.merge(dfdelta, dftruemax, on=['Fspectrum','SourceFile',"Peptide"], how='right')
    ww = ww[['Fspectrum','SourceFile',"deltascore","PP.Charge","score","key_x","striptrue","PEP.StrippedSequence"]]
    
    df = pd.concat([zz, ww])
    df = df.drop_duplicates()
    
    # Filter for target sequences only and apply cutoff
    df = df.loc[df["striptrue"] == df["PEP.StrippedSequence"]]
    df["deltascore"] = df["deltascore"].astype("float")
    df = df[df["deltascore"] >= args.cutoff]
    
    
    # Clean key for phosphosite analysis
    df["key"] = df["key_x"].str.replace("2",'',regex=False)
    df["key"] = df["key"].str.replace("3",'',regex=False) 
    df["key"] = df["key"].str.replace("4",'',regex=False)
    df.reset_index(drop=True, inplace=True)
    
    # Extract phosphorylation positions
    for k in range(0, len(df)):
        sequence = df.loc[k,"key"]
        sequence = list(sequence)
        m = 0
        index_list = []
        for i in range(len(sequence)):
            if sequence[i] == "1":
                i = i - m
                index_list.append(i)
                m += 1
        index_list = list(map(str, index_list))
        index_list = ";".join(index_list)
        df.loc[k,"Index"] = index_list
    
    # Split multiple phosphorylation positions into separate rows
    df = df.drop('Index', axis=1).join(
        df['Index'].str.split(";", expand=True).stack().reset_index(level=1, drop=True).rename('Index'))
    df["Index"] = df["Index"].astype("int")
    dfmodel = df.copy()
    
    
    # Prepare model results
    dfmodel = dfmodel[['Fspectrum', 'SourceFile', "PP.Charge",'deltascore', 'score', 'key_x', 'striptrue',
           'key', 'Index']]
    dfmodel.columns = ['Spectrum','SourceFile',"PP.Charge", 'deltascore_model', 'score_model', 'key_model_1234', 'striptrue',
           'key_model_1', 'Index_model']
    
    # Read MaxQuant results
    dfmsms = pd.read_csv(args.inputfile1, delimiter="\t")
    # 自动识别磷酸化位点 ID 列，兼容 Phospho(STY)/Phospho(Y)/Phospho(XXXX) 命名
    phospho_site_cols = [
        c for c in dfmsms.columns
        if re.match(r'^Phospho\s*\(.+\)\s*site IDs$', str(c))
    ]
    if len(phospho_site_cols) == 0:
        raise ValueError(
            f"未在 msms.txt 中找到类似 'Phospho(XXXX) site IDs' 的列，当前列名包括：{list(dfmsms.columns)}"
        )
    phospho_site_col = phospho_site_cols[0]

    dfmsms = dfmsms[['Raw file', 'Scan number',  'Sequence',  'Modified sequence', 'Charge',
                     'id', 'Peptide ID', 'Mod. peptide ID', 'Evidence ID', phospho_site_col]]
    dfmsms.columns = ['SourceFile', 'Spectrum',  'striptrue',  'Peptide', 'PP.Charge',
                      'MSMS_ID', 'Peptide ID', 'Mod. peptide ID', 'Evidence ID', 'Phospho_site_IDs']
    
    # 统一 SourceFile 为 rawX 字符串，保证与 MaxQuant 结果匹配
    dfmodel["SourceFile"] = dfmodel["SourceFile"].astype(str)
    if dfmodel["SourceFile"].str.fullmatch(r"\d+").all():
        dfmodel["SourceFile"] = "raw" + dfmodel["SourceFile"]
    dfmsms["SourceFile"] = dfmsms["SourceFile"].astype(str)
    dfmsms.drop_duplicates(keep="first", inplace=True)
    dfmsms.reset_index(drop=True, inplace=True)
    
    # Merge with MaxQuant data
    df = pd.merge(dfmodel, dfmsms, on=["SourceFile", 'Spectrum',  "PP.Charge",'striptrue'], how="left")
    df = df.dropna(subset=["Phospho_site_IDs"])
    df.reset_index(drop=True, inplace=True)
    
    # Read MaxQuant phosphosite data
    df1 = pd.read_table(args.inputfile2, delimiter="\t")
    # 自动识别磷酸化概率列，兼容 Phospho(STY)/Phospho(Y)/Phospho(XXXX) 命名
    phospho_prob_cols = [
        c for c in df1.columns
        if re.match(r'^Phospho\s*\(.+\)\s*Probabilities$', str(c))
    ]
    if len(phospho_prob_cols) == 0:
        raise ValueError(
            f"未在 Phospho_Sites 文件中找到类似 'Phospho(XXXX) Probabilities' 的列，当前列名包括：{list(df1.columns)}"
        )
    phospho_prob_col = phospho_prob_cols[0]

    df1 = df1[['Proteins', 'Positions within proteins', 'Leading proteins', 'Protein',
               phospho_prob_col, 'Position in peptide', 'Positions', 'Position',
               'MS/MS IDs', 'Best localization MS/MS ID', 'Best score scan number', "id"]]
    df1.columns = ['Proteins', 'Positions within proteins', 'Leading proteins', 'Protein',
                   'Phospho_Probabilities', 'Position in peptide', 'Positions', 'Position',
                   'MS/MS IDs', 'Best localization MS/MS ID', 'Best score scan number', "Phossite_IDs_maxq"]
    
    # 从 Phospho_Probabilities 提取纯序列用于匹配（去除磷酸化标记）
    df1["striptrue_sites"] = df1["Phospho_Probabilities"].str.replace("(", "", regex=False)
    df1["striptrue_sites"] = df1["striptrue_sites"].str.replace(")", "", regex=False)
    df1["striptrue_sites"] = df1["striptrue_sites"].str.replace(r"(\d+\.*\d*)", "", regex=True)
    
    # 通过 (序列 + 磷酸化位置) 组合匹配 Phospho Sites 表，避免跨 raw 文件的错误匹配
    df1["Position in peptide"] = df1["Position in peptide"].astype(int)
    df = pd.merge(
        df, df1,
        left_on=["striptrue", "Index_model"],
        right_on=["striptrue_sites", "Position in peptide"],
        how="left"
    )
    df.drop_duplicates(keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Calculate protein positions for DeepFLR predictions
    for k in range(len(df)):
        position_model = df.loc[k,"Index_model"]
        position_imply = df.loc[k, "Position in peptide"]
        position_protein = df.loc[k, "Position"]
        
        if not np.isnan(position_model) and not np.isnan(position_imply):
            position_delta = int(position_model) - int(position_imply)
            position_protein_model = int(position_delta) + int(position_protein)
            df.loc[k,"position_protein_model"] = position_protein_model
    
    # 过滤掉未匹配到 Phospho Sites 的行
    df = df.dropna(subset=["Protein"])
    
    # Prepare final output
    df = df[['Spectrum', 'SourceFile', 'PP.Charge', 'deltascore_model',
           'score_model', 'key_model_1234', 'striptrue', 'key_model_1',
           'Index_model', 'Peptide', 'MSMS_ID', 'Peptide ID', 'Mod. peptide ID',
           'Evidence ID', 'Proteins',
            'Leading proteins', 'Protein','position_protein_model']]
    df = df.drop_duplicates(keep="first")
    df.reset_index(drop=True, inplace=True)
    
    # 创建蛋白位点标识（向量化，避免 apply 返回多列导致赋值报错）
    def _fmt_pos(v):
        if pd.isna(v):
            return "na"
        try:
            return str(int(v))
        except Exception:
            return "na"

    df["model_proteinsite"] = df["Protein"].astype(str) + "_" + df["position_protein_model"].apply(_fmt_pos)
    
    # Filter out reverse and contaminant proteins
    df = df.loc[~df["Protein"].str.contains("REV__")]
    df = df.loc[~df["Protein"].str.contains("CON__")]
    
    df.to_csv(args.outputresult, index=False)
    print(f"磷酸化位点: {len(df)}")

if __name__ == "__main__":
    main()
