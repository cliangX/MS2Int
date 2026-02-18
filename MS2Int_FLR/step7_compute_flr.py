#!/usr/bin/env python3
"""Compute FLR (False Localization Rate) curve from scored target/decoy candidates."""

import pandas as pd
import numpy as np
import argparse

def ace(instance):
    """Adjust acetyl modification position"""
    if instance[1] == "4":
        instance = list(instance)
        instance.remove("4")
        instance.insert(0, "4")
        instance = ''.join(instance)
    return instance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelresultfile", type=str, required=True)
    parser.add_argument("--sequencefile", type=str, required=True)
    parser.add_argument("--outputfile", type=str, default="step4_FLRPSM.csv")
    parser.add_argument("--psm_outputfile", type=str, default="step7_unique_psm.csv")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.modelresultfile)
    
    # Handle score column format (compatibility with old tensor format)
    if df["score"].dtype == 'object':
        df["score"] = df["score"].str.replace("tensor(",'',regex=False)
        df["score"] = df["score"].str.replace(".)",'',regex=False) 
        df["score"] = df["score"].str.replace(")",'',regex=False)
    df["score"] = df["score"].astype("float")
    
    df1 = pd.read_csv(args.sequencefile)
    
    # Standardize column names for sequencefile (as done in original Step4)
    df1.columns = ["SourceFile","Fspectrum","PP.Charge","exp_strip_sequence","Peptide","key_x"]
    
    df = pd.merge(df, df1, on=["SourceFile","Fspectrum","PP.Charge","key_x"])
    df = df[["SourceFile","Fspectrum","Peptide","PEP.StrippedSequence","key_x","score"]]
    
    # Use key_x stripping for target/decoy classification
    df["strip_key"] = df["key_x"].str.replace("1",'',regex=False)
    df["strip_key"] = df["strip_key"].str.replace("2",'',regex=False)
    df["strip_key"] = df["strip_key"].str.replace("3",'',regex=False)
    df["strip_key"] = df["strip_key"].str.replace("4",'',regex=False)
    df["score"] = df["score"].astype("float")
    
    # Sort by score (highest first)
    df = df.sort_values(by='score', ascending=False)
    df.reset_index(drop=True, inplace=True)
    
    # Get best scoring hit per spectrum-peptide combination
    dfmax = df.drop_duplicates(subset=['SourceFile', 'Fspectrum', 'Peptide'])
    dftruemax = dfmax.loc[dfmax["strip_key"] == dfmax["PEP.StrippedSequence"]]  # Target hits
    dffalsemax = dfmax.loc[dfmax["strip_key"] != dfmax["PEP.StrippedSequence"]]  # Decoy hits
    dftrue = df.loc[df["strip_key"] == df["PEP.StrippedSequence"]]  # All target hits
    dfdecoy = df.loc[df["strip_key"] != df["PEP.StrippedSequence"]]  # All decoy hits
    
    dftruenum = len(dftrue)
    dfdecoynum = len(dfdecoy)
    
    # Combine max hits with all true hits for delta score calculation
    df = pd.concat([dfmax, dftrue])
    df = df.drop_duplicates()
    
    # Calculate delta scores (best score - second best score per spectrum-peptide)
    f = lambda x: x.score.iloc[0] - x.score.iloc[1] if len(x) >= 2 else x.score.iloc[0]
    dfdelta = df.groupby(["Fspectrum","SourceFile","Peptide"]).apply(f)
    dfdelta = pd.DataFrame(dfdelta).reset_index(drop=False)
    dfdelta.columns = ['Fspectrum','SourceFile',"Peptide",'deltascore']
    
    # Merge delta scores with max hits
    zz = pd.merge(dfdelta, dffalsemax, on=['Fspectrum','SourceFile','Peptide'], how='right')
    zz = zz[['Fspectrum','SourceFile','deltascore','score','key_x','strip_key','PEP.StrippedSequence']]
    
    ww = pd.merge(dfdelta, dftruemax, on=['Fspectrum','SourceFile','Peptide'], how='right')
    ww = ww[['Fspectrum','SourceFile','deltascore','score','key_x','strip_key','PEP.StrippedSequence']]
    
    df = pd.concat([zz, ww])
    df["deltascore"] = df["deltascore"].astype("float")
    
    df["is_target"] = df["strip_key"] == df["PEP.StrippedSequence"]
    df[["SourceFile", "Fspectrum", "key_x", "PEP.StrippedSequence", "deltascore", "is_target"]].to_csv(
        args.psm_outputfile, index=False
    )
    print(f"Unique PSMs: {len(df)} -> {args.psm_outputfile}")
    
    # Initialize FLR calculation
    out = pd.DataFrame(columns=["cutoff","esti_FLR","PSMs"])
    down = 0
    
    # Clean key_x for analysis
    df["key_x"] = df["key_x"].str.replace("2",'',regex=False)
    df["key_x"] = df["key_x"].str.replace("3",'',regex=False)
    df["key_x"] = df["key_x"].str.replace("4",'',regex=False)
    
    # Get unique delta scores for cutoff analysis
    d = 1
    dftarget = df["deltascore"].drop_duplicates().sort_values(ascending=True)
    dftarget = np.array(dftarget).tolist()
    
    for a in dftarget:
        dfcutoff = df[df["deltascore"] >= a]
        if len(dfcutoff) == 0:
            break
            
        dfcutoffdecoy = dfcutoff.loc[dfcutoff["strip_key"] != dfcutoff["PEP.StrippedSequence"]]
        
        # Calculate estimated FLR
        if dfdecoynum == 0:
            # Fallback when there are no decoys in the dataset
            decoy_score = (len(dfcutoffdecoy) / len(dfcutoff)) if len(dfcutoff) > 0 else 0.0
        else:
            decoy_score = ((dftruenum + dfdecoynum) / dfdecoynum) * len(dfcutoffdecoy) / (len(dfcutoff))
            if d <= decoy_score:
                decoy_score = d
            d = min(d, decoy_score)
        
        PSMs = len(dfcutoff) - len(dfcutoffdecoy)
        out.loc[down] = [a, decoy_score, PSMs]
        down += 1
        
        if decoy_score == 0:
            break
    
    out.to_csv(args.outputfile, index=False)

if __name__ == "__main__":
    main()
