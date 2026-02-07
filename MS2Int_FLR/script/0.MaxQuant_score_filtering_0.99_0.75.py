"""
MaxQuant定位概率筛选和统计脚本

运行示例:
python script/MaxQuant_score_filtering_0.99_0.75.py \
  --inputfile demo_data/msms_sample.txt \
  --inputfile1 demo_data/Phospho_STY_Sites.txt \
  --outputfile demo_data/MaxQuant_filtered_sites

功能:
- 基于MaxQuant的Localization prob列进行筛选
- 应用两个阈值: >= 0.99 和 >= 0.75
- 统计每个阈值下的唯一磷酸化位点数量
- 应用与DeepFLR相同的过滤策略（单磷酸化，可磷酸化位点>=2）
"""

import pandas as pd
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--inputfile",
    default=None,
    type=str,
    required=True,
    help="inputfile,searching result from Maxquant(msms.txt)",
)
parser.add_argument(
    "--inputfile1",
    default=None,
    type=str,
    required=True,
    help="inputfile,searching result from Maxquant(Phospho (STY)Sites.txt)",
)
parser.add_argument(
    "--outputfile",
    default="MaxQuant_filtered_sites",
    type=str,
    required=False,
    help="output filename prefix, will generate two files with _0.99 and _0.75 suffix",
)
args = parser.parse_args()

inputfile = args.inputfile
inputfile1 = args.inputfile1
outputfile_prefix = args.outputfile

# 读取msms.txt
df = pd.read_csv(inputfile, delimiter="\t")
df = df[['Raw file', 'Scan number', 'Sequence', 'Modified sequence', "Charge",
         'id', 'Peptide ID', 'Mod. peptide ID', 'Evidence ID', "Phospho (STY) Probabilities", 'Phospho (STY) site IDs']]
df.columns = ['SourceFile', 'Spectrum', 'striptrue', 'Peptide', "PP.Charge",
              'MSMS_ID', 'Peptide ID', 'Mod. peptide ID', 'Evidence ID', "Score", 'Phospho (STY) site IDs']
df = df.dropna(subset=["Phospho (STY) site IDs"])
df = df.drop_duplicates(keep="first")
df.reset_index(drop=True, inplace=True)

# 处理Peptide序列
df["Peptide"] = df["Peptide"].str.replace("_", '', regex=False)
df["Peptide"] = df["Peptide"].str.replace("(Phospho (STY))", '1', regex=False)
df["Peptide"] = df["Peptide"].str.replace("(Oxidation (M))", '2', regex=False)
df["Peptide"] = df["Peptide"].str.replace("(Acetyl (Protein N-term))", '4', regex=False)
df = df.loc[~df["Peptide"].str.contains("4M2")]
df = df.loc[~df["Peptide"].str.contains("4S")]
df = df.loc[~df["Peptide"].str.contains("4T")]
df = df.loc[~df["Peptide"].str.contains("4Y")]
df = df.loc[~df["Peptide"].str.contains("4C")]
df["Peptide"] = df["Peptide"].str.replace("4", '', regex=False)
df["Peptide"] = df["Peptide"].str.replace("2", '', regex=False)
df["key"] = df["Peptide"].str.replace("1", '', regex=False)
df["phos_num"] = df["Peptide"].apply(lambda x: x.count("1"))

# 应用与DeepFLR相同的过滤策略
# 1. 只保留单磷酸化修饰
df = df.loc[df["Peptide"].str.count("1") == 1]

# 2. 计算去除修饰后的序列（用于检查可磷酸化位点数量）
df["exp_strip_sequence"] = df["Peptide"].str.replace("1", '', regex=False)
df["exp_strip_sequence"] = df["exp_strip_sequence"].str.replace("4", '', regex=False)
df["exp_strip_sequence"] = df["exp_strip_sequence"].str.replace("3", '', regex=False)
df["exp_strip_sequence"] = df["exp_strip_sequence"].str.replace("2", '', regex=False)

# 3. 可磷酸化位点数量 > 1（即 >= 2）
df = df.loc[(df["exp_strip_sequence"].str.count("S") + df["exp_strip_sequence"].str.count("T") + df["exp_strip_sequence"].str.count("Y")) > 1]
df.reset_index(drop=True, inplace=True)

# 保存原始的Phospho (STY) Probabilities用于后续提取每个位点的分数
df["Original_Score"] = df["Score"].copy()

# 提取位点索引
df["striptrue"] = df["Peptide"].str.replace("1", '', regex=False)
df.reset_index(drop=True, inplace=True)

for k in range(0, len(df)):
    sequence = df.loc[k, "Peptide"]
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
    df.loc[k, "Index_Maxquant"] = index_list

# 展开多磷酸化位点
df = df.drop('Index_Maxquant', axis=1).join(
    df['Index_Maxquant'].str.split(";", expand=True).stack().reset_index(level=1, drop=True).rename('Index_Maxquant'))
df["Index_Maxquant"] = df["Index_Maxquant"].astype("int")

# 展开Phospho (STY) site IDs
df["Phospho (STY) site IDs"] = df["Phospho (STY) site IDs"].astype("str")
df = df.drop('Phospho (STY) site IDs', axis=1).join(
    df['Phospho (STY) site IDs'].str.split(";", expand=True).stack().reset_index(level=1, drop=True).rename('Phossite_IDs_maxq'))

# 为每个位点提取对应的分数
# MaxQuant的Phospho (STY) Probabilities格式通常是"(0.99;0.80)"，分数按位点在peptide中的顺序排列
def extract_site_score(row):
    """为每个位点提取对应的定位概率分数"""
    original_score_str = str(row["Original_Score"])
    # 提取所有分数，保持原始顺序（不排序，因为MaxQuant按位点顺序排列）
    scores = re.findall(r"(\d+\.*\d*)", original_score_str)
    scores = [float(i) for i in scores]
    
    # 获取当前位点在peptide中的索引位置
    peptide = row["Peptide"]
    index_in_peptide = row["Index_Maxquant"]
    
    # 找到所有磷酸化位点的位置（按在peptide中的顺序）
    sequence = list(peptide)
    phos_positions = []
    m = 0
    for i in range(len(sequence)):
        if sequence[i] == "1":
            pos = i - m  # 去除修饰符号后的位置
            phos_positions.append(pos)
            m += 1
    
    # 找到当前Index_Maxquant在phos_positions中的顺序（第几个位点）
    try:
        site_order = phos_positions.index(index_in_peptide)
        # 根据位点顺序取对应的分数
        if site_order < len(scores):
            return scores[site_order]
        else:
            # 如果分数数量不够，返回0.0
            return 0.0
    except (ValueError, IndexError):
        # 如果找不到对应位点，返回0.0
        return 0.0

df["Score"] = df.apply(extract_site_score, axis=1)
df["Score"] = df["Score"].astype("float")
df.reset_index(drop=True, inplace=True)

# 读取Phospho (STY)Sites.txt
df1 = pd.read_table(inputfile1, delimiter="\t")
df1 = df1[['Proteins', 'Positions within proteins', 'Leading proteins', 'Protein',
           'Phospho (STY) Probabilities', 'Position in peptide', 'Positions', 'Position',
           'MS/MS IDs', 'Best localization MS/MS ID', 'Best score scan number', "id", "Localization prob"]]
df1.columns = ['Proteins', 'Positions within proteins', 'Leading proteins', 'Protein',
               'Phospho (STY) Probabilities', 'Position in peptide', 'Positions', 'Position',
               'MS/MS IDs', 'Best localization MS/MS ID', 'Best score scan number', "Phossite_IDs_maxq", "Localization prob"]
df1["Phossite_IDs_maxq"] = df1["Phossite_IDs_maxq"].astype("str")

# 合并数据
df = pd.merge(df, df1, on=["Phossite_IDs_maxq"], how="left")
df.drop_duplicates(keep="first", inplace=True)
df.reset_index(drop=True, inplace=True)

# 使用Localization prob列作为筛选分数，转换为数值类型
df["Localization prob"] = pd.to_numeric(df["Localization prob"], errors='coerce')
df = df.dropna(subset=["Localization prob"])

# 计算蛋白位置
for k in range(len(df)):
    position_model = df.loc[k, "Index_Maxquant"]
    position_imply = df.loc[k, "Position in peptide"]
    position_protein = df.loc[k, "Position"]
    if not np.isnan(position_model) and not np.isnan(position_imply) and not np.isnan(position_protein):
        position_delta = int(position_model) - int(position_imply)
        position_protein_model = int(position_delta) + int(position_protein)
        df.loc[k, "position_protein_Maxquant"] = position_protein_model

# 验证peptide匹配
df["stripPeptide_phosprob"] = df["Phospho (STY) Probabilities"].str.replace("(", "", regex=False)
df["stripPeptide_phosprob"] = df["stripPeptide_phosprob"].str.replace(")", "", regex=False)
df["stripPeptide_phosprob"] = df["stripPeptide_phosprob"].str.replace(r"(\d+\.*\d*)", "", regex=True)
df = df.loc[df["stripPeptide_phosprob"] == df["striptrue"]]

# 过滤REV__和CON__蛋白
df = df.loc[~df["Protein"].str.contains("REV__", na=False)]
df = df.loc[~df["Protein"].str.contains("CON__", na=False)]

# 创建唯一位点标识
def combine(instance):
    x = instance["Protein"]
    y = instance["position_protein_Maxquant"]
    if pd.notna(x) and pd.notna(y):
        return str(x) + "_" + str(int(y))
    return None

df["Maxquant_proteinsite"] = df.apply(combine, axis=1)
df = df.dropna(subset=["Maxquant_proteinsite"])

# 定义筛选和统计函数
def filter_and_statistics(df, threshold, threshold_name):
    """根据阈值筛选并统计磷酸化位点"""
    df_filtered = df[df["Localization prob"] >= threshold].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    
    # 统计唯一磷酸化位点数量
    unique_sites = df_filtered["Maxquant_proteinsite"].nunique()
    
    # 准备输出列
    output_columns = ['Spectrum', 'SourceFile', 'PP.Charge', "Localization prob", 'striptrue',
                      'Index_Maxquant', 'Peptide', 'MSMS_ID', 'Peptide ID', 'Mod. peptide ID',
                      'Evidence ID', 'Proteins',
                      'Leading proteins', 'Protein', 'position_protein_Maxquant', 'Maxquant_proteinsite']
    
    # 确保所有列都存在
    available_columns = [col for col in output_columns if col in df_filtered.columns]
    df_output = df_filtered[available_columns].copy()
    df_output = df_output.drop_duplicates(keep="first")
    df_output.reset_index(drop=True, inplace=True)
    
    return df_output, unique_sites

# 应用两个阈值进行筛选
thresholds = [0.99, 0.75]
results = {}

for threshold in thresholds:
    df_filtered, unique_count = filter_and_statistics(df, threshold, str(threshold))
    results[threshold] = {
        'dataframe': df_filtered,
        'unique_sites': unique_count,
        'total_psms': len(df_filtered)
    }

# 输出统计信息
print("=" * 60)
print("MaxQuant定位概率筛选统计结果")
print("=" * 60)
print(f"\n阈值 >= 0.99:")
print(f"  唯一磷酸化位点数量: {results[0.99]['unique_sites']}")
print(f"  总PSM数量: {results[0.99]['total_psms']}")
print(f"\n阈值 >= 0.75:")
print(f"  唯一磷酸化位点数量: {results[0.75]['unique_sites']}")
print(f"  总PSM数量: {results[0.75]['total_psms']}")
print("=" * 60)

# 输出详细列表
output_file_099 = f"{outputfile_prefix}_0.99.csv"
output_file_075 = f"{outputfile_prefix}_0.75.csv"

results[0.99]['dataframe'].to_csv(output_file_099, index=False)
results[0.75]['dataframe'].to_csv(output_file_075, index=False)

print(f"\n详细结果已保存:")
print(f"  - {output_file_099} (阈值 >= 0.99)")
print(f"  - {output_file_075} (阈值 >= 0.75)")

