#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 SpecId 文件（输出自定义文件名的 TSV）

SpecId 定义:
- 由 `Raw file-Scan number-Modified sequence-Charge` 拼接而成
- 在生成前会将 `Modified sequence` 中的 C 替换为 C[UNIMOD:4]

使用示例:
    python utils/1.make_specid.py \
        /mnt/data_nas/lcy/project_MS2predict/1.data/rescore_PXD000561/output/msms.filtered_unmodified_lenle29.txt \
        /mnt/data_nas/lcy/project_MS2predict/1.data/rescore_PXD000561/output/msms_specid.tsv
"""

import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="根据 MaxQuant msms.txt 生成 SpecId 文件"
    )

    parser.add_argument("input", help="输入 msms.txt 文件路径")
    parser.add_argument("output", help="输出文件路径 (包含文件名，如: /path/to/output/msms_specid.tsv)")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # 自动创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ===== 读入数据 =====
    df = pd.read_csv(input_path, sep='\t')

    required_cols = ['Raw file', 'Scan number', 'Modified sequence', 'Charge']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"文件缺少必须的列: {col}")

    # ===== Modified sequence 处理：C → C[UNIMOD:4] =====
    df['Modified sequence'] = df['Modified sequence'].astype(str).str.replace(
        "C",
        "C[UNIMOD:4]",
        regex=False
    )

    # ===== 生成 SpecId =====
    df['SpecId'] = (
        df['Raw file'].astype(str)
        + "-" + df['Scan number'].astype(str)
        + "-" + df['Modified sequence'].astype(str)
        + "-" + df['Charge'].astype(str)
    )

    # ===== 导出结果 =====
    df_out = df[['SpecId']]
    df_out.to_csv(output_path, sep='\t', index=False)

    print(f"处理完成，结果已保存到: {output_path}")

if __name__ == "__main__":
    main()