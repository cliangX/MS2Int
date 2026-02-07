#!/usr/bin/env python3
"""
DeepFLR3.1: CSV to H5 Converter
Usage Examples:
    python script/DeepFLR3.1_csv_to_h5.py --input script/demo_modelresult_template.csv --output mamba_input.h5 --collision_energy 27 --fragmentation HCD
    python script/DeepFLR3.1_csv_to_h5.py --input step1_target_decoy_processed.csv --output my_mamba_input.h5 --collision_energy 30 --fragmentation CID
    python script/DeepFLR3.1_csv_to_h5.py --input demo_modelresult_template.csv --output mamba_low_energy.h5 --collision_energy 20 --fragmentation ETD
"""

import pandas as pd
import numpy as np
import h5py
import argparse
import os
import re
from tqdm import tqdm

def convert_deepflr_to_mamba_sequence(key_x):
    modification_map = {
        '1': '[Phospho]',
        '2': '[Oxidation]',
        '3': '[Carbamidomethyl]',
        '4': '[Acetyl]'
    }
    
    mamba_sequence = key_x
    
    if mamba_sequence.startswith('4'):
        mamba_sequence = '[Acetyl]-' + mamba_sequence[1:]
    
    for deepflr_code, mamba_mod in modification_map.items():
        if deepflr_code != '4':
            pattern = f'([A-Z]){deepflr_code}'
            replacement = f'\\1{mamba_mod}'
            mamba_sequence = re.sub(pattern, replacement, mamba_sequence)
    
    return mamba_sequence

def calculate_sequence_length_from_stripped(stripped_sequence):
    return len(stripped_sequence.strip())

def main():
    parser = argparse.ArgumentParser(description="Convert DeepFLR CSV template to Mamba H5 format")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output H5 file")
    parser.add_argument("--collision_energy", type=int, required=True, help="Collision energy (20-35)")
    parser.add_argument("--fragmentation", type=str, required=True, choices=['HCD', 'CID', 'ETD'], help="Fragmentation method")
    parser.add_argument(
        "--ref_h5",
        type=str,
        default=None,
        help="可选：参考 H5（通常为 rescore/origin_data.h5），"
             "若提供且包含与输入 CSV 行数一致的 train_data，则一并复制到输出 H5 中",
    )
    parser.add_argument("--quiet", action="store_true", help="精简输出，仅打印关键数量信息")
    
    args = parser.parse_args()
    
    if not args.quiet:
        print(f"Reading input CSV: {args.input}")
    df = pd.read_csv(args.input)
    
    required_columns = ['SourceFile', 'Fspectrum', 'PP.Charge', 'key_x', 'PEP.StrippedSequence']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Found {len(df)} peptide candidates")

    n_samples = len(df)
    sequences = []
    lengths = []
    charges = []

    # 预取关键列，避免在循环中做 Series 索引
    col_key_x = df['key_x'].to_numpy()
    col_strip = df['PEP.StrippedSequence'].to_numpy()
    col_charge = df['PP.Charge'].to_numpy()

    if not args.quiet:
        print("Processing peptide sequences with tqdm...")
        iterator = tqdm(range(n_samples), desc="Processing", ncols=100)
    else:
        iterator = range(n_samples)
    for idx in iterator:
        mamba_seq = convert_deepflr_to_mamba_sequence(str(col_key_x[idx]))
        sequences.append(mamba_seq.encode('utf-8'))

        seq_length = calculate_sequence_length_from_stripped(str(col_strip[idx]))
        lengths.append(seq_length)

        charges.append(int(col_charge[idx]))
    
    sequences_array = np.array(sequences, dtype='S128')
    lengths_array = np.array(lengths, dtype=np.int32)
    charges_array = np.array(charges, dtype=np.int32)
    raw_files_array = np.array(df['SourceFile'].astype(str).str.encode('utf-8'), dtype='S100')
    ms2_scan_array = pd.to_numeric(df['Fspectrum'], errors='coerce').fillna(-1).astype(np.int32).to_numpy()
    
    
    if not args.quiet:
        print(f"Creating H5 file: {args.output}")
    with h5py.File(args.output, 'w') as f:
        # 基本字段：序列 / 电荷 / 扫描信息
        f.create_dataset('Sequence', data=sequences_array)
        f.create_dataset('Length', data=lengths_array)
        f.create_dataset('Charge', data=charges_array)
        f.create_dataset('Raw_file', data=raw_files_array)
        f.create_dataset('MS2_Scan_Number', data=ms2_scan_array)

        # 默认使用命令行提供的碰撞能量与碎裂方式，后续如 ref_h5 中存在逐 PSM 信息则覆盖
        ce_data = np.full((n_samples,), args.collision_energy, dtype=np.int32)
        frag_data = np.array(
            [args.fragmentation.encode('utf-8')] * n_samples, dtype='S10'
        )
        analyzer_data = None

        # 若提供了参考 H5（通常为 rescore/origin_data.h5），尝试复制其中的 train_data
        # 以及 collision_energy / Fragmentation / Mass_analyzer 等元数据
        if args.ref_h5 is not None:
            ref_path = args.ref_h5
            if not os.path.isfile(ref_path):
                print(f"[WARN] 指定的 ref_h5 不存在，跳过从中复制元数据: {ref_path}")
            else:
                try:
                    with h5py.File(ref_path, "r") as f_ref:
                        # train_data
                        if "train_data" not in f_ref:
                            print(f"[WARN] ref_h5 中未找到 'train_data' 数据集，跳过复制: {ref_path}")
                        else:
                            ref_train = f_ref["train_data"]
                            if ref_train.shape[0] != n_samples:
                                print(
                                    f"[WARN] ref_h5 中 train_data 行数 ({ref_train.shape[0]}) "
                                    f"与输入 CSV 样本数 ({n_samples}) 不一致，跳过复制。"
                                )
                            else:
                                data = ref_train[()]
                                f.create_dataset("train_data", data=data)
                                print(
                                    f"[INFO] 已从 ref_h5 复制 train_data 到输出 H5，形状: {data.shape}"
                                )

                        # collision_energy
                        if "collision_energy" in f_ref:
                            ref_ce = f_ref["collision_energy"]
                            if ref_ce.shape[0] == n_samples:
                                ce_data = ref_ce[()].astype(np.float32)
                                print("[INFO] 已从 ref_h5 继承逐 PSM 的 collision_energy")
                            else:
                                print(
                                    f"[WARN] ref_h5.collision_energy 行数 ({ref_ce.shape[0]}) "
                                    f"与输入样本数 ({n_samples}) 不一致，保留命令行常量值。"
                                )

                        # Fragmentation
                        if "Fragmentation" in f_ref:
                            ref_frag = f_ref["Fragmentation"]
                            if ref_frag.shape[0] == n_samples:
                                frag_data = ref_frag[()]
                                print("[INFO] 已从 ref_h5 继承逐 PSM 的 Fragmentation")
                            else:
                                print(
                                    f"[WARN] ref_h5.Fragmentation 行数 ({ref_frag.shape[0]}) "
                                    f"与输入样本数 ({n_samples}) 不一致，保留命令行常量值。"
                                )

                        # Mass_analyzer（可选）
                        if "Mass_analyzer" in f_ref and f_ref["Mass_analyzer"].shape[0] == n_samples:
                            analyzer_data = f_ref["Mass_analyzer"][()]
                            print("[INFO] 已从 ref_h5 继承逐 PSM 的 Mass_analyzer")
                except Exception as e:
                    print(f"[WARN] 从 ref_h5 复制元数据失败，保留命令行常量配置: {e}")

        # 写入碰撞能量 / 碎裂方式 / 质量分析器
        f.create_dataset('collision_energy', data=ce_data)
        f.create_dataset('Fragmentation', data=frag_data)
        if analyzer_data is not None:
            f.create_dataset('Mass_analyzer', data=analyzer_data)

        f.attrs['description'] = 'H5 file generated from DeepFLR modelresult template'
        f.attrs['source'] = args.input
        f.attrs['created_by'] = 'DeepFLR3.1_csv_to_h5.py'
        f.attrs['n_samples'] = n_samples
        # attr 中仍保留命令行给定的“默认配置”，便于追踪
        f.attrs['collision_energy'] = args.collision_energy
        f.attrs['fragmentation'] = args.fragmentation
    
    print("✅ H5 file created successfully!")
    print(f"Samples: {n_samples}, Length range: {lengths_array.min()}-{lengths_array.max()}, Charge range: {charges_array.min()}-{charges_array.max()}")
    if not args.quiet:
        print(f"Next step: bash script/DeepFLR3.2_mamba_predict.sh {args.output}")

if __name__ == "__main__":
    main()
