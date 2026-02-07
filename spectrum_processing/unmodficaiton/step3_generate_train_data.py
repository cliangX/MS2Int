import os
import pandas as pd
import numpy as np
import warnings
import gc
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# 忽略 PerformanceWarning
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

# -----------------------------
# Prebuild annotation matrix and index mapping (module-level)
# -----------------------------
ION_ROWS = 31
ION_COLS = 29

def _build_annotation():
    """Build the 31x29 annotation matrix and name->flat-index mapping.
    Keeps the same layout/semantics as original code (including IH/IR/IF/IY)."""
    mat = np.full((ION_ROWS, ION_COLS), '', dtype=object)
    # b series
    for i in range(ION_COLS):
        mat[0, i] = f'b{i+1}'
    # b^2 series
    for i in range(ION_COLS):
        mat[1, i] = f'b{i+1}^2'
    # y series
    for i in range(ION_COLS):
        mat[2, i] = f'y{i+1}'
    # y^2 series
    for i in range(ION_COLS):
        mat[3, i] = f'y{i+1}^2'
    # m ranges
    for row in range(4, ION_ROWS):
        for col in range(ION_COLS):
            m_start = col + 2
            m_end = m_start + (row - 2)
            if m_end <= ION_COLS:
                mat[row, col] = f'm{m_start}:{m_end}'
    # immonium ions override last column
    mat[0, 28] = 'IH'
    mat[1, 28] = 'IR'
    mat[2, 28] = 'IF'
    mat[3, 28] = 'IY'

    ion_order = mat.ravel().tolist()
    index = {name: i for i, name in enumerate(ion_order) if name}
    return mat, ion_order, index

ANNOTATION_MATRIX, ION_ORDER, ION_TO_IDX = _build_annotation()

# 将process_file函数移到外部
def process_file(file_path, ylabel_df_dir):
    with pd.HDFStore(file_path, 'r') as store:
        df = store['combined_data']
        # 不再填充空值：保留 Step 2 产出的原始正确列
        # 仅在后保存前做必要的列筛选以减小体积

        # Build train_data as a list, avoid per-row df.at overhead
        n = len(df)
        train_data = [None] * n
        flat_len = ION_ROWS * ION_COLS
        for idx in range(n):
            if pd.isna(df['Mass analyzer'].iat[idx]):
                # Keep None to mirror original behavior
                continue
            fragments = df['theoretical_fragments_int'].iat[idx]
            if not fragments:
                # No fragments; return zeros matrix
                train_data[idx] = np.zeros((ION_ROWS, ION_COLS), dtype=float)
                continue

            # Fill vector using first-occurence semantics per ion name
            vec = np.zeros(flat_len, dtype=float)
            filled = np.zeros(flat_len, dtype=bool)
            try:
                for name, inten in fragments:
                    j = ION_TO_IDX.get(name)
                    if j is not None and not filled[j]:
                        vec[j] = float(inten)
                        filled[j] = True
            except Exception:
                # If fragments is malformed, fallback to zeros
                pass
            train_data[idx] = vec.reshape((ION_ROWS, ION_COLS))

        df['train_data'] = train_data

        # 仅保留 Step 4 会用到的列 + 生成的 train_data
        keep_cols = [
            'Sequence','Length','Modifications','Modified_sequence','Charge',
            'MS2_Scan_Number','Score','Raw_file','annotate','RT','instrument',
            'collision_energy','Mass analyzer','Fragmentation','Reverse','train_data'
        ]
        # 若片段强度列仍在，保存前丢弃（Step 4 不需要）
        df = df[[c for c in keep_cols if c in df.columns]].copy()

        output_path = os.path.join(ylabel_df_dir, df['Raw_file'].iloc[0] + '.h5')
        df.to_hdf(output_path, key='combined_data', mode='w')
        gc.collect()
        return output_path

def run_step3(config):
    # 从配置中获取路径和性能参数
    result_base_path = os.path.dirname(config['paths']['df_h5_dir'])
    df_h5_dir = config['paths']['df_h5_dir']
    ylabel_df_dir = config['paths']['ylabel_df_dir']
    num_work = config['performance']['num_workers']
    
    # 创建必要的目录
    os.makedirs(ylabel_df_dir, exist_ok=True)
    
    def process_project():
        directory_path = df_h5_dir
        file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.h5')]

        # 使用partial传递额外参数
        process_func = partial(process_file, ylabel_df_dir=ylabel_df_dir)

        with Pool(processes=num_work) as pool:
            # 设置tqdm的mininterval为0.5秒，chunksize为1
            results = list(tqdm(pool.imap_unordered(process_func, file_paths, chunksize=1), 
                               total=len(file_paths), desc="Processing files", mininterval=0.5))
        return results

    # 执行处理
    results = process_project()
    gc.collect()

    print("3.3完成")

if __name__ == "__main__":
    # 仅用于直接运行此脚本的测试
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    run_step3(config)
