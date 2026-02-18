import os
import pandas as pd
import numpy as np
import warnings
import gc
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
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

def _normalize_ion_name(name: str):
    if not name:
        return None

    base = str(name)
    charge_suffix = ""
    if "^" in base:
        parts = base.split("^", 1)
        base, charge_suffix = parts[0], "^" + parts[1]

    if base.endswith("-H3PO4"):
        base = base[: -len("-H3PO4")]

    if not base:
        return None

    if base.startswith("m"):
        return base

    return base + charge_suffix


def process_file(file_path, ylabel_df_dir, mode: str = "unmodified"):
    with pd.HDFStore(file_path, 'r') as store:
        df = store['combined_data']
        n = len(df)
        train_data = [None] * n
        flat_len = ION_ROWS * ION_COLS
        mode = str(mode or "unmodified").strip().lower()
        for idx in range(n):
            if pd.isna(df['Mass analyzer'].iat[idx]):
                # Keep None to mirror original behavior
                continue
            fragments = df['theoretical_fragments_int'].iat[idx]
            if not fragments:
                # No fragments; return zeros matrix
                train_data[idx] = np.zeros((ION_ROWS, ION_COLS), dtype=float)
                continue

            # unmodified: keep first occurrence per ion name; phospho: normalize ion name then accumulate
            vec = np.zeros(flat_len, dtype=float)
            filled = np.zeros(flat_len, dtype=bool) if mode != "phospho" else None
            for fragment in fragments:
                if not isinstance(fragment, (list, tuple)) or len(fragment) != 2:
                    continue

                name, inten = fragment
                inten_val = pd.to_numeric(inten, errors="coerce")
                if pd.isna(inten_val):
                    continue
                inten_val = float(inten_val)

                if mode == "phospho":
                    norm_name = _normalize_ion_name(name)
                    if norm_name is None:
                        continue
                    j = ION_TO_IDX.get(norm_name)
                    if j is not None:
                        vec[j] += inten_val
                else:
                    j = ION_TO_IDX.get(name)
                    if j is not None and not filled[j]:
                        vec[j] = inten_val
                        filled[j] = True
            train_data[idx] = vec.reshape((ION_ROWS, ION_COLS))

        df['train_data'] = train_data

        keep_cols = [
            'Sequence','Length','Modifications','Modified_sequence','Charge',
            'MS2_Scan_Number','Score','Raw_file','annotate','RT','instrument',
            'collision_energy','Mass analyzer','Fragmentation','Reverse','train_data'
        ]
        df = df[[c for c in keep_cols if c in df.columns]].copy()

        output_path = os.path.join(ylabel_df_dir, df['Raw_file'].iloc[0] + '.h5')
        df.to_hdf(output_path, key='combined_data', mode='w')
        gc.collect()
        return output_path

def run_step3(config):
    df_h5_dir = config['paths']['df_h5_dir']
    ylabel_df_dir = config['paths']['ylabel_df_dir']
    num_work = config['performance']['num_workers']
    mode = str(config.get("mode", "unmodified")).strip().lower()

    os.makedirs(ylabel_df_dir, exist_ok=True)

    def process_project():
        directory_path = df_h5_dir
        file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.h5')]

        process_func = partial(process_file, ylabel_df_dir=ylabel_df_dir, mode=mode)

        with Pool(processes=num_work) as pool:
            results = list(tqdm(pool.imap_unordered(process_func, file_paths, chunksize=1),
                               total=len(file_paths), desc="Processing files", mininterval=0.5))
        return results

    results = process_project()
    gc.collect()

if __name__ == "__main__":
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    run_step3(config)
