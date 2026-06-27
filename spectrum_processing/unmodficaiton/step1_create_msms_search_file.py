import os
import pandas as pd
import glob
import warnings

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

MAX_PEPTIDE_LENGTH = 40


def run_step1(config):
    mode = str(config.get("mode", "unmodified")).strip().lower()
    COMBINED_MSMS_PATH = config['paths'].get('msms') or config['paths']['combined_msms']
    MSMS_OUTPUT_DIR = config['paths']['msms_dir']
    SEARCH_OUTPUT_DIR = config['paths']['search_dir']
    FILTERED_MSMS_DIR = config['paths']['msms_filtered_dir']

    os.makedirs(MSMS_OUTPUT_DIR, exist_ok=True)

    with open(COMBINED_MSMS_PATH, 'r') as f:
        header = f.readline().strip()
        output_files = {}
        line_counts = {}

        for line in f:
            raw_file = line.split('\t')[0]
            if raw_file not in output_files:
                output_path = os.path.join(MSMS_OUTPUT_DIR, f"{raw_file}.txt")
                output_files[raw_file] = open(output_path, 'w')
                output_files[raw_file].write(header + '\n')
                line_counts[raw_file] = 0

            output_files[raw_file].write(line)
            line_counts[raw_file] += 1

    total_lines = 0
    for raw_file, file_handle in output_files.items():
        file_handle.close()
        total_lines += line_counts[raw_file]
    print(f"MSMS split: files={len(output_files)}, total_lines={total_lines}")

    os.makedirs(SEARCH_OUTPUT_DIR, exist_ok=True)

    unique_raw_files = set()
    with open(COMBINED_MSMS_PATH, 'r') as fin:
        header_cols = fin.readline().rstrip('\n').split('\t')
        try:
            raw_col_idx = header_cols.index('Raw file')
        except ValueError:
            raw_col_idx = 0
        for line in fin:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= raw_col_idx:
                continue
            unique_raw_files.add(parts[raw_col_idx])

    for raw_name in sorted(unique_raw_files):
        txt_path = os.path.join(SEARCH_OUTPUT_DIR, f"{raw_name}.txt")
        with open(txt_path, 'w'):
            pass

    os.makedirs(FILTERED_MSMS_DIR, exist_ok=True)
    txt_files = glob.glob(os.path.join(MSMS_OUTPUT_DIR, "*.txt"))

    processed = 0
    failed = 0
    total_before = 0
    total_after = 0
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            before = len(df)
            df['Length'] = pd.to_numeric(df.get('Length'), errors='coerce')
            mods = df.get('Modifications')
            if mods is None:
                mods = pd.Series([""] * len(df), index=df.index)

            if mode == "phospho":
                mods_str = mods.astype(str).str.strip().str.lower()
                is_phospho_sty = mods_str.str.contains(r"phospho\s*\(\s*sty\s*\)")
                is_phospho_y = mods_str.str.contains(r"phospho\s*\(\s*y\s*\)")
                is_keep = is_phospho_sty | is_phospho_y

                seq_str = df.get('Sequence')
                if seq_str is None:
                    seq_str = pd.Series([""] * len(df), index=df.index)
                seq_str = seq_str.apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
                )
                no_u = ~seq_str.str.contains('U', regex=False)

                df = df[is_keep & (df['Length'] <= MAX_PEPTIDE_LENGTH) & no_u]
            else:
                is_unmodified = mods.astype(str).str.strip().eq('Unmodified')
                seq_str = df.get('Sequence')
                if seq_str is None:
                    seq_str = pd.Series([""] * len(df), index=df.index)
                seq_str = seq_str.apply(
                    lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
                )
                no_u = ~seq_str.str.contains('U', regex=False)

                df = df[is_unmodified & (df['Length'] <= MAX_PEPTIDE_LENGTH) & no_u]
            after = len(df)
            output_path = os.path.join(FILTERED_MSMS_DIR, file_name)
            df.to_csv(output_path, sep='\t', index=False)
            processed += 1
            total_before += before
            total_after += after
        except (pd.errors.ParserError, OSError, UnicodeDecodeError, ValueError) as e:
            failed += 1
            print(f"Warning: failed to process {file_name}: {e}")

    print(f"MSMS filter: ok={processed}, failed={failed}, before={total_before}, after={total_after}")

if __name__ == "__main__":
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    run_step1(config)
