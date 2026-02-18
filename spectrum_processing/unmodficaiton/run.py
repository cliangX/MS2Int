"""Spectrum processing pipeline runner

Usage:
    python run.py --msms msms.txt --mzml-dir ./mzml --mode unmodified
    python run.py --msms msms.txt --mzml-dir ./mzml --mode phospho
    # Output single file (recommended): auto-infer output dir and rename batch1 product
    python run.py --msms msms.txt --mzml-dir ./mzml --output ./out/train.h5

Steps:
    1. Split msms.txt by raw file and filter (mode-specific, length <= 30)
    2. Match theoretical fragments to experimental spectra
    3. Generate training data matrices
    4. Merge into final HDF5
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def _infer_repo_root() -> str:
    script_path = Path(__file__).resolve()
    for p in [script_path.parent, *script_path.parent.parents]:
        if (p / "README.md").is_file() and (p / "MS2Int").is_dir():
            return str(p)
    return str(script_path.parent.parent)

REPO_ROOT = _infer_repo_root()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def _print_step_header(tag: str, title: str) -> None:
    line = "-" * 60
    print("\n" + line)
    print(f"[START] Step {tag} - {title}")
    print(line)

def _clean_directory(path):
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        p = os.path.join(path, name)
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)
        except OSError as e:
            print(f"Warning: failed to delete {p}: {e}")

def cleanup_intermediate_dirs(config):
    targets = [
        config['paths']['msms_dir'],
        config['paths']['search_dir'],
        config['paths']['msms_filtered_dir'],
        config['paths']['df_h5_dir'],
        config['paths']['ylabel_df_dir'],
    ]
    for d in targets:
        _clean_directory(d)

def _rename_final_h5(output_dir: str, dataset_name: str, final_h5: str) -> None:
    if not final_h5:
        return

    src = os.path.join(output_dir, f"{dataset_name}_batch1.h5")
    if not os.path.isfile(src):
        candidates = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith("_batch1.h5")
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"No batch1 output found. Expected: {src}; candidates: {candidates}"
            )
        src = candidates[0]

    dst = os.path.abspath(final_h5)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    if os.path.abspath(src) == dst:
        return

    os.replace(src, dst)
    print(f"[OK] Final H5 renamed: {dst}")

def _resolve_output_paths(args) -> tuple[str, str]:
    """Resolve output directory and final H5 path from --output argument.

    - If --output is a file path (.h5/.hdf5/.hdf): output dir = parent dir, rename batch1 to that file.
    - If --output is a directory path: output dir = that directory, no renaming.
    - If --output is omitted: output dir = ./output, no renaming.
    """

    output = (getattr(args, "output", "") or "").strip()

    if output:
        # Expand ~ in path
        output = os.path.expanduser(output)

        # Determine type by extension: .h5/.hdf5/.hdf = file, otherwise directory
        lower = output.lower()
        looks_like_file = lower.endswith((".h5", ".hdf5", ".hdf"))

        # Allow directory paths: e.g. --output ./out/ or --output ./out
        if (not looks_like_file) or output.endswith(os.sep) or (
            os.path.exists(output) and os.path.isdir(output)
        ):
            return output, ""

        # File path: e.g. --output data/training/train.h5
        out_dir = os.path.dirname(output) or os.getcwd()
        return out_dir, output

    return os.path.join(os.getcwd(), "output"), ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spectrum data processing pipeline')
    parser.add_argument('--msms', type=str, required=True, help='Path to msms.txt')
    parser.add_argument('--mzml-dir', type=str, required=True, help='Directory containing mzML files')
    parser.add_argument(
        '--mode',
        type=str,
        choices=('unmodified', 'phospho'),
        default='unmodified',
        help="Processing mode: unmodified or phospho (default: unmodified)",
    )
    parser.add_argument('--dataset-name', type=str, default=None, help='Output dataset name (auto-inferred if omitted)')
    parser.add_argument('--num-workers', type=int, default=32, help='Number of parallel workers (default: 32)')
    parser.add_argument('--batch-size', type=int, default=400, help='Batch size (default: 400)')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help=(
            'Output path: recommended as a single file, e.g. data/training/train.h5; '
            'or a directory, e.g. ./output/ (outputs batch file without renaming)'
        ),
    )
    parser.add_argument('--skip-step1', action='store_true', help='Skip step 1')
    parser.add_argument('--skip-step2', action='store_true', help='Skip step 2')
    parser.add_argument('--skip-step3', action='store_true', help='Skip step 3')
    parser.add_argument('--skip-step4', action='store_true', help='Skip step 4')
    parser.add_argument('--keep-tmp', action='store_true', help='Keep tmp/ directory after completion')
    args = parser.parse_args()

    output_dir, final_h5 = _resolve_output_paths(args)

    dataset_name = (args.dataset_name or "").strip()
    if not dataset_name:
        dataset_name = os.path.basename(os.path.normpath(args.mzml_dir))
        if dataset_name.lower() in ('mzml', 'mzmls', 'mgf', 'mgfs', 'raw', 'raws', 'data'):
            parent = os.path.basename(os.path.dirname(os.path.normpath(args.mzml_dir)))
            if parent:
                dataset_name = parent
            dataset_name = os.path.splitext(os.path.basename(args.msms))[0]


    cwd = os.getcwd()
    tmp_root = os.path.join(cwd, 'tmp')
    config = {
        'mode': args.mode,
        'dataset': {
            'name': dataset_name,
        },
        'paths': {
            'msms': args.msms,
            'mzml_dir': args.mzml_dir,
            'msms_dir': os.path.join(tmp_root, 'MSMS'),
            'search_dir': os.path.join(tmp_root, 'Search'),
            'msms_filtered_dir': os.path.join(tmp_root, 'MSMS_filtered'),
            'df_h5_dir': os.path.join(tmp_root, '0.process_df_h5'),
            'ylabel_df_dir': os.path.join(tmp_root, '1.ylabel_df'),
            'final_dir': output_dir,
        },
        'performance': {
            'num_workers': args.num_workers,
            'batch_size': args.batch_size,
        },
    }

    for k in ('msms_dir', 'search_dir', 'msms_filtered_dir', 'df_h5_dir', 'ylabel_df_dir', 'final_dir'):
        os.makedirs(config['paths'][k], exist_ok=True)


    try:
        if not args.skip_step1:
            _print_step_header("3.1", "Create MSMS search files")
            from step1_create_msms_search_file import run_step1
            run_step1(config)

        if not args.skip_step2:
            _print_step_header("3.2", "Process spectrum data")
            from step2_process_df_h5 import run_step2
            run_step2(config)

        if not args.skip_step3:
            _print_step_header("3.3", "Generate training data")
            from step3_generate_train_data import run_step3
            run_step3(config)

        if not args.skip_step4:
            _print_step_header("3.4", "Merge final data")
            from step4_merge_final_data import run_step4
            run_step4(config)
            if final_h5:
                _rename_final_h5(output_dir, dataset_name, final_h5)

        if args.keep_tmp:
            print(f"[INFO] Keeping tmp directory: {tmp_root}")
        else:
            cleanup_intermediate_dirs(config)

            try:
                _tmp_root = os.path.join(os.getcwd(), 'tmp')
                if os.path.basename(_tmp_root) == 'tmp' and os.path.isdir(_tmp_root):
                    shutil.rmtree(_tmp_root)
                    print(f"[OK] Removed tmp directory: {_tmp_root}")
            except OSError as e:
                print(f"Warning: failed to remove {_tmp_root}: {e}")

        print("[OK] All steps completed")
    finally:
        print("[OK] Done")
