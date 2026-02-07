"""
运行示例 (Usage Examples)
--------------------------------
1) 默认运行全部步骤，并在当前目录生成 tmp/ 与 output/：
   python scripts/run.py \
     --msms /absolute/path/to/msms.txt \
     --mzml-dir /absolute/path/to/MZML \
     --num-workers 32 --batch-size 400 --output-dir ./output

2) 仅运行步骤1+2，跳过步骤3和4：
   python scripts/run.py \
     --msms /absolute/path/to/msms.txt \
     --mzml-dir /absolute/path/to/MZML \
     --skip-step3 --skip-step4

3) 运行全部步骤，但保留 tmp 临时目录用于调试：
   python scripts/run.py \
     --msms /absolute/path/to/msms.txt \
     --mzml-dir /absolute/path/to/MZML \
     --keep-tmp

参数说明：
- --msms           合并后的 msms.txt 路径（原 combined_msms）
- --mzml-dir       mzML 文件所在目录
- --num-workers    并行进程数，默认 32
- --batch-size     batch 大小，默认 400
- --output-dir     最终输出目录，默认 ./output
- --skip-stepX     跳过步骤 X（不加表示默认全部开启）
- --keep-tmp       保留 tmp/ 临时目录（默认完成后自动删除）

中间与输出目录：
- tmp/MSMS
- tmp/Search
- tmp/MSMS_filtered
- tmp/0.process_df_h5
- tmp/1.ylabel_df
- output
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def _infer_repo_root() -> str:
    """
    推断 MS2Int 仓库根目录：
    - 向上查找同时包含 README.md 与 MS2Int/ 目录的父目录；
    - 作为兜底，使用脚本所在目录的上两级（适配当前目录结构）。
    """
    script_path = Path(__file__).resolve()
    for p in [script_path.parent, *script_path.parent.parents]:
        if (p / "README.md").is_file() and (p / "MS2Int").is_dir():
            return str(p)
    return str(script_path.parent.parent)


REPO_ROOT = _infer_repo_root()
DEFAULT_FINAL_H5 = os.path.join(REPO_ROOT, "data", "MS2Int_input.h5")

# 让本目录（以及需要时的仓库根）在 import 搜索路径中
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)



# 统一步骤头部打印
def _print_step_header(tag: str, title: str) -> None:
    line = "-" * 60
    print("\n" + line)
    print(f"[START] Step {tag} - {title}")
    print(line)

# 简单的目录清理工具：删除目录下的所有文件和子目录，但保留目录本身
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
        except Exception as e:
            print(f"Warning: 无法删除 {p}: {e}")

def cleanup_intermediate_dirs(config):
    targets = [
        config['paths']['msms_dir'],
        config['paths']['search_dir'],
        config['paths']['msms_filtered_dir'],
        config['paths']['df_h5_dir'],
        config['paths']['ylabel_df_dir'],
    ]
    for d in targets:
        # 静默清理中间目录，避免冗余日志
        _clean_directory(d)


def _rename_final_h5(output_dir: str, dataset_name: str, final_h5: str) -> None:
    """
    将 Step4 生成的 `{dataset_name}_batch1.h5` 重命名为用户指定的最终路径。
    说明：该流程默认按 batch 输出文件名；在小样本/单 batch 场景下常希望得到固定文件名。
    """
    if not final_h5:
        return

    src = os.path.join(output_dir, f"{dataset_name}_batch1.h5")
    if not os.path.isfile(src):
        # 兜底：如果 dataset_name 推断不一致，尝试在输出目录中找唯一的 *_batch1.h5
        candidates = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith("_batch1.h5")
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"未找到可重命名的 batch1 产物。期望: {src}；实际候选: {candidates}"
            )
        src = candidates[0]

    dst = os.path.abspath(final_h5)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)

    if os.path.abspath(src) == dst:
        return

    # 覆盖写入（符合“再次测试”的预期）
    os.replace(src, dst)
    print(f"[OK] 最终 H5 已重命名: {dst}")


# 将运行入口放到文件头部
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='质谱数据处理项目')
    parser.add_argument('--msms', type=str, required=True, help='合并后的 msms.txt 路径')
    parser.add_argument('--mzml-dir', type=str, required=True, help='mzML 文件所在目录')
    parser.add_argument('--dataset-name', type=str, default=None, help='可选：显式指定输出数据集名称（默认自动推断）')
    parser.add_argument('--num-workers', type=int, default=32, help='并行进程数 (默认: 32)')
    parser.add_argument('--batch-size', type=int, default=400, help='批大小 (默认: 400)')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.getcwd(), 'output'), help='最终输出目录 (默认: ./output)')
    parser.add_argument('--final-h5', type=str, default=DEFAULT_FINAL_H5, help='可选：将最终 batch1 输出重命名为该路径（默认: data/MS2Int_input.h5；如需关闭可传空字符串 ""）')
    parser.add_argument('--skip-step1', action='store_true', help='跳过步骤1')
    parser.add_argument('--skip-step2', action='store_true', help='跳过步骤2')
    parser.add_argument('--skip-step3', action='store_true', help='跳过步骤3')
    parser.add_argument('--skip-step4', action='store_true', help='跳过步骤4')
    parser.add_argument('--keep-tmp', action='store_true', help='保留临时目录 tmp/ 不删除')
    args = parser.parse_args()

    # 数据集名称：优先使用用户显式指定，其次自动推断
    dataset_name = (args.dataset_name or "").strip()
    if not dataset_name:
        # 自动推断数据集名称（从 mzML 目录名为主，若目录名通用则用其父目录名；兜底用 msms 文件名）
        dataset_name = os.path.basename(os.path.normpath(args.mzml_dir))
        if dataset_name.lower() in ('mzml', 'mzmls', 'mgf', 'mgfs', 'raw', 'raws', 'data'):
            parent = os.path.basename(os.path.dirname(os.path.normpath(args.mzml_dir)))
            if parent:
                dataset_name = parent
        if not dataset_name:
            dataset_name = os.path.splitext(os.path.basename(args.msms))[0]


    cwd = os.getcwd()
    tmp_root = os.path.join(cwd, 'tmp')
    config = {
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
            'final_dir': args.output_dir,
        },
        'performance': {
            'num_workers': args.num_workers,
            'batch_size': args.batch_size,
        },
    }

    for k in ('msms_dir', 'search_dir', 'msms_filtered_dir', 'df_h5_dir', 'ylabel_df_dir', 'final_dir'):
        os.makedirs(config['paths'][k], exist_ok=True)


    try:
        # 根据配置运行各个步骤
        if not args.skip_step1:
            _print_step_header("3.1", "创建MSMS搜索文件")
            from step1_create_msms_search_file import run_step1
            run_step1(config)

        if not args.skip_step2:
            _print_step_header("3.2", "处理 h5 数据")
            from step2_process_df_h5 import run_step2
            run_step2(config)

        if not args.skip_step3:
            _print_step_header("3.3", "生成训练数据")
            from step3_generate_train_data import run_step3
            run_step3(config)

        if not args.skip_step4:
            _print_step_header("3.4", "合并最终数据")
            from step4_merge_final_data import run_step4
            run_step4(config)
            if args.final_h5:
                _rename_final_h5(args.output_dir, dataset_name, args.final_h5)

        # 运行完成后，清理中间产物目录（除非用户指定 --keep-tmp）
        if args.keep_tmp:
            print(f"[INFO] 保留临时目录: {tmp_root}")
        else:
            cleanup_intermediate_dirs(config)

            # 额外：在步骤全部完成并清理子目录后，删除 tmp 根目录本身
            # 安全防护：仅当目录名确为 'tmp' 且存在时才执行删除
            try:
                _tmp_root = os.path.join(os.getcwd(), 'tmp')
                if os.path.basename(_tmp_root) == 'tmp' and os.path.isdir(_tmp_root):
                    shutil.rmtree(_tmp_root)
                    print(f"[OK] 已删除临时目录: {_tmp_root}")
            except Exception as e:
                print(f"Warning: 删除临时目录失败 {_tmp_root}: {e}")

        print("[OK] 3.x 全部步骤完成")
    finally:
        print("[OK] 处理完成")
