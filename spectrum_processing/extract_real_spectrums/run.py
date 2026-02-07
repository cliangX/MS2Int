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

# Ensure we can import from project root (for `utils` package)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



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

# 将运行入口放到文件头部
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='质谱数据处理项目')
    parser.add_argument('--msms', type=str, required=True, help='合并后的 msms.txt 路径')
    parser.add_argument('--mzml-dir', type=str, required=True, help='mzML 文件所在目录')
    parser.add_argument('--num-workers', type=int, default=32, help='并行进程数 (默认: 32)')
    parser.add_argument('--batch-size', type=int, default=400, help='批大小 (默认: 400)')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.getcwd(), 'output'), help='最终输出目录 (默认: ./output)')
    parser.add_argument('--skip-step1', action='store_true', help='跳过步骤1')
    parser.add_argument('--skip-step2', action='store_true', help='跳过步骤2')
    parser.add_argument('--skip-step3', action='store_true', help='跳过步骤3')
    parser.add_argument('--skip-step4', action='store_true', help='跳过步骤4')
    parser.add_argument('--keep-tmp', action='store_true', help='保留临时目录 tmp/ 不删除')
    args = parser.parse_args()

    # 自动推断数据集名称（从 mzML 目录名为主，若目录名通用则用其父目录名；兜底用 msms 文件名）
    dataset_name = os.path.basename(os.path.normpath(args.mzml_dir))
    if dataset_name.lower() in ('mzml', 'mzmls', 'mgf', 'mgfs', 'raw', 'raws', 'data'):
        parent = os.path.basename(os.path.dirname(os.path.normpath(args.mzml_dir)))
        if parent:
            dataset_name = parent
    if not dataset_name:
        dataset_name = os.path.splitext(os.path.basename(args.msms))[0]
    
    # 强制设置为固定名称 origin_data
    dataset_name = "origin_data"


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
