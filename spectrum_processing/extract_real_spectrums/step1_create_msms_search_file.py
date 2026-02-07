import os
import pandas as pd
import glob
import warnings

# keep output quiet: suppress pandas dtype noise
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

def run_step1(config):
    # 从配置中获取路径（兼容两个键名，优先使用新键 'msms'）
    COMBINED_MSMS_PATH = config['paths'].get('msms') or config['paths']['combined_msms']
    MSMS_OUTPUT_DIR = config['paths']['msms_dir']
    SEARCH_OUTPUT_DIR = config['paths']['search_dir']
    FILTERED_MSMS_DIR = config['paths']['msms_filtered_dir']

    # 创建输出目录（如果不存在）
    os.makedirs(MSMS_OUTPUT_DIR, exist_ok=True)
    
    # 打开输入文件
    with open(COMBINED_MSMS_PATH, 'r') as f:
        # 读取标题行
        header = f.readline().strip()
        
        # 文件句柄字典，用于存储每个raw file对应的输出文件
        output_files = {}
        
        # 计数器
        line_counts = {}
        
        # 逐行处理数据
        for line in f:
            # 从行中提取raw file名称（第一列）
            raw_file = line.split('\t')[0]
            
            # 如果这是第一次遇到这个raw file，创建对应的输出文件
            if raw_file not in output_files:
                # 使用原始raw文件名，但改为.txt后缀
                output_filename = f"{raw_file}.txt"
                output_path = os.path.join(MSMS_OUTPUT_DIR, output_filename)
                output_files[raw_file] = open(output_path, 'w')
                output_files[raw_file].write(header + '\n')
                line_counts[raw_file] = 0
            
            # 将行写入对应的输出文件
            output_files[raw_file].write(line)
            line_counts[raw_file] += 1

    # 关闭所有输出文件并打印统计信息（仅统计性输出）
    total_lines = 0
    min_lines = None
    max_lines = None
    for raw_file, file_handle in output_files.items():
        file_handle.close()
        cnt = line_counts[raw_file]
        total_lines += cnt
        min_lines = cnt if min_lines is None else min(min_lines, cnt)
        max_lines = cnt if max_lines is None else max(max_lines, cnt)

    num_files = len(output_files)
    avg_lines = (total_lines / num_files) if num_files else 0
    print(f"MSMS拆分: 文件数={num_files}, 总行数={total_lines}")

    # 创建输出目录（如果不存在）
    os.makedirs(SEARCH_OUTPUT_DIR, exist_ok=True)

    # 通过 COMBINED_MSMS_PATH 中的 'Raw file' 列创建占位 Search 文件
    # 为了避免整表读入内存，这里流式读取并定位 'Raw file' 列索引
    created_empty = 0
    unique_raw_files = set()
    with open(COMBINED_MSMS_PATH, 'r') as fin:
        header_cols = fin.readline().rstrip('\n').split('\t')
        try:
            raw_col_idx = header_cols.index('Raw file')
        except ValueError:
            # 兜底：如果找不到列名，退回第一列（与上面的拆分逻辑一致）
            raw_col_idx = 0
        for line in fin:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= raw_col_idx:
                continue
            unique_raw_files.add(parts[raw_col_idx])

    for raw_name in sorted(unique_raw_files):
        txt_filename = f"{raw_name}.txt"
        txt_path = os.path.join(SEARCH_OUTPUT_DIR, txt_filename)
        # 创建空文件占位
        with open(txt_path, 'w'):
            pass
        created_empty += 1



    # 创建目标文件夹（如果不存在）
    os.makedirs(FILTERED_MSMS_DIR, exist_ok=True)

    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(MSMS_OUTPUT_DIR, "*.txt"))

    # 处理每个文件（仅统计性输出）
    processed = 0
    failed = 0
    total_before = 0
    total_after = 0
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            before = len(df)
            # Filter: 仅保留磷酸化肽段（Phospho (STY) 或 Phospho(Y)）、Length <= 30，且去除含 U 的序列
            df['Length'] = pd.to_numeric(df.get('Length'), errors='coerce')
            mods = df['Modifications'].astype(str).str.strip().str.lower()
            # 兼容两种标记方式：
            #   - Phospho (STY)：MaxQuant 经典 STY 磷酸
            #   - Phospho(Y)：Y 特异性磷酸
            is_phospho_sty = mods.str.contains(r"phospho\s*\(\s*sty\s*\)")
            is_phospho_y = mods.str.contains(r"phospho\s*\(\s*y\s*\)")
            is_phospho = is_phospho_sty | is_phospho_y
            seq_str = df['Sequence'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
            no_U = ~seq_str.str.contains('U', regex=False)
            df = df[is_phospho & (df['Length'] <= 30) & no_U]
            after = len(df)
            output_path = os.path.join(FILTERED_MSMS_DIR, file_name)
            df.to_csv(output_path, sep='\t', index=False)
            processed += 1
            total_before += before
            total_after += after
        except Exception as e:
            failed += 1

    removed = total_before - total_after
    print(f"MSMS过滤: 成功={processed}, 失败={failed}, 原始行={total_before}, 过滤后行={total_after}, 去除={removed}")

    print("3.1完成")

if __name__ == "__main__":
    # 仅用于直接运行此脚本的测试
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    run_step1(config)
