import gc
import os
import warnings
from functools import partial
from multiprocessing import Pool

import h5py
import pandas as pd
from einops import rearrange
from tqdm import tqdm

# 忽略 PerformanceWarning
warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)


# 将process_file函数移到外部
def process_file(path, dataset_name):
    try:
        df = pd.read_hdf(path, "combined_data")
        # 将dataset设置为配置中的数据集名称
        df["dataset"] = dataset_name
        return df
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None
    finally:
        gc.collect()


def run_step4(config):
    # 获取配置信息
    result_base_path = config["paths"]["final_dir"]
    ylabel_df_dir = config["paths"]["ylabel_df_dir"]
    dataset_name = config["dataset"]["name"]
    batch_size = config["performance"]["batch_size"]
    num_workers = config["performance"]["num_workers"]

    def get_file_paths(directory):
        """获取指定目录下所有h5文件的路径"""
        file_paths = []
        if os.path.isdir(directory):
            for file in os.listdir(directory):
                if file.endswith(".h5"):
                    file_paths.append(os.path.join(directory, file))
        return file_paths

    def process_batch(file_paths, batch_idx):
        # 使用partial传递额外参数
        process_func = partial(process_file, dataset_name=dataset_name)

        # 使用多进程加速读取文件
        with Pool(processes=num_workers) as pool:
            dataframes = list(
                tqdm(
                    pool.imap(process_func, file_paths),
                    total=len(file_paths),
                    desc="Processing files",
                )
            )

        # 合并所有DataFrame
        combined_df = pd.concat(
            [df for df in dataframes if df is not None], ignore_index=True
        )
        print("Combined DataFrame shape:", combined_df.shape)

        output_file = os.path.join(
            result_base_path, f"{dataset_name}_batch{batch_idx + 1}.h5"
        )

        # 打开或创建HDF5文件
        f = h5py.File(output_file, "w")

        # 创建数据集并添加数据
        # 字符串列转换为字节串（与现有读取代码兼容）
        dset = f.create_dataset(
            "dataset", data=combined_df["dataset"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "数据集名称"

        dset = f.create_dataset(
            "Sequence", data=combined_df["Sequence"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "肽链的序列信息"

        dset = f.create_dataset("Length", data=combined_df["Length"].tolist())
        dset.attrs["description"] = "肽链的长度"

        dset = f.create_dataset(
            "Modifications",
            data=combined_df["Modifications"].astype(str).values.astype("S"),
        )
        dset.attrs["description"] = "肽链的修饰信息"

        dset = f.create_dataset(
            "Mass_analyzer",
            data=combined_df["Mass analyzer"].astype(str).values.astype("S"),
        )
        dset.attrs["description"] = "HCD"

        dset = f.create_dataset(
            "Fragmentation",
            data=combined_df["Fragmentation"].astype(str).values.astype("S"),
        )
        dset.attrs["description"] = "FTIM"

        dset = f.create_dataset(
            "Modified_sequence",
            data=combined_df["Modified_sequence"].astype(str).values.astype("S"),
        )
        dset.attrs["description"] = "包含修饰的完整肽链序列"

        dset = f.create_dataset("Charge", data=combined_df["Charge"].tolist())
        dset.attrs["description"] = "肽链的电荷状态"
        dset = f.create_dataset(
            "MS2_Scan_Number", data=combined_df["MS2_Scan_Number"].tolist()
        )
        dset.attrs["description"] = "二级质谱扫描的编号"
        dset = f.create_dataset("Score", data=combined_df["Score"].tolist())
        dset.attrs["description"] = "肽链鉴定的分数"

        dset = f.create_dataset(
            "Raw_file", data=combined_df["Raw_file"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "原始数据文件的名称"

        dset = f.create_dataset(
            "annotate", data=combined_df["annotate"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "附加的注释信息"

        dset = f.create_dataset("RT", data=combined_df["RT"].tolist())
        dset.attrs["description"] = "肽链在色谱中的保留时间"

        dset = f.create_dataset(
            "instrument", data=combined_df["instrument"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "使用的质谱仪器"
        dset = f.create_dataset(
            "collision_energy", data=combined_df["collision_energy"].tolist()
        )
        dset.attrs["description"] = "在MS2实验中使用的碰撞能量"
        
        dset = f.create_dataset(
            "Reverse", data=combined_df["Reverse"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "反向数据库匹配标记"
        
        tmp = combined_df["train_data"].tolist()
        tmp = rearrange(tmp, "n h w -> n w h")
        dset = f.create_dataset("train_data", data=tmp)
        dset.attrs["description"] = "用于训练模型的数据"

        # 关闭文件
        f.close()
        print(f"Created output file: {output_file}")

    def main():
        file_paths = get_file_paths(ylabel_df_dir)

        # 如果文件数大于batch_size，则分批次处理
        num_batches = (
            len(file_paths) + batch_size - 1
        ) // batch_size  # 计算需要多少批次

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(file_paths))
            batch_file_paths = file_paths[start_idx:end_idx]
            print(
                f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_file_paths)} files"
            )
            process_batch(batch_file_paths, batch_idx)

    # 执行主函数
    main()
    print("3.4完成")


if __name__ == "__main__":
    # 仅用于直接运行此脚本的测试
    import os

    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    run_step4(config)
