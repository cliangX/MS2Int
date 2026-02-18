import gc
import os
import warnings
from functools import partial
from multiprocessing import Pool

import h5py
import pandas as pd
from einops import rearrange
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.io.pytables.PerformanceWarning)


def process_file(path, dataset_name):
    try:
        df = pd.read_hdf(path, "combined_data")
        df["dataset"] = dataset_name
        return df
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None
    finally:
        gc.collect()


def run_step4(config):
    result_base_path = config["paths"]["final_dir"]
    ylabel_df_dir = config["paths"]["ylabel_df_dir"]
    dataset_name = config["dataset"]["name"]
    batch_size = config["performance"]["batch_size"]
    num_workers = config["performance"]["num_workers"]

    def get_file_paths(directory):
        file_paths = []
        if os.path.isdir(directory):
            for file in os.listdir(directory):
                if file.endswith(".h5"):
                    file_paths.append(os.path.join(directory, file))
        return file_paths

    def process_batch(file_paths, batch_idx):
        process_func = partial(process_file, dataset_name=dataset_name)

        with Pool(processes=num_workers) as pool:
            dataframes = list(
                tqdm(
                    pool.imap(process_func, file_paths),
                    total=len(file_paths),
                    desc="Processing files",
                )
            )

        combined_df = pd.concat(
            [df for df in dataframes if df is not None], ignore_index=True
        )
        print("Combined DataFrame shape:", combined_df.shape)

        output_file = os.path.join(
            result_base_path, f"{dataset_name}_batch{batch_idx + 1}.h5"
        )

        f = h5py.File(output_file, "w")

        dset = f.create_dataset(
            "dataset", data=combined_df["dataset"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "Dataset name"

        dset = f.create_dataset(
            "Sequence", data=combined_df["Sequence"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "Peptide sequence"

        dset = f.create_dataset("Length", data=combined_df["Length"].tolist())
        dset.attrs["description"] = "Peptide length"

        dset = f.create_dataset(
            "Modifications",
            data=combined_df["Modifications"].astype(str).values.astype("S"),
        )
        dset.attrs["description"] = "Modifications"

        dset = f.create_dataset(
            "Mass_analyzer",
            data=combined_df["Mass analyzer"].astype(str).values.astype("S"),
        )
        dset.attrs["description"] = "Mass analyzer type"

        dset = f.create_dataset(
            "Fragmentation",
            data=combined_df["Fragmentation"].astype(str).values.astype("S"),
        )
        dset.attrs["description"] = "Fragmentation method"

        dset = f.create_dataset(
            "Modified_sequence",
            data=combined_df["Modified_sequence"].astype(str).values.astype("S"),
        )
        dset.attrs["description"] = "Modified peptide sequence"

        dset = f.create_dataset("Charge", data=combined_df["Charge"].tolist())
        dset.attrs["description"] = "Charge state"
        dset = f.create_dataset(
            "MS2_Scan_Number", data=combined_df["MS2_Scan_Number"].tolist()
        )
        dset.attrs["description"] = "MS2 scan number"
        dset = f.create_dataset("Score", data=combined_df["Score"].tolist())
        dset.attrs["description"] = "Identification score"

        dset = f.create_dataset(
            "Raw_file", data=combined_df["Raw_file"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "Raw file name"

        dset = f.create_dataset(
            "annotate", data=combined_df["annotate"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "ProForma annotation"

        dset = f.create_dataset("RT", data=combined_df["RT"].tolist())
        dset.attrs["description"] = "Retention time"

        dset = f.create_dataset(
            "instrument", data=combined_df["instrument"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "Instrument name"
        dset = f.create_dataset(
            "collision_energy", data=combined_df["collision_energy"].tolist()
        )
        dset.attrs["description"] = "Collision energy"
        
        dset = f.create_dataset(
            "Reverse", data=combined_df["Reverse"].astype(str).values.astype("S")
        )
        dset.attrs["description"] = "Reverse decoy flag"
        
        tmp = combined_df["train_data"].tolist()
        tmp = rearrange(tmp, "n h w -> n w h")
        dset = f.create_dataset("train_data", data=tmp)
        dset.attrs["description"] = "Training intensity matrix"

        f.close()
        print(f"Created output file: {output_file}")

    def main():
        file_paths = get_file_paths(ylabel_df_dir)
        num_batches = (len(file_paths) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(file_paths))
            batch_file_paths = file_paths[start_idx:end_idx]
            print(
                f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_file_paths)} files"
            )
            process_batch(batch_file_paths, batch_idx)

    main()


if __name__ == "__main__":
    import os

    import yaml

    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    run_step4(config)
