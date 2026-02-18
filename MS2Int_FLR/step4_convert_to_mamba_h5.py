#!/usr/bin/env python3
"""Convert TD DataFrame CSV to Mamba H5 input format."""

import pandas as pd
import numpy as np
import h5py
import argparse
import os
import re


def convert_deepflr_to_mamba_sequence(key_x):
    modification_map = {
        '1': '[Phospho]',
        '2': '[Oxidation]',
        '3': '[Carbamidomethyl]',
        '4': '[Acetyl]',
    }
    mamba_sequence = key_x
    if mamba_sequence.startswith('4'):
        mamba_sequence = '[Acetyl]-' + mamba_sequence[1:]
    for code, mod in modification_map.items():
        if code != '4':
            mamba_sequence = re.sub(f'([A-Z]){code}', f'\\1{mod}', mamba_sequence)
    return mamba_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--collision_energy", type=int, required=True)
    parser.add_argument("--fragmentation", type=str, required=True, choices=['HCD', 'CID', 'ETD'])
    parser.add_argument("--ref_h5", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required_columns = ['SourceFile', 'Fspectrum', 'PP.Charge', 'key_x', 'PEP.StrippedSequence']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_samples = len(df)
    print(f"Found {n_samples} peptide candidates")

    col_key_x = df['key_x'].to_numpy()
    col_strip = df['PEP.StrippedSequence'].to_numpy()
    col_charge = df['PP.Charge'].to_numpy()

    sequences = []
    lengths = []
    charges = []
    for idx in range(n_samples):
        sequences.append(convert_deepflr_to_mamba_sequence(str(col_key_x[idx])).encode('utf-8'))
        lengths.append(len(str(col_strip[idx]).strip()))
        charges.append(int(col_charge[idx]))

    sequences_array = np.array(sequences, dtype='S128')
    lengths_array = np.array(lengths, dtype=np.int32)
    charges_array = np.array(charges, dtype=np.int32)
    raw_files_array = np.array(df['SourceFile'].astype(str).str.encode('utf-8'), dtype='S100')
    ms2_scan_array = pd.to_numeric(df['Fspectrum'], errors='coerce').fillna(-1).astype(np.int32).to_numpy()

    with h5py.File(args.output, 'w') as f:
        f.create_dataset('Sequence', data=sequences_array)
        f.create_dataset('Length', data=lengths_array)
        f.create_dataset('Charge', data=charges_array)
        f.create_dataset('Raw_file', data=raw_files_array)
        f.create_dataset('MS2_Scan_Number', data=ms2_scan_array)

        ce_data = np.full((n_samples,), args.collision_energy, dtype=np.int32)
        frag_data = np.array([args.fragmentation.encode('utf-8')] * n_samples, dtype='S10')
        analyzer_data = None

        if args.ref_h5 is not None and os.path.isfile(args.ref_h5):
            with h5py.File(args.ref_h5, "r") as f_ref:
                if "train_data" in f_ref and f_ref["train_data"].shape[0] == n_samples:
                    f.create_dataset("train_data", data=f_ref["train_data"][()])
                if "collision_energy" in f_ref and f_ref["collision_energy"].shape[0] == n_samples:
                    ce_data = f_ref["collision_energy"][()].astype(np.float32)
                if "Fragmentation" in f_ref and f_ref["Fragmentation"].shape[0] == n_samples:
                    frag_data = f_ref["Fragmentation"][()]
                if "Mass_analyzer" in f_ref and f_ref["Mass_analyzer"].shape[0] == n_samples:
                    analyzer_data = f_ref["Mass_analyzer"][()]

        f.create_dataset('collision_energy', data=ce_data)
        f.create_dataset('Fragmentation', data=frag_data)
        if analyzer_data is not None:
            f.create_dataset('Mass_analyzer', data=analyzer_data)

        f.attrs['source'] = args.input
        f.attrs['n_samples'] = n_samples
        f.attrs['collision_energy'] = args.collision_energy
        f.attrs['fragmentation'] = args.fragmentation

    print(f"Samples: {n_samples}, Length: {lengths_array.min()}-{lengths_array.max()}, Charge: {charges_array.min()}-{charges_array.max()}")


if __name__ == "__main__":
    main()
