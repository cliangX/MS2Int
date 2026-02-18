# Rescore (Mamba + MS2PIP features with m-ion)

Mokapot rescoring pipeline using **Mamba predictions (Intpredict)** and **MS2PIP-style features (including m-ions)**.

Phase 2 automatically adds **Basic / MaxQuant** features via `ms2rescore` (disable with `--no-basic` / `--no-maxquant`).

---

## Quick Start

```bash
bash spectrum_processing/rescore/run_pipeline.sh \
  "/path/to/WORKDIR" \
  "/path/to/model.pth" \
  "mamba_dev"
```

Output: `WORKDIR/rescore/mokapot/`

---

## Scripts

- **step01_filter_msms_unmodified_len30.py** — Filter msms.txt (Unmodified, Length<=30, no selenocysteine)
- **step02_make_msms_specid.py** — Generate SpecId TSV from Raw file / Scan number / Sequence / Charge
- **step03_calc_ms2pip_features_m.py** — Compute MS2PIP features (with m-ions; auto-generates SpecId if missing)
- **step04_rescore_mamba_ms2pip_m.py** — Two-stage mokapot rescoring (Phase 2 adds Basic/MaxQuant by default)
- **run_pipeline.sh** — End-to-end pipeline wrapper
- **add_specid_to_h5.py** — Standalone SpecId writer for H5 (usually not needed; step03 handles this)

---

## Input / Output

### Input (WORKDIR)

```
WORKDIR/
  txt/msms.txt
  mzml/*.mzML
```

### Output (WORKDIR/rescore/)

- `1.msms_filtered_unmodified_lenle30.txt`
- `msms_specid.tsv`
- `rescore_batch1.h5` (from `spectrum_processing/unmodficaiton/run.py`)
- `rescore.h5` (with `Intpredict` from `MS2Int/predict.py`)
- `.features_tmp*/` (temporary; auto-cleaned by pipeline)
- `mokapot/mokapot.psms.txt`
- `mokapot/mokapot.peptides.txt`

---

## Manual Step-by-Step

Run from the MS2Int repo root with conda env `mamba_dev`.

1) Filter msms.txt
```bash
conda run -n mamba_dev python spectrum_processing/rescore/step01_filter_msms_unmodified_len30.py \
  -i /path/to/WORKDIR/txt/msms.txt \
  -o /path/to/WORKDIR/rescore/1.msms_filtered_unmodified_lenle30.txt
```

2) Generate SpecId TSV
```bash
conda run -n mamba_dev python spectrum_processing/rescore/step02_make_msms_specid.py \
  /path/to/WORKDIR/rescore/1.msms_filtered_unmodified_lenle30.txt \
  /path/to/WORKDIR/rescore/msms_specid.tsv
```

3) Build H5 (observed spectra + train_data)
```bash
cd /path/to/WORKDIR
conda run -n mamba_dev python /path/to/MS2Int/spectrum_processing/unmodficaiton/run.py \
  --msms rescore/1.msms_filtered_unmodified_lenle30.txt \
  --mzml-dir mzml \
  --dataset-name rescore \
  --output rescore/rescore_batch1.h5
```

4) MS2Int prediction (writes Intpredict)
```bash
cd /path/to/WORKDIR
conda run -n mamba_dev python /path/to/MS2Int/MS2Int/predict.py \
  --ckpt /path/to/model.pth \
  --input rescore/rescore_batch1.h5 \
  --output rescore/rescore.h5
```

5) Compute MS2PIP features (with m-ions)
```bash
cd /path/to/WORKDIR
conda run -n mamba_dev python /path/to/MS2Int/spectrum_processing/rescore/step03_calc_ms2pip_features_m.py \
  --h5_path rescore/rescore.h5 \
  --tsv_path rescore/msms_specid.tsv \
  --output rescore/.features_tmp/ms2pip_features_m.tsv
```

6) Mokapot rescoring
```bash
cd /path/to/WORKDIR
conda run -n mamba_dev python /path/to/MS2Int/spectrum_processing/rescore/step04_rescore_mamba_ms2pip_m.py \
  --msms_path rescore/1.msms_filtered_unmodified_lenle30.txt \
  --tsv_path rescore/.features_tmp/ms2pip_features_m.tsv \
  --rng 42 --folds 2 --max_workers 2 -v
```

---

## Dependencies

In addition to MS2Int requirements:
- `mokapot`
- `ms2rescore`
- `psm-utils`

```bash
conda run -n mamba_dev pip install mokapot ms2rescore psm-utils
```
