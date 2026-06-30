# Data Guide

This repository does not commit raw or processed data files because they are large and reproducible.

## Raw data

Expected local path:

- `Rule-based Scene/data/raw/mick_df_full_raw.csv`

You can recreate it from the Hugging Face dataset with:

```bash
cd /Users/dienmayhaituyet/Documents/AVM
python3 - <<'PY'
from datasets import load_dataset
from pathlib import Path

out = Path("Rule-based Scene/data/raw/mick_df_full_raw.csv")
out.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("MikeGreen2710/mick_df")
split_name = "train" if "train" in ds else list(ds.keys())[0]
df = ds[split_name].to_pandas()
df.to_csv(out, index=False)
print("wrote", out, df.shape)
PY
```

## Processed data

Processed files are created by running:

- `Rule-based Scene/notebooks/01_process_data.ipynb`

or:

```bash
cd /Users/dienmayhaituyet/Documents/AVM/Rule-based\ Scene/src
MAX_RAW_ROWS=250000 python3 process_data.py
```

That command will create local files under:

- `Rule-based Scene/data/processed/`

By default the scene uses a `250000` row sample cap for reproducible local execution. Set `MAX_RAW_ROWS=0` if you want to attempt a full raw-data pass.
