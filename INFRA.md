# Infrastructure

Where code, data, checkpoints, and experiments physically live. This is
operational, not scientific — kept out of PLAN.md on purpose.

## Three locations, clear roles

| Location | Role | Lifetime |
|---|---|---|
| **GitHub (this repo)** | Source of truth for code, configs, text | Permanent |
| **AutoDL GPU instance** | Runs simulation + training | Ephemeral — SSD wiped on shutdown |
| **HuggingFace Hub** | Persistent storage for datasets and model weights | Permanent |

The AutoDL box is treated as a scratch compute node. Anything that
needs to survive a shutdown must be pushed to GitHub (code) or
HuggingFace (data / weights).

## Python environments — **two venvs** (plan B)

Separate environments for simulation and training to avoid JAX↔PyTorch
CUDA / library conflicts.

```
/root/envs/
  sim/     # jax[cuda12], brax, numpy, jupyter
  train/   # torch, numpy, pyyaml, tqdm, huggingface_hub, tensorboard
```

Setup:

```bash
python -m venv /root/envs/sim   && source /root/envs/sim/bin/activate   && pip install -e .[sim]
python -m venv /root/envs/train && source /root/envs/train/bin/activate && pip install -e .[train]
```

`pyproject.toml` should expose `sim` and `train` as optional-dependency
extras so a single `pip install -e .[sim]` picks up everything needed
for that venv.

Neither venv ever imports the other's framework. Data moves between
them via `.npz` files on disk.

## Datasets — HuggingFace Datasets

- **Small springs set (~50MB)**: regenerate on demand with
  `make data CONFIG=springs_small`. Seed pinned, output byte-identical.
  Not uploaded.
- **Full dataset used for published results**: pushed to
  `tsuikahk/fabric-<system>-<date>` on HF Datasets. Pinned by commit
  hash in `configs/*.yaml`.

Upload pattern:

```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="data/springs",
    repo_id="tsuikahk/fabric-springs-v1",
    repo_type="dataset",
)
```

Download pattern in `fabric/data/dataset.py`:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("tsuikahk/fabric-springs-v1", repo_type="dataset")
```

## Model checkpoints — HuggingFace Hub

- During training: keep the last 3 checkpoints on the AutoDL SSD, delete
  older ones to save disk.
- On checkpoint-best-val: push to HF Hub as
  `tsuikahk/fabric-<system>-<run_id>`.
- Each HF model repo must include: the `configs/*.yaml` used, the git
  SHA of this repo at training time, the dataset HF repo id + revision.

Upload helper: `fabric/utils/hub.py::push_checkpoint(state_dict, run_id)`.

## Reproducibility protocol

A result is only "real" if someone can rerun it from scratch. Every
result in `RESULTS.md` must link to:

1. A git SHA in this repo.
2. An HF dataset repo + revision.
3. An HF model repo + revision.
4. A config file path that reproduces the exact run.

`scripts/reproduce.sh <run_id>` should be a one-shot that:

```
git checkout <sha>
source /root/envs/sim/bin/activate   && make data   CONFIG=<config>
source /root/envs/train/bin/activate && make train  CONFIG=<config>
source /root/envs/train/bin/activate && make verify CHECKPOINT=<hf_repo>
```

## AutoDL workflow loop

Day-to-day rhythm once M0 is done:

1. Edit code on Claude Code (web) → `git push` to this repo.
2. SSH into AutoDL → `git pull`.
3. `source /root/envs/sim/bin/activate && make data` (generates / uploads).
4. `source /root/envs/train/bin/activate && make train` (trains / uploads).
5. Pull the numbers back to the repo: `make pull-results` fetches
   final metrics JSON from HF → writes into `results/<run_id>.json` →
   `git commit` → `git push`.
6. Claude Code (web) updates `RESULTS.md` from the new JSON.

## Costs (rough)

- AutoDL 4090: ~¥1.8 / hour. Budget ¥500–1500 for the whole project.
- HF Hub: free for public repos; private repos are free under size
  limits we won't hit.
- arXiv / OpenReview: free.
- Total cash: dominated by AutoDL. Set a `BUDGET.md` entry when spend
  goes past ¥500 so we don't sleepwalk past ¥1500.
