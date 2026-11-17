# Training and evaluation entry points

_The entry points are curated from the existing project; algorithms are not rewritten for packaging._

---

## 📋 Main stages

| Stage | Entry point | Reference settings |
| --- | --- | --- |
| State normalization | `compute_stats.py` | 1993-2016 |
| Deterministic training | `train_direct.py` | `configs/hpc_direct.yaml`, up to 120 epochs |
| Deterministic rollout fine-tuning | `train_rollout.py` | 7-day rollout, 8-epoch limit, 30-day validation |
| Residual normalization | `compute_residual_stats.py` | Frozen deterministic checkpoint |
| Residual diffusion training | `train_residual_diffusion.py` | 40 epochs in archived submission, batch size 1 |
| Residual rollout fine-tuning | `train_residual_rollout.py` | 12-epoch limit, 16 windows, 30-day rollout, 2 members, 4 mean steps |
| Frozen-model cycle | `ems_cycle.py` / `ems_observer.py` | Recorded recipe defines guidance and diagnostics |
| Experiment orchestration | `ems_experiments.py` | Tune/calibrate/validate phases with immutable records |
| Partial-validation support audit | `ems_validate61_support.py` | Explicit 61-case policy; no imputation |
| Availability screen | `ems_robust_quick.py` | Original 12-case guided/nudging screen, separate 10-step budget |
| Section extraction | `ems_sections_2d.py` | Winter day 90, 17.25°N and 135°E |

The source residual and fine-tuned residual checkpoints are both retained. The original pre-fine-tuning deterministic checkpoint is not present in the available project; reproduce it with deterministic training before rollout fine-tuning, or reuse the released final deterministic model for inference. Training from scratch does not claim an identical optimizer trajectory or final checkpoint bit pattern.

## ⚙️ Protocols and portability

The data archive contains original configs, recipes and fingerprints. Root `configs/` in this repository are portable copies with storage roots replaced; `tools/configure_release.py` sets real locations. Scientific source modules otherwise retain their original algorithms, including protocol validation. The assembly manifest lists file-level hashes.

Never point a fresh orchestration run at the archived data output folders. Historical completion/dependency checks may intentionally reject changed configurations or dependencies. Direct recipe relocation is supported for ordinary 90-day cases; the quick observer additionally validates its fixed quick-screen protocol and is not automatically relocated by that helper.

The paper's Deterministic availability comparison is an analysis of six guided cases plus two existing controls, not a new label for the original nudging runs. `tools/reproduce_tables.py` and the Figure 8 script perform the current comparison directly from the supplied matching daily sources.

## 🧪 Testing and hardware

The original Slurm scripts used one GPU, 8 CPU workers and up to 128 GB memory in several stages. These are archived requests, not measured universal minimum requirements. The release wrapper deliberately omits site-specific partition/account names. CUDA PyTorch must match the target cluster; CPU unit tests are distinct from production 50-member/90-day execution.

`RELEASE_VALIDATION.md` records the checks actually run for this package. A declared dependency range is not a lockfile for the historical HPC environment. New experiment runs should record Python/PyTorch/CUDA/GSW/NumPy versions, hardware, command line, seeds and input hashes.
