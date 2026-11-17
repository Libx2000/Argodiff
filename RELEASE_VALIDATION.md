# Local release validation

_Checks performed on 17 November 2026. These describe this release candidate, not a new production experiment._

---

## ✅ Completed checks

| Check | Outcome |
| --- | --- |
| Scientific unit/regression tests | 33 passed |
| Synthetic NetCDF/U-Net end-to-end smoke test | 1 passed; train, validate, save checkpoints and run two-day cycles |
| Python parsing | 80 files parsed successfully |
| Tables 2–6 | Reconstructed numeric cells match the current manuscript |
| Figures 2–8 | All regenerated successfully as readable single-page PDFs |
| Figure 3/4/8 exported numerical tables | Match the supplied source tables within stated floating-point tolerance |
| Figure 8 panel geometry | Local 2-by-2 alignment/label check passed |
| Archived output integrity | 2,729 outputs match original completion-record hashes; 137 execution logs intentionally omitted |
| Portable configuration helper | Data-root changes checked in a separate disposable checkout |
| Archived recipe relocation | Ordinary 90-day SA-Diff recipe prepared successfully without executing it |
| GitHub large-file check | No model weights in code; largest source asset below 100 KiB |
| Original manuscript | Unchanged SHA-256: `93250b1a07c0e342b092a1d0381c67f73e5b24fb7699969627386103b025903a` |

The data companion includes the structured validation report, final JUnit reports, table-cell comparison and source-integrity audit. SHA-256 inventories are generated after documentation is finalized. Figure PDFs can differ across fonts/library versions; the numerical source checks do not require byte-identical PDF files.

## ⚙️ Validation environment

Windows, Python 3.11.15, PyTorch 2.13.0+cpu, NumPy 2.3.5, netCDF4 1.7.4, Matplotlib 3.11.1, PyYAML 6.0.3, GSW 3.6.23, pytest 9.2.0 and pypdf 7.1.0. Small test-only dependencies were installed outside this repository. This records the local validation environment, not the exact historical HPC environment.

```bash
python -m pytest -q tests/test_direct.py tests/test_residual_rollout_finetune.py \
  tests/test_spread_calibration.py test_ems_v1.py test_ems_lowmem.py test_ems_sections_2d.py
python -m pytest -q tests/test_residual_rollout_end_to_end.py
```

## 🔍 Scope of validation

No new full-scale GPU training, real-observation assimilation cycle, 90-day rerun or final-year test was performed while packaging. The original input archives remain necessary for those tasks. Model checkpoint serialization was not executed merely to calculate archive checksums.

The revised Figure 4 depth summary and archived daily scores retain distinct provenance. Passing table reconstruction verifies the reported summary calculation; it does not supply the missing revised daily series or independently validate the author-supplied correction. The author subsequently confirmed CC BY 4.0 for original data and MIT for original code; those licence notices are included. Creator credit and public identifiers still require completion before publication.
