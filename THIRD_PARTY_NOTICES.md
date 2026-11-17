# Third-party components and provenance

---

## 📦 Runtime dependencies

Python, PyTorch, NumPy, netCDF4, PyYAML, tqdm, TensorBoard, GSW, Matplotlib, Pillow, pypdf and pytest are installed separately. They are not vendored in this archive and retain their own licences. The requirements files are installation specifications, not licence grants for those projects.

## 🔍 Release additions

Scientific project modules and selected plotting scripts were copied from the authors' local research snapshot. Portable configurations change storage roots, not scientific parameters. New `tools/` helpers, the portable submission wrapper, and documentation support packaging and reconstruction.

The original Figure 8 script referenced an external general-purpose panel-audit utility. That utility is **not** redistributed. `figures/figure8_support/audit_panel_alignment.py` is a newly written, release-local adapter that checks this figure's four-panel geometry and labels. It does not change scientific values or plotted artists. This is a packaging dependency replacement, not a new scientific method.

## 🔗 External scientific inputs

GLORYS and Argo data terms and attribution remain independent of this repository's MIT licence. No raw archive, proprietary plotting application, or Elsevier template is included in this code candidate. Figure PDFs and numerical source data are in the companion data archive; original research outputs there use CC BY 4.0, subject to the third-party exclusions in that archive's rights statement.
