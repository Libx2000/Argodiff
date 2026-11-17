# External input data contract

_Required inputs for model execution; not all are included in the companion archive._

---

## 💾 GLORYS files

Copied configurations expect the following paths under the configured reanalysis root:

```text
thetao_glor_1.5deg/thetao_glor_YEAR_0_1.5deg.nc
so_glor_1.5deg/so_glor_YEAR_0_1.5deg.nc
```

Variables are `thetao_glor` and `so_glor`; coordinate names are `time`, `elevation`, `lat`, `lon`. The archive has 120-by-240 horizontal points and 15 retained depths. Training/normalization spans 1993-2016; configurations also list later validation/test ranges that must not be confused with completed manuscript cases.

The exact regridding/coarsening operator from the native GLORYS12 product is not fully documented in the available project. Therefore this repository supports the preprocessed input format but does not supply an invented raw-product-to-training-grid conversion. Obtain the source product from Copernicus Marine Service and verify coordinates, depth order, missing-value masks and provenance against the released normalization archive.[^1]

## 📊 Argo observations

`argo_daily_profile_stats.py` inventories exported daily NetCDF files; `prepare_paper_argo.py` creates a fixed spatial split and interpolates/aggregates guidance profiles. `prepare_oe_argo.py`/`audit_oe_argo.py` retain source-declaration checks. Read each entry point's `--help` for required inventory and input paths.

Expected observation layout is `Argo/argo_derived/paper_split_v1/guidance` and `.../holdout`. The spatial split uses 9° blocks, holdout target 0.30 and one-grid-cell guidance buffer; defaults and seeds are recorded in the source. These are block fractions, not enforced per-day profile counts.

The particular original Argo export's temperature convention/native QC/adjusted-value provenance remains unresolved. Public Argo access guidance is an upstream route, not proof that an arbitrary export is thermodynamically equivalent.[^2] Do not mark `human_confirmed=true` without actual verification.

## 🔧 Checks before inference

- Match yearly time coordinates and the three-day history needed at initialization
- Match latitude, longitude, stored depths, channel order and masks to the frozen checkpoint/statistics
- Install the released residual statistics and the selected 2018 calibration file
- Retain the original declared observation variable convention, or document and validate a new preprocessing branch
- Use a fresh output directory; do not overwrite historical completion records

[^1]: Copernicus Marine Service. Global Ocean Physics Reanalysis. https://doi.org/10.48670/moi-00021
[^2]: Argo programme. Data access. https://argo.ucsd.edu/data/
