# Code and numerical data

A compact directional-lobe representation for repeated urban shading queries

## Contents

- `final_results/`: frozen sample manifests, accepted geometry, calibration,
  fitted candidates, quantised records, query arrays, measured comparator costs,
  thermal inputs/outputs, and full-precision result summaries.
- `focused_analysis/`: common-cohort external bias attribution, patch-specific
  summaries of the existing capacity sweep, and selective-caster examples.
- `code/`: scientific calculation, analysis, and figure scripts.
- `data/`: the input geometry snapshots and weather files used for the study.
- `source/original_package/urban_shading_method_article/`: shared computational
  primitives, synthetic scene inputs, and the inputs needed for the reference
  audit. Manuscripts and editorial correspondence are excluded.
- `figures/`: the scientific figures used in the article.
- `INPUT_PROVENANCE.json`: input names, source catalogue, weather URL and hashes.
- `FILE_MANIFEST.json`: SHA-256 hashes of packaged files.

The authoritative measured results are in `final_results/`.
For workflow costs use `final_results/analysis/complete_workflow_costs.csv`;
`workflow_costs.csv` is an intermediate accounting table without the separately
measured proxy-construction and archive-encoding terms. The primary calibration
is fixed at R = 250 m, K = 160, biweight kernel and gamma = 0.6. The exploratory
capacity sweep describes sensitivity on the comparator subset.

## Reading and checking the results

Install the dependencies in `requirements-reproduction.txt` in a separate Python
environment. Run commands from this directory:

```bash
python code/reproduce_study.py check
python code/focused_diagnostics.py
```

The first command checks the file hashes, geometry/aggregation checks, sample
counts and headline errors from saved arrays. The second recomputes the focused
summaries and stored-caster examples; it does not refit models or remeasure timing.
The example predictions are compared with the frozen quantised predictions.

Useful result entry points are `scientific_summary.json`, `report_values.json`,
`kind_metrics.csv`, `stratified_error_conditions.csv`, `stage_worst_cases.json`,
`hourly_energy_metrics.csv`, `thermal_differences.csv`, and
`exploratory_K_sensitivity.csv`, all under `final_results/analysis/`.
The direction-level table is `directional_results.csv.gz`.

The primary grid contains 256 receivers and 117,051 queries. The comparator and
hourly subsets contain 32 and 64 receivers. The thermal experiment contains 32
paired reference/lobe simulations of the common test cell. Scene and receiver
identifiers link the tables to their arrays and geometric records.

## Full computation

The recorded environment and package versions are provided with the results and
requirements. The measured platform was Linux on AMD EPYC 9V74, with nine logical
CPUs available. Ray comparisons use Embree through `embreex`. Timing results
depend on the implementation and machine.

EnergyPlus 26.1.0 (commit 6f2e40d102) is obtained separately from its official
release: https://github.com/NatLabRockies/EnergyPlus/releases/tag/v26.1.0
Place its distribution under `dependencies/EnergyPlus-*/` so that an `energyplus`
executable is available there, or adapt the EP assignment in `thermal_study.py`.

```bash
python code/reproduce_study.py fresh /absolute/path/to/new_reproduction
```

The destination must not already exist. This launcher copies the inputs and code,
rebuilds geometry, runs calibration and validation, then the hourly, thermal and
comparator calculations and analyses. The supplied results were obtained through
the individual phase scripts. The launcher is provided for a separate fresh run.
Preserve the frozen measurements: `extended_diagnostics.py` and `report_values.py`
remeasure some construction/encoding costs when run.

Delivered EnergyPlus `cell.epJSON` files retain the executed schedule paths.
For manual reruns, set each `Schedule:File` file_name to its adjacent `gains.csv`.
The fresh-run launcher writes schedule paths for its destination automatically.

Geometry sources are attributed to TU Delft 3D Geoinformation / 3DBAG and the
CityJSON datasets catalogue; weather is from European Commission JRC PVGIS.
Their source terms continue to apply. Pickle files contain Python-serialized
objects and should be loaded only from a trusted copy of the package.
