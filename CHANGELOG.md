# Changelog

All notable changes to this repository are documented here. Entries below are
relative to `origin/main`.

## Unreleased

### Added

- `fov.ipynb` now also generates a **Reduced Inscribed Circle FOV**
  (`reduced-inscribed-circle.ds9`), a smaller inscribed circle, scaled by
  `FOV_OVERLAP_FACTOR`, used to increase overlap between tiles for
  time-domain science.
- New `supplemental_notebooks/RollVectorField.ipynb`: computes
  `m4opt.dynamics.nominal_roll` for every skygrid field over a range of
  times and renders it as an animated vector field on the sky, to visually
  confirm the roll field is only ill-defined at the Sun and anti-Sun points.
- `report.ipynb`: new diagnostics --  a weekly time-usage breakdown
  (observing / slewing / downlinking / block slack / empty blocks), survey
  completeness by constituent region over time, and slew duration/angle
  distributions split out by adjacency to a downlink.

### Changed

- `fov.ipynb`, `main.ipynb`, `report.ipynb`, `skyblocks.ipynb`, and
  `skygrid.ipynb` substantially rewritten for modularity and readability,
  with expanded markdown documentation of the scheduling model (block-to-slot
  matching, within-block TSP sequencing, observability constraints, and why
  the survey is scheduled in blocks rather than individual fields).
- `report.ipynb`'s cadence pair-separation analysis replaced an
  O(N_visits^2) `itertools.combinations` scan (which runs out of memory on a
  multi-year, many-pass survey) with a HEALPix-binned, per-pixel approach.
- Data products are being reorganized out of the flat `fov/`, `tables/`,
  `visualizations/` layout into per-run output directories (e.g.
  `preliminary_survey/`); `Makefile` targets are being updated to match
  (in progress).

### Fixed

- `skygrid.ipynb`: corrected a `$t_{\rm slew, fast}$` vs.
  `$t_{\rm slew, short}$` terminology inconsistency in the block-size
  derivation -- the code already computed a `SHORT_SLEW_DURATION`, but the
  formula's rendered math didn't match.
- Restored imports (`m4opt.fov`, `matplotlib.animation.FuncAnimation`,
  `tqdm`, `regions.Region`, `circle_to_polygon`, `PatchCollection`) in
  `fov.ipynb`, `main.ipynb`, and `skygrid.ipynb` that `ruff`'s unused-import
  autofix had incorrectly stripped: these names are referenced only inside
  `%%skipif`-gated animation cells, which ruff's notebook linting doesn't
  see into. Marked with `# noqa: F401` to prevent recurrence on the next
  `pre-commit run`.

### Chores

- `.gitignore`: ignore `.DS_Store`, `.idea/`, and `.ruff_cache/`.
