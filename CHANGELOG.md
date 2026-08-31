# Changelog

All notable changes to this repository are documented here. Entries below are
relative to `origin/main`.

## Unreleased

### Added

- New `notebooks/survey_utils.py`: a `SurveyProgram` dataclass (name, region, `mode`
  (`"block"`/`"field"`/`"mc"`), and either a fixed `visits` count or a `required_depth` to derive
  one from), plus `compute_visits_to_depth` (moved here from `skyblocks.ipynb`) and its inverse,
  `compute_depth_to_visits`. Ported/adapted from upstream `m4opt/uvex-scheduler`'s `survey.py`.
- `skyblocks.ipynb`: region membership, per-field visit goals, and the all-sky-base leftover flag
  are now driven generically by a `SURVEY_PROGRAMS` list (`SurveyProgram`s from
  `survey_utils.py`), defined in its own "Survey Programs" cell immediately after the
  settings/configuration cell so the survey's science requirements (which regions, how many
  visits, depth- vs. cadence-driven, block/field/mc mode) stay in one easy-to-find, easy-to-edit
  place rather than scattered across several hardcoded cells further down. The Magellanic
  Clouds' dedicated top-N-by-overlap block selection runs inline in the same "Determine If a
  Field is in a Region" loop as every other program, dispatched on `mode == "mc"`, mirroring
  upstream's single-loop `survey_programs` structure (upstream currently has this branch
  commented out; this fork's version is live).
- `report.ipynb`: new **visit map** diagnostic -- a HEALPix sky map of per-pixel visit count for
  each candidate FOV shape, complementing the existing visit-multiplicity histogram/CDF with
  *where* under/over-visited pixels actually are. Written to `visit-map.pdf`.
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

- `pyproject.toml`: bumped `m4opt>=2.11.0` to `>=2.12.0`, tracking upstream
  `m4opt/uvex-scheduler`'s dependency bump. (`uv.lock` was not regenerated in this environment --
  no `uv` available -- run `uv lock` before executing the pipeline.)
- `skyblocks.ipynb`: `ALL_SKY_VISITS` 6 -> 3, and `LMLZ_WIDE`/`LMLZ_DEEP` now specify
  `visits=10`/`visits=85` directly instead of deriving them from a required-depth target, to
  match upstream `m4opt/uvex-scheduler`'s `survey.py` (which hardcodes all four programs' visit
  counts with no depth formula at all). This fork's previous `LMLZ_WIDE_REQUIRED_DEPTH`/
  `LMLZ_DEEP_REQUIRED_DEPTH` (25.75/25.75 and 27.0/27.0 mag) implied 16 and 159 visits under
  `compute_visits_to_depth` given `SINGLE_VISIT_DEPTH` -- not 10 and 85 -- so those depth
  constants are removed rather than reconciled; `compute_depth_to_visits` now reports what depth
  the hardcoded visit counts actually reach (26.50/25.50 mag for LMLZ Wide, 27.66/26.66 mag for
  LMLZ Deep) as a diagnostic instead.
- `skyblocks.ipynb`/`main.ipynb`: LMLZ Deep (and its point-source companions) is no longer
  merged into the spatial block partition. Each LMLZ Deep field now gets its own dedicated
  **field-mode block** that revisits that one field for `MAX_DWELLS_PER_BLOCK_NO_SLEW`
  consecutive dwells per visit (tracked via a new `BLOCK_DWELLS` field column), needing only
  `ceil(EXPECTED_VISITS / MAX_DWELLS_PER_BLOCK_NO_SLEW)` block-visits instead of
  `EXPECTED_VISITS` single-dwell ones. Ported from upstream `m4opt/uvex-scheduler`'s
  `survey.py`/`mode="field"` design, adapted to this fork's `BLOCK_ID`/`BLOCKS` schema instead
  of introducing a separate `skyblocks.ecsv` join table. `main.ipynb`'s `plan_block` now also
  zeroes slew time between duplicate (identical-coordinate) targets, needed for a field-mode
  block's repeated dwells to sequence back-to-back instead of paying a spurious TSP "slew".
- `skyblocks.ipynb`: the block size distribution and padding over-visitation diagnostics now
  account for field-mode blocks instead of conflating them with spatially-tiled ones. Field-mode
  blocks (`DWELLS_PER_VISIT > 1`) are single-field by design, so they're excluded from the
  spatial block-size histogram/efficiency figure (tallied separately instead), and their
  ceiling-rounding over-delivery is reported as a distinct "rounding over-visitation" diagnostic
  rather than folded into the spatial "padding over-visitation" total -- on a real run, spatial
  padding excess was 1.66% vs. field-mode rounding excess of 17.65%, a distinction the old
  combined ~5.2% figure hid entirely.
- `report.ipynb`: updated the "Survey Completeness" section's description of `TOTAL_FIELD_VISITS`
  to include the `DWELLS_PER_VISIT` factor now used in `skyblocks.ipynb`.
- Makefile: `skyblocks.ipynb`'s target now also depends on `notebooks/survey_utils.py`, mirroring
  upstream's own `notebooks/survey.py` dependency; added `visit-map.pdf` to the report target.
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
- Not ported from upstream `m4opt/uvex-scheduler`: its re-accelerated cadence pair-separation
  scan (`m4opt.utils.numpy.count_intersect1d_combinations` /
  `ligo.skymap.util.progress_map_vectorized`, both new in m4opt 2.12). This fork's own
  HEALPix-binned per-pixel approach (above) already solves the same O(N²) problem and
  additionally produces the pair-count sensitivity curve upstream's version doesn't have, so it
  was kept as-is rather than replaced.

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
