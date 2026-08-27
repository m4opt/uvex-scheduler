"""
Base classes for representing and validating survey schedules.

A survey schedule is a chronological table of spacecraft actions (observations, slews,
downlinks, ...). :class:`SurveySchedule` wraps such a table and enforces a schema against it:
per-column type/unit/dtype checks (:class:`~uvex_transient_toolkit.surveys.utils.QTableColumnSpec`),
plus per-action required-column and custom checks
(:class:`~uvex_transient_toolkit.surveys.utils.ActionSpec`). All validation errors are collected and
reported together via :class:`ScheduleValidationError`, rather than stopping at the first
failure.
"""

from collections.abc import Iterator
from pathlib import Path

import astropy_healpix as ah
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, SkyOffsetFrame
from astropy.table import QTable, Row, vstack
from astropy.time import Time
from astropy.utils.masked import Masked
from m4opt.fov import contains, footprint_healpix
from regions import CircleSkyRegion, RectangleSkyRegion, Regions, SkyRegion
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from uvex_transient_toolkit.utils import logger

from .utils import ActionSpec, QTableColumnSpec

# =========================================================================== #
# EXCEPTIONS                                                                  #
# =========================================================================== #


class ScheduleValidationError(ValueError):
    """
    Raised when a survey schedule table fails schema validation.

    Collects every validation failure found across both the column-level and
    action-level validation stages, rather than surfacing only the first one, so a user
    fixing an invalid schedule can address every problem in a single pass.
    """

    def __init__(self, errors: list[str]):
        self.errors = list(errors)

        message = (
            "Survey schedule failed validation with the following errors:\n"
            + "\n".join(f"  - {error}" for error in self.errors)
        )
        super().__init__(message)


def _sanitize_masked_value(value):
    """
    Rebuild a value using only :class:`~astropy.utils.masked.MaskedANDArray`.

    Discards whatever ndarray subclass a masked value's wrapper happened to pick up
    upstream (e.g. through repeated :func:`~astropy.table.vstack`), so that
    :meth:`SurveySchedule.to_disk` always writes columns in the one form astropy's
    ECSV reader is guaranteed to be able to reconstruct. Non-masked values pass
    through unchanged.
    """
    if isinstance(value, Masked):
        return Masked(np.asarray(np.ma.getdata(value)), mask=np.ma.getmaskarray(value))

    return value


def _bounding_radius(region: SkyRegion) -> u.Quantity:
    """
    Angular radius of the smallest cone, centered on ``region``, that fully contains it.

    Used by :attr:`SurveySchedule.bounding_radius` as a cheap pre-filter -- plain
    angular separation -- before falling back to an exact but more expensive
    containment test such as :func:`m4opt.fov.contains`.

    Parameters
    ----------
    region
        The field-of-view region, defined at RA=0deg/Dec=0deg/PA=0deg (the
        convention used throughout ``m4opt``, e.g. :attr:`m4opt.missions.Mission.fov`).

    Returns
    -------
    ~astropy.units.Quantity
        The bounding radius, inflated by 1% to stay conservative at the
        boundary.

    Raises
    ------
    TypeError
        If ``region`` is not one of the supported shapes.
    """
    if isinstance(region, RectangleSkyRegion):
        radius = np.hypot(region.width, region.height) / 2
    elif isinstance(region, CircleSkyRegion):
        radius = region.radius
    else:
        raise TypeError(
            f"Cannot compute a bounding radius for FOV region of type "
            f"{type(region).__name__!r}; supported types are "
            f"RectangleSkyRegion and CircleSkyRegion."
        )

    return 1.01 * radius


# =========================================================================== #
# SURVEY SCHEDULE                                                             #
# =========================================================================== #
class SurveySchedule:
    """
    A validated, chronological table of scheduled spacecraft actions.

    Wraps an :class:`~astropy.table.QTable` whose rows describe individual scheduled
    actions (observations, slews, downlinks, ...) and enforces the schema declared in
    :attr:`_SCHEMA` (per-column checks) and :attr:`_ACTION_SCHEMA` (per-action required
    columns and custom checks). See :class:`ScheduleValidationError` for how validation
    failures are reported.
    """

    _SCHEMA = {
        "start_time": QTableColumnSpec(
            column_type=Time,
            description="Start time of the scheduled action.",
        ),
        "duration": QTableColumnSpec(
            unit=u.s,
            validator=lambda col: np.all(col > 0 * u.s),
            description="Duration of the scheduled action.",
        ),
        "observer_location": QTableColumnSpec(
            column_type=EarthLocation,
            description="Observer position.",
        ),
        "action": QTableColumnSpec(
            dtype=np.str_,
            description="Scheduled action type.",
        ),
        "target_coord": QTableColumnSpec(
            column_type=SkyCoord,
            description="Target coordinate, masked for non-target actions.",
        ),
        "roll": QTableColumnSpec(
            unit=u.deg,
            description="Spacecraft roll angle.",
        ),
        "field_id": QTableColumnSpec(
            dtype=np.integer,
            description="Survey field identifier.",
        ),
        "block_id": QTableColumnSpec(
            dtype=np.integer,
            description="Scheduling block identifier.",
        ),
        "phase": QTableColumnSpec(
            dtype=np.str_,
            required=False,
            description=(
                "Optional label identifying which combined survey phase a row "
                "originated from (see SurveySchedule.with_phase/__add__)."
            ),
        ),
    }
    _ACTION_COLUMN = "action"
    _ACTION_SCHEMA: dict[str, ActionSpec] = {
        "observe": ActionSpec(
            required_columns=(
                "target_coord",
                "roll",
                "field_id",
                "block_id",
            ),
            validator=lambda rows: (
                True
                if np.all(rows["duration"] > 0)
                else "observation durations must be positive"
            ),
            description="Science observation.",
        ),
        "slew": ActionSpec(
            required_columns=("observer_location",),
            description="Spacecraft slew.",
        ),
        "downlink": ActionSpec(
            required_columns=("observer_location",),
            description="Communications downlink.",
        ),
    }

    # ----------------------------------------- #
    # Initialization                            #
    # ----------------------------------------- #
    def __init__(self, schedule_table: QTable, instrument_fov: SkyRegion, **kwargs):
        """
        Construct and validate a survey schedule.

        Parameters
        ----------
        schedule_table
            A chronological table of scheduled spacecraft actions. Copied on
            construction, so mutating it afterwards has no effect on this instance.
        instrument_fov
            The instrument's field of view, defined at RA=0deg/Dec=0deg/PA=0deg
            (the convention used throughout ``m4opt``, e.g.
            :attr:`m4opt.missions.Mission.fov`). Must be a `RectangleSkyRegion` or
            `CircleSkyRegion` -- see :func:`_bounding_radius`.
        **kwargs
            Forwarded to :meth:`_validate_table_columns` and
            :meth:`_validate_table_semantics`, so subclasses that override those
            hooks to add extra validation rules can accept extra constructor
            arguments without touching ``__init__`` itself.

        Raises
        ------
        TypeError
            If ``schedule_table`` is not a `~astropy.table.QTable`, or
            ``instrument_fov`` is not a `~regions.SkyRegion` of a supported shape.
        ScheduleValidationError
            If ``schedule_table`` fails schema validation.
        """
        # Ensure that the schedule is a QTable and make a copy to avoid mutating the caller's table.
        if not isinstance(schedule_table, QTable):
            raise TypeError(
                f"Parameter 'schedule_table' must be of type QTable, not {type(schedule_table)}"
            )

        self._schedule_table = schedule_table.copy()
        self._ensure_chronological()

        errors = [
            *self._validate_table_columns(**kwargs),
            *self._validate_table_semantics(**kwargs),
        ]
        if errors:
            raise ScheduleValidationError(errors)

        # Read in the instrument FOV and ensure that it is a valid region.
        if not isinstance(instrument_fov, SkyRegion):
            raise TypeError(
                f"Parameter 'instrument_fov' must be of type SkyRegion, not {type(instrument_fov)}"
            )

        self._instrument_fov = instrument_fov
        self._instrument_bounding_radius = _bounding_radius(instrument_fov)

        # Cached once here (rather than recomputed per query) for `get_rows_between_times`'s
        # binary-search lower bound -- see that method for why the longest action in the
        # whole schedule is the relevant quantity.
        self._max_action_duration = (
            self._schedule_table["duration"].max()
            if len(self._schedule_table)
            else 0 * u.s
        )
        self._max_action_duration_days = self._max_action_duration.to_value(u.day)

        # A plain float64 view of `start_time`'s Julian date, in a fixed scale, cached once
        # here so `get_rows_between_times` can binary-search a real numpy array instead of
        # calling `np.searchsorted` on the `Time` column directly. `astropy.time.Time` has no
        # vectorized `searchsorted` of its own, so `np.searchsorted` against a `Time` array
        # dispatches through a generic, unvectorized fallback that reconstructs the array
        # element by element -- for a single query against a schedule of a few thousand rows
        # this turns an O(log n) bisection into thousands of `Time` object reconstructions.
        # Query boundary times are converted into this same scale (an O(1) conversion, not
        # O(n)) before being compared against this array -- see `get_rows_between_times`.
        self._start_time_scale = self._schedule_table["start_time"].scale
        self._start_time_jd = np.asarray(
            self._schedule_table["start_time"].jd, dtype=np.float64
        )

        # Lazily built by `get_healpix_coverage_index`, keyed by `(nside, order)`; see
        # that method's docstring.
        self._HPX_MAP_CACHE = {}

    # ----------------------------------------- #
    # Schema Validation                         #
    # ----------------------------------------- #
    def _ensure_chronological(self) -> None:
        """Sort the schedule table by ``start_time`` in place, if it isn't already."""
        if "start_time" not in self._schedule_table.colnames:
            return

        start_time = self._schedule_table["start_time"]

        if not isinstance(start_time, Time) or start_time.ndim != 1:
            return

        order = np.argsort(start_time)

        if not np.array_equal(order, np.arange(len(order))):
            self._schedule_table = self._schedule_table[order]

    def _validate_table_columns(self, **_) -> list[str]:
        """Validate each column against :attr:`_SCHEMA`, collecting every error found."""
        errors: list[str] = []

        for column_name, column_spec in self._SCHEMA.items():
            errors.extend(
                column_spec.validate_against_table(self._schedule_table, column_name)
            )

        return errors

    def _validate_table_semantics(self, **_) -> list[str]:
        """
        Validate that the table's actions satisfy :attr:`_ACTION_SCHEMA`.

        Checks that every value in the action column is declared, then delegates to each
        :class:`~uvex_transient_toolkit.surveys.utils.ActionSpec` to check its action's required
        columns and any custom rules.
        """
        table = self._schedule_table

        if self._ACTION_COLUMN not in table.colnames:
            return [f"Missing required action column {self._ACTION_COLUMN!r}."]

        actions = np.asarray(table[self._ACTION_COLUMN]).astype(str)
        declared_actions = set(self._ACTION_SCHEMA)
        unknown_actions = set(actions) - declared_actions

        errors: list[str] = []

        if unknown_actions:
            errors.append(
                f"Unknown action values: {sorted(unknown_actions)}. Allowed values are {sorted(declared_actions)}."
            )

        for action_name, spec in self._ACTION_SCHEMA.items():
            errors.extend(
                spec.validate_against_table(
                    table,
                    action_name=action_name,
                    action_column=self._ACTION_COLUMN,
                )
            )

        return errors

    # ----------------------------------------- #
    # Dunder Methods                            #
    # ----------------------------------------- #
    def __len__(self) -> int:
        return len(self._schedule_table)

    def __iter__(self) -> Iterator[Row]:
        return iter(self._schedule_table)

    def __getitem__(self, key: str | int | slice | np.ndarray):
        return self._schedule_table[key]

    def __contains__(self, action: str) -> bool:
        return action in self.actions

    def __add__(self, other: "SurveySchedule") -> "SurveySchedule":
        """
        Concatenate two chronologically non-overlapping schedules.

        ``self + other`` appends ``other`` onto the end of ``self``, so ``other``
        must pick up where ``self`` leaves off: its earliest action may not start
        before :attr:`self.end_time <end_time>`. Combine more than two schedules by
        chaining, e.g. ``a + b + c``, since each ``+`` is evaluated left to right.

        Rows keep whatever ``"phase"`` label they already carry -- use
        :meth:`with_phase` beforehand to tag either schedule's rows with a label
        (e.g. so the combined schedule can be grouped or sorted by which input
        survey each row came from).

        Parameters
        ----------
        other
            The schedule to append after ``self``. Must share the same
            :attr:`fov` as ``self``.

        Returns
        -------
        SurveySchedule
            A new schedule, of ``type(self)``, whose table is the chronological
            concatenation of both inputs.

        Raises
        ------
        ValueError
            If ``other`` does not share ``self``'s :attr:`fov`, or if ``other``
            has any action, and ``self`` has any action, and ``other``'s
            earliest action starts before ``self``'s latest action ends.
        """
        if not isinstance(other, SurveySchedule):
            return NotImplemented

        if self.fov != other.fov:
            raise ValueError(
                "Cannot combine schedules with different instrument fields of view."
            )

        if len(self) and len(other) and other.start_time < self.end_time:
            raise ValueError(
                f"Cannot combine schedules: the second schedule's first action starts at "
                f"{other.start_time.iso!r}, before the first schedule's last action ends at "
                f"{self.end_time.iso!r}."
            )

        combined_table = vstack([self.table, other.table], join_type="outer")

        return type(self)(combined_table, self.fov)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} n_actions={self.n_actions} "
            f"start_time={self.start_time.iso!r} end_time={self.end_time.iso!r}>"
        )

    # ----------------------------------------- #
    # Properties                                #
    # ----------------------------------------- #
    @property
    def table(self) -> QTable:
        """QTable: The underlying, validated schedule table."""
        return self._schedule_table

    @property
    def actions(self) -> np.ndarray:
        """numpy.ndarray: The action label of every row, as strings."""
        return np.asarray(self._schedule_table[self._ACTION_COLUMN]).astype(str)

    @property
    def n_actions(self) -> int:
        """int: The total number of scheduled actions."""
        return len(self)

    @property
    def start_time(self) -> Time:
        """~astropy.time.Time: The start of the earliest scheduled action."""
        return self._schedule_table["start_time"].min()

    @property
    def end_time(self) -> Time:
        """~astropy.time.Time: The end of the latest scheduled action."""
        end_times = (
            self._schedule_table["start_time"] + self._schedule_table["duration"]
        )
        return end_times.max()

    @property
    def duration(self) -> u.Quantity:
        """~astropy.units.Quantity: The total elapsed time spanned by the survey."""
        return (self.end_time - self.start_time).to(u.s)

    @property
    def observing_time(self) -> u.Quantity:
        """~astropy.units.Quantity: Total time spent on ``observe`` actions."""
        return self.time_spent("observe")

    @property
    def action_summary(self) -> QTable:
        """QTable: Per-action row counts and total durations, one row per action type present."""
        actions = self.actions
        unique_actions = sorted(set(actions))

        return QTable(
            {
                "action": unique_actions,
                "count": [int(np.sum(actions == name)) for name in unique_actions],
                "total_duration": u.Quantity(
                    [self.time_spent(name) for name in unique_actions]
                ),
            }
        )

    @property
    def summary(self) -> QTable:
        """QTable: A single-row overview of the whole survey (start/end time, action count, ...)."""
        return QTable(
            {
                "start_time": [self.start_time],
                "end_time": [self.end_time],
                "duration": u.Quantity([self.duration]),
                "n_actions": [self.n_actions],
                "observing_time": u.Quantity([self.observing_time]),
            }
        )

    @property
    def fov(self) -> SkyRegion:
        """~regions.SkyRegion: The instrument field of view at RA=0deg/Dec=0deg/PA=0deg."""
        return self._instrument_fov

    @property
    def observing_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask, one per row of :attr:`table`, flagging ``"observe"`` actions."""
        return self.actions == "observe"

    @property
    def bounding_radius(self) -> u.Quantity:
        """
        ~astropy.units.Quantity: Angular radius of the smallest cone that fully contains :attr:`fov`.

        Inflated by 1% for a conservative margin. Used internally (e.g. by
        :meth:`get_observations_of`) as a cheap pre-filter -- plain angular
        separation -- before falling back to an exact but more expensive
        containment test such as :func:`m4opt.fov.contains`.
        """
        return self._instrument_bounding_radius

    # ----------------------------------------- #
    # Utility Methods                           #
    # ----------------------------------------- #
    def _resolve_tstart_tend(
        self,
        start_time: Time | None = None,
        end_time: Time | None = None,
    ) -> tuple[Time, Time]:
        """
        Fill in omitted survey-interval bounds with the survey's own extent.

        Shared by every ``compute_*``/query method that accepts optional
        ``start_time``/``end_time`` bounds, so ``None`` uniformly means "the
        whole survey" rather than each call site re-deriving that default.

        Parameters
        ----------
        start_time
            Start of the interval, or `None` to use :attr:`start_time`.
        end_time
            End of the interval, or `None` to use :attr:`end_time`.

        Returns
        -------
        tuple[~astropy.time.Time, ~astropy.time.Time]
            ``(start_time, end_time)`` with defaults filled in.
        """
        if start_time is None:
            start_time = self.start_time

        if end_time is None:
            end_time = self.end_time

        return start_time, end_time

    def with_phase(self, phase: str) -> "SurveySchedule":
        """
        Return a copy of this schedule with every row labeled by ``phase``.

        Sets (or overwrites) the optional ``"phase"`` column so that, once this
        schedule has been combined with another via :meth:`__add__`, rows can be
        told apart by which input schedule they came from -- e.g. to sort or
        group the combined table by phase.

        Parameters
        ----------
        phase
            Label to assign to every row.

        Returns
        -------
        SurveySchedule
            A new schedule, of ``type(self)``, identical to this one except for
            its ``"phase"`` column.
        """
        table = self.table.copy()
        table["phase"] = phase

        return type(self)(table, self.fov)

    def time_spent(self, action: str) -> u.Quantity:
        """
        Total duration spent on a given action across the whole survey.

        Parameters
        ----------
        action
            One of the action names declared in :attr:`_ACTION_SCHEMA`.

        Returns
        -------
        ~astropy.units.Quantity
            The summed ``duration`` of every row with this action, or ``0 * u.s`` if the
            action does not occur in the schedule.
        """
        mask = self.actions == action

        if not np.any(mask):
            return 0 * u.s

        return np.sum(self._schedule_table["duration"][mask])

    def get_row_at_time(self, time: Time) -> Row:
        """
        Return the scheduled action active at a given time.

        An action with ``start_time`` ``s`` and ``duration`` ``d`` is treated as covering
        the half-open interval ``[s, s + d)``.

        Parameters
        ----------
        time
            A scalar time to look up.

        Returns
        -------
        ~astropy.table.Row
            The row of :attr:`table` whose interval contains ``time``.

        Raises
        ------
        ValueError
            If ``time`` is not scalar, or no scheduled action covers it.
        """
        if not time.isscalar:
            raise ValueError("Parameter 'time' must be a scalar Time.")

        action_start = self._schedule_table["start_time"]
        action_end = action_start + self._schedule_table["duration"]
        matches = np.flatnonzero((action_start <= time) & (time < action_end))

        if matches.size == 0:
            raise ValueError(f"No scheduled action covers time {time.iso!r}.")

        return self._schedule_table[matches[0]]

    def get_rows_between_times(self, start_time: Time, end_time: Time) -> QTable:
        """
        Return the scheduled actions overlapping a time window.

        An action with ``start_time`` ``s`` and ``duration`` ``d`` is treated as covering
        the half-open interval ``[s, s + d)``; it is included if that interval overlaps
        ``[start_time, end_time)``.

        Parameters
        ----------
        start_time
            Start of the query window (inclusive).
        end_time
            End of the query window (exclusive).

        Returns
        -------
        QTable
            The subset of :attr:`table` overlapping the window, in chronological order.

        Raises
        ------
        ValueError
            If ``end_time`` is not after ``start_time``.
        """
        start_time, end_time = self._resolve_tstart_tend(start_time, end_time)
        if end_time <= start_time:
            raise ValueError(
                f"Parameter 'end_time' ({end_time.iso!r}) must be after 'start_time' ({start_time.iso!r})."
            )

        if len(self._start_time_jd) == 0:
            return self._schedule_table

        # `self._start_time_jd` is sorted ascending (see `_ensure_chronological`, which runs
        # before it's cached in `__init__`), so narrow down to a small candidate slice via
        # binary search before falling back to an exact boolean mask, instead of scanning
        # every row in the schedule on every call. The lower bound is pushed back by
        # `_max_action_duration` -- the longest action anywhere in the schedule -- since an
        # action starting before `start_time` can still overlap the window if it runs long
        # enough (e.g. a downlink), even though a plain `>= start_time` search on
        # `action_start` alone would miss it.
        #
        # The search itself runs against `self._start_time_jd` -- a plain float64 array
        # cached once in `__init__` -- rather than `np.searchsorted` on the `Time` column
        # directly: `Time` has no vectorized `searchsorted`, so searching the column itself
        # dispatches through an unvectorized fallback that reconstructs `Time` objects
        # element by element (see `__init__`'s comment for the cache). Only the two query
        # boundaries need an (O(1)) scale conversion to match the cached array's scale.
        scale = self._start_time_scale
        lo_bound_jd = getattr(start_time, scale).jd - self._max_action_duration_days
        hi_bound_jd = getattr(end_time, scale).jd

        lo = np.searchsorted(self._start_time_jd, lo_bound_jd, side="left")
        hi = np.searchsorted(self._start_time_jd, hi_bound_jd, side="left")

        candidates = self._schedule_table[lo:hi]
        if len(candidates) == 0:
            return candidates

        candidate_start = candidates["start_time"]
        candidate_end = candidate_start + candidates["duration"]
        mask = (candidate_start < end_time) & (candidate_end > start_time)

        return candidates[mask]

    @property
    def observe_rows(self) -> QTable:
        """
        QTable: The ``"observe"`` subset of :attr:`table`, in schedule order.

        Recomputed on every access -- a boolean-mask slice, cheap enough that it isn't
        worth caching alongside :meth:`get_healpix_coverage_index`'s pixel index, which
        its row indices (as returned by :meth:`get_observation_indices_of`) are
        relative to.
        """
        return self._schedule_table[self.actions == "observe"]

    def get_observed_regions(
        self,
        start_time: Time,
        end_time: Time,
    ) -> list[SkyRegion]:
        """
        Return the instrument footprints observed during a time interval.

        Schedule rows are included when their action interval overlaps the
        half-open query interval ``[start_time, end_time)``. Only rows whose
        action is ``"observe"`` contribute footprints.

        Parameters
        ----------
        start_time
            Beginning of the query interval, inclusive.
        end_time
            End of the query interval, exclusive.

        Returns
        -------
        list[regions.SkyRegion]
            One positioned and rotated footprint per selected observation.
        """
        rows = self.get_rows_between_times(start_time, end_time)

        actions = np.asarray(rows[self._ACTION_COLUMN]).astype(str)
        rows = rows[actions == "observe"]

        if len(rows) == 0:
            return []

        if not isinstance(self._instrument_fov, RectangleSkyRegion):
            raise TypeError(
                "`get_observed_regions` currently requires `instrument_fov` to be a RectangleSkyRegion."
            )

        template = self._instrument_fov

        return [
            RectangleSkyRegion(
                center=row["target_coord"],
                width=template.width,
                height=template.height,
                angle=row["roll"],
                visual=dict(template.visual),
                meta=dict(template.meta),
            )
            for row in rows
        ]

    def get_observed_region(
        self,
        start_time: Time,
        end_time: Time,
    ) -> SkyRegion | None:
        """
        Return the union of every instrument footprint observed during a time interval.

        Parameters
        ----------
        start_time
            Beginning of the query interval, inclusive.
        end_time
            End of the query interval, exclusive.

        Returns
        -------
        regions.SkyRegion or None
            The union (:meth:`regions.SkyRegion.union`) of every footprint
            returned by :meth:`get_observed_regions`, or `None` if no
            observation occurred during the interval.
        """
        observed_regions = self.get_observed_regions(start_time, end_time)

        if not observed_regions:
            return None

        return np.logical_or.reduce(observed_regions)

    def get_healpix_coverage_index(
        self,
        nside: int = 256,
        order: str = "nested",
        cache: bool = True,
        overwrite: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Lazily build the HEALPix coverage index as a CSR (compressed-sparse-row) structure.

        Rasterizes *every* ``"observe"`` row's rolled footprint in a single, vectorized
        :func:`~m4opt.fov.footprint_healpix` call -- not one call per row -- then groups
        the resulting ``(row, pixel)`` hits by pixel via one vectorized sort plus
        :func:`numpy.bincount`/:func:`numpy.cumsum`, into two flat arrays rather than a
        Python ``dict``: pixel id doubles directly as an array index into
        ``pixel_offsets``, so a lookup is a memory offset, not a hash-table probe, and a
        whole batch of pixel ids can be resolved in one fancy-index call instead of a
        per-query Python loop.

        `footprint_healpix` follows HEALPix's own convention of pixel-*center*
        membership, not full-pixel overlap: a row is registered under pixel ``p`` only
        if ``p``'s center falls inside that row's rolled footprint. A query point can
        therefore sit inside a footprint while its own pixel goes unregistered (a
        one-directional miss, never a false claim -- see :meth:`get_observation_indices_of`,
        which confirms every candidate this index returns with an exact geometric test,
        so nothing reached through this index is ever wrongly included, only
        occasionally left out near a footprint's edge). Raise ``nside`` to shrink how
        often that happens; there is no dilation margin built into the index itself.

        Parameters
        ----------
        nside
            HEALPix resolution parameter.
        order
            HEALPix pixel ordering scheme, ``"nested"`` or ``"ring"``.
        cache
            If `True` (the default), reuse a previously built index for this
            ``(nside, order)`` when available, and store the freshly built one for later
            reuse. If `False`, always rebuild and never store the result -- useful for a
            one-off query at a resolution not worth caching.
        overwrite
            If `True`, rebuild even if a cached index for this ``(nside, order)``
            already exists. Ignored if ``cache`` is `False`, since every call already
            rebuilds in that case.

        Returns
        -------
        pixel_offsets : numpy.ndarray
            ``int64`` array of shape ``(12 * nside**2 + 1,)``. Row indices whose
            footprint covers pixel ``p`` are
            ``sorted_rows[pixel_offsets[p]:pixel_offsets[p + 1]]`` -- an empty slice if
            pixel ``p`` is not covered by any observation.
        sorted_rows : numpy.ndarray
            ``int64`` array of row indices into :attr:`observe_rows`, grouped
            contiguously by pixel and ordered to match ``pixel_offsets``.
        """
        if cache and not overwrite:
            cached = self._HPX_MAP_CACHE.get((nside, order))
            if cached is not None:
                return cached

        observe_rows = self.observe_rows
        npix = ah.nside_to_npix(nside)

        if len(observe_rows) == 0:
            pixel_offsets = np.zeros(npix + 1, dtype=np.int64)
            sorted_rows = np.array([], dtype=np.int64)
        else:
            hpx = ah.HEALPix(
                nside=nside, order=order, frame=observe_rows["target_coord"].frame
            )
            pixel_arrays = footprint_healpix(
                hpx,
                self._instrument_fov,
                observe_rows["target_coord"],
                observe_rows["roll"],
            )

            # Flatten every (row, pixel) hit, then group by pixel with one vectorized
            # sort plus a bincount/cumsum offsets array, instead of a Python
            # dict.setdefault/append per hit -- the sort and bincount are each a single
            # C-level pass over every pixel hit in the whole schedule, rather than a
            # Python-level dict operation for each one.
            pixel_counts = np.fromiter(
                (len(pixels) for pixels in pixel_arrays),
                dtype=np.int64,
                count=len(pixel_arrays),
            )
            row_indices = np.repeat(
                np.arange(len(pixel_arrays), dtype=np.int64), pixel_counts
            )
            pixels = np.concatenate(pixel_arrays)

            sort_order = np.argsort(pixels, kind="stable")
            sorted_pixels = pixels[sort_order]
            sorted_rows = row_indices[sort_order]

            counts = np.bincount(sorted_pixels, minlength=npix)
            pixel_offsets = np.empty(npix + 1, dtype=np.int64)
            pixel_offsets[0] = 0
            np.cumsum(counts, out=pixel_offsets[1:])

        result = (pixel_offsets, sorted_rows)

        if cache:
            self._HPX_MAP_CACHE[(nside, order)] = result

        return result

    def get_observed_healpix_ids(
        self,
        start_time: Time,
        end_time: Time,
        nside: int = 128,
        order: str = "nested",
    ) -> np.ndarray:
        """
        Return the HEALPix pixel indices covered by observations in a time interval.

        Rasterizes the actual (rolled) instrument footprint at each observed
        pointing -- via :func:`m4opt.fov.footprint_healpix` -- rather than
        approximating it with a bounding-circle cone search.

        Parameters
        ----------
        start_time
            Beginning of the query interval, inclusive.
        end_time
            End of the query interval, exclusive.
        nside
            HEALPix resolution parameter.
        order
            HEALPix pixel ordering scheme, ``"nested"`` or ``"ring"``.

        Returns
        -------
        numpy.ndarray
            Sorted, deduplicated HEALPix pixel indices covered by any
            observation in the interval; empty if there were none.
        """
        # Extract the rows associated with this time range and then filter them to
        # only include the observations.
        subtable = self.get_rows_between_times(start_time, end_time)
        subtable = subtable[subtable[self._ACTION_COLUMN] == "observe"]

        if len(subtable) == 0:
            return np.array([], dtype=np.int64)

        # Instantiate a healpix object.
        hpx = ah.HEALPix(nside=nside, order=order, frame=subtable["target_coord"].frame)

        # Rasterize the actual (rolled) instrument footprint at each observed
        # pointing, rather than approximating it with a bounding-circle cone search.
        pixel_arrays = footprint_healpix(
            hpx, self._instrument_fov, subtable["target_coord"], subtable["roll"]
        )

        return np.unique(np.concatenate(pixel_arrays))

    def get_observation_indices_of(
        self,
        coord: SkyCoord,
        nside: int = 256,
        order: str = "nested",
        start_time: Time | None = None,
        end_time: Time | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Row indices of observations whose footprint covers each query position.

        The batched, index-returning counterpart to :meth:`get_observations_of`: each
        query position's own HEALPix pixel is looked up directly in the cached
        coverage index (see :meth:`get_healpix_coverage_index`) -- a plain,
        single-pixel broad-phase filter, not the final answer -- and just that
        (typically small) candidate set is confirmed with the same exact, vectorized
        :func:`m4opt.fov.contains` test :meth:`get_observations_of` uses, run once over
        every query position at once rather than in a Python loop.

        Because :meth:`get_healpix_coverage_index` registers a pointing under a pixel
        only if that pixel's *center* falls inside the rolled footprint, a query point
        can sit inside a footprint while its own pixel goes unregistered -- such a
        match is silently missed here, never falsely included (the exact `contains`
        test can only remove a candidate, never add one back once the pixel lookup
        already missed it). Raise ``nside`` to shrink how often that happens; there is
        no neighbour-pixel margin here to compensate for a coarse index.

        Parameters
        ----------
        coord
            Sky position(s) to test -- scalar or array, any shape (flattened
            internally).
        nside, order
            Resolution and pixel ordering of the coverage index to query -- see
            :meth:`get_healpix_coverage_index`.
        start_time, end_time
            Optional time window to restrict matches to. Must be given together. Each
            may be scalar (one shared window for every query position) or an array the
            same length as flattened ``coord`` (one window per query position, e.g. an
            explosion time and duration limit that differs per event -- the batched
            equivalent of calling this once per event with its own window). If both
            are `None` (the default), every ``"observe"`` row in the schedule is
            considered.

        Returns
        -------
        query_index : numpy.ndarray
            ``int64`` array, one entry per matched observation, giving which element
            of flattened ``coord`` it belongs to (values in ``[0, coord.size)`` --
            always ``0`` if ``coord`` is scalar). Chronological within each query
            position.
        row_index : numpy.ndarray
            ``int64`` array, the same length as ``query_index``, of row indices into
            :attr:`observe_rows` for the matched observations.

        Raises
        ------
        ValueError
            If exactly one of ``start_time``/``end_time`` is given, or either is an
            array not the same length as flattened ``coord``.
        """
        if (start_time is None) != (end_time is None):
            raise ValueError(
                "Parameters 'start_time' and 'end_time' must be given together."
            )

        pixel_offsets, sorted_rows = self.get_healpix_coverage_index(
            nside=nside, order=order
        )
        observe_rows = self.observe_rows

        empty = np.array([], dtype=np.int64)
        if len(observe_rows) == 0:
            return empty, empty

        coord_array = coord.reshape((-1,))
        n = len(coord_array)

        if start_time is not None and not start_time.isscalar:
            if len(start_time) != n or len(end_time) != n:
                raise ValueError(
                    "Array 'start_time'/'end_time' must be the same length as "
                    f"flattened 'coord' ({n}), got {len(start_time)} and "
                    f"{len(end_time)}."
                )

        hpx = ah.HEALPix(
            nside=nside, order=order, frame=observe_rows["target_coord"].frame
        )
        query_pixel = np.asarray(hpx.skycoord_to_healpix(coord_array), dtype=np.int64)

        # Each query's candidate range is resolved by fancy-indexing `pixel_offsets`
        # with the whole `query_pixel` array at once -- no per-query Python loop. What
        # remains is gathering each query's own (variable-length) slice of
        # `sorted_rows` and concatenating them in query order; that ragged gather is
        # itself vectorized via the standard repeat/arange/cumsum trick below.
        starts = pixel_offsets[query_pixel]
        ends = pixel_offsets[query_pixel + 1]
        counts = ends - starts
        total = int(counts.sum())

        if total == 0:
            return empty, empty

        query_index = np.repeat(np.arange(n, dtype=np.int64), counts)
        group_start_in_output = np.zeros(n, dtype=np.int64)
        np.cumsum(counts[:-1], out=group_start_in_output[1:])
        within_group_index = (
            np.arange(total, dtype=np.int64) - group_start_in_output[query_index]
        )
        row_index = sorted_rows[starts[query_index] + within_group_index]

        # Exact confirmation: the same geometric test `get_observations_of` uses, run
        # once over every candidate pair at once instead of once per query position.
        # This can only remove a candidate the pixel lookup shouldn't have offered, not
        # discard a genuine match -- see the docstring's note on this method's
        # one-directional (miss, never false-claim) error.
        candidate_rows = observe_rows[row_index]
        local_frame = SkyOffsetFrame(
            origin=candidate_rows["target_coord"], rotation=candidate_rows["roll"]
        )
        local_coord = coord_array[query_index].transform_to(local_frame)
        local_coord_as_icrs = SkyCoord(local_coord.lon, local_coord.lat, frame="icrs")
        contains_mask = np.asarray(
            contains(self._instrument_fov, local_coord_as_icrs), dtype=bool
        )

        query_index = query_index[contains_mask]
        row_index = row_index[contains_mask]
        candidate_rows = candidate_rows[contains_mask]

        if start_time is not None:
            matched_start = candidate_rows["start_time"]
            # Scalar window broadcasts against every candidate as-is; a per-query
            # window is indexed by `query_index` so each candidate is checked against
            # *its own* query's window, not a shared one.
            if start_time.isscalar:
                window_start, window_end = start_time, end_time
            else:
                window_start = start_time[query_index]
                window_end = end_time[query_index]
            time_mask = (matched_start >= window_start) & (matched_start < window_end)
            query_index = query_index[time_mask]
            row_index = row_index[time_mask]
            candidate_rows = candidate_rows[time_mask]

        # Chronological within each query position.
        sort_order = np.lexsort((candidate_rows["start_time"].jd, query_index))
        return query_index[sort_order], row_index[sort_order]


    def get_observations_of(
        self,
        coord: SkyCoord,
        start_time: Time | None = None,
        end_time: Time | None = None,
    ) -> QTable:
        """
        Return the ``"observe"`` rows whose footprint covers a sky position.

        Useful for checking whether -- and when -- the survey observed a given
        position, e.g. to evaluate detectability of a simulated transient.

        As a performance optimization, candidate rows are first narrowed down with a
        cheap angular-separation pre-filter using :attr:`bounding_radius`, then
        confirmed with a single, vectorized exact containment test
        (:func:`m4opt.fov.contains`): rather than rotating :attr:`instrument_fov` out
        to each candidate pointing and testing ``coord`` against each rotated copy in
        a loop (which rebuilds a WCS -- the dominant cost -- once per candidate row),
        ``coord`` is transformed into each candidate's own local pointing frame and
        tested against the single, unrotated ``instrument_fov`` all at once. This is
        the same geometric test, just inverted to build one WCS per call instead of
        one per candidate row.

        Parameters
        ----------
        coord
            Scalar sky position to test.
        start_time, end_time
            Optional time window to restrict the search to (see
            :meth:`get_rows_between_times`). Must be given together. If both are
            `None` (the default), every ``"observe"`` row in the schedule is
            considered.

        Returns
        -------
        QTable
            The subset of :attr:`table` whose observed footprint contains
            ``coord``, in chronological order. Empty if there is no such row.

        Raises
        ------
        ValueError
            If ``coord`` is not scalar, or exactly one of ``start_time``/``end_time``
            is given.
        """
        if not coord.isscalar:
            raise ValueError("Parameter 'coord' must be a scalar SkyCoord.")

        if (start_time is None) != (end_time is None):
            raise ValueError(
                "Parameters 'start_time' and 'end_time' must be given together."
            )

        rows = (
            self.get_rows_between_times(start_time, end_time)
            if start_time is not None
            else self._schedule_table
        )

        actions = np.asarray(rows[self._ACTION_COLUMN]).astype(str)
        rows = rows[actions == "observe"]

        if len(rows) == 0:
            return rows

        candidate_mask = coord.separation(rows["target_coord"]) <= self.bounding_radius
        rows = rows[candidate_mask]

        if len(rows) == 0:
            return rows

        # Transform `coord` into each candidate row's own pointing frame (an ordinary,
        # vectorized SkyCoord transform -- cheap), then reinterpret those per-row
        # offset angles as literal ICRS coordinates (the same trick `m4opt.fov`'s own
        # `footprint`/`skycoord_to_offset` uses, just inverted: there, a *local* FOV
        # vertex offset gets reinterpreted as sitting *at* the rotated pointing; here,
        # the *global* query point's local-frame offset gets reinterpreted as sitting
        # on `instrument_fov`'s own unrotated, RA=0/Dec=0-centered copy). `contains`
        # is not itself a coordinate-frame-aware test -- it treats both its region and
        # its target_coord as literal ICRS-like positions -- so this only works
        # because both sides of the comparison are consistently expressed that way.
        local_frame = SkyOffsetFrame(origin=rows["target_coord"], rotation=rows["roll"])
        local_coord = coord.transform_to(local_frame)
        local_coord_as_icrs = SkyCoord(local_coord.lon, local_coord.lat, frame="icrs")
        contains_mask = np.asarray(
            contains(self._instrument_fov, local_coord_as_icrs), dtype=bool
        )

        return rows[contains_mask]







    # ----------------------------------------- #
    # Cadence Calculations                      #
    # ----------------------------------------- #
    # These functions are concerned with computing various features of the overlap cadence
    # for analysis.
    def compute_visit_count(
        self,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> np.ndarray:
        """
        Compute the number of visits to each HEALPix pixel.

        Each ``"observe"`` action contributes one visit to every HEALPix pixel
        covered by the rolled instrument footprint.

        Parameters
        ----------
        start_time, end_time
            Optional time range over which to compute visit counts. If omitted,
            the full survey duration is used.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        numpy.ndarray
            Integer array of shape ``(12 * nside**2,)``. Entry ``i`` gives the
            number of observations whose footprint covered HEALPix pixel ``i``.
        """
        start_time, end_time = self._resolve_tstart_tend(start_time, end_time)

        rows = self.get_rows_between_times(start_time, end_time)
        rows = rows[np.asarray(rows[self._ACTION_COLUMN]).astype(str) == "observe"]

        hpx = ah.HEALPix(
            nside=nside,
            order=order,
            frame=self._schedule_table["target_coord"].frame,
        )

        if len(rows) == 0:
            return np.zeros(hpx.npix, dtype=np.int64)

        pixel_arrays = footprint_healpix(
            hpx,
            self._instrument_fov,
            rows["target_coord"],
            rows["roll"],
        )

        pixels = np.concatenate(pixel_arrays)

        return np.bincount(
            pixels,
            minlength=hpx.npix,
        ).astype(np.int64, copy=False)

    def compute_visit_times(
        self,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> tuple[u.Quantity, np.ndarray]:
        """
        Compute observation times for each HEALPix pixel.

        Returns
        -------
        visit_times : ~astropy.units.Quantity
            Flattened elapsed observation times, grouped by HEALPix pixel.
        offsets : numpy.ndarray
            Integer offsets of shape ``(npix + 1,)``. Observation times for
            pixel ``i`` are given by
            ``visit_times[offsets[i]:offsets[i + 1]]``.
        """
        start_time, end_time = self._resolve_tstart_tend(start_time, end_time)

        rows = self.get_rows_between_times(start_time, end_time)
        rows = rows[np.asarray(rows[self._ACTION_COLUMN]).astype(str) == "observe"]

        hpx = ah.HEALPix(
            nside=nside,
            order=order,
            frame=self._schedule_table["target_coord"].frame,
        )

        if len(rows) == 0:
            return (
                np.array([]) * u.day,
                np.zeros(hpx.npix + 1, dtype=np.int64),
            )

        pixel_arrays = footprint_healpix(
            hpx,
            self._instrument_fov,
            rows["target_coord"],
            rows["roll"],
        )

        pixels = np.concatenate(pixel_arrays)

        elapsed_times = (rows["start_time"] - start_time).to(u.day)

        times = np.repeat(
            elapsed_times,
            [len(pixel_ids) for pixel_ids in pixel_arrays],
        )

        # Group entries by pixel.
        order_idx = np.argsort(pixels, kind="stable")
        pixels = pixels[order_idx]
        times = times[order_idx]

        counts = np.bincount(pixels, minlength=hpx.npix)

        offsets = np.empty(hpx.npix + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        return times, offsets

    def compute_cadence_time_differences(
        self,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> tuple[u.Quantity, np.ndarray]:
        """
        Compute all unique pairwise observation-time separations for each HEALPix pixel.

        For a pixel observed at times ``t_0, ..., t_N``, the cadence time
        differences are all positive pairwise separations

        ``t_j - t_i`` for ``j > i``.

        Parameters
        ----------
        start_time, end_time
            Optional time range over which to compute cadence separations.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        time_differences : ~astropy.units.Quantity
            Flattened pairwise time differences, grouped by HEALPix pixel.
        offsets : numpy.ndarray
            Integer offsets of shape ``(npix + 1,)``. Pairwise differences
            for pixel ``i`` are given by

            ``time_differences[offsets[i]:offsets[i + 1]]``.

            Pixels with fewer than two observations have no entries.
        """
        visit_times, visit_offsets = self.compute_visit_times(
            start_time=start_time,
            end_time=end_time,
            nside=nside,
            order=order,
        )

        npix = len(visit_offsets) - 1
        unit = visit_times.unit

        # Number of visits to each pixel.
        visit_counts = np.diff(visit_offsets)

        # A pixel with N visits has N(N - 1) / 2 unique pairs.
        pair_counts = visit_counts * (visit_counts - 1) // 2

        pair_offsets = np.empty(npix + 1, dtype=np.int64)
        pair_offsets[0] = 0
        np.cumsum(pair_counts, out=pair_offsets[1:])

        time_differences = np.empty(pair_offsets[-1], dtype=float)

        for pixel in np.flatnonzero(pair_counts):
            times = visit_times[
                visit_offsets[pixel] : visit_offsets[pixel + 1]
            ].to_value(unit)

            # The visit times are already chronological, so taking the upper
            # triangle gives every unique positive pairwise separation.
            i, j = np.triu_indices(len(times), k=1)

            time_differences[pair_offsets[pixel] : pair_offsets[pixel + 1]] = (
                times[j] - times[i]
            )

        return time_differences * unit, pair_offsets

    def compute_cadence_statistics(
        self,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> dict[str, u.Quantity]:
        """
        Compute per-pixel statistics of all pairwise temporal baselines.

        Cadence is represented by every unique pairwise separation between
        observations of the same HEALPix pixel, rather than only separations
        between consecutive observations.

        Parameters
        ----------
        start_time, end_time
            Optional time range over which to compute cadence statistics.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        dict[str, ~astropy.units.Quantity]
            Full-sky HEALPix maps containing the ``mean``, ``median``,
            ``min``, ``max``, and ``std`` of the pairwise temporal baselines
            for each pixel.

            Pixels observed fewer than twice are assigned ``NaN``.
        """
        time_differences, offsets = self.compute_cadence_time_differences(
            start_time=start_time,
            end_time=end_time,
            nside=nside,
            order=order,
        )

        npix = len(offsets) - 1
        unit = time_differences.unit
        values = time_differences.to_value(unit)

        mean = np.full(npix, np.nan)
        median = np.full(npix, np.nan)
        minimum = np.full(npix, np.nan)
        maximum = np.full(npix, np.nan)
        std = np.full(npix, np.nan)

        for pixel in np.flatnonzero(np.diff(offsets)):
            baselines = values[offsets[pixel] : offsets[pixel + 1]]

            mean[pixel] = np.mean(baselines)
            median[pixel] = np.median(baselines)
            minimum[pixel] = np.min(baselines)
            maximum[pixel] = np.max(baselines)
            std[pixel] = np.std(baselines)

        return {
            "mean": mean * unit,
            "median": median * unit,
            "min": minimum * unit,
            "max": maximum * unit,
            "std": std * unit,
        }

    @staticmethod
    def _validate_timescale(timescale: u.Quantity) -> None:
        """
        Check that ``timescale`` is a single, positive `~astropy.units.Quantity`.

        Raises
        ------
        ValueError
            If ``timescale`` is not scalar, or is not strictly positive.
        """
        if not u.Quantity(timescale).isscalar:
            raise ValueError("Timescale must be scalar.")

        if timescale <= 0 * timescale.unit:
            raise ValueError("Timescale must be positive.")

    @staticmethod
    def _validate_pair_window_factors(
        minimum_factor: float, maximum_factor: float
    ) -> None:
        """
        Check that a pair-separation window's bounding factors are well-formed.

        Raises
        ------
        ValueError
            If ``minimum_factor`` is negative, or ``maximum_factor`` does not
            exceed ``minimum_factor``.
        """
        if minimum_factor < 0:
            raise ValueError("Parameter 'minimum_factor' must be non-negative.")

        if maximum_factor <= minimum_factor:
            raise ValueError(
                "Parameter 'maximum_factor' must be greater than 'minimum_factor'."
            )

    @staticmethod
    def _pixel_pair_count(visit_times: np.ndarray, lo: float, hi: float) -> int:
        """
        Count one pixel's own visit pairs with separation in ``[lo, hi]``.

        Uses two vectorized :func:`numpy.searchsorted` calls (one query per
        visit, all evaluated at once) rather than materializing every pairwise
        difference the way :meth:`compute_cadence_time_differences` does. That
        distinction matters once a pixel has been revisited across many
        repeated full-sky survey passes: the number of *all* pairs grows with
        the square of the number of passes even though only nearby ones are
        ever relevant to a given timescale, so this keeps per-pixel cost at
        O(n log n) regardless of how many times the survey has covered the
        whole sky.

        Parameters
        ----------
        visit_times
            Sorted elapsed observation times for one pixel.
        lo, hi
            Bounds of the qualifying pair-separation window, in the same units
            as ``visit_times``.

        Returns
        -------
        int
            Number of qualifying pairs for this pixel.
        """
        if len(visit_times) < 2:
            return 0

        # Earlier visits satisfying lo <= later - earlier <= hi lie in
        # [later - hi, later - lo]. `minimum_factor` (and hence `lo`) may be
        # exactly 0, in which case `visit_times - lo` collides with
        # `visit_times` itself, so `hi_idx` is explicitly capped at each
        # visit's own index to exclude pairing a visit with itself.
        lo_idx = np.searchsorted(visit_times, visit_times - hi, side="left")
        hi_idx = np.searchsorted(visit_times, visit_times - lo, side="right")
        hi_idx = np.minimum(hi_idx, np.arange(len(visit_times)))

        return int(np.sum(np.maximum(hi_idx - lo_idx, 0)))

    @staticmethod
    def _validate_visibility_factor(visibility_factor: float) -> None:
        """
        Check that a control-time visibility factor is well-formed.

        Raises
        ------
        ValueError
            If ``visibility_factor`` is not strictly positive.
        """
        if visibility_factor <= 0:
            raise ValueError("Parameter 'visibility_factor' must be positive.")

    @staticmethod
    def _pixel_control_time(
        visit_times: np.ndarray,
        minimum_separation: float,
        maximum_separation: float,
        visibility_window: float,
        maximum_time: float,
    ) -> float:
        """
        Total control-time duration for one pixel's sorted visit times.

        For each visit, the *latest* earlier visit satisfying
        ``minimum_separation <= later - earlier <= maximum_separation``
        defines an interval of transient start times,
        ``[later - visibility_window, earlier]`` (clipped to
        ``[0, maximum_time]``), for which that pair would provide useful
        temporal sampling -- the latest qualifying earlier visit is used
        because it produces the largest such interval, which therefore
        contains the interval any earlier qualifying visit would have
        produced. Overlapping intervals across all visits are merged before
        summing their total duration, so no candidate start time is double
        counted.

        Both the per-visit interval construction (via array-form
        :func:`numpy.searchsorted`, one query per visit, evaluated all at
        once) and the interval merge (via a running-max trick, since the
        intervals already come out in non-decreasing start-time order) are
        fully vectorized -- no Python-level loop over visits -- so this stays
        O(n log n) per pixel regardless of how many times a pixel has been
        revisited (e.g. across repeated full-sky survey passes), unlike the
        sequential per-visit Python loop this replaces.

        Parameters
        ----------
        visit_times
            Sorted elapsed observation times for one pixel.
        minimum_separation, maximum_separation
            Bounds of the qualifying pair-separation window, in the same
            units as ``visit_times``.
        visibility_window
            Maximum time after transient onset over which an observation is
            considered useful, in the same units as ``visit_times``.
        maximum_time
            Upper boundary for allowed transient start times, in the same
            units as ``visit_times``.

        Returns
        -------
        float
            Total control time, in the same units as ``visit_times``.
        """
        n = len(visit_times)

        if n < 2:
            return 0.0

        t = visit_times

        lo_idx = np.searchsorted(t, t - maximum_separation, side="left")
        hi_idx = np.searchsorted(t, t - minimum_separation, side="right")
        hi_idx = np.minimum(hi_idx, np.arange(n))

        valid = lo_idx < hi_idx

        if not np.any(valid):
            return 0.0

        earlier_time = t[np.clip(hi_idx - 1, 0, n - 1)]

        interval_start = np.maximum(t - visibility_window, 0.0)
        interval_end = np.minimum(earlier_time, maximum_time)

        keep = valid & (interval_end > interval_start)

        if not np.any(keep):
            return 0.0

        starts = interval_start[keep]
        ends = interval_end[keep]

        # `starts` is already non-decreasing -- as the visit index (and hence
        # `t`) increases, `t - visibility_window` (clipped at 0) can only
        # increase or stay flat -- so the intervals are already in the
        # chronological order a merge pass needs, without an explicit sort.
        running_max_end = np.maximum.accumulate(ends)
        prev_max_end = np.concatenate(([-np.inf], running_max_end[:-1]))
        new_group = starts > prev_max_end

        group_start_indices = np.flatnonzero(new_group)
        group_starts = starts[group_start_indices]
        group_end_indices = np.concatenate((group_start_indices[1:], [len(starts)])) - 1
        group_ends = running_max_end[group_end_indices]

        return float(np.sum(group_ends - group_starts))

    def compute_pair_counts(
        self,
        timescale: u.Quantity,
        minimum_factor: float = 0.5,
        maximum_factor: float = 2.0,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> tuple[np.ndarray, u.Quantity]:
        """
        Compute per-pixel counts of visit pairs bracketing a transient timescale.

        For each HEALPix pixel, counts the number of unique observation pairs
        separated by

        ``minimum_factor * timescale <= dt <= maximum_factor * timescale``,

        i.e. pairs of visits able to catch a timescale-``timescale`` transient
        rising and/or fading. This is the same idea as LSST/Rubin's pair-count
        cadence metrics (e.g. ``rubin_sim.maf.metrics.PairMetric``, used to assess
        sensitivity to kilonova- and fast-transient-like timescales): a cheap
        proxy for "can this survey's cadence constrain a timescale-``T``
        transient here?" that avoids the interval-union bookkeeping (and
        per-pixel, per-timescale cost) of an exact control-time calculation.

        Built on :meth:`compute_visit_times` and :meth:`_pixel_pair_count`
        rather than :meth:`compute_cadence_time_differences`, so cost scales as
        O(n log n) in the number of visits to each pixel rather than O(n^2) --
        important for a survey that covers the whole sky many times over,
        where a given pixel's total visit count (and hence its number of
        *all* pairs) grows with the number of repeated full-sky passes.

        Parameters
        ----------
        timescale
            Characteristic transient timescale.
        minimum_factor, maximum_factor
            Bounds of the qualifying pair-separation window, relative to
            ``timescale``.
        start_time, end_time
            Optional survey interval to restrict the calculation to.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        pair_counts : numpy.ndarray
            Full-sky HEALPix map of qualifying pair counts, shape
            ``(12 * nside**2,)``.
        sensitive_area : ~astropy.units.Quantity
            Total solid angle of pixels with at least one qualifying pair --
            the area of sky with any cadence sensitivity to this timescale.
        """
        self._validate_timescale(timescale)
        self._validate_pair_window_factors(minimum_factor, maximum_factor)

        visit_times, offsets = self.compute_visit_times(
            start_time=start_time,
            end_time=end_time,
            nside=nside,
            order=order,
        )

        unit = visit_times.unit
        times = visit_times.to_value(unit)
        npix = len(offsets) - 1
        visit_counts = np.diff(offsets)

        lo = (minimum_factor * timescale).to_value(unit)
        hi = (maximum_factor * timescale).to_value(unit)

        pair_counts = np.zeros(npix, dtype=np.int64)

        for pixel in np.flatnonzero(visit_counts >= 2):
            pixel_times = times[offsets[pixel] : offsets[pixel + 1]]
            pair_counts[pixel] = self._pixel_pair_count(pixel_times, lo, hi)

        pixel_area = (4 * np.pi / npix) * u.sr
        sensitive_area = np.count_nonzero(pair_counts) * pixel_area

        return pair_counts, sensitive_area

    def compute_pair_count_curve(
        self,
        timescales: u.Quantity,
        minimum_factor: float = 0.5,
        maximum_factor: float = 2.0,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> u.Quantity:
        """
        Compute sensitive sky area over a sequence of transient timescales.

        The fast, curve-friendly counterpart to :meth:`compute_pair_counts`:
        the (expensive, HEALPix-footprint-rasterizing) per-pixel visit times
        are computed once and reused for every timescale, rather than redoing
        that rasterization inside a loop the way an exact interval-union
        control-time curve would.

        Each timescale still re-scans every observed pixel (via
        :meth:`_pixel_pair_count`), so cost scales as
        O(n_timescales * n_observed_pixels * log(visits per pixel)) -- no
        repeated rasterization, and no O(n^2) blowup with the number of
        repeated full-sky passes, but also no way to get a timescale "for
        free" the way a precomputed, timescale-independent statistic could.
        If this loop itself becomes the bottleneck for very fine timescale
        grids on a very deep multi-pass survey, the per-pixel loop can be
        vectorized away too (encoding pixel id and time into one sortable key
        so the whole sky is searched in a single vectorized pass) -- not done
        here since it adds real complexity that isn't needed until it is.

        Parameters
        ----------
        timescales
            Sequence of characteristic transient timescales.
        minimum_factor, maximum_factor
            Bounds of the qualifying pair-separation window, relative to each
            timescale.
        start_time, end_time
            Optional survey interval to restrict the calculation to.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        ~astropy.units.Quantity
            Sensitive sky area (see :meth:`compute_pair_counts`) for each input
            timescale.
        """
        timescales = u.Quantity(timescales, copy=False, ndmin=1)
        self._validate_pair_window_factors(minimum_factor, maximum_factor)

        visit_times, offsets = self.compute_visit_times(
            start_time=start_time,
            end_time=end_time,
            nside=nside,
            order=order,
        )

        unit = visit_times.unit
        times = visit_times.to_value(unit)
        npix = len(offsets) - 1
        visit_counts = np.diff(offsets)
        observed_pixels = np.flatnonzero(visit_counts >= 2)
        pixel_area = (4 * np.pi / npix) * u.sr

        sensitive_pixel_counts = []

        with logging_redirect_tqdm(loggers=[logger]):
            for timescale in tqdm(
                timescales, desc="Computing pair-count curve", unit="timescale"
            ):
                self._validate_timescale(timescale)

                lo = (minimum_factor * timescale).to_value(unit)
                hi = (maximum_factor * timescale).to_value(unit)

                n_sensitive = 0
                for pixel in observed_pixels:
                    pixel_times = times[offsets[pixel] : offsets[pixel + 1]]
                    if self._pixel_pair_count(pixel_times, lo, hi) > 0:
                        n_sensitive += 1

                logger.debug(
                    "Timescale %.3g %s: %d/%d observed pixels sensitive.",
                    timescale.to_value(unit),
                    unit,
                    n_sensitive,
                    len(observed_pixels),
                )
                sensitive_pixel_counts.append(n_sensitive)

        return u.Quantity(sensitive_pixel_counts) * pixel_area

    def compute_control_time(
        self,
        timescale: u.Quantity,
        minimum_factor: float = 0.5,
        maximum_factor: float = 2.0,
        visibility_factor: float = 3.0,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> tuple[u.Quantity, u.Quantity]:
        """
        Compute transient control time as a function of sky position.

        For each HEALPix pixel, the control time is the total *duration* over
        which a transient with characteristic timescale ``timescale`` could
        begin and still receive useful temporal sampling from the survey: at
        least one pair of observations separated by

        ``minimum_factor * timescale <= dt <= maximum_factor * timescale``,

        with the later observation occurring within
        ``visibility_factor * timescale`` of transient onset. Overlapping
        intervals of allowed start times are merged before summing, so no
        candidate start time is counted twice.

        This is the statistic to reach for when you need to know not just
        whether a pixel can ever catch a timescale-``T`` transient (see
        :meth:`compute_pair_counts`, which only answers that existence
        question), but for how much of the survey it remains able to --
        important once a pixel is revisited many times (e.g. across repeated
        full-sky passes), since a pixel hit by one lucky pair and a pixel with
        continuous cadence support look identical under an existence-only
        statistic but very different here.

        Built on :meth:`compute_visit_times` and :meth:`_pixel_control_time`,
        both the per-visit interval construction and the interval merge are
        fully vectorized (no Python-level loop over visits), so cost stays
        O(n log n) per pixel rather than the O(n) sequential Python loop the
        original implementation of this statistic used.

        Parameters
        ----------
        timescale
            Characteristic transient timescale.
        minimum_factor
            Minimum useful observation separation relative to ``timescale``.
        maximum_factor
            Maximum useful observation separation relative to ``timescale``.
        visibility_factor
            Duration over which the transient is assumed useful for temporal
            characterization, relative to ``timescale``.
        start_time, end_time
            Optional survey interval over which to calculate control time.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        control_time : ~astropy.units.Quantity
            Full-sky HEALPix map of control times with shape
            ``(12 * nside**2,)``.
        exposure : ~astropy.units.Quantity
            Survey-integrated area-time exposure,

            ``sum(control_time * pixel_area)``,

            with dimensions of solid angle times time.
        """
        self._validate_timescale(timescale)
        self._validate_pair_window_factors(minimum_factor, maximum_factor)
        self._validate_visibility_factor(visibility_factor)

        start_time, end_time = self._resolve_tstart_tend(start_time, end_time)

        visit_times, offsets = self.compute_visit_times(
            start_time=start_time,
            end_time=end_time,
            nside=nside,
            order=order,
        )

        unit = visit_times.unit
        times = visit_times.to_value(unit)
        npix = len(offsets) - 1
        visit_counts = np.diff(offsets)

        minimum_separation = (minimum_factor * timescale).to_value(unit)
        maximum_separation = (maximum_factor * timescale).to_value(unit)
        visibility_window = (visibility_factor * timescale).to_value(unit)
        maximum_time = (end_time - start_time).to_value(unit)

        control_time = np.zeros(npix, dtype=float)

        for pixel in np.flatnonzero(visit_counts >= 2):
            pixel_times = times[offsets[pixel] : offsets[pixel + 1]]
            control_time[pixel] = self._pixel_control_time(
                pixel_times,
                minimum_separation,
                maximum_separation,
                visibility_window,
                maximum_time,
            )

        control_time = control_time * unit

        pixel_area = (4 * np.pi / npix) * u.sr
        exposure = np.sum(control_time) * pixel_area

        return control_time, exposure

    def compute_control_time_curve(
        self,
        timescales: u.Quantity,
        minimum_factor: float = 0.5,
        maximum_factor: float = 2.0,
        visibility_factor: float = 3.0,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> u.Quantity:
        """
        Compute survey area-time exposure over a sequence of transient timescales.

        The curve-friendly counterpart to :meth:`compute_control_time`:
        per-pixel visit times (the expensive, HEALPix-footprint-rasterizing
        part) are computed once and reused for every timescale, rather than
        redone inside the loop the way the original implementation of this
        curve did. Each timescale still re-scans every observed pixel to
        rebuild and re-merge its intervals, since the merged intervals
        themselves depend on the timescale.

        Parameters
        ----------
        timescales
            Sequence of characteristic transient timescales.
        minimum_factor, maximum_factor
            Bounds of the qualifying pair-separation window, relative to each
            timescale.
        visibility_factor
            Duration over which the transient is assumed useful for temporal
            characterization, relative to each timescale.
        start_time, end_time
            Optional survey interval over which to calculate control time.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        ~astropy.units.Quantity
            Area-time exposure for each input timescale.
        """
        timescales = u.Quantity(timescales, copy=False, ndmin=1)
        self._validate_pair_window_factors(minimum_factor, maximum_factor)
        self._validate_visibility_factor(visibility_factor)

        start_time, end_time = self._resolve_tstart_tend(start_time, end_time)

        visit_times, offsets = self.compute_visit_times(
            start_time=start_time,
            end_time=end_time,
            nside=nside,
            order=order,
        )

        unit = visit_times.unit
        times = visit_times.to_value(unit)
        npix = len(offsets) - 1
        visit_counts = np.diff(offsets)
        observed_pixels = np.flatnonzero(visit_counts >= 2)
        maximum_time = (end_time - start_time).to_value(unit)
        pixel_area = (4 * np.pi / npix) * u.sr

        exposures = []

        # A single progress bar over timescales, not one nested per-pixel bar per
        # timescale: `observed_pixels` is re-scanned in full for every timescale (see the
        # docstring above), so a fresh inner `tqdm` on each iteration would be re-created
        # thousands of times over a fine timescale grid, adding real overhead for a bar
        # that starts over each time rather than tracking overall progress.
        with logging_redirect_tqdm(loggers=[logger]):
            progress = tqdm(
                timescales, desc="Computing control-time curve", unit="timescale"
            )
            for timescale in progress:
                self._validate_timescale(timescale)

                minimum_separation = (minimum_factor * timescale).to_value(unit)
                maximum_separation = (maximum_factor * timescale).to_value(unit)
                visibility_window = (visibility_factor * timescale).to_value(unit)

                total_control_time = 0.0

                for pixel in observed_pixels:
                    pixel_times = times[offsets[pixel] : offsets[pixel + 1]]
                    total_control_time += self._pixel_control_time(
                        pixel_times,
                        minimum_separation,
                        maximum_separation,
                        visibility_window,
                        maximum_time,
                    )

                logger.debug(
                    "Timescale %.3g %s: total control time %.3g %s over %d observed pixels.",
                    timescale.to_value(unit),
                    unit,
                    total_control_time,
                    unit,
                    len(observed_pixels),
                )
                exposures.append(total_control_time)
                progress.set_postfix(exposure=f"{total_control_time:.3g} {unit}")

        return u.Quantity(exposures, unit=unit) * pixel_area

    def compute_visit_count_histogram(
        self,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the distribution of HEALPix pixels by visit count.

        Parameters
        ----------
        start_time, end_time
            Optional time range over which to compute visit counts.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        visit_counts : numpy.ndarray
            Possible numbers of visits: ``0, 1, ..., N``.
        pixel_counts : numpy.ndarray
            Number of HEALPix pixels receiving each corresponding number of visits.

        Notes
        -----
        The zero-visit bin includes every HEALPix pixel on the sky that was not
        observed during the requested interval.
        """
        visits_per_pixel = self.compute_visit_count(
            start_time=start_time,
            end_time=end_time,
            nside=nside,
            order=order,
        )

        pixel_counts = np.bincount(visits_per_pixel)
        visit_counts = np.arange(pixel_counts.size, dtype=np.int64)

        return visit_counts, pixel_counts

    def compute_visit_count_cdf(
        self,
        start_time: Time | None = None,
        end_time: Time | None = None,
        nside: int = 128,
        order: str = "nested",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the complementary cumulative visit-count distribution.

        The returned fraction gives the fraction of *observed* HEALPix pixels
        receiving at least a given number of visits.

        Parameters
        ----------
        start_time, end_time
            Optional time range over which to compute visit counts.
        nside
            HEALPix resolution parameter.
        order
            HEALPix ordering scheme, either ``"nested"`` or ``"ring"``.

        Returns
        -------
        visit_counts : numpy.ndarray
            Visit-count thresholds ``1, 2, ..., N``.
        fractions : numpy.ndarray
            Fraction of observed HEALPix pixels receiving at least the
            corresponding number of visits.
        """
        visit_counts, pixel_counts = self.compute_visit_count_histogram(
            start_time=start_time,
            end_time=end_time,
            nside=nside,
            order=order,
        )

        # Exclude pixels that were never observed. For cadence analysis, including
        # the rest of the full sky in the denominator usually obscures the useful
        # distribution over the actual survey footprint.
        observed_pixel_counts = pixel_counts[1:]
        visit_counts = visit_counts[1:]

        if observed_pixel_counts.size == 0 or observed_pixel_counts.sum() == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=float),
            )

        # Reverse cumulative sum:
        #
        #   fractions[k] = P(number of visits >= visit_counts[k])
        #
        cumulative = np.cumsum(observed_pixel_counts[::-1])[::-1]

        return visit_counts, cumulative / cumulative[0]

    # ----------------------------------------- #
    # IO Methods                                #
    # ----------------------------------------- #
    def _sanitized_table(self) -> QTable:
        """
        Return a copy of :attr:`table` with every mixin column rebuilt via :func:`_sanitize_masked_value`.

        Used by :meth:`to_disk` so that a written file's masked columns always use
        astropy's officially ECSV-round-trippable
        :class:`~astropy.utils.masked.MaskedANDArray`, regardless of what ndarray
        subclass they may have picked up upstream (e.g. through repeated
        :func:`~astropy.table.vstack` while a schedule is being assembled).
        """
        table = self._schedule_table.copy()

        for column_name, spec in self._SCHEMA.items():
            if column_name not in table.colnames:
                continue

            col = table[column_name]

            if spec.column_type is EarthLocation:
                table[column_name] = EarthLocation.from_geocentric(
                    _sanitize_masked_value(col.x.value) * col.x.unit,
                    _sanitize_masked_value(col.y.value) * col.y.unit,
                    _sanitize_masked_value(col.z.value) * col.z.unit,
                )
            elif spec.column_type is SkyCoord:
                table[column_name] = SkyCoord(
                    _sanitize_masked_value(col.ra.value) * col.ra.unit,
                    _sanitize_masked_value(col.dec.value) * col.dec.unit,
                    frame=col.frame.name,
                )
            elif spec.unit is not None:
                table[column_name] = _sanitize_masked_value(col.value) * col.unit

        return table

    def to_disk(
        self,
        path: str | Path,
        fov_path: str | Path | None = None,
        table_format: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Write the schedule table -- and, optionally, the instrument FOV -- to disk.

        See :meth:`_sanitized_table` for why this doesn't just call
        ``self.table.write(...)`` directly: without it, a file can fail to read
        back with ``ValueError: unsupported class for construct ...`` if a masked
        column's value picked up an ndarray subclass along the way (e.g. through
        repeated :func:`~astropy.table.vstack`) that isn't one astropy's ECSV
        reader knows how to reconstruct.

        :meth:`from_disk` always requires a companion ``fov_path`` to reconstruct
        :attr:`fov`, so pass ``fov_path`` here too if you intend to round-trip via
        :meth:`from_disk` -- otherwise :attr:`fov` is not persisted at all.

        Parameters
        ----------
        path
            Destination path for the schedule table.
        fov_path
            Destination path for :attr:`fov`. If `None` (the default), the FOV is
            not written.
        table_format
            Passed through to :meth:`~astropy.table.QTable.write`; if `None`,
            the format is inferred from ``path``'s suffix.
        overwrite
            Whether to overwrite an existing file at ``path`` and ``fov_path``.
        """
        self._sanitized_table().write(
            Path(path), format=table_format, overwrite=overwrite
        )

        if fov_path is not None:
            Regions([self._instrument_fov]).write(str(fov_path), overwrite=overwrite)

    @classmethod
    def from_disk(
        cls,
        path: str | Path,
        fov_path: str | Path,
        table_format: str | None = None,
        **kwargs,
    ) -> "SurveySchedule":
        """
        Read a schedule table and its companion instrument FOV back from disk.

        The inverse of :meth:`to_disk` (when it was called with a ``fov_path``).

        Parameters
        ----------
        path
            Path to the schedule table, as written by :meth:`to_disk`.
        fov_path
            Path to the instrument FOV region file, as written by :meth:`to_disk`.
        table_format
            Passed through to :meth:`~astropy.table.QTable.read`; if `None`, the
            format is inferred from ``path``'s suffix.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        SurveySchedule

        Raises
        ------
        FileNotFoundError
            If ``path`` or ``fov_path`` does not exist.
        ScheduleValidationError
            If the table read from ``path`` fails schema validation.
        """
        # Ensure that the path exists before proceeding.
        path = Path(path)
        fov_path = Path(fov_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not fov_path.exists():
            raise FileNotFoundError(f"File not found: {fov_path}")

        # Read the Qtable ecsv file from disk.
        survey_table = QTable.read(path, format=table_format)

        # read the FOV file.
        (fov_region,) = Regions.read(fov_path)

        return cls(survey_table, fov_region, **kwargs)
