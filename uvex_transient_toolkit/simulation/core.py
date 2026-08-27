"""
Monte Carlo sampling of transient populations against a survey schedule.

`SurveySimulator` does exactly one job: given a collection of transient types, a
`~uvex_transient_toolkit.surveys.base.SurveySchedule`, and a root seed, sample a Monte
Carlo realization of every registered transient type into an `EventCatalog`
(:meth:`SurveySimulator.generate_events`). Real per-event synthetic photometry lives
one level down, on `~uvex_transient_toolkit.simulation.event.Event`
(:meth:`~uvex_transient_toolkit.simulation.event.Event.simulate_photometry`) -- it's too
expensive to run over a whole freshly-sampled population, which is typically dominated
by events far too faint to ever matter. Two progressively more expensive screening
steps narrow the catalog down first: `filter_by_limiting_magnitude` never touches the
schedule at all (only "could this event, at its brightest, ever clear a fixed magnitude
limit" over a shared per-type phase grid); `filter_by_snr` does consult the schedule --
"is this event, at any observation the schedule actually made of it, ever detected
above a given SNR" -- batched across many events at once the same way
`Event.simulate_photometry` batches across one event's own observations.
"""

from functools import partial
from typing import Union

import astropy_healpix as ah
import numpy as np
from astropy import units as u
from astropy.table import vstack
from astropy.time import Time
from m4opt.missions import Mission
from m4opt.synphot import observing
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from uvex_transient_toolkit.dust import dust_map, log_attenuation
from uvex_transient_toolkit.utils import get_seed_sequence, logger

from ..surveys.base import SurveySchedule
from ..transients.base import ExtragalacticTransient
from .event_catalog import EventCatalog

_SeedType = Union[np.random.SeedSequence, int, None]


def _sample_parameters_from_seeds(sed, seeds) -> dict:
    """
    Regenerate each event's own physical SED parameters from its stored `parameter_seed`.

    Draws one independent `numpy.random.Generator` per row, seeded from that row's own
    `parameter_seed` -- exactly matching
    `~uvex_transient_toolkit.simulation.event.Event.sample_parameters` row for row (same
    `sed.sample_parameters(size=1, rng=np.random.default_rng(int(seed)))` call). Unlike a
    single shared per-type stream, this guarantees the parameters used for screening here
    are the *same* ones a kept event's own `Event.mag`/`Event.simulate_photometry` will
    later reconstruct -- there is no independent, unbatched fallback for `seeds` to fall
    back on.

    Parameters
    ----------
    sed : ~uvex_transient_toolkit.models.core._base.SpectralModel
        The transient type's SED model to sample parameters from.
    seeds : array-like of int
        Each event's own `parameter_seed`, in catalog row order.

    Returns
    -------
    dict
        ``{name: samples}``, each with shape ``(len(seeds),)``, in the same row order as
        `seeds`.
    """
    if len(seeds) == 0:
        return {name: np.array([]) for name in sed.sample_parameters(size=0)}
    per_event = [
        sed.sample_parameters(size=1, rng=np.random.default_rng(int(seed)))
        for seed in seeds
    ]
    return {
        name: np.concatenate([sample[name] for sample in per_event])
        for name in per_event[0]
    }


class SurveySimulator:
    """Samples transient populations against a survey schedule."""

    def __init__(
        self,
        survey_schedule: SurveySchedule,
        transients: dict[str, ExtragalacticTransient] | None = None,
        simulation_seed: int | None = None,
    ):
        # Ensure that the survey schedule is valid.
        if not isinstance(survey_schedule, SurveySchedule):
            raise TypeError(
                f"'survey_schedule' must be an instance of SurveySchedule, got {type(survey_schedule)} instead."
            )

        self._survey_schedule = survey_schedule
        self._simulation_seed = simulation_seed

        # If we are given any transients to start with, we'll add them to the dictionary. Otherwise
        # we'll just pass through.
        self._transients = {}

        if transients is not None:
            for _transient_type_name, _transient_type in transients.items():
                if _transient_type_name in self._transients:
                    raise ValueError(
                        f"Transient type '{_transient_type_name}' is already used."
                    )
                if not isinstance(_transient_type, ExtragalacticTransient):
                    raise TypeError(
                        f"Transient type '{_transient_type_name}' must be an instance of "
                        f"ExtragalacticTransient, got {type(_transient_type)} instead."
                    )

                self._transients[_transient_type_name] = _transient_type

    # ---------------------------------------------- #
    # Properties and Accessors                       #
    # ---------------------------------------------- #
    @property
    def survey_schedule(self) -> SurveySchedule:
        return self._survey_schedule

    @property
    def transient_collection(self) -> dict[str, ExtragalacticTransient]:
        return self._transients

    @property
    def simulation_seed(self) -> _SeedType:
        return self._simulation_seed

    # -------------------------------------------------- #
    # Event Generation                                   #
    # -------------------------------------------------- #
    def _resolve_time_bins(self, time_bins: Time | int) -> Time:
        """Resolve `generate_events`'s `time_bins` argument down to a concrete array of edges."""
        if isinstance(time_bins, Time):
            if time_bins.isscalar or time_bins.size < 2:
                raise ValueError(
                    "`time_bins`, given as a Time array, must contain at least 2 edges."
                )
            return time_bins

        if isinstance(time_bins, bool) or not isinstance(time_bins, (int, np.integer)):
            raise TypeError(
                f"`time_bins` must be an astropy Time array of bin edges or a positive int, got {type(time_bins)!r}."
            )

        if time_bins < 1:
            raise ValueError(
                f"`time_bins`, given as an int, must be a positive number of bins, got {time_bins!r}."
            )

        schedule = self._survey_schedule
        return (
            schedule.start_time
            + np.linspace(0.0, 1.0, time_bins + 1) * schedule.duration
        )

    def generate_events(
        self,
        time_bins: Time | int,
        nside: int = 128,
        order: str = "nested",
    ) -> EventCatalog:
        """
        Sample a Monte Carlo realization of every registered transient type over a time grid.

        For each transient type and each bin ``[t_k, t_{k+1})`` of `time_bins`, events are
        sampled only within the HEALPix pixels the survey actually observes at some point
        between ``t_k`` and ``t_{k+1} + transient.duration_limit`` -- i.e. only where an event
        exploding in this bin could plausibly still be caught by an observation before it fades
        below relevance. Explosion times themselves are drawn only within ``[t_k, t_{k+1})``, so
        no event is double-counted across adjacent bins.

        Two columns are computed once here, at generation time, so nothing downstream ever
        re-derives them: ``luminosity_distance`` (interpolated per transient type off its own
        cached
        :attr:`~uvex_transient_toolkit.transients.base.ExtragalacticTransient.luminosity_distance_grid`/
        :attr:`~uvex_transient_toolkit.transients.base.ExtragalacticTransient.redshift_grid`, rather than
        a fresh `cosmology.luminosity_distance` call per event) and ``ebv`` (one vectorized
        Milky Way dust-map query over every sampled position at once).

        Parameters
        ----------
        time_bins : ~astropy.time.Time or int
            Either an explicit, monotonically increasing `Time` array of ``n + 1`` bin edges, or
            a positive int giving the number of evenly-spaced bins to divide
            `survey_schedule`'s full span into.
        nside : int, optional
            HEALPix resolution used both to query the observed footprint and to sample event
            positions. The default is 128.
        order : str, optional
            HEALPix pixel ordering scheme (``"nested"`` or ``"ring"``). The default is
            ``"nested"``.

        Returns
        -------
        EventCatalog
        """
        # Validate the inputs and ensure that there are actually registered transients to model.
        if not self._transients:
            raise ValueError(
                "No transient types registered in `transient_collection`; nothing to sample."
            )

        # Set up the time bins so that we can perform the windowing analysis properly.
        edges = self._resolve_time_bins(time_bins)
        n_bins = len(edges) - 1

        # Set up the RNG. Because the various sampled transients will need to each be assigned a seed
        # from the single seed provided here, we need to create a seed sequence.
        root_seed = get_seed_sequence(self._simulation_seed)
        sorted_names = sorted(self._transients)
        type_seeds = root_seed.spawn(len(sorted_names))

        # --- Begin Iteration Section --- #
        # In this code-section, we iterate through each window (t_i, t_i+1 + duration) and through each
        # of the transient types to construct the sample of events. This is ALL events within the redshift limit
        # which occur within the footprint of the survey. At this stage, the only event reduction is based in the
        # redshift limit and the survey footprint.
        tables = []

        with (
            tqdm(
                total=len(sorted_names) * n_bins, desc="Generating events", unit="bin"
            ) as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for name, type_seed in zip(sorted_names, type_seeds):
                transient = self._transients[name]
                bin_seeds = type_seed.spawn(n_bins)

                for k in range(n_bins):
                    pbar.set_postfix(type=name, bin=f"{k + 1}/{n_bins}")

                    t_start, t_end = edges[k], edges[k + 1]

                    # Determine which set of the healpix IDs intersect at all with the
                    # FOV of the survey.
                    pixel_ids = self._survey_schedule.get_observed_healpix_ids(
                        t_start,
                        t_end + transient.duration_limit,
                        nside=nside,
                        order=order,
                    )

                    # Determine the tiled area for this timestep and provide the debug info to console.
                    N_PIXELS_VISITED = len(pixel_ids)
                    SOLID_ANGLE_VISITED = (
                        ah.nside_to_pixel_area(nside) * N_PIXELS_VISITED
                    )
                    logger.debug(
                        "Transient Type: %s, Bin: %d/%d, Time Window: %s to %s, N Pixels Visited: %d, Solid Angle Visited: %.3f deg^2",
                        name,
                        k + 1,
                        n_bins,
                        t_start.iso,
                        t_end.iso,
                        N_PIXELS_VISITED,
                        SOLID_ANGLE_VISITED.to_value(u.deg**2),
                    )

                    # Extract a random sample of transients which occur within the
                    # relevant footprint.
                    table = transient.sample_events_on_healpix_grid(
                        nside,
                        t_start=t_start,
                        t_end=t_end,
                        pixel_ids=pixel_ids,
                        order=order,
                        seed=bin_seeds[k],
                    )

                    # Interpolate this transient type's own cached D_L(z) grid at each sampled
                    # event's redshift, right here, rather than a fresh `cosmology.luminosity_distance`
                    # call per event later -- see `ExtragalacticTransient.luminosity_distance_grid`.
                    if len(table) > 0:
                        table["luminosity_distance"] = (
                            np.interp(
                                table["redshift"],
                                transient.redshift_grid,
                                transient.luminosity_distance_grid.to_value(u.Mpc),
                            )
                            * u.Mpc
                        )
                    else:
                        table["luminosity_distance"] = u.Quantity([], u.Mpc)

                    table["transient_type"] = name
                    table["time_bin"] = k
                    tables.append(table)
                    pbar.update(1)

        combined = vstack(tables, metadata_conflicts="silent")
        combined.sort("t_explosion")
        combined["event_id"] = np.arange(len(combined), dtype=np.int64)

        # One vectorized Milky Way dust-map query over every sampled position at once, rather
        # than per-event later (see `EventCatalog.ebv`).
        combined["ebv"] = (
            np.asarray(dust_map().query(combined["coord"]), dtype=np.float64)
            if len(combined) > 0
            else np.array([], dtype=np.float64)
        )

        return EventCatalog(
            table=combined,
            nside=nside,
            order=order,
            time_bins=edges,
            seed=self._simulation_seed,
        )

    # -------------------------------------------------- #
    # Filtering                                          #
    # -------------------------------------------------- #
    def filter_by_limiting_magnitude(
        self,
        catalog: EventCatalog,
        mission: Mission,
        mag_limit: float,
        bands: list[str] | None = None,
        n_phase: int = 50,
        chunk_size: int = 5000,
        n_visits: int = 1,
    ) -> EventCatalog:
        """
        Cheaply cut an `EventCatalog` down to events that could ever plausibly be seen.

        Deliberately not synthetic photometry: no schedule, no background, no SNR formula,
        no per-event noise realization. For each transient type present in `catalog`, this
        regenerates each event's own physical SED parameters from its stored
        `parameter_seed` (see `_sample_parameters_from_seeds`), so a kept event's later
        `simulate_photometry` realization is guaranteed to match what was screened here --
        not an independent draw from the same population -- then evaluates
        :meth:`~uvex_transient_toolkit.models.core._base.SpectralModel.flux_band` over a shared
        ``linspace(0, duration_limit, n_phase)`` phase grid -- the same grid for every event
        of a type, regardless of whether the schedule ever actually pointed there at that
        phase -- broadcasting every event and every requested band at once. At each phase
        sample, an event's brightest band is compared against `mag_limit`; an event survives
        only if at least `n_visits` phase samples clear the limit.

        A freshly sampled catalog is typically dominated by faint, easily-rejected events
        and can easily run into the hundreds of thousands of rows (e.g. a multi-year,
        all-sky schedule) -- `flux_band` broadcasts every event and phase sample into one
        dense ``(n_phase, n_events, n_wavelength)`` array, so evaluating that in a single
        shot over the *whole* catalog can allocate tens of gigabytes and exhaust memory.
        `chunk_size` bounds this: events of each type are evaluated `chunk_size` at a time,
        keeping peak memory roughly constant regardless of catalog size, at the cost of
        some vectorization efficiency (still fully vectorized within a chunk).

        Parameters
        ----------
        catalog : EventCatalog
            Typically produced by `generate_events`; must already carry the
            ``luminosity_distance``/``ebv``/``parameter_seed`` columns that method fills
            in.
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` whose named bandpasses `bands` selects
            from.
        mag_limit : float
            AB magnitude limit; a phase sample clears the limit if the event's brightest
            evaluated band is at or below this value at that phase.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every bandpass
            the detector has.
        n_phase : int, optional
            Number of phase-grid samples per transient type, spanning
            ``[0, transient.duration_limit]``. The default is 50.
        chunk_size : int, optional
            Number of events (per transient type) evaluated per `flux_band` call. The
            default is 5000 (at `n_phase=50` and a ~100-point bandpass wavelength grid,
            roughly a few hundred MB of peak array memory per band); lower it further for a
            very fine `n_phase` or a very densely sampled bandpass.
        n_visits : int, optional
            Minimum number of phase-grid samples that must clear `mag_limit` for an event to
            survive. The default is 1, i.e. an event survives if it is ever bright enough at
            even a single sampled phase. Raising this discards events that are only
            momentarily bright enough to clear the limit, at (roughly) one sampled phase or
            fewer -- a coarse stand-in for "detected on at least `n_visits` visits", since this
            method never consults the actual schedule.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only (same `nside`/`order`/`time_bins`/
            `seed` as `catalog`; original `event_id` values are preserved, not renumbered).
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(
                f"'catalog' must be an EventCatalog, got {type(catalog)} instead."
            )
        if not isinstance(mission, Mission):
            raise TypeError(
                f"'mission' must be an m4opt.missions.Mission, got {type(mission)} instead."
            )
        if (
            isinstance(n_visits, bool)
            or not isinstance(n_visits, (int, np.integer))
            or n_visits < 1
        ):
            raise ValueError(f"'n_visits' must be a positive int, got {n_visits!r}.")

        detector = mission.detector
        if detector is None:
            raise ValueError(f"Mission {mission.name!r} has no detector configured.")

        band_names = list(detector.bandpasses) if bands is None else list(bands)
        unknown_bands = [band for band in band_names if band not in detector.bandpasses]
        if unknown_bands:
            raise ValueError(
                f"Unknown bandpass(es) {unknown_bands}; available: {list(detector.bandpasses)}."
            )

        table = catalog.table
        missing = [
            col
            for col in ("luminosity_distance", "ebv", "parameter_seed")
            if col not in table.colnames
        ]
        if missing:
            raise ValueError(
                f"'catalog' is missing column(s) {missing}; regenerate it via `generate_events`."
            )

        if len(table) == 0:
            return EventCatalog(
                table=table,
                nside=catalog.nside,
                order=catalog.order,
                time_bins=catalog.time_bins,
                seed=catalog.seed,
            )

        # Bandpass wavelength/throughput/frequency grids are the same for every transient
        # type and event -- sampled once here, not per type.
        band_grids = {}
        for band in band_names:
            bp = detector.bandpasses[band]
            wave = bp.waveset
            band_grids[band] = (
                wave,
                bp(wave),
                wave.to(u.Hz, equivalencies=u.spectral()),
            )

        transient_type = np.asarray(table["transient_type"]).astype(str)
        type_names = sorted(set(transient_type) & set(self._transients))
        unknown_types = sorted(set(transient_type) - set(self._transients))
        if unknown_types:
            raise ValueError(
                f"'catalog' contains transient type(s) {unknown_types} not in `transient_collection`."
            )

        keep = np.zeros(len(table), dtype=bool)

        # Precompute chunk counts per type up front so the progress bar can report a single
        # total spanning every chunk of every type, not just one tick per type.
        type_idx = {name: np.flatnonzero(transient_type == name) for name in type_names}
        n_chunks_by_type = {
            name: -(-idx.size // chunk_size) for name, idx in type_idx.items()
        }
        total_chunks = sum(n_chunks_by_type.values())

        with (
            tqdm(
                total=total_chunks, desc="Filtering by mag limit", unit="chunk"
            ) as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for name in type_names:
                transient = self._transients[name]
                idx = type_idx[name]
                n = idx.size
                n_chunks = n_chunks_by_type[name]

                redshift_all = np.asarray(table["redshift"])[idx]
                luminosity_distance_all = table["luminosity_distance"][idx]
                ebv_all = np.asarray(table["ebv"])[idx]
                seeds_all = np.asarray(table["parameter_seed"])[idx]

                # Sampling physical parameters for the whole type at once is cheap (a
                # handful of floats per event); it's only the `flux_band` evaluation below
                # -- an (n_phase, n_chunk, n_wavelength) array per band -- that can blow up
                # memory for a large catalog, so only *that* is chunked (see `chunk_size`).
                # Each event's own `parameter_seed` (not a shared per-type stream) keeps
                # this screening pass consistent with `Event.mag`/`Event.simulate_photometry`
                # -- see `_sample_parameters_from_seeds`.
                sed_params_all = _sample_parameters_from_seeds(transient.sed, seeds_all)

                t_grid = np.linspace(0.0, 1.0, n_phase) * transient.duration_limit

                for chunk_num, start in enumerate(range(0, n, chunk_size), start=1):
                    stop = min(start + chunk_size, n)
                    chunk_idx = idx[start:stop]

                    pbar.set_postfix(type=name, chunk=f"{chunk_num}/{n_chunks}")

                    redshift = redshift_all[start:stop]
                    luminosity_distance = luminosity_distance_all[start:stop]
                    ebv = ebv_all[start:stop]
                    sed_params = {
                        param_name: value[start:stop]
                        for param_name, value in sed_params_all.items()
                    }

                    # Brightest flux across bands *at each sampled phase* (not collapsed
                    # across phase yet), so visits can be counted per phase sample below.
                    best_flux_by_phase = None
                    for band in band_names:
                        wave, throughput, nu = band_grids[band]
                        flux = transient.sed.flux_band(
                            nu,
                            throughput,
                            t_grid[:, None],
                            redshift=redshift,
                            luminosity_distance=luminosity_distance,
                            log_attenuation=log_attenuation(nu, ebv),
                            **sed_params,
                        )  # shape (n_phase, chunk_size)
                        best_flux_by_phase = (
                            flux
                            if best_flux_by_phase is None
                            else np.maximum(best_flux_by_phase, flux)
                        )

                    with np.errstate(invalid="ignore", divide="ignore"):
                        mag_by_phase = best_flux_by_phase.to_value(
                            u.ABmag
                        )  # shape (n_phase, chunk_size)

                    n_visits_cleared = np.count_nonzero(
                        mag_by_phase <= mag_limit, axis=0
                    )  # shape (chunk_size,)
                    keep[chunk_idx[n_visits_cleared >= n_visits]] = True

                    pbar.update(1)

        return EventCatalog(
            table=table[keep],
            nside=catalog.nside,
            order=catalog.order,
            time_bins=catalog.time_bins,
            seed=catalog.seed,
        )

    def filter_by_snr(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        bands: list[str] | None = None,
        chunk_size: int = 2000,
        n_visits: int = 1,
    ) -> EventCatalog:
        """
        Cut an `EventCatalog` down to events the schedule actually detects.

        Unlike `filter_by_limiting_magnitude` (which never consults the schedule, only
        a shared phase grid), this asks the real question: over every observation the
        schedule actually made of an event's position while it was active, is it ever
        detected above `snr_threshold`? For each transient type present in `catalog`,
        events are processed `chunk_size` at a time:

        1. One `~uvex_transient_toolkit.surveys.base.SurveySchedule.get_observation_indices_of`
           call for the *whole chunk at once* finds which observations covered each
           event while active, each against its own `(t_explosion, t_explosion +
           transient.duration_limit)` window. The broad-phase HEALPix pixel lookup it's
           built on is queried at `catalog`'s own `nside`/`order`, but unlike the
           coarser screening this replaced, every candidate it returns is confirmed
           with an exact polygon-containment test before being handed back -- so this
           step never mistakenly keeps an event the schedule didn't truly observe; it
           can only ever miss one whose position happened to fall in a HEALPix pixel
           the covering footprint didn't register (see
           `~uvex_transient_toolkit.surveys.base.SurveySchedule.get_healpix_coverage_index`'s
           docstring for that one-directional tradeoff, tunable via `catalog.nside`).
           The first call for a given `(nside, order)` pays a one-time cost to index
           the *whole* schedule; every call after that, for any event (even from a
           different catalog, as long as the resolution matches), is a cheap lookup.
           An event the schedule never observed at all is discarded here, before any
           photometry.
        2. Every (event, observation) pair surviving that -- however many observations
           each individual event happens to have -- is flattened into one combined
           table, tagged by which chunk-local event it came from.
        3. Physical SED parameters are regenerated from each event's own stored
           `parameter_seed` (see `_sample_parameters_from_seeds` -- the same helper
           `filter_by_limiting_magnitude` uses, so both screening passes and a
           surviving event's later `Event.mag`/`Event.simulate_photometry` all agree on
           its physical parameters), repeated out to one row per (event, observation)
           pair alongside `redshift`/`luminosity_distance`/`ebv`/the event's own sky
           position, then the *whole* flattened chunk is evaluated in a single
           vectorized `get_snr` call per band -- exactly the batching
           `Event.simulate_photometry` does for one event's own observations, just
           extended across many events at once.
        4. Each row's best-band SNR is compared against `snr_threshold`; an event
           survives if at least `n_visits` of its own rows clear it (mirroring
           `filter_by_limiting_magnitude`'s own `n_visits`: bands are collapsed to the
           best one *before* counting, so a single observation bright enough in two
           bands at once still counts as one visit, not two).

        Parameters
        ----------
        catalog : EventCatalog
            Typically the (already magnitude-filtered) output of
            `filter_by_limiting_magnitude` -- running this directly on a freshly
            sampled catalog works too, just more slowly, since every event still costs
            one `get_observations_of` call regardless of how faint it is.
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` `bands` selects from.
        snr_threshold : float
            An event survives if at least `n_visits` observations clear this SNR, in
            their best band.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every
            bandpass the detector has.
        chunk_size : int, optional
            Number of events (per transient type) whose observations are gathered and
            evaluated together, bounding the size of each flattened `get_snr` batch.
            The default is 2000; see `filter_by_limiting_magnitude`'s docstring for
            the same memory-vs-vectorization tradeoff on the evaluation side (the
            schedule lookup itself is cheap now, so unlike before this no longer needs
            to be small purely to bound how many schedule queries happen before the
            next batch).
        n_visits : int, optional
            Minimum number of observations that must clear `snr_threshold` (in their
            best band) for an event to survive. The default is 1: an event survives if
            it is ever detected at all.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only (same `nside`/`order`/`time_bins`/
            `seed` as `catalog`; original `event_id` values are preserved, not renumbered).
        """
        # Validate the event catalog and the mission along with the necessary number of visits
        # to be considered worthwhile.
        if not isinstance(catalog, EventCatalog):
            raise TypeError(
                f"'catalog' must be an EventCatalog, got {type(catalog)} instead."
            )
        if not isinstance(mission, Mission):
            raise TypeError(
                f"'mission' must be an m4opt.missions.Mission, got {type(mission)} instead."
            )
        if (
            isinstance(n_visits, bool)
            or not isinstance(n_visits, (int, np.integer))
            or n_visits < 1
        ):
            raise ValueError(f"'n_visits' must be a positive int, got {n_visits!r}.")

        detector = mission.detector
        if detector is None:
            raise ValueError(f"Mission {mission.name!r} has no detector configured.")

        band_names = list(detector.bandpasses) if bands is None else list(bands)
        unknown_bands = [band for band in band_names if band not in detector.bandpasses]
        if unknown_bands:
            raise ValueError(
                f"Unknown bandpass(es) {unknown_bands}; available: {list(detector.bandpasses)}."
            )

        # Load the event catalog table and
        table = catalog.table
        missing = [
            col
            for col in ("luminosity_distance", "ebv", "healpix_id", "parameter_seed")
            if col not in table.colnames
        ]
        if missing:
            raise ValueError(
                f"'catalog' is missing column(s) {missing}; regenerate it via `generate_events`."
            )

        if len(table) == 0:
            return EventCatalog(
                table=table,
                nside=catalog.nside,
                order=catalog.order,
                time_bins=catalog.time_bins,
                seed=catalog.seed,
            )

        schedule = self._survey_schedule

        transient_type = np.asarray(table["transient_type"]).astype(str)
        type_names = sorted(set(transient_type) & set(self._transients))
        unknown_types = sorted(set(transient_type) - set(self._transients))
        if unknown_types:
            raise ValueError(
                f"'catalog' contains transient type(s) {unknown_types} not in `transient_collection`."
            )

        keep = np.zeros(len(table), dtype=bool)

        coord_all = table["coord"]
        t_explosion_all = table["t_explosion"]
        healpix_nside, healpix_order = catalog.nside, catalog.order

        # Precompute chunk counts per type up front so the progress bar can report a
        # single total spanning every chunk of every type, not just one tick per type.
        type_idx = {name: np.flatnonzero(transient_type == name) for name in type_names}
        n_chunks_by_type = {
            name: -(-idx.size // chunk_size) for name, idx in type_idx.items()
        }
        total_chunks = sum(n_chunks_by_type.values())

        with (
            tqdm(total=total_chunks, desc="Filtering by SNR", unit="chunk") as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for name in type_names:
                transient = self._transients[name]
                idx = type_idx[name]
                n = idx.size
                n_chunks = n_chunks_by_type[name]

                redshift_all = np.asarray(table["redshift"])[idx]
                luminosity_distance_all = table["luminosity_distance"][idx]
                ebv_all = np.asarray(table["ebv"])[idx]
                seeds_all = np.asarray(table["parameter_seed"])[idx]
                coord_type = coord_all[idx]
                t_explosion_type = t_explosion_all[idx]

                # As in `filter_by_limiting_magnitude`: each event's own `parameter_seed`,
                # not a shared per-type stream -- see `_sample_parameters_from_seeds`.
                sed_params_all = _sample_parameters_from_seeds(transient.sed, seeds_all)

                for chunk_num, start in enumerate(range(0, n, chunk_size), start=1):
                    stop = min(start + chunk_size, n)
                    chunk_idx = idx[start:stop]
                    chunk_size_actual = stop - start

                    pbar.set_postfix(type=name, chunk=f"{chunk_num}/{n_chunks}")

                    # --- Step 1 & 2: gather every observation of every event in this
                    # chunk in one shot (one `get_observation_indices_of` call for the
                    # whole chunk, backed by the cached coverage index and doing
                    # exactly one QTable construction -- indexing `observe_rows` once
                    # by the whole chunk's row indices -- for the whole batch, *not*
                    # one query and one small QTable per event: at chunk_size ~ a few
                    # thousand, that per-event QTable-construction overhead was itself
                    # the dominant cost of this whole method, dwarfing the actual SNR
                    # computation below).
                    coord_chunk = coord_type[start:stop]
                    t_explosion_chunk = t_explosion_type[start:stop]

                    event_index, row_index = schedule.get_observation_indices_of(
                        coord_chunk,
                        nside=healpix_nside,
                        order=healpix_order,
                        start_time=t_explosion_chunk,
                        end_time=t_explosion_chunk + transient.duration_limit,
                    )
                    if len(row_index) == 0:
                        pbar.update(1)
                        continue

                    flat = schedule.observe_rows[row_index]
                    flat["event_index"] = event_index
                    flat["t_since_explosion"] = (
                        flat["start_time"] - t_explosion_chunk[event_index]
                    ).to(u.day)

                    # --- Step 3: repeat each event's own quantities out to one row per
                    # (event, observation) pair, then evaluate the whole flattened
                    # chunk in one batch. Every batched quantity here (`t`, `redshift`,
                    # `luminosity_distance`, every SED param) shares the exact same flat
                    # row axis, so *those* need this reserved trailing axis -- otherwise
                    # it collides with the wavelength axis introduced later inside
                    # `get_snr` (see `Event.simulate_photometry`'s own comment on this;
                    # the same broadcasting rule applies here to every one of these
                    # arrays, not just `t`). `ebv` is the one exception: unlike
                    # `_eval`/`_eval_flux`'s own plain-broadcasting parameters,
                    # `dust.log_attenuation` reserves *its own* trailing axis internally
                    # (`Av[..., np.newaxis]`) -- giving it one here too would reserve it
                    # twice, silently producing an extra broadcast axis (verified: this
                    # produced a wrong, doubled-up `(chunk, chunk)`-shaped SNR before the
                    # fix). `ebv_flat` must stay flat, matching the same
                    # `test_synthetic_photometry_with_batched_dust_extinction` distinction.
                    redshift_chunk = redshift_all[start:stop]
                    luminosity_distance_chunk = luminosity_distance_all[start:stop]
                    ebv_chunk = ebv_all[start:stop]
                    sed_params_chunk = {
                        param_name: value[start:stop]
                        for param_name, value in sed_params_all.items()
                    }

                    t_obs_flat = flat["t_since_explosion"][:, np.newaxis]
                    redshift_flat = redshift_chunk[event_index][:, np.newaxis]
                    luminosity_distance_flat = luminosity_distance_chunk[event_index][
                        :, np.newaxis
                    ]
                    ebv_flat = ebv_chunk[event_index]
                    sed_params_flat = {
                        param_name: value[event_index][:, np.newaxis]
                        for param_name, value in sed_params_chunk.items()
                    }
                    coord_flat = coord_chunk[event_index]

                    spectra = transient.sed.as_source_spectrum(
                        t_obs_flat,
                        redshift=redshift_flat,
                        luminosity_distance=luminosity_distance_flat,
                        log_attenuation=partial(log_attenuation, Ebv=ebv_flat),
                        **sed_params_flat,
                    )

                    best_snr = None
                    with observing(
                        flat["observer_location"], coord_flat, flat["start_time"]
                    ):
                        for band in band_names:
                            snr = detector.get_snr(flat["duration"], spectra, band)
                            best_snr = (
                                snr if best_snr is None else np.maximum(best_snr, snr)
                            )

                    # --- Step 4: count, per chunk-local event, how many of its own
                    # rows clear `snr_threshold` in their best band.
                    above = best_snr > snr_threshold
                    visit_count = np.zeros(chunk_size_actual, dtype=np.int64)
                    np.add.at(visit_count, event_index[above], 1)
                    keep[chunk_idx[visit_count >= n_visits]] = True

                    pbar.update(1)

        return EventCatalog(
            table=table[keep],
            nside=catalog.nside,
            order=catalog.order,
            time_bins=catalog.time_bins,
            seed=catalog.seed,
        )
