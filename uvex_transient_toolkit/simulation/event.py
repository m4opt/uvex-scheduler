"""
A single sampled transient event, reconstructed from one `EventCatalog` row.

An :class:`Event` wraps everything needed to evaluate one sampled transient's
detectability against a real survey schedule: its position, redshift, the
luminosity distance and E(B-V) already cached in the catalog at generation time
(see :meth:`~uvex_transient_toolkit.simulation.core.SurveySimulator.generate_events` --
neither is re-derived here), its SED parameter seed, and the schedule's own record
of which observations actually covered its position during its active window.

Normally reconstructed via :meth:`~uvex_transient_toolkit.simulation.event_catalog.EventCatalog.get_events`,
not constructed directly.
"""

from collections.abc import Hashable
from functools import partial

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.units import Quantity
from m4opt.missions import Mission
from m4opt.synphot import observing

from uvex_transient_toolkit.dust import log_attenuation
from uvex_transient_toolkit.utils import get_rng

from ..surveys.base import SurveySchedule
from ..transients.base import TransientBase


class Event:
    """
    One sampled transient event, tied to a real survey schedule.

    Building an `Event` runs exactly one query against `schedule`
    (:meth:`~uvex_transient_toolkit.surveys.base.SurveySchedule.get_observations_of`) to find
    which scheduled observations actually covered this event's position during
    ``[t_explosion, t_explosion + transient.duration_limit)`` -- no photometry is done
    here. :meth:`simulate_photometry` does the (comparatively expensive) per-observation,
    per-band synthetic photometry, on demand.
    """

    def __init__(
        self,
        event_id: int,
        schedule: SurveySchedule,
        transient: TransientBase,
        coord: SkyCoord,
        redshift: float,
        t_explosion: Time,
        seed: int,
        *,
        luminosity_distance: Quantity | None = None,
        ebv: float | None = None,
        transient_type: str | None = None,
    ):
        """
        Parameters
        ----------
        event_id : int
            This event's unique id, as assigned by `EventCatalog`.
        schedule : ~uvex_transient_toolkit.surveys.base.SurveySchedule
            The survey schedule to check this event's visibility against.
        transient : ~uvex_transient_toolkit.transients.base.TransientBase
            The transient type this event is a realization of -- supplies `sed` and
            `duration_limit`.
        coord : ~astropy.coordinates.SkyCoord
            Scalar sky position of the event.
        redshift : float
            Cosmological redshift.
        t_explosion : ~astropy.time.Time
            Scalar time of explosion.
        seed : int
            This event's stored parameter seed; deterministically regenerates its
            physical SED parameters (see :meth:`sample_parameters`).
        luminosity_distance : ~astropy.units.Quantity, optional
            Luminosity distance, if already known (e.g. cached by `EventCatalog` at
            generation time). If not given, it's derived from `redshift` via
            `transient.cosmology` -- a real cosmology call, so passing it in when
            it's already on hand is the whole point of caching it.
        ebv : float, optional
            Milky Way foreground E(B-V) at `coord`, if already known (cached by
            `EventCatalog`). If not given, reddening is treated as zero.
        transient_type : str, optional
            This event's transient-type name, for `__repr__` only.
        """
        if not isinstance(schedule, SurveySchedule):
            raise TypeError(
                f"'schedule' must be a SurveySchedule, got {type(schedule)}."
            )
        if not isinstance(transient, TransientBase):
            raise TypeError(
                f"'transient' must be a TransientBase, got {type(transient)}."
            )

        self._event_id = int(event_id)
        self._schedule = schedule
        self._transient = transient
        self._transient_type = transient_type
        self._coord = coord
        self._redshift = float(redshift)
        self._t_explosion = t_explosion
        self._seed = int(seed)
        self._luminosity_distance = (
            luminosity_distance
            if luminosity_distance is not None
            else transient.cosmology.luminosity_distance(self._redshift)
        )
        self._ebv = 0.0 if ebv is None else float(ebv)

        # The one query this class exists to make: which scheduled observations actually
        # covered this event, while it was active. `get_observations_of`'s own window can
        # include a row that *started* slightly before `t_explosion` but overlaps into it
        # (e.g. a long downlink); that row is dropped here since only observations that
        # began at or after the explosion are ever useful for this event's lightcurve.
        candidate = schedule.get_observations_of(
            coord, t_explosion, t_explosion + transient.duration_limit
        )
        self._observations = candidate[candidate["start_time"] >= t_explosion]

    def __repr__(self) -> str:
        return (
            f"<Event id={self._event_id} type={self._transient_type!r} "
            f"z={self._redshift:.4g} n_observations={len(self._observations)}>"
        )

    # ------------------------------ #
    # Properties                     #
    # ------------------------------ #
    @property
    def event_id(self) -> int:
        """int: This event's unique id."""
        return self._event_id

    @property
    def transient(self) -> TransientBase:
        """TransientBase: The transient type this event is a realization of."""
        return self._transient

    @property
    def transient_type(self) -> str | None:
        """Str or None: This event's transient-type name."""
        return self._transient_type

    @property
    def coord(self) -> SkyCoord:
        """~astropy.coordinates.SkyCoord: Sky position of the event."""
        return self._coord

    @property
    def redshift(self) -> float:
        """float: Cosmological redshift."""
        return self._redshift

    @property
    def luminosity_distance(self) -> Quantity:
        """~astropy.units.Quantity: Luminosity distance to the event."""
        return self._luminosity_distance

    @property
    def ebv(self) -> float:
        """float: Milky Way foreground E(B-V) at `coord` (0 if never supplied)."""
        return self._ebv

    @property
    def t_explosion(self) -> Time:
        """~astropy.time.Time: Time of explosion."""
        return self._t_explosion

    @property
    def seed(self) -> int:
        """int: This event's stored parameter seed."""
        return self._seed

    @property
    def observations(self) -> QTable:
        """QTable: The schedule's ``"observe"`` rows that covered this event while active.

        One row per candidate observation, chronological, with the same columns as
        `SurveySchedule.table` (``start_time``, ``duration``, ``observer_location``
        -- i.e. where the spacecraft was -- ``target_coord``, ``roll``, ...). Empty if
        the survey never observed this event's position during its active window.
        """
        return self._observations

    @property
    def n_observations(self) -> int:
        """int: Number of candidate observations (``len(observations)``)."""
        return len(self._observations)

    # ------------------------------ #
    # Parameters                     #
    # ------------------------------ #
    def sample_parameters(self) -> dict:
        """
        dict: This event's physical SED parameters, deterministically regenerated from `seed`.

        A pure, idempotent "peek" -- every call reseeds a fresh generator from `seed`
        rather than sharing state with `simulate_photometry`, so this always returns
        the same values regardless of how many times it (or `simulate_photometry`)
        has already been called.
        """
        rng = get_rng(self._seed)
        return {
            name: value[0]
            for name, value in self._transient.sed.sample_parameters(
                size=1, rng=rng
            ).items()
        }

    # ------------------------------ #
    # Theoretical Photometry         #
    # ------------------------------ #
    def _resolve_bandpass(self, mission: Mission, band: Hashable | None):
        """The `~synphot.SpectralElement` named `band` in `mission.detector.bandpasses`.

        `band` may be omitted only if the detector has exactly one bandpass -- mirrors
        `~m4opt.synphot.Detector`'s own bandpass-resolution rule (see
        `~m4opt.synphot.Detector.get_snr`).
        """
        detector = mission.detector
        if detector is None:
            raise ValueError(f"Mission {mission.name!r} has no detector configured.")
        if band is not None:
            return detector.bandpasses[band]
        if len(detector.bandpasses) == 1:
            (bandpass,) = detector.bandpasses.values()
            return bandpass
        raise ValueError(
            f"Mission {mission.name!r} has more than one bandpass. Please specify "
            f"one of them: {list(detector.bandpasses)}."
        )

    def _pivot_nu_and_log_attenuation(self, bandpass) -> tuple[Quantity, np.ndarray]:
        """`bandpass`'s pivot frequency, and this event's own dust attenuation there.

        The same single-wavelength approximation `simulate_photometry` uses for its
        noiseless flux (`SpectralElement.pivot`), not a full bandpass-throughput
        integral -- see :meth:`mag`'s docstring for why that distinction matters here.
        """
        nu = bandpass.pivot().to(u.Hz, equivalencies=u.spectral())
        return nu, log_attenuation(nu, self._ebv)

    def mag(
        self, t: Quantity, mission: Mission, band: Hashable | None = None
    ) -> Quantity:
        """
        Theoretical (noiseless) apparent AB magnitude of this event's SED at time(s) `t`.

        This is the literal noiseless curve :meth:`simulate_photometry`'s noisy points
        scatter around, not merely a physically similar but numerically different
        quantity: like that method, the flux is evaluated at `band`'s pivot
        wavelength (`~synphot.SpectralElement.pivot`), *not* integrated across the
        full bandpass throughput, and this event's own foreground dust (`ebv`) is
        applied, in addition to its `redshift`/`luminosity_distance`/SED parameters
        (see :meth:`sample_parameters`) -- so only a time grid and a mission/band are
        needed. Useful for plotting a theory curve alongside real observations, not
        for simulating a detection.

        Parameters
        ----------
        t
            Observed time(s) since explosion, any shape.
        mission
            Supplies the `~m4opt.synphot.Detector` `band` selects a bandpass from.
        band
            Which of `mission.detector`'s bandpasses to evaluate. Required unless the
            detector has exactly one.

        Returns
        -------
        ~astropy.units.Quantity
            The apparent AB magnitude, as an :attr:`~astropy.units.ABmag` Quantity,
            with `t`'s shape.
        """
        bandpass = self._resolve_bandpass(mission, band)
        nu, attenuation = self._pivot_nu_and_log_attenuation(bandpass)
        return self._transient.sed.mag(
            nu,
            t,
            redshift=self._redshift,
            luminosity_distance=self._luminosity_distance,
            log_attenuation=attenuation,
            **self.sample_parameters(),
        )

    def flux(
        self, t: Quantity, mission: Mission, band: Hashable | None = None
    ) -> Quantity:
        """
        Theoretical (noiseless) observed flux density of this event's SED, at `band`'s pivot wavelength.

        Same inputs, and the same pivot-wavelength-plus-dust definition matching
        :meth:`simulate_photometry` exactly, as :meth:`mag` -- see its docstring.

        Returns
        -------
        ~astropy.units.Quantity
            The observed flux density, with `t`'s shape.
        """
        bandpass = self._resolve_bandpass(mission, band)
        nu, attenuation = self._pivot_nu_and_log_attenuation(bandpass)
        return self._transient.sed.flux(
            nu,
            t,
            redshift=self._redshift,
            luminosity_distance=self._luminosity_distance,
            log_attenuation=attenuation,
            **self.sample_parameters(),
        )

    def luminosity(self, t: Quantity) -> Quantity:
        """
        Bolometric luminosity of this event's SED at time(s) `t` since explosion.

        Intrinsic (rest-frame, distance-independent) -- unlike :meth:`mag`/:meth:`flux`,
        no `mission`/`band` is involved. This event's own SED parameters (see
        :meth:`sample_parameters`) are supplied automatically. Wraps
        `~uvex_transient_toolkit.models.core._base.SpectralModel.eval_bolometric`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\\mathrm{bol}(t)`, in erg/s, with `t`'s shape.
        """
        return self._transient.sed.eval_bolometric(t, **self.sample_parameters())

    # ------------------------------ #
    # Photometry                     #
    # ------------------------------ #
    @staticmethod
    def _empty_photometry_table() -> QTable:
        table = QTable()
        table["event_id"] = np.array([], dtype=np.int64)
        table["obs_time"] = Time([], format="jd")
        table["exptime"] = u.Quantity([], u.s)
        # A wide, explicit itemsize -- `dtype=str` alone infers itemsize 1 from
        # an empty array (silently truncating any band name to one character
        # once populated rows are ever `vstack`'d onto this one).
        table["band"] = np.array([], dtype="<U32")
        table["snr"] = np.array([], dtype=np.float64)
        table["flux"] = u.Quantity([], u.Jy)
        table["flux_err"] = u.Quantity([], u.Jy)
        table["flux_upper"] = u.Quantity([], u.Jy)
        table["flux_lower"] = u.Quantity([], u.Jy)
        table["ab_mag"] = np.array([], dtype=np.float64)
        table["mag_err"] = np.array([], dtype=np.float64)
        table["mag_upper"] = np.array([], dtype=np.float64)
        table["mag_lower"] = np.array([], dtype=np.float64)
        return table

    def simulate_photometry(
        self,
        mission: Mission,
        bands: list | None = None,
        n_sigma: float = 5.0,
    ) -> QTable:
        """
        Evaluate this event's detectability at every observation in `observations`.

        Builds one `~synphot.SourceSpectrum` for this whole event -- batched over
        every candidate observation's own time since explosion, via
        `~uvex_transient_toolkit.models.core._base.SpectralModel.as_source_spectrum`
        (dust folded in through its ``log_attenuation``, from this event's own cached
        `ebv`) -- and reuses it, unmodified, for every requested band:
        a `~synphot.SourceSpectrum` is purely a function of wavelength/time, so
        "band" only enters once it's integrated against a bandpass, via
        `~m4opt.synphot.Detector.get_snr`. That turns what used to be a Python loop
        over every (observation, band) pair, each doing its own scalar
        `SpectralModel.flux`/`astropy.stats.signal_to_noise_oir_ccd` call, into one
        vectorized `get_snr` call per band, regardless of how many observations
        there are.

        The reported flux/magnitude at each (observation, band) is then a Gaussian
        realization of the true (noiseless) flux -- evaluated from that same
        `SourceSpectrum` at the band's pivot wavelength -- at that SNR's implied
        uncertainty; a synthetic measurement, not the ground truth. Physical SED
        parameters are sampled once, from `seed`; every band's noise draws are one
        vectorized call over all observations at once, in `bands` order, on the
        same stream that draw consumed -- so the whole event still replays
        identically given the same `seed`, but *not* row-for-row identically to an
        older, unbatched implementation, since the draws are now grouped per band
        across every observation rather than interleaved observation-by-observation.

        ``mag_err`` is the usual linearized (first-order) propagation of ``flux_err``
        through the magnitude log transform -- a good description of the uncertainty
        only while it's small relative to ``flux``, i.e. at high SNR. It is *not* a
        substitute for a real confidence interval: because magnitude is a nonlinear
        (logarithmic) function of flux, a symmetric interval in flux is an asymmetric
        one in magnitude, and that asymmetry grows as SNR drops -- comparing against
        ``ab_mag``/``mag_err`` as if they were a plain Gaussian pull systematically
        reads as biased at low SNR even when the underlying flux draw has no bias at
        all. ``flux_upper``/``flux_lower`` and ``mag_upper``/``mag_lower`` are the
        actual ``n_sigma`` interval, built the correct way around: bound `flux`
        symmetrically first (where the noise is actually Gaussian), then transform
        each bound to magnitude separately, rather than propagating one linearized
        width through the transform.

        Parameters
        ----------
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` (bandpasses, background, ...) evaluated
            against.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every
            bandpass the detector has.
        n_sigma : float, optional
            Width, in multiples of ``flux_err``, of the ``flux_upper``/``flux_lower``/
            ``mag_upper``/``mag_lower`` interval. Default is 5.

        Returns
        -------
        astropy.table.QTable
            One row per (observation, band), sorted by ``obs_time`` then ``band``,
            with columns ``event_id``, ``obs_time``, ``exptime``, ``band``, ``snr``,
            ``flux``/``flux_err`` (Jy), ``flux_upper``/``flux_lower`` (Jy, ``flux ±
            n_sigma*flux_err``), ``ab_mag``/``mag_err``, and ``mag_upper``/``mag_lower``
            -- the ``n_sigma`` interval transformed to magnitude, brighter bound first:
            ``mag_lower`` (from ``flux_upper``) is always finite when ``flux_upper>0``;
            ``mag_upper`` (from ``flux_lower``) is ``nan`` whenever ``flux_lower<=0``,
            i.e. whenever the source isn't securely distinguished from zero flux at
            ``n_sigma`` -- the correct behavior is a one-sided (no faint bound) result
            there, not a spuriously finite one. ``flux``/``flux_err``/``ab_mag``/
            ``mag_err`` are ``nan`` wherever ``snr`` is non-positive or non-finite
            (``ab_mag``/``mag_err`` are also ``nan`` wherever the noisy ``flux``
            realization itself came out non-positive). Empty (but correctly typed) if
            `observations` is empty.
        """
        detector = mission.detector
        if detector is None:
            raise ValueError(f"Mission {mission.name!r} has no detector configured.")

        band_names = list(detector.bandpasses) if bands is None else list(bands)
        unknown = [band for band in band_names if band not in detector.bandpasses]
        if unknown:
            raise ValueError(
                f"Unknown bandpass(es) {unknown}; available: {list(detector.bandpasses)}."
            )

        n_obs = len(self._observations)
        if n_obs == 0:
            return self._empty_photometry_table()

        # One RNG, seeded from this event's own stored `seed`, drives both the parameter
        # draw and every band's noise realization below, in `band_names` order -- so the
        # whole event replays identically from `seed` alone. Deliberately *not*
        # `self.sample_parameters()` (which reseeds fresh every call): the noise draws
        # below must continue on the same stream the parameter draw already consumed.
        rng = get_rng(self._seed)
        sed_params = {
            name: value[0]
            for name, value in self._transient.sed.sample_parameters(
                size=1, rng=rng
            ).items()
        }

        # `as_source_spectrum` does not auto-insert batch axes -- this trailing axis
        # is what keeps `t_obs`'s per-observation batch from colliding with whatever
        # wavelength grid it's later called with (e.g. a bandpass's `waveset`).
        t_obs = (self._observations["start_time"] - self._t_explosion).to(u.day)[
            :, np.newaxis
        ]

        # One SourceSpectrum for this whole event, batched over every candidate
        # observation's own time since explosion -- reused, unmodified, across every
        # band below.
        spectra = self._transient.sed.as_source_spectrum(
            t_obs,
            redshift=self._redshift,
            luminosity_distance=self._luminosity_distance,
            log_attenuation=partial(log_attenuation, Ebv=self._ebv),
            **sed_params,
        )

        tables = []

        with observing(
            self._observations["observer_location"],
            self._coord,
            self._observations["start_time"],
        ):
            for band in band_names:
                snr = detector.get_snr(self._observations["duration"], spectra, band)

                pivot = detector.bandpasses[band].pivot()
                with np.errstate(invalid="ignore", divide="ignore"):
                    # `pivot` is scalar, so it broadcasts against `t_obs`'s own
                    # reserved trailing axis rather than colliding with it --
                    # squeeze that axis back out to get one flux per observation.
                    true_flux = np.squeeze(
                        spectra(pivot, flux_unit=u.Jy).to_value(u.Jy), axis=-1
                    )

                    valid = np.isfinite(snr) & (snr > 0)
                    safe_snr = np.where(valid, snr, np.nan)
                    flux_err = true_flux / safe_snr
                    flux = rng.normal(true_flux, np.where(valid, np.abs(flux_err), 1.0))
                    flux = np.where(valid, flux, np.nan)
                    mag_err = 2.5 / (np.log(10) * safe_snr)
                    mag = np.where(flux > 0, (flux * u.Jy).to_value(u.ABmag), np.nan)

                    # The actual n_sigma interval: bound `flux` symmetrically first
                    # (where the noise is actually Gaussian), then transform each bound
                    # to magnitude separately -- rather than propagating one linearized
                    # width through the nonlinear log transform, as `mag_err` does. A
                    # larger flux is a *smaller* (brighter) magnitude, so `flux_upper`
                    # maps to `mag_lower` and vice versa. `flux_lower` can go
                    # non-positive at low SNR -- that's not an error, it means the
                    # source isn't securely distinguished from zero flux at `n_sigma`,
                    # so there is no finite faint bound (`mag_upper` is correctly `nan`
                    # there, not a substitute finite value).
                    flux_upper = flux + n_sigma * flux_err
                    flux_lower = flux - n_sigma * flux_err
                    mag_lower = np.where(
                        flux_upper > 0, (flux_upper * u.Jy).to_value(u.ABmag), np.nan
                    )
                    mag_upper = np.where(
                        flux_lower > 0, (flux_lower * u.Jy).to_value(u.ABmag), np.nan
                    )

                band_table = QTable()
                band_table["event_id"] = np.full(n_obs, self._event_id, dtype=np.int64)
                band_table["obs_time"] = self._observations["start_time"]
                band_table["exptime"] = self._observations["duration"]
                band_table["band"] = np.full(n_obs, band)
                band_table["snr"] = snr
                band_table["flux"] = flux * u.Jy
                band_table["flux_err"] = flux_err * u.Jy
                band_table["flux_upper"] = flux_upper * u.Jy
                band_table["flux_lower"] = flux_lower * u.Jy
                band_table["ab_mag"] = mag
                band_table["mag_err"] = mag_err
                band_table["mag_upper"] = mag_upper
                band_table["mag_lower"] = mag_lower
                tables.append(band_table)

        table = vstack(tables)
        # `Table.sort(["obs_time", "band"])` is >1000x slower here than this --
        # astropy's multi-key sort falls back to pairwise `Time.__lt__`
        # comparisons once a second sort key is involved (a `Time` column
        # sorted alone, or `np.lexsort` on its fast numeric `.jd` proxy, are
        # both fine; combining a `Time` key with another column through
        # `Table.sort` is what's slow). `band`'s exact string encoding doesn't
        # matter for sort order, only that equal strings compare equal.
        order = np.lexsort((table["band"], table["obs_time"].jd))
        return table[order]
