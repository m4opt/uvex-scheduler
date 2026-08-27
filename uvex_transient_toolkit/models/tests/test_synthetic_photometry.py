"""
This test suite verifies that the model module can plug into the synphot machinery.
"""

from functools import partial

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from m4opt.missions._uvex import uvex
from uvex_transient_toolkit.dust import log_attenuation
from uvex_transient_toolkit.models import VanVelzenTDESED, VillarCoolingBlackbodySED
from m4opt.skygrid import _geodesic
from m4opt.synphot import observing
from m4opt.synphot.extinction import DustExtinction


def test_synthetic_photometry_batches_events_and_times():
    """
    Per-event, per-time SNR light curves via a batched `SourceSpectrum`, round-tripped against `flux`.

    Mirrors the batching pattern a real scheduling run uses -- one parameter
    realization per sky-grid event, a light curve of several times since
    explosion per event -- at a size small enough to run as a fast unit
    test. Unlike `test_synthetic_photometry_with_dust_extinction` below, this
    does not multiply in dust extinction, since batching a per-event SED
    through that path is a known, separate limitation.
    """
    rng = np.random.default_rng(0)

    n_events = 5
    n_times = 4
    redshift = 0.1
    band = "FUV"
    exptime = 900 * u.s

    model = VillarCoolingBlackbodySED()

    # One parameter realization per event, kept as plain (n_events,) arrays;
    # each consumer below adds however many trailing axes *it* needs to keep
    # the event axis from colliding with its own batch axes (time,
    # wavelength).
    raw_params = model.sample_parameters(n_events, rng=rng)
    t_since_explosion = np.linspace(1, 200, n_times) * u.day

    # --- Band-averaged magnitude light curves, via the model's own
    # vectorized quadrature (not synphot's Observation/effstim, which
    # chokes on batched spectra).
    bandpass = uvex.detector.bandpasses[band]
    wave = bandpass.waveset
    nu = wave.to(u.Hz, equivalencies=u.spectral())
    throughput = bandpass(wave)

    mag_params = {name: value[:, np.newaxis] for name, value in raw_params.items()}
    mag = model.mag_band(
        nu, throughput, t_since_explosion, redshift=redshift, **mag_params
    )
    assert mag.shape == (n_events, n_times)
    assert np.all(np.isfinite(mag.value))

    # --- SNR light curves, via a batched `SourceSpectrum` fed to the
    # detector. `as_source_spectrum` does not auto-insert axes, so every
    # batch axis needs its own manually reserved trailing axis: two here
    # (time, then wavelength) for the parameters, one (wavelength) for `t`.
    source_params = {
        name: value[:, np.newaxis, np.newaxis] for name, value in raw_params.items()
    }
    t_batched = t_since_explosion[:, np.newaxis]
    source_spectra = model.as_source_spectrum(
        t=t_batched, redshift=redshift, **source_params
    )

    base_time = Time("2025-01-01T00:00:00", scale="utc")
    obs_times = base_time + t_since_explosion
    locations = uvex.observer_location(obs_times)

    with observing(locations, uvex.skygrid[0], obs_times):
        snr = uvex.detector.get_snr(exptime, source_spectra, band)

    assert snr.shape == (n_events, n_times)
    assert np.all(np.isfinite(snr))
    assert np.all(snr > 0)

    # --- Round trip: sample the batched `SourceSpectrum` -- the same object
    # just fed to `get_snr` -- at the bandpass wavelength grid, then confirm
    # one (event, time, wavelength) slot of that batch matches `flux`
    # computed directly and independently for that same single realization.
    # This is what actually confirms that broadcasting several extra batch
    # axes through `as_source_spectrum` lines up each event/time slot with
    # the right parameter values, rather than merely checking finiteness.
    #
    # A bare scalar frequency trips an astropy `Model.__call__` bug
    # unrelated to this (it assumes a scalar input broadcasts to a scalar
    # output, which doesn't hold once extra batch axes are bound into the
    # model via closure) -- sample the whole wavelength grid instead and
    # index into it.
    i_event, i_time, i_freq = 2, 1, len(nu) // 2
    scalar_params = {name: value[i_event] for name, value in raw_params.items()}
    expected = model.flux(
        nu[i_freq], t_since_explosion[i_time], redshift=redshift, **scalar_params
    )
    actual = source_spectra(nu, flux_unit=expected.unit)[i_event, i_time, i_freq]
    np.testing.assert_allclose(actual.value, expected.value, rtol=1e-6)


@pytest.mark.xfail(
    reason=(
        "m4opt.synphot._math.countrate's cubic-interpolation shortcut for "
        "dust extinction (skygrid sizes >= 512) assumes the source spectrum "
        "carries no batch axis of its own; np.vectorize(otypes=[float]) "
        "chokes when the per-Ebv countrate comes back array-valued instead "
        "of scalar, as happens for a per-event batched SED. Needs a fix in "
        "_math.py to interpolate along a trailing Ebv axis instead of "
        "assuming a scalar result."
    ),
    strict=True,
)
def test_synthetic_photometry_with_dust_extinction():

    # Generate a skygrid of points so that we can place a synthetic event
    # at each of them.
    skygrid = _geodesic.for_subdivision(21, 4, "icosahedron")
    n_events = len(skygrid)

    # Choose a photometry model. For this instance, we'll just use
    # a simple Van Vezlen + 2021 TDE model that has pre-set priors.
    SED_MODEL = VanVelzenTDESED()

    # Sample a set of parameters, one realization per event, from a fixed
    # seed for reproducibility. Each parameter gets a trailing axis so that
    # its per-event batch axis broadcasts against the (unbatched) wavelength
    # axis the spectrum is called with, rather than colliding with it.
    PARAMETERS = {
        name: value[:, np.newaxis]
        for name, value in SED_MODEL.sample_parameters(n_events, rng=0).items()
    }

    # Observe every event 10 days after explosion, at a single observing epoch.
    TIME_SINCE_EXPLOSION = 10 * u.day
    OBSTIME = Time("2025-01-01T00:00:00", scale="utc")

    # Every event sits at the same fiducial redshift for this test.
    REDSHIFT = 0.05

    # Create one batched SourceSpectrum, fixed at TIME_SINCE_EXPLOSION, ready
    # to feed into the detector -- still fully vectorized over all n_events
    # parameter realizations via ordinary NumPy broadcasting. This is the
    # observed, redshifted and distance-diluted flux (not the rest-frame
    # luminosity that `as_astropy_model` would give without distance
    # keywords), since that's what a detector actually sees. Multiply in
    # Milky Way dust extinction, looked up per sky position from the
    # `observing()` state below (see DustExtinctionForSkyCoord).
    spectra = SED_MODEL.as_source_spectrum(
        TIME_SINCE_EXPLOSION, redshift=REDSHIFT, **PARAMETERS
    )
    spectra = spectra * DustExtinction()

    with observing(
        uvex.observer_location(OBSTIME),
        skygrid,
        OBSTIME,
    ):
        SNR = uvex.detector.get_snr(900 * u.s, spectra, "FUV")

    assert np.all(SNR >= 0), "All events should have positive SNR in the FUV band."


def test_synthetic_photometry_with_batched_dust_extinction():
    """
    Per-event dust extinction via `as_source_spectrum`'s `log_attenuation`, fully batched.

    The workaround for `test_synthetic_photometry_with_dust_extinction`'s xfail:
    rather than multiplying a batched `SourceSpectrum` by a separate
    `DustExtinction()` `SpectralElement` (which routes through
    `m4opt.synphot._math.countrate`'s per-Ebv interpolation shortcut, and
    chokes on a per-event batch axis), `uvex_transient_toolkit.dust.log_attenuation`
    (bound to each event's own E(B-V) via `functools.partial`) is folded
    directly into the flux -- as plain NumPy arithmetic inside
    `as_source_spectrum`'s own evaluation kernel -- before the
    `SourceSpectrum` is ever built. That keeps the whole thing one ordinary
    broadcast, so it batches over events exactly as cleanly as every other
    parameter already does in `test_synthetic_photometry_batches_events_and_times`.
    """
    rng = np.random.default_rng(0)

    skygrid = _geodesic.for_subdivision(21, 4, "icosahedron")
    n_events = len(skygrid)

    model = VanVelzenTDESED()

    parameters = {
        name: value[:, np.newaxis]
        for name, value in model.sample_parameters(n_events, rng=rng).items()
    }
    # One E(B-V) per event. `dust.log_attenuation` derives its own output
    # shape from `Ebv`'s shape directly (no manually reserved trailing
    # axis needed here, unlike `parameters` above), so a flat (n_events,)
    # array already broadcasts correctly against wavelength.
    ebv = rng.uniform(0.0, 0.3, size=n_events)

    time_since_explosion = 10 * u.day
    obstime = Time("2025-01-01T00:00:00", scale="utc")
    redshift = 0.05

    spectra_with_dust = model.as_source_spectrum(
        time_since_explosion,
        redshift=redshift,
        log_attenuation=partial(log_attenuation, Ebv=ebv),
        **parameters,
    )
    spectra_without_dust = model.as_source_spectrum(
        time_since_explosion, redshift=redshift, **parameters
    )

    with observing(uvex.observer_location(obstime), skygrid, obstime):
        snr_with_dust = uvex.detector.get_snr(900 * u.s, spectra_with_dust, "FUV")
        snr_without_dust = uvex.detector.get_snr(
            900 * u.s, spectra_without_dust, "FUV"
        )

    assert snr_with_dust.shape == (n_events,)
    assert np.all(np.isfinite(snr_with_dust))
    assert np.all(snr_with_dust >= 0)
    # Every event has EBV > 0, so extinction should only ever suppress flux.
    assert np.all(snr_with_dust <= snr_without_dust)

    # Cross-check one event/wavelength slot against a manual, unbatched
    # computation, the same way `test_synthetic_photometry_batches_events_and_times`
    # confirms broadcasting actually lines up event slots with the right values.
    i_event, i_freq = 3, 5
    wave = uvex.detector.bandpasses["FUV"].waveset
    scalar_params = {name: value[i_event, 0] for name, value in parameters.items()}
    scalar_spectrum = VanVelzenTDESED().as_source_spectrum(
        time_since_explosion,
        redshift=redshift,
        log_attenuation=partial(log_attenuation, Ebv=ebv[i_event]),
        **scalar_params,
    )
    expected = scalar_spectrum(wave[i_freq])
    actual = spectra_with_dust(wave)[i_event, i_freq]
    np.testing.assert_allclose(actual.value, expected.value, rtol=1e-6)
