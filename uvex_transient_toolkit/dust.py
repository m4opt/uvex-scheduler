"""
Milky Way foreground dust extinction.

This module provides access to the PlanckGNILC E(B-V) map (`dust_map`), the adopted
Gordon+2023 (`G23`) reddening law, and a single vectorized entry point
(`log_attenuation`) that turns the two into a natural-log attenuation array ready to
add directly into a `~uvex_transient_toolkit.models.core._base.SpectralModel` flux calculation
(its ``log_attenuation`` keyword argument).

There is deliberately no `synphot.SpectralElement`/`astropy.modeling.Model` wrapping
here anymore: nothing in the `models` package needs a multiplicative transmission
curve as an object to hand elsewhere, only a plain array to add in log-space, so
building one is pure overhead. `dust_extinction.parameter_averages.G23` is still an
`astropy.modeling.Model` under the hood (that's `dust_extinction`'s own API, not
something this module controls), and it still enforces its native wavelength range by
*raising* `ValueError` if any input is out of range -- `log_attenuation` masks
out-of-range frequencies to `NaN` before calling it (comparisons against `NaN` are
always `False`, so `dust_extinction`'s own range check never sees them) purely to turn
that hard failure into a graceful per-element `NaN`, exactly as the code it replaces
did.
"""

from functools import cache
from typing import Protocol, Union, runtime_checkable

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.utils.data import download_file
from dust_extinction.parameter_averages import G23
from dustmaps.planck import PlanckGNILCQuery
from numpy.typing import NDArray


# =========================================================================== #
# TYPING MANAGEMENT                                                           #
# =========================================================================== #
@runtime_checkable
class DustMapLike(Protocol):
    """Structural type for anything queryable for E(B-V) at a sky position, like `PlanckGNILCQuery`."""

    def query(self, coord: SkyCoord) -> float: ...


Reddening = Union[float, Quantity, NDArray[np.float64], DustMapLike]
"""An explicit E(B-V) value/array (`float`/dimensionless `Quantity`/`numpy.ndarray`), or a
`DustMapLike` to query at a given `~astropy.coordinates.SkyCoord` -- see `resolve_ebv`."""


# =========================================================================== #
# DUST MAPS                                                                    #
# =========================================================================== #

# Same two mirrors (IRSA, then ESA as fallback) and the same `astropy` download
# cache that m4opt's own `m4opt.synphot.extinction._dust.dust_map` uses. Reusing
# the exact URLs means that if `m4opt prime` (or any other m4opt code path) has
# already warmed the astropy download cache for this file, we get a cache hit
# instead of a second multi-hundred-megabyte download -- even though this
# loader is otherwise fully independent of m4opt.
_GNILC_SOURCES = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/"
    "COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits",
    "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CompMap_Dust-GNILC-Model-Opacity_2048_R2.01.fits",
)


@cache
def dust_map() -> PlanckGNILCQuery:
    """
    Return the Planck GNILC E(B-V) sky map, downloading and caching it on first use.

    The map is fetched via :func:`astropy.utils.data.download_file` (cached to disk,
    with the ESA mirror as a fallback source), and the resulting :class:`PlanckGNILCQuery`
    is itself process-cached via :func:`functools.cache`, so repeated calls -- even across
    millions of per-event reddening lookups -- pay the download/load cost only once.

    Returns
    -------
    dustmaps.planck.PlanckGNILCQuery
        Queryable dust map: ``dust_map().query(coord)`` returns E(B-V) at ``coord``,
        vectorized over an array-valued ``coord``.
    """
    path = download_file(_GNILC_SOURCES[-1], cache=True, sources=list(_GNILC_SOURCES))
    return PlanckGNILCQuery(path)


# =========================================================================== #
# REDDENING LAW                                                               #
# =========================================================================== #
# To model the extinction for UVEX, we adopt the Gordon+2023 dust model, which covers
# IR through FUV and is suitable for our needs.
_reddening_law = G23()


# =========================================================================== #
# REDDENING RESOLUTION                                                       #
# =========================================================================== #
def resolve_ebv(reddening: Reddening, coord: SkyCoord | None = None) -> NDArray[np.float64]:
    """
    Resolve a `Reddening` value into a plain E(B-V) array.

    Parameters
    ----------
    reddening : float, ~astropy.units.Quantity, numpy.ndarray, or DustMapLike
        Either an explicit :math:`E(B-V)` value/array, or an object exposing
        ``.query(coord) -> E(B-V)`` (e.g. `dust_map`'s return value).
    coord : ~astropy.coordinates.SkyCoord, optional
        Sky position(s) to query, any shape -- required if ``reddening`` is a
        `DustMapLike`; ignored (and may be omitted) otherwise.

    Returns
    -------
    numpy.ndarray
        Dimensionless E(B-V), shape matching ``reddening`` (if already a value) or
        ``coord`` (if queried from a map).

    Raises
    ------
    TypeError
        If ``reddening`` is a `DustMapLike` and ``coord`` isn't given.
    """
    if isinstance(reddening, DustMapLike):
        if coord is None:
            raise TypeError("`coord` is required to resolve a `DustMapLike` `reddening`.")
        return np.asarray(reddening.query(coord), dtype=np.float64)

    if isinstance(reddening, Quantity):
        return np.asarray(reddening.to_value(u.dimensionless_unscaled), dtype=np.float64)

    return np.asarray(reddening, dtype=np.float64)


def log_attenuation(
    nu: Quantity,
    Ebv: float | Quantity | NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""
    Natural log of the Milky Way foreground attenuation, :math:`\ln(10^{-0.4\,A(\nu)})`, at ``nu``.

    Takes an already-resolved E(B-V) -- resolving one from a dust map and a sky
    position is a separate, prior step (`resolve_ebv`), deliberately kept out of
    this function: fixing ``Ebv``'s shape once, up front, is what lets this stay a
    single, fully vectorized computation (no per-position or per-frequency Python
    loop, and, unlike building a `~synphot.SpectralElement` per position, no
    repeated `~astropy.modeling.Model` construction either) -- the reddening law's
    dimensionless shape :math:`A(\nu)/A(V)` is evaluated once at every ``nu``
    sample (shape ``(K,)``), and combined with ``Ebv`` (shape ``S``) by plain numpy
    broadcasting into a result of shape ``S + (K,)``.

    Add this directly to a `~uvex_transient_toolkit.models.core._base.SpectralModel` flux
    calculation's log flux (its ``log_attenuation`` keyword) -- e.g., having already
    called ``ebv = resolve_ebv(dust_map(), coord)`` once,
    ``model.flux_log(nu, t, ..., log_attenuation=log_attenuation(nu, ebv))`` --
    rather than multiplying a linear transmission fraction in, since everything on
    that side is already computed in log space.

    `SpectralModel.as_source_spectrum` takes the same shape one level removed: bind
    ``Ebv`` with `functools.partial` first (its wavelength grid isn't known until
    the resulting `~synphot.SourceSpectrum` is actually called), e.g.
    ``model.as_source_spectrum(t, ..., log_attenuation=functools.partial(log_attenuation,
    Ebv=ebv))``.

    Parameters
    ----------
    nu : ~astropy.units.Quantity
        Frequency(ies) (or frequency-equivalent, e.g. wavelength) to evaluate the
        reddening law at, any shape ``(K,)`` (or scalar, in which case the trailing
        frequency axis below is squeezed away).
    Ebv : float, ~astropy.units.Quantity, or numpy.ndarray
        Already-resolved, dimensionless :math:`E(B-V)`, any shape ``S`` -- see
        `resolve_ebv` for turning a dust map + sky position into this.

    Returns
    -------
    numpy.ndarray
        Natural log of the dimensionless attenuation, ``NaN`` wherever ``nu`` falls
        outside `dust_extinction.parameter_averages.G23`'s native range (roughly 900
        Angstrom to 32 microns). Shape ``S + (K,)``, or plain ``S`` if ``nu`` was
        scalar.
    """
    Ebv = (
        np.asarray(Ebv.to_value(u.dimensionless_unscaled), dtype=np.float64)
        if isinstance(Ebv, Quantity)
        else np.asarray(Ebv, dtype=np.float64)
    )

    x = np.atleast_1d(np.asarray(nu.to_value(1 / u.micron, equivalencies=u.spectral()), dtype=np.float64))

    # `G23.__call__` (inherited from `dust_extinction.baseclasses.BaseExtModel`) raises
    # `ValueError` if *any* input falls outside its native range, rather than returning
    # `NaN` for just the offending samples -- so out-of-range `x` is masked out *before*
    # the call. It's tempting to pass `NaN` through for those samples and let the range
    # check's own `<=`/`>=` comparisons (always `False` against `NaN`) silently wave them
    # through -- the previous version of this module did exactly that -- but verify before
    # relying on it: empirically, G23's spline evaluates `NaN` in as ``0`` (zero
    # extinction) out, not `NaN` out, so that trick only avoids the exception, it doesn't
    # produce the documented `NaN`. This masks the *output* explicitly instead, using
    # `valid` computed from `x` directly, which is correct regardless of what G23 happens
    # to do internally with a `NaN` input.
    lo, hi = _reddening_law.x_range
    delta = 1e-6
    valid = (x > lo - delta) & (x < hi + delta)
    x_safe = np.where(valid, x, lo)

    axav = np.where(valid, np.asarray(_reddening_law(x_safe / u.micron), dtype=np.float64), np.nan)
    Av = _reddening_law.Rv.value * Ebv

    result = -0.4 * np.log(10.0) * Av[..., np.newaxis] * axav

    return result if nu.shape else result[..., 0]
