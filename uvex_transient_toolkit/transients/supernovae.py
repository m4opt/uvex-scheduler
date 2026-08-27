"""Core-collapse supernova population."""

from typing import Union

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from uvex_transient_toolkit.models.supernovae import TopHatCCSNeSED, TypeIbSNeSED, TypeIcSNeSED, TypeIIPSNeSED
from .base import ExtragalacticTransient

# Local, volume-limited subtype fractions of the *total* CC SNe rate below, from
# Li et al. 2011 (MNRAS 412, 1441, Table 9; approximate -- verify before treating as
# precise). Deliberately left un-renormalized: II-P + Ib + Ic sum to ~71.5%, not 100%,
# of `_core_collapse_rate` -- the remaining ~28.5% (IIb, IIn, IIL, Ic-BL, ...) isn't
# modeled by any class here yet, rather than being silently folded into these three.
_TYPE_IIP_FRACTION = 0.482
_TYPE_IB_FRACTION = 0.076
_TYPE_IC_FRACTION = 0.157


def _core_collapse_rate(z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    The total (all-subtype) volumetric core-collapse SNe rate at redshift(s) `z`.

    Shared by `CoreCollapseSNe.event_rate` and each subtype class's own
    `event_rate` below, so the per-subtype rates always stay a fixed fraction of
    the same underlying total rather than risking independent drift.

    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to evaluate the event rate.

    Returns
    -------
    float or array-like
        The volumetric event rate at the specified redshift(s), in events per cubic megaparsec per year.
    """
    z = np.asarray(z)

    # Maude & Dickenson coefficient * CC rate from LGS 2015. (sic -- verify citation before relying on it)
    _coefficient = 0.0001365
    rate = _coefficient * (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)

    return rate if z.ndim > 0 else rate.item()  # Return scalar if input was scalar.


class CoreCollapseSNe(ExtragalacticTransient):
    """
    A single, monolithic core-collapse SNe population: constant bolometric luminosity for
    a fixed duration, no Ib/Ic/II-P subtype distinction. See `TypeIbSNe`/`TypeIcSNe`/
    `TypeIIPSNe` below for the subtype-resolved alternative (a Villar rise/plateau/decline
    lightcurve times a cooling blackbody, with subtype-specific priors and rates) -- this
    class is kept alongside them as a simpler, faster fallback, not superseded by them.
    """

    DEFAULT_MODEL = TopHatCCSNeSED
    DEFAULT_DURATION = 10 * u.day

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric event rate of core-collapse supernovae (CCSNe) at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the event rate.

        Returns
        -------
        float or array-like
            The volumetric event rate of CCSNe at the specified redshift(s), in units of events per cubic megaparsec per year.
        """
        return _core_collapse_rate(z)


class TypeIbSNe(ExtragalacticTransient):
    """Type Ib core-collapse SNe: `TypeIbSNeSED` (Villar lightcurve x cooling blackbody), Li+2011 rate fraction."""

    DEFAULT_MODEL = TypeIbSNeSED
    DEFAULT_DURATION = 100 * u.day

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """Volumetric event rate: `_core_collapse_rate(z)` times the Type Ib fraction (see module docstring)."""
        return _TYPE_IB_FRACTION * _core_collapse_rate(z)


class TypeIcSNe(ExtragalacticTransient):
    """Type Ic core-collapse SNe: `TypeIcSNeSED` (Villar lightcurve x cooling blackbody), Li+2011 rate fraction."""

    DEFAULT_MODEL = TypeIcSNeSED
    DEFAULT_DURATION = 100 * u.day

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """Volumetric event rate: `_core_collapse_rate(z)` times the Type Ic fraction (see module docstring)."""
        return _TYPE_IC_FRACTION * _core_collapse_rate(z)


class TypeIIPSNe(ExtragalacticTransient):
    """Type II-P core-collapse SNe: `TypeIIPSNeSED` (Villar lightcurve x cooling blackbody), Li+2011 rate fraction."""

    DEFAULT_MODEL = TypeIIPSNeSED
    DEFAULT_DURATION = 200 * u.day

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """Volumetric event rate: `_core_collapse_rate(z)` times the Type II-P fraction (see module docstring)."""
        return _TYPE_IIP_FRACTION * _core_collapse_rate(z)
