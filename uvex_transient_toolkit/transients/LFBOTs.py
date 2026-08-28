"""Luminous fast blue optical transient (LFBOT) population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transient_toolkit.models.lfbots import LFBOTCoolingBlackbodySED

from .base import ExtragalacticTransient

# Constant (redshift-independent) volumetric rate, Perley et al. 2026 / Ho & Lu
# et al. 2026: 10 Gpc^-3 yr^-1.
_LFBOT_RATE: Quantity = 10 / (u.Gpc**3 * u.yr)


class LuminousFastBlueOpticalTransient(ExtragalacticTransient):
    """
    Luminous fast blue optical transient (LFBOT) population, e.g. AT2018cow-like events.

    Modeled with `LFBOTCoolingBlackbodySED` (a Gaussian-rise/power-law-decline
    light curve with a smoothly cooling blackbody photosphere), a constant
    volumetric rate of 10 Gpc^-3 yr^-1 out to z=4 (Perley et al. 2026; Ho & Lu
    et al. 2026), and a 100-day duration window -- generous relative to the
    SED's own rise/decline timescales, to safely bound the slowly fading
    power-law tail.
    """

    DEFAULT_MODEL = LFBOTCoolingBlackbodySED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 1

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric event rate of LFBOTs at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the event rate.

        Returns
        -------
        float or array-like
            The volumetric event rate of LFBOTs at the specified redshift(s), in units of
            events per cubic megaparsec per year. Constant in `z` (Perley et al. 2026;
            Ho & Lu et al. 2026).
        """
        z = np.asarray(z)

        rate = np.full_like(z, _LFBOT_RATE.to_value(u.Mpc**-3 * u.yr**-1), dtype=np.float64)

        return rate if z.ndim > 0 else rate.item()  # Return scalar if input was scalar.
