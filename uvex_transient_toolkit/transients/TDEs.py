"""Tidal disruption event population."""

from typing import Union

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from uvex_transient_toolkit.models.tdes import VanVelzenTDESED
from .base import ExtragalacticTransient


class TidalDisruptionEvent(ExtragalacticTransient):
    DEFAULT_MODEL = VanVelzenTDESED
    DEFAULT_DURATION = 200 * u.day
    DEFAULT_Z_LIM = 1

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric event rate of tidal disruption events (TDEs) at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the event rate.

        Returns
        -------
        float or array-like
            The volumetric event rate of TDEs at the specified redshift(s), in units of events per cubic megaparsec per year.
        """
        # Coerce z to an array.
        z = np.asarray(z)

        # Yao+2023 value, taken as constant with redshift.
        rate = np.full_like(z, 3.1e-7, dtype=np.float64)

        return rate if z.ndim > 0 else rate.item()  # Return scalar if input was scalar.
