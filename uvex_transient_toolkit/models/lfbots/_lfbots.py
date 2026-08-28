r"""Luminous Fast Blue Optical Transient (LFBOT) SED model."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transient_toolkit.models._typing import CGSParameterValue, FloatArray
from uvex_transient_toolkit.models.core._base import SpectralModel
from uvex_transient_toolkit.models.core._parameters import Parameter
from uvex_transient_toolkit.models.core.priors import LogNormalPrior, NormalPrior
from uvex_transient_toolkit.models.lightcurves.generic import GaussianRisePowerLawLightcurve
from uvex_transient_toolkit.models.spectra.thermal import BlackbodySpectrum
from uvex_transient_toolkit.models._utils import to_cgs_value

__all__ = ["LFBOTCoolingBlackbodySED"]


class LFBOTCoolingBlackbodySED(SpectralModel):
    r"""
    A Gaussian-rise/power-law-decline light curve with a Villar-type cooling blackbody photosphere.

    This model pairs
    :class:`~uvex_transient_toolkit.models.lightcurves.generic.GaussianRisePowerLawLightcurve`
    with a time-dependent blackbody spectral shape,

    .. math::

        L_\nu(\nu, t)
        =
        L_\mathrm{bol}(t)\,
        S_\mathrm{BB}\!\left[\nu, T(t)\right],

    where :math:`S_\mathrm{BB}` is the normalized blackbody spectrum provided
    by :class:`~uvex_transient_toolkit.models.spectra.thermal.BlackbodySpectrum`.
    The photosphere follows the same cooling law as
    :class:`~uvex_transient_toolkit.models.supernovae.VillarCoolingBlackbodySED`,

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})(1 + t/\tau_T)^{-\alpha_T},

    which has the correct :math:`T \to T_\mathrm{floor} + (T_0 -
    T_\mathrm{floor})(t/\tau_T)^{-\alpha_T}` power-law asymptote at late times.

    Both this model's default priors and its choice of light curve/cooling-law
    shape are informed by Ho & Lu et al. (2026)'s sample of bolometric
    light curves and blackbody temperature evolution for known LFBOTs:

    - The bolometric light curve peaks (log-normally, with fairly small
      scatter) around 10 d and declines post-peak roughly as :math:`t^{-3}`
      -- some events (e.g. CSS161010) peak noticeably earlier, but most
      published light curves are caught only near or after peak, so the
      earlier rise is comparatively poorly constrained.
    - The blackbody temperature declines post-peak roughly as
      :math:`t^{-1/3}` for most events (e.g. AT2018cow); CSS161010 was
      closer to constant temperature -- both are within this cooling law's
      :math:`\alpha_T` range.
    - Peak bolometric luminosities span roughly :math:`10^{44}` erg/s
      (CSS161010) to a few :math:`\times 10^{45}` erg/s.

    Because the blackbody shape is normalized independently of luminosity,
    the bolometric and spectral components remain exactly separable:

    .. math::

        \int L_\nu(\nu,t)\,d\nu = L_\mathrm{bol}(t).

    Consequently, :meth:`_eval_bolometric` delegates directly to
    :class:`~uvex_transient_toolkit.models.lightcurves.generic.GaussianRisePowerLawLightcurve`,
    while :meth:`_eval_spectrum` evaluates
    :class:`~uvex_transient_toolkit.models.spectra.thermal.BlackbodySpectrum` at
    the time-dependent temperature :math:`T(t)`. No numerical frequency
    integration is required to recover the bolometric luminosity.

    .. rubric:: Parameters

    The model parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity since explosion.
       * - ``sigma_rise``
         - :math:`\sigma_\mathrm{rise}`
         - Gaussian width of the rise.
       * - ``decline_index``
         - :math:`\alpha_\mathrm{decline}`
         - Positive post-peak power-law decline index of the bolometric
           light curve.
       * - ``T0``
         - :math:`T_0`
         - Photospheric temperature at t=0 (the T(t) -> T0 limit, not
           literally T at peak).
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Asymptotic late-time photospheric temperature (T(t) -> T_floor
           as t -> infinity).
       * - ``tau_T``
         - :math:`\tau_T`
         - Photospheric cooling timescale.
       * - ``alpha_T``
         - :math:`\alpha_T`
         - Late-time photospheric cooling power-law index.

    Notes
    -----
    ``T0``/``T_floor`` are broad, phenomenological starting points -- unlike
    ``t_peak``/``decline_index``/``alpha_T``, Ho & Lu et al. (2026) does not
    report characteristic absolute temperature values for this sample.

    The temperature law assumes :math:`T_0 > T_\mathrm{floor}` for a cooling
    photosphere, although this ordering is not enforced by the model itself.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=44.65, sigma=0.35),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Peak bolometric luminosity. log10(L_0/[erg/s]) ~ N(44.65, 0.35^2), "
            "spanning the ~1e44 (CSS161010) to a few 1e45 erg/s range reported by Ho & Lu et al. 2026.",
            latex=r"L_0",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.2),
            scale=10.0 * u.day,
            description="Time of peak bolometric luminosity since explosion. Log-normal around "
            "10 d with fairly small scatter (Ho & Lu et al. 2026).",
            latex=r"t_\mathrm{peak}",
        ),
        "sigma_rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=2.0 * u.day,
            description="Gaussian width of the pre-peak rise.",
            latex=r"\sigma_\mathrm{rise}",
        ),
        "decline_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.15),
            scale=3.0 * u.dimensionless_unscaled,
            description="Positive post-peak power-law decline index; L_bol ~ t^-decline_index. "
            "Late-time light curves are broadly consistent with a t^-3 decline (Ho & Lu et al. 2026).",
            latex=r"\alpha_\mathrm{decline}",
        ),
        "T0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.1),
            scale=2e4 * u.K,
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.1),
            scale=1.2e4 * u.K,
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=5.0 * u.day,
            description="Photospheric cooling timescale.",
            latex=r"\tau_T",
        ),
        "alpha_T": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=1.0 / 3.0,
            description="Late-time photospheric cooling power-law index; T ~ t^-alpha_T for t >> tau_T. "
            "~1/3 is a reasonable fit for most events (e.g. AT2018cow); CSS161010 was closer to "
            "constant temperature (Ho & Lu et al. 2026).",
            latex=r"\alpha_T",
        ),
    }

    # -------------------------------------- #
    # Cooling Law: T(t)                       #
    # -------------------------------------- #
    @classmethod
    def _temperature_cgs(
        cls,
        t: FloatArray,
        *,
        T0: CGSParameterValue,
        T_floor: CGSParameterValue,
        tau_T: CGSParameterValue,
        alpha_T: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r""":math:`T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})(1 + t/\tau_T)^{-\alpha_T}`."""
        return T_floor + (T0 - T_floor) * (1.0 + t / tau_T) ** (-alpha_T)

    @classmethod
    def temperature(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`T(t)` in Kelvin."""
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls._temperature_cgs(t.cgs.value, **cgs_parameters) * u.K

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log L_\mathrm{bol}(t)`, delegated directly to :class:`GaussianRisePowerLawLightcurve`. Exact -- no integration needed."""
        lightcurve_parameters = {
            name: parameters[name] for name in GaussianRisePowerLawLightcurve._DEFAULT_PARAMETERS
        }
        return GaussianRisePowerLawLightcurve._eval(t, **lightcurve_parameters)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log S(\nu, T(t))`, delegated to :class:`BlackbodySpectrum` at this ``t``'s own cooling-law temperature."""
        temperature = cls._temperature_cgs(t, **parameters)
        return BlackbodySpectrum._eval(nu, temperature=temperature)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu, T(t))`."""
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)
