r"""Supernova SED models from :footcite:t:`2019ApJ...884...83V`."""

from typing import ClassVar

from astropy import units as u

from uvex_transient_toolkit.models._typing import CGSParameterValue, FloatArray
from uvex_transient_toolkit.models.core._base import SpectralModel
from uvex_transient_toolkit.models.core._parameters import Parameter
from uvex_transient_toolkit.models.core.priors import LogNormalPrior, NormalPrior, UniformPrior
from uvex_transient_toolkit.models.lightcurves.generic import VillarLightcurve
from uvex_transient_toolkit.models.spectra.thermal import BlackbodySpectrum

__all__ = [
    "TypeIbSNeSED",
    "TypeIcSNeSED",
    "TypeIIPSNeSED",
    "VillarCoolingBlackbodySED",
]


class VillarCoolingBlackbodySED(SpectralModel):
    r"""
    Supernova light curve with an evolving blackbody photosphere.

    This model combines the phenomenological bolometric light curve of
    :class:`~uvex_transient_toolkit.models.lightcurves.generic.VillarLightcurve` (the
    parametric form introduced by :footcite:p:`2019ApJ...884...83V`) with a
    time-dependent blackbody spectral shape,

    .. math::

        L_\nu(\nu, t)
        =
        L_\mathrm{bol}(t)\,
        S_\mathrm{BB}\!\left[\nu, T(t)\right],

    where :math:`S_\mathrm{BB}` is the normalized blackbody spectrum provided
    by :class:`~uvex_transient_toolkit.models.spectra.thermal.BlackbodySpectrum`.

    The photospheric temperature evolves according to

    .. math::

        T(t)
        =
        T_\mathrm{floor}
        +
        \left(T_0-T_\mathrm{floor}\right)
        \left(1+\frac{t}{\tau_T}\right)^{-\alpha_T}.

    Thus, the temperature begins at :math:`T_0` at :math:`t=0` and approaches
    :math:`T_\mathrm{floor}` asymptotically at late times. The parameters
    :math:`\tau_T` and :math:`\alpha_T` control the characteristic cooling
    timescale and the rate of the decline, respectively.

    Because the blackbody shape is normalized independently of luminosity,
    the bolometric and spectral components remain exactly separable:

    .. math::

        \int L_\nu(\nu,t)\,d\nu = L_\mathrm{bol}(t).

    Consequently, :meth:`_eval_bolometric` delegates directly to
    :class:`~uvex_transient_toolkit.models.lightcurves.generic.VillarLightcurve`, while
    :meth:`_eval_spectrum` evaluates
    :class:`~uvex_transient_toolkit.models.spectra.thermal.BlackbodySpectrum` at the
    time-dependent temperature :math:`T(t)`. No numerical frequency
    integration is required to recover the bolometric luminosity.

    The model is intended as a generic phenomenological description of
    supernova-like transients whose continuum can be approximated by a cooling
    photosphere. It does not attempt to model photospheric-radius evolution,
    line blanketing, recombination physics, nebular emission, or detailed
    radiative transfer.

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
         - Overall luminosity normalization.
       * - ``t0``
         - :math:`t_0`
         - Reference time at which the logistic rise is centered.
       * - ``gamma``
         - :math:`\gamma`
         - Duration of the plateau, measured from t0.
       * - ``beta``
         - :math:`\beta`
         - Linear slope of the plateau.
       * - ``tau_rise``
         - :math:`\tau_\mathrm{rise}`
         - Logistic rise timescale.
       * - ``tau_fall``
         - :math:`\tau_\mathrm{fall}`
         - Exponential decline timescale, after the plateau.
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
         - Photospheric cooling power-law index.

    Notes
    -----
    The default parameter distributions are broad, phenomenological starting
    points rather than subtype-specific empirical priors.

    Subclasses may define only the entries of :attr:`_DEFAULT_PARAMETERS` they
    wish to modify. Their parameter definitions are merged with the complete
    parameter set of ``VillarCoolingBlackbodySED`` during subclass creation.
    This makes it straightforward to define supernova-subtype variants by
    changing only the parameters whose distributions differ between
    populations.

    The temperature law assumes :math:`T_0 > T_\mathrm{floor}` for a cooling
    photosphere, although this ordering is not enforced by the model itself.

    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **VillarLightcurve._DEFAULT_PARAMETERS,
        "T0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=1.2e4 * u.K,
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=6e3 * u.K,
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=15.0 * u.day,
            description="Photospheric cooling timescale.",
            latex=r"\tau_T",
        ),
        "alpha_T": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=1.5,
            description="Photospheric cooling power-law index.",
            latex=r"\alpha_T",
        ),
    }

    # -------------------------------------- #
    # Subclass Merging                        #
    # -------------------------------------- #
    def __init_subclass__(cls, **kwargs) -> None:
        """Merge a subtype subclass's own `_DEFAULT_PARAMETERS` on top of this class's full parameter set."""
        super().__init_subclass__(**kwargs)

        own = cls.__dict__.get("_DEFAULT_PARAMETERS")
        if own is not None:
            cls._DEFAULT_PARAMETERS = {
                **VillarCoolingBlackbodySED._DEFAULT_PARAMETERS,
                **own,
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

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(
        cls, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r""":math:`\log L_\mathrm{bol}(t)`, delegated directly to :class:`VillarLightcurve`. Exact -- no integration needed."""
        lightcurve_parameters = {
            name: parameters[name] for name in VillarLightcurve._DEFAULT_PARAMETERS
        }
        return VillarLightcurve._eval(t, **lightcurve_parameters)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(
        cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r""":math:`\log S(\nu, T(t))`, delegated to :class:`BlackbodySpectrum` at this ``t``'s own cooling-law temperature."""
        temperature = cls._temperature_cgs(t, **parameters)
        return BlackbodySpectrum._eval(nu, temperature=temperature)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(
        cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r""":math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu, T(t))`."""
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(
            nu, t, **parameters
        )


class TypeIbSNeSED(VillarCoolingBlackbodySED):
    """
    Type Ib SNe: a Villar rise/short-or-no-plateau/decline lightcurve, cooling from a hot,
    line-blanketing-free-by-assumption photosphere with no H-recombination floor.

    The ``tau_rise``/``gamma``/``beta`` ranges are the starting simulation ranges
    (not yet literature-calibrated population priors) from the survey's own
    design discussion: rise ~10-25 d, little-to-no plateau (0-20 d), and a
    post-peak slope steep enough that the lightcurve is peaked rather than
    flat-topped. ``amplitude``, ``T0``, ``T_floor``, ``tau_T`` are order-of-magnitude
    literature starting points (stripped-envelope SN peak bolometric
    luminosities and photospheric temperatures are broadly ~10^42-10^42.5 erg/s
    and ~10-13 kK; e.g. Drout et al. 2011, Lyman et al. 2016) -- refine once
    real samples are fit, exactly as with the timing parameters.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=42.3, sigma=0.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Overall luminosity normalization. log10(amplitude/[erg/s]) ~ N(42.3, 0.3^2).",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=UniformPrior(10.0, 25.0),
            scale=1.0 * u.day,
            description="Approximate peak epoch; same starting range as tau_rise.",
            latex=r"t_0",
        ),
        "tau_rise": Parameter(
            prior=UniformPrior(10.0, 25.0),
            scale=1.0 * u.day,
            description="Rise timescale, 10-25 d starting range.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "beta": Parameter(
            prior=UniformPrior(5e-3, 3e-2),
            scale=1.0 / u.day,
            description="Post-peak slope, 0.005-0.03/d starting range (beta*gamma < 1 for gamma <= 20 d).",
            latex=r"\beta",
        ),
        "gamma": Parameter(
            prior=UniformPrior(0.0, 20.0),
            scale=1.0 * u.day,
            description="Little-to-no plateau, 0-20 d starting range.",
            latex=r"\gamma",
        ),
        "tau_fall": Parameter(
            prior=UniformPrior(30.0, 80.0),
            scale=1.0 * u.day,
            description="Radioactive-tail decline timescale, 30-80 d starting range.",
            latex=r"\tau_\mathrm{fall}",
        ),
        "T0": Parameter(
            prior=NormalPrior(mean=4.10, sigma=0.05),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0. log10(T0/K) ~ N(4.10, 0.05^2), ~12.6 kK.",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=NormalPrior(mean=3.85, sigma=0.05),
            scale=1.0 * u.K,
            transform="log10",
            description="Asymptotic late-time temperature; no H-recombination floor, so hotter/less "
            "sharp than II-P. log10(T_floor/K) ~ N(3.85, 0.05^2), ~7.1 kK.",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=NormalPrior(mean=1.0, sigma=0.2),
            scale=1.0 * u.day,
            transform="log10",
            description="Cooling timescale. log10(tau_T/day) ~ N(1.0, 0.2^2), ~10 d.",
            latex=r"\tau_T",
        ),
    }


class TypeIcSNeSED(VillarCoolingBlackbodySED):
    """
    Type Ic SNe: same rise/plateau/decline and cooling-law family as `TypeIbSNeSED` --
    stripped-envelope SNe photometrically overlap Ib substantially, differing mainly
    spectroscopically (He absence) -- but on average somewhat more luminous and slightly
    hotter (Lyman et al. 2016, Taddia et al. 2018).
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=42.5, sigma=0.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Overall luminosity normalization. log10(amplitude/[erg/s]) ~ N(42.5, 0.3^2).",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=UniformPrior(10.0, 25.0),
            scale=1.0 * u.day,
            description="Approximate peak epoch; same starting range as tau_rise.",
            latex=r"t_0",
        ),
        "tau_rise": Parameter(
            prior=UniformPrior(10.0, 25.0),
            scale=1.0 * u.day,
            description="Rise timescale, 10-25 d starting range.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "beta": Parameter(
            prior=UniformPrior(5e-3, 3e-2),
            scale=1.0 / u.day,
            description="Post-peak slope, 0.005-0.03/d starting range (beta*gamma < 1 for gamma <= 20 d).",
            latex=r"\beta",
        ),
        "gamma": Parameter(
            prior=UniformPrior(0.0, 20.0),
            scale=1.0 * u.day,
            description="Little-to-no plateau, 0-20 d starting range.",
            latex=r"\gamma",
        ),
        "tau_fall": Parameter(
            prior=UniformPrior(30.0, 80.0),
            scale=1.0 * u.day,
            description="Radioactive-tail decline timescale, 30-80 d starting range.",
            latex=r"\tau_\mathrm{fall}",
        ),
        "T0": Parameter(
            prior=NormalPrior(mean=4.12, sigma=0.05),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0. log10(T0/K) ~ N(4.12, 0.05^2), ~13.2 kK.",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=NormalPrior(mean=3.85, sigma=0.05),
            scale=1.0 * u.K,
            transform="log10",
            description="Asymptotic late-time temperature; no H-recombination floor. "
            "log10(T_floor/K) ~ N(3.85, 0.05^2), ~7.1 kK.",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=NormalPrior(mean=1.0, sigma=0.2),
            scale=1.0 * u.day,
            transform="log10",
            description="Cooling timescale. log10(tau_T/day) ~ N(1.0, 0.2^2), ~10 d.",
            latex=r"\tau_T",
        ),
    }


class TypeIIPSNeSED(VillarCoolingBlackbodySED):
    """
    Type II-P SNe: a long, near-flat plateau (large ``gamma``, ``beta`` ~ 0) followed by a
    radioactive-tail decline, cooling toward the ~5-6 kK H-recombination temperature that
    sets the plateau's characteristic photospheric floor.

    Bolometrically the plateau is close to flat by construction (small ``beta``), but the
    photosphere keeps cooling underneath it -- so the *UV* flux (this SED's whole point)
    keeps fading throughout the plateau even while ``L_bol(t)`` looks constant; see
    :class:`VillarCoolingBlackbodySED`.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=42.0, sigma=0.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Overall luminosity normalization. log10(amplitude/[erg/s]) ~ N(42.0, 0.3^2).",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=UniformPrior(5.0, 20.0),
            scale=1.0 * u.day,
            description="Approximate peak epoch; same starting range as tau_rise.",
            latex=r"t_0",
        ),
        "tau_rise": Parameter(
            prior=UniformPrior(5.0, 20.0),
            scale=1.0 * u.day,
            description="Rise timescale, 5-20 d starting range.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "beta": Parameter(
            prior=UniformPrior(-2e-3, 2e-3),
            scale=1.0 / u.day,
            description="Near-flat plateau slope, +-0.002/d starting range (beta*gamma < 1 for gamma <= 100 d).",
            latex=r"\beta",
        ),
        "gamma": Parameter(
            prior=UniformPrior(70.0, 100.0),
            scale=1.0 * u.day,
            description="Plateau duration, 70-100 d starting range.",
            latex=r"\gamma",
        ),
        "tau_fall": Parameter(
            prior=UniformPrior(50.0, 120.0),
            scale=1.0 * u.day,
            description="Post-plateau radioactive-tail decline timescale, 50-120 d starting range.",
            latex=r"\tau_\mathrm{fall}",
        ),
        "T0": Parameter(
            prior=NormalPrior(mean=4.00, sigma=0.05),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0. log10(T0/K) ~ N(4.00, 0.05^2), ~10 kK.",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=NormalPrior(mean=3.75, sigma=0.03),
            scale=1.0 * u.K,
            transform="log10",
            description="H-recombination photospheric floor. log10(T_floor/K) ~ N(3.75, 0.03^2), ~5.6 kK.",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=NormalPrior(mean=1.2, sigma=0.2),
            scale=1.0 * u.day,
            transform="log10",
            description="Cooling timescale. log10(tau_T/day) ~ N(1.2, 0.2^2), ~16 d.",
            latex=r"\tau_T",
        ),
    }
