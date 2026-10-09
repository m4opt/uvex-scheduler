from dataclasses import dataclass
from typing import Literal

from astropy import units as u
from m4opt.constraints import ZodiacalBackgroundConstraint
from m4opt.missions import Mission, uvex
from m4opt.synphot import Detector
from regions import Regions, SkyRegion

mission = Mission(
    name=uvex.name,
    fov=uvex.fov,
    constraints=uvex.constraints,
    observer_location=uvex.observer_location,
    slew=uvex.slew,
    skygrid=uvex.skygrid,
    detector=Detector(
        plate_scale=uvex.detector.plate_scale,
        bandpasses=uvex.detector.bandpasses,
        background=uvex.detector.background,
        dark_noise=uvex.detector.dark_noise,
        read_noise=uvex.detector.read_noise,
        npix=uvex.detector.npix,
        aperture_correction=uvex.detector.aperture_correction,
        gain=uvex.detector.gain,
    ),
)
"""Mission configuration."""


@dataclass
class SurveyProgram:
    name: str
    """Name of the program"""

    region: SkyRegion | Regions | None
    """Sky region or collection of regions defining the project"""

    visits: int
    """Minimum number of visits"""

    mode: Literal["block", "field"]
    """Whether this program is observed by sky block or by field"""

    max_visits: int = -1
    """Maximum number of visits, or -1 to observe exactly the given number of visits"""

    min_cadence: u.Quantity[u.physical.time] = 0 * u.day
    """Minimum time between repeated visits"""


survey_programs = [
    SurveyProgram(
        name="allsky",
        region=None,
        visits=3,
        mode="block",
        min_cadence=2 * u.day,
    ),
    SurveyProgram(
        name="lmlz_wide",
        region=Regions.read("../survey-footprints/lmlz-wide.ds9"),
        visits=10,
        max_visits=50,
        mode="block",
        min_cadence=2 * u.day,
    ),
    SurveyProgram(
        name="lmlz_deep",
        region=Regions.read("../survey-footprints/lmlz-deep.ds9"),
        visits=85,
        mode="field",
    ),
    SurveyProgram(
        name="mc",
        region=Regions.read("../survey-footprints/magellanic-clouds.ds9"),
        visits=50,
        mode="block",
        min_cadence=14 * u.day,
    ),
]

extra_constraints = ZodiacalBackgroundConstraint(22.25)
"""Additional constraints applied to survey mode observations."""
