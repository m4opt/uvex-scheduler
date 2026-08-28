from dataclasses import dataclass
from typing import Literal

from regions import Regions, SkyRegion


@dataclass
class SurveyProgram:
    name: str
    """Name of the program"""

    region: SkyRegion | Regions | None
    """Sky region or collection of regions defining the project"""

    visits: int
    """Required number of visits"""

    mode: Literal["block", "field"]
    """Whether this program is observed by sky block or by field"""


survey_programs = [
    SurveyProgram(name="allsky", region=None, visits=3, mode="block"),
    SurveyProgram(
        name="lmlz_wide",
        region=Regions.read("../survey-footprints/lmlz-wide.ds9"),
        visits=10,
        mode="block",
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
        visits=52,
        mode="block",
    ),
]
