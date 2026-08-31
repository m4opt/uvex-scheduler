"""Survey-program definitions shared by ``skyblocks.ipynb`` (and, via the ``fields.ecsv``/
``blocks.ecsv`` tables it produces, ``main.ipynb``).

A :class:`SurveyProgram` describes one named component of the UVEX survey (e.g. the all-sky base
cadence, LMLZ Wide, LMLZ Deep, the Magellanic Clouds): the sky region it covers, how many visits
it needs, and how those visits should be scheduled (see ``mode`` below). The actual list of
programs (``SURVEY_PROGRAMS``) is defined directly in ``skyblocks.ipynb``, immediately after its
settings/configuration cell, so it stays easy to find and edit without touching this file.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from regions import Regions, SkyRegion

__all__ = [
    "SurveyProgram",
    "compute_visits_to_depth",
    "compute_depth_to_visits",
]


def compute_visits_to_depth(
    required_depth: tuple[float, float], single_visit_depth: tuple[float, float]
) -> int:
    """
    Compute the number of visits required to reach a specified depth given a single-visit depth.

    Parameters
    ----------
    required_depth : tuple
        The required depths for the NUV and FUV bands, in AB magnitudes.
    single_visit_depth : tuple
        The depths achieved in a single visit for the NUV and FUV bands, in AB magnitudes.

    Returns
    -------
    int
        The number of visits required to reach the specified depth.
    """
    # Background-limited co-addition: SNR (and thus limiting flux) scales as sqrt(N_visits), so
    # the depth gain per decade of visits is 2.5 * log10(sqrt(N)) = 1.25 * log10(N) mag.
    visits_nuv = np.ceil(10 ** ((required_depth[0] - single_visit_depth[0]) / 1.25))
    visits_fuv = np.ceil(10 ** ((required_depth[1] - single_visit_depth[1]) / 1.25))

    # NUV and FUV are imaged simultaneously every dwell, so the harder-to-reach band sets the
    # visit count for both.
    return int(max(visits_nuv, visits_fuv))


def compute_depth_to_visits(
    n_visits: int, single_visit_depth: tuple[float, float]
) -> tuple[float, float]:
    """
    Compute the depth reached after a given number of visits, given a single-visit depth.

    Inverse of :func:`compute_visits_to_depth`: useful for a program that specifies its visit
    count directly (e.g. a fixed cadence) when you want to know what depth that cadence actually
    buys.

    Parameters
    ----------
    n_visits : int
        The number of visits.
    single_visit_depth : tuple
        The depths achieved in a single visit for the NUV and FUV bands, in AB magnitudes.

    Returns
    -------
    tuple
        The depths reached in the NUV and FUV bands, in AB magnitudes, after ``n_visits``.
    """
    # Inverse of the co-addition scaling in compute_visits_to_depth: depth gain per decade of
    # visits is 1.25 * log10(N) mag.
    gain = 1.25 * np.log10(n_visits)
    return (single_visit_depth[0] + gain, single_visit_depth[1] + gain)


@dataclass
class SurveyProgram:
    """
    One named component of the survey: a sky region, how many visits it needs, and how those
    visits should be scheduled.
    """

    name: str
    """Short, lowercase identifier for the program (e.g. ``"lmlz_deep"``). Used to build column
    names (``IN_<NAME>``, ``OVERLAP_<NAME>``) and the resolved ``CATEGORY`` value on ``FIELDS``,
    so it should be a valid, upper-case-able identifier."""

    region: Regions | SkyRegion | None
    """Sky region (or collection of regions) defining the program's footprint. ``None`` means
    "the rest of the sky" -- the all-sky base program that every field not claimed by a named
    region falls back to."""

    mode: Literal["block", "field", "mc"]
    """How this program's fields are scheduled:

    - ``"block"``: tiled together with spatial neighbors into ordinary multi-field blocks by the
      partitioner in ``skyblocks.ipynb``.
    - ``"field"``: each field gets its own dedicated single-field block, revisited repeatedly --
      see ``skyblocks.ipynb``'s "Field-Mode Blocks" section.
    - ``"mc"``: the Magellanic Clouds -- fields still end up in an ordinary block (like
      ``"block"``), but membership isn't a simple overlap threshold. Because MC is the tightest
      angular cluster in the survey, ``skyblocks.ipynb``'s "Determine If a Field is in a Region"
      loop instead ranks fields by decreasing overlap and searches for the largest number that
      provably fit -- via a real TSP solve -- within a single block's available time.
    """

    visits: int | None = None
    """Directly-specified number of visits (e.g. a fixed-cadence program like the all-sky base or
    the Magellanic Clouds). Mutually exclusive with ``required_depth``."""

    required_depth: tuple[float, float] | None = None
    """(NUV, FUV) limiting AB magnitude this program must reach; the number of visits needed is
    derived from the mission's single-visit depth via :func:`compute_visits_to_depth`. Mutually
    exclusive with ``visits``."""

    overlap_threshold: float | None = None
    """Minimum fractional footprint overlap with ``region`` for a field to be considered a member
    of this program. ``None`` means any overlap at all counts, appropriate for a point-source
    region (e.g. LMLZ Deep's individual high-priority targets), where "overlap fraction" isn't a
    meaningful quantity."""

    def get_visits(self, single_visit_depth: tuple[float, float]) -> int:
        """Resolve this program's required number of visits."""
        if self.visits is not None:
            return self.visits
        if self.required_depth is not None:
            return compute_visits_to_depth(self.required_depth, single_visit_depth)
        raise ValueError(
            f"SurveyProgram {self.name!r} specifies neither `visits` nor `required_depth`."
        )
