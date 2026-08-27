"""
Composite supernova SED models.
"""

__all__ = [
    "TopHatCCSNeSED",
    "TypeIbSNeSED",
    "TypeIcSNeSED",
    "TypeIIPSNeSED",
    "VillarCoolingBlackbodySED",
]

from ._tophat import TopHatCCSNeSED
from ._villar import TypeIbSNeSED, TypeIcSNeSED, TypeIIPSNeSED, VillarCoolingBlackbodySED
