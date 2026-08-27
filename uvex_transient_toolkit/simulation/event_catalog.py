"""
A Monte Carlo realization of sampled transient events, produced by `SurveySimulator`.

`EventCatalog` is a pure data table -- it holds no live references to the
`~uvex_transients.surveys.base.SurveySchedule` or transient-type instances it was
generated against (see :meth:`EventCatalog.get_events`, which takes those in
explicitly rather than storing them), so it round-trips to/from disk cleanly and
stays trivially picklable/shareable on its own.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity

from ..surveys.base import SurveySchedule
from ..transients.base import TransientBase
from .event import Event

_SeedType = Union[np.random.SeedSequence, int, None]


def _seed_to_meta(seed: _SeedType) -> int | None:
    """Reduce a root seed to something ECSV-header-serializable, for `EventCatalog.to_disk`."""
    if seed is None or isinstance(seed, (int, np.integer)):
        return None if seed is None else int(seed)

    if isinstance(seed, np.random.SeedSequence):
        entropy = seed.entropy
        return int(entropy) if isinstance(entropy, int) else None

    return None


@dataclass
class EventCatalog:
    """
    A Monte Carlo realization of sampled transient events, tagged by type and time bin.

    Produced by :meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events`.
    Each event carries a ``parameter_seed`` rather than sampled physical SED parameters
    (see :meth:`~uvex_transients.transients.base.ExtragalacticTransient.sample_events_on_healpix_grid`)
    -- those are regenerated lazily, per event, by :meth:`Event.sample_parameters`/
    :meth:`Event.simulate_photometry` once an event is reconstructed via :meth:`get_events`.
    """

    table: QTable
    """QTable: One row per sampled event.

    Columns are those of ``sample_events_on_healpix_grid``'s own event table (``healpix_id``,
    ``healpix_dx``, ``healpix_dy``, ``coord``, ``redshift``, ``t_explosion``,
    ``parameter_seed``), plus ``transient_type`` (str), ``time_bin`` (the index into
    :attr:`time_bins` during which the event exploded), a unique ``event_id``, and two
    columns computed once at generation time so they never need re-deriving per event:
    ``luminosity_distance`` (from the event's own transient type's cached
    :attr:`~uvex_transients.transients.base.ExtragalacticTransient.luminosity_distance_grid`)
    and ``ebv`` (Milky Way foreground E(B-V) at the event's exact position).
    """

    nside: int
    """int: HEALPix resolution used both to query the observed footprint and to sample events."""

    order: str
    """str: HEALPix pixel ordering scheme (``"nested"`` or ``"ring"``)."""

    time_bins: Time
    """~astropy.time.Time: The ``n + 1`` bin edges events were sampled within."""

    seed: _SeedType = None
    """numpy.random.SeedSequence, int, or None: The root seed this catalog was generated from."""

    # ----------------------------------------- #
    # Dunder Methods                            #
    # ----------------------------------------- #
    def __len__(self) -> int:
        return len(self.table)

    # ----------------------------------------- #
    # Array Export                              #
    # ----------------------------------------- #
    def column(self, name: str) -> np.ndarray:
        """
        A table column as a plain array, rather than an `astropy.table.Column`/`Quantity` view.

        Parameters
        ----------
        name : str
            Column name; must be one of `table.colnames`.

        Returns
        -------
        numpy.ndarray or ~astropy.units.Quantity
            `numpy.asarray` of the column, preserving units if the column is a
            `Quantity` column (e.g. ``luminosity_distance``).
        """
        if name not in self.table.colnames:
            raise KeyError(
                f"No column {name!r} in this catalog; available: {self.table.colnames}."
            )

        col = self.table[name]
        return col if isinstance(col, (Quantity, Time, SkyCoord)) else np.asarray(col)

    @property
    def event_id(self) -> np.ndarray:
        """numpy.ndarray: Every event's unique id, shape ``(n_events,)``."""
        return self.column("event_id")

    @property
    def healpix_id(self) -> np.ndarray:
        """numpy.ndarray: Every event's HEALPix pixel index, at this catalog's own :attr:`nside`/:attr:`order`."""
        return self.column("healpix_id")

    @property
    def transient_type(self) -> np.ndarray:
        """numpy.ndarray of str: Every event's transient-type name."""
        return np.asarray(self.table["transient_type"]).astype(str)

    @property
    def time_bin(self) -> np.ndarray:
        """numpy.ndarray: The time-bin index each event was sampled within."""
        return self.column("time_bin")

    @property
    def coord(self) -> SkyCoord:
        """~astropy.coordinates.SkyCoord: Every event's sky position."""
        return self.column("coord")

    @property
    def redshift(self) -> np.ndarray:
        """numpy.ndarray: Every event's redshift."""
        return self.column("redshift")

    @property
    def luminosity_distance(self) -> Quantity:
        """~astropy.units.Quantity: Every event's luminosity distance (cached at generation time)."""
        return self.column("luminosity_distance")

    @property
    def ebv(self) -> np.ndarray:
        """numpy.ndarray: Every event's Milky Way foreground E(B-V) (cached at generation time)."""
        return self.column("ebv")

    @property
    def t_explosion(self) -> Time:
        """~astropy.time.Time: Every event's explosion time."""
        return self.column("t_explosion")

    @property
    def parameter_seed(self) -> np.ndarray:
        """numpy.ndarray: Every event's stored SED parameter seed."""
        return self.column("parameter_seed")

    # ----------------------------------------- #
    # Event Reconstruction                      #
    # ----------------------------------------- #
    def get_events(
        self,
        ids: int | np.ndarray | list,
        transients: dict[str, TransientBase],
        schedule: SurveySchedule,
    ) -> Event | list[Event]:
        """
        Reconstruct one or more `Event` objects from their `event_id`.

        `EventCatalog` itself holds no live reference to `transients`/`schedule` (see
        the module docstring), so both must be supplied here -- typically the same
        `dict`/`SurveySchedule` the catalog was generated from.

        Parameters
        ----------
        ids : int or array_like of int
            One event id, or several. A scalar `ids` returns a single `Event`; anything
            array-like returns a `list` of `Event`, in the order given.
        transients : dict[str, TransientBase]
            Transient-type instances, keyed by the same names used in the ``transient_type``
            column (e.g. `SurveySimulator.transient_collection`).
        schedule : ~uvex_transients.surveys.base.SurveySchedule
            The survey schedule to check each reconstructed event's visibility against.

        Returns
        -------
        Event or list[Event]

        Raises
        ------
        KeyError
            If an id isn't present in this catalog, or its ``transient_type`` isn't a key
            of `transients`.
        """
        scalar = np.ndim(ids) == 0
        id_array = np.atleast_1d(np.asarray(ids, dtype=np.int64))

        table = self.table
        event_id_col = np.asarray(table["event_id"])
        has_distance = "luminosity_distance" in table.colnames
        has_ebv = "ebv" in table.colnames

        events: list[Event] = []
        for eid in id_array:
            matches = np.flatnonzero(event_id_col == eid)
            if matches.size == 0:
                raise KeyError(f"No event with id {int(eid)!r} in this catalog.")

            row = table[int(matches[0])]
            name = str(row["transient_type"])
            if name not in transients:
                raise KeyError(
                    f"No transient type {name!r} in 'transients'; available: {list(transients)}."
                )

            events.append(
                Event(
                    event_id=int(row["event_id"]),
                    schedule=schedule,
                    transient=transients[name],
                    coord=row["coord"],
                    redshift=float(row["redshift"]),
                    t_explosion=row["t_explosion"],
                    seed=int(row["parameter_seed"]),
                    luminosity_distance=row["luminosity_distance"]
                    if has_distance
                    else None,
                    ebv=float(row["ebv"]) if has_ebv else None,
                    transient_type=name,
                )
            )

        return events[0] if scalar else events

    # ----------------------------------------- #
    # IO Methods                                #
    # ----------------------------------------- #
    def to_disk(
        self, path: str | Path, table_format: str | None = None, overwrite: bool = False
    ) -> None:
        """
        Write this catalog's event table to disk as ECSV, with its provenance in the header.

        Parameters
        ----------
        path
            Destination path.
        table_format
            Passed through to :meth:`~astropy.table.QTable.write`; if `None`, inferred from
            ``path``'s suffix.
        overwrite
            Whether to overwrite an existing file at ``path``.
        """
        table = self.table.copy()
        table.meta.update(
            {
                "nside": int(self.nside),
                "order": self.order,
                "time_bins": self.time_bins,
                "seed": _seed_to_meta(self.seed),
            }
        )
        table.write(Path(path), format=table_format, overwrite=overwrite)

    @classmethod
    def from_disk(
        cls, path: str | Path, table_format: str | None = None
    ) -> "EventCatalog":
        """
        Read an event catalog back from disk, as written by :meth:`to_disk`.

        Parameters
        ----------
        path
            Path to the event table, as written by :meth:`to_disk`.
        table_format
            Passed through to :meth:`~astropy.table.QTable.read`; if `None`, inferred from
            ``path``'s suffix.

        Returns
        -------
        EventCatalog
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        table = QTable.read(path, format=table_format)
        meta = dict(table.meta)
        table.meta.clear()

        return cls(
            table=table,
            nside=int(meta.pop("nside")),
            order=meta.pop("order"),
            time_bins=meta.pop("time_bins"),
            seed=meta.pop("seed", None),
        )
