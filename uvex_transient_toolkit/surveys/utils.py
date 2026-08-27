"""Schema-validation primitives used by :class:`~uvex_transient_toolkit.surveys.base.SurveySchedule`."""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.units import UnitBase

ActionValidator = Callable[[QTable], bool | str | list[str]]


@dataclass(frozen=True)
class QTableColumnSpec:
    """Validation rules for one QTable column."""

    column_type: type | tuple[type, ...] | None = None
    dtype: np.dtype | type | None = None
    unit: UnitBase | str | None = None
    required: bool = True
    ndim: int | None = 1
    validator: Callable[[Any], bool | str] | None = None
    description: str = ""

    def validate_against_table(
        self,
        table: QTable,
        column_name: str,
    ) -> list[str]:
        """
        Validate this specification against one column in a QTable.

        Returns
        -------
        list[str]
            Validation errors. An empty list indicates success.
        """
        errors: list[str] = []

        # Column existence
        if column_name not in table.colnames:
            if self.required:
                errors.append(f"Missing required column {column_name!r}.")
            return errors

        column = table[column_name]

        # Astropy/Python column class
        if self.column_type is not None and not isinstance(column, self.column_type):
            errors.append(
                f"Column {column_name!r} must be an instance of "
                f"{self._format_types(self.column_type)}, "
                f"got {type(column).__name__}."
            )

            # Dtype/unit checks may produce confusing secondary errors when
            # the fundamental column type is wrong.
            return errors

        # Dimensionality
        if self.ndim is not None:
            actual_ndim = getattr(column, "ndim", None)

            if actual_ndim != self.ndim:
                errors.append(f"Column {column_name!r} must have ndim={self.ndim}, got {actual_ndim}.")

        # NumPy dtype
        if self.dtype is not None:
            actual_dtype = getattr(column, "dtype", None)

            if actual_dtype is None:
                errors.append(f"Column {column_name!r} has no NumPy-compatible dtype.")
            elif not self._dtype_matches(actual_dtype):
                errors.append(
                    f"Column {column_name!r} must have dtype compatible with {self.dtype!r}, got {actual_dtype}."
                )

        # Astropy unit
        if self.unit is not None:
            expected_unit = u.Unit(self.unit)
            actual_unit = getattr(column, "unit", None)

            if actual_unit is None:
                errors.append(f"Column {column_name!r} must have units equivalent to {expected_unit}, but is unitless.")
            else:
                try:
                    unit_matches = actual_unit.is_equivalent(expected_unit)
                except (AttributeError, TypeError, ValueError) as exc:
                    errors.append(f"Could not validate units for column {column_name!r}: {exc}")
                else:
                    if not unit_matches:
                        errors.append(
                            f"Column {column_name!r} must have units equivalent to {expected_unit}, got {actual_unit}."
                        )

        # User-supplied value validation
        if self.validator is not None:
            try:
                result = self.validator(column)
            except Exception as exc:
                errors.append(f"Custom validation for column {column_name!r} raised {type(exc).__name__}: {exc}")
            else:
                if isinstance(result, str):
                    errors.append(f"Column {column_name!r}: {result}")
                elif not isinstance(result, (bool, np.bool_)):
                    errors.append(
                        f"Custom validator for column {column_name!r} "
                        f"must return bool or str, got "
                        f"{type(result).__name__}."
                    )
                elif not result:
                    errors.append(f"Column {column_name!r} failed custom validation.")

        return errors

    def _dtype_matches(
        self,
        actual_dtype: np.dtype,
    ) -> bool:
        """Return whether a concrete dtype satisfies this specification."""
        try:
            return bool(np.issubdtype(actual_dtype, self.dtype))
        except TypeError:
            try:
                return np.dtype(actual_dtype) == np.dtype(self.dtype)
            except TypeError:
                return False

    @staticmethod
    def _format_types(
        expected: type | tuple[type, ...],
    ) -> str:
        """Format one or more expected classes for an error message."""
        if isinstance(expected, tuple):
            return " or ".join(cls.__name__ for cls in expected)

        return expected.__name__


@dataclass(frozen=True)
class ActionSpec:
    """
    Validation rules for one survey-schedule action.

    Parameters
    ----------
    required_columns
        Columns that must contain valid, unmasked values for every row
        associated with this action.
    validator
        Optional custom validator called with the subset of rows containing
        this action. It may return:

        - ``True`` for success;
        - ``False`` for a generic validation failure;
        - a string containing one validation error;
        - a list of validation-error strings.
    description
        Human-readable description of the action.
    """

    required_columns: Collection[str] = ()
    validator: ActionValidator | None = None
    description: str = ""

    def validate_against_table(
        self,
        table: QTable,
        *,
        action_name: str,
        action_column: str = "action",
    ) -> list[str]:
        """
        Validate rows corresponding to one action.

        Parameters
        ----------
        table
            Complete survey schedule.
        action_name
            Action value governed by this specification.
        action_column
            Name of the column containing action labels.

        Returns
        -------
        list[str]
            Validation errors. An empty list indicates success.
        """
        errors: list[str] = []

        if action_column not in table.colnames:
            return [f"Cannot validate action {action_name!r}: missing action column {action_column!r}."]

        actions = np.asarray(table[action_column]).astype(str)
        row_mask = actions == action_name

        # It is valid for a declared action to have no rows.
        if not np.any(row_mask):
            return errors

        action_rows = table[row_mask]

        for column_name in self.required_columns:
            if column_name not in table.colnames:
                errors.append(f"Action {action_name!r} requires missing column {column_name!r}.")
                continue

            column = action_rows[column_name]
            missing = self._missing_mask(column)

            if np.any(missing):
                row_indices = np.flatnonzero(row_mask)
                missing_rows = row_indices[missing]

                errors.append(
                    f"Action {action_name!r} requires column "
                    f"{column_name!r}, but it is missing at table rows "
                    f"{missing_rows.tolist()}."
                )

        if self.validator is not None:
            try:
                result = self.validator(action_rows)
            except Exception as exc:
                errors.append(f"Custom validation for action {action_name!r} raised {type(exc).__name__}: {exc}")
            else:
                errors.extend(
                    self._normalize_validator_result(
                        result,
                        action_name=action_name,
                    )
                )

        return errors

    @staticmethod
    def _missing_mask(column) -> np.ndarray:
        """
        Return a row-wise mask identifying missing values.

        Supports ordinary columns, masked columns, Quantity columns, and
        common Astropy mixin columns such as SkyCoord.
        """
        # MaskedColumn, MaskedQuantity, and many Astropy mixins expose
        # either ``mask`` directly or masks on their coordinate components.
        mask = getattr(column, "mask", None)

        if mask is not None:
            mask = np.asarray(mask)

            if mask.dtype.names:
                # Structured masks (e.g. EarthLocation's combined x/y/z mask) carry
                # one sub-mask per field rather than one bool per row; a row is
                # missing if any of its fields are.
                mask = np.logical_or.reduce([mask[name] for name in mask.dtype.names])

            if mask.ndim == 0:
                return np.full(len(column), bool(mask))

            if mask.ndim == 1:
                return mask.astype(bool, copy=False)

            # If each row contains multiple masked components, consider
            # the row missing when any component is masked.
            return np.any(mask, axis=tuple(range(1, mask.ndim)))

        # SkyCoord mixin columns may expose masks through components.
        component_masks = []

        for component_name in ("ra", "dec", "lon", "lat", "distance"):
            component = getattr(column, component_name, None)

            if component is None:
                continue

            component_mask = np.ma.getmaskarray(component)

            if component_mask.shape == (len(column),):
                component_masks.append(component_mask)

        if component_masks:
            return np.logical_or.reduce(component_masks)

        # Plain, unmasked columns have no missing values structurally.
        return np.zeros(len(column), dtype=bool)

    @staticmethod
    def _normalize_validator_result(
        result: bool | str | list[str],
        *,
        action_name: str,
    ) -> list[str]:
        """Convert a custom-validator result into error messages."""
        if isinstance(result, str):
            return [f"Action {action_name!r}: {result}"]

        if isinstance(result, list):
            if not all(isinstance(item, str) for item in result):
                return [f"Validator for action {action_name!r} returned a list containing non-string values."]

            return [f"Action {action_name!r}: {message}" for message in result]

        if isinstance(result, (bool, np.bool_)):
            if result:
                return []

            return [f"Action {action_name!r} failed custom validation."]

        return [
            f"Validator for action {action_name!r} must return bool, str, or list[str], got {type(result).__name__}."
        ]
