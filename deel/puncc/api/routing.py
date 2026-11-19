# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Low-level building blocks for capacity-constrained batch routing.

The routing problem solved here is a capacitated assignment problem: every
record is assigned to exactly one of the categories allowed by its conformal
prediction set, or to the manual review queue. Each category queue accepts a
minimum and a maximum number of records, and the total number of records sent
to manual review is bounded. Costs are integers and the returned assignment
minimizes the *global* total cost (as opposed to a per-record greedy choice,
which can violate the capacity constraints).

The solver is exact: it relies on the HiGHS mixed-integer optimizer shipped
with SciPy, and the returned plan is verified (integrality, capacity bounds,
objective value) before being exposed. Any failure raises an exception, so
callers never receive a partial routing plan.
"""

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

#: Sentinel used to denote the manual review queue in assignments.
REVIEW = -1

#: Default business label of the manual review queue.
DEFAULT_REVIEW_LABEL = "manual_review"


class RoutingError(Exception):
    """Base class of all routing related errors."""


class RoutingValidationError(RoutingError, ValueError):
    """Raised when routing inputs are inconsistent or malformed."""


class RoutingPredictionError(RoutingError, RuntimeError):
    """Raised when prediction or calibration fails.

    When this error is raised, no routing plan is produced.
    """


class RoutingSolverError(RoutingError, RuntimeError):
    """Raised when the routing optimizer fails or returns an invalid plan.

    When this error is raised, no routing plan is produced.
    """


class RoutingInfeasibleError(RoutingError):
    """Raised when no routing plan satisfies all the requested constraints.

    :param str message: human readable explanation.
    :param list reasons: specific constraint violations that were detected.
    :param dict diagnostics: per-category availability statistics.

    When this error is raised, no routing plan is produced.
    """

    def __init__(
        self,
        message: str,
        reasons: Optional[Iterable[str]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.reasons = list(reasons) if reasons is not None else []
        self.diagnostics = diagnostics if diagnostics is not None else {}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input normalization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _integral_array(
    values: Any,
    *,
    name: str,
    ndim: int,
    length: Optional[int] = None,
    second_length: Optional[int] = None,
) -> np.ndarray:
    """Convert ``values`` to an integer NumPy array after strict checks.

    Integral floats (e.g. ``3.0``) are accepted and converted; non-integral
    values (e.g. ``3.5``) and booleans are rejected.
    """
    try:
        arr = np.asarray(values)
    except Exception as exc:  # pragma: no cover - defensive
        raise RoutingValidationError(f"`{name}` cannot be converted to an array.") from exc

    if arr.ndim != ndim:
        raise RoutingValidationError(f"`{name}` must be {ndim}-dimensional, got shape {arr.shape}.")
    if length is not None and arr.shape[0] != length:
        raise RoutingValidationError(f"`{name}` must have {length} rows, got {arr.shape[0]}.")
    if second_length is not None and arr.shape[1] != second_length:
        raise RoutingValidationError(
            f"`{name}` must have {second_length} columns, got {arr.shape[1]}."
        )

    if arr.dtype.kind == "b":
        raise RoutingValidationError(f"`{name}` must contain integers, not booleans.")
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise RoutingValidationError(f"`{name}` must not contain NaN or infinity.")
        rounded = np.rint(arr)
        if not np.all(arr == rounded):
            raise RoutingValidationError(f"`{name}` must only contain integer values.")
        return rounded.astype(np.int64)
    raise RoutingValidationError(f"`{name}` must contain integers, got dtype {arr.dtype}.")


def normalize_costs(
    costs: Optional[Iterable],
    n_records: int,
    n_categories: int,
    *,
    name: str = "costs",
) -> np.ndarray:
    """Normalize the routing costs to an ``(n_records, n_categories)`` array.

    ``costs`` may be ``None`` (all zero costs), a vector of ``n_categories``
    integers shared by all records, or a matrix with one row per record.
    Integer values are required (integral floats are accepted).

    :raises RoutingValidationError: malformed costs.
    """
    if costs is None:
        return np.zeros((n_records, n_categories), dtype=np.int64)

    if isinstance(costs, Mapping):
        keys = list(costs.keys())
        if len(keys) != n_categories or set(keys) != set(range(n_categories)):
            raise RoutingValidationError(
                f"`{name}` must map every class index in 0..{n_categories - 1} "
                "to an integer cost."
            )
        costs = [costs[i] for i in range(n_categories)]

    raw = np.asarray(costs)
    if raw.dtype == object:
        raise RoutingValidationError(
            f"`{name}` must contain integers, got a ragged or non-numeric object."
        )
    arr = _integral_array(costs, name=name, ndim=raw.ndim)
    if arr.ndim == 1:
        if arr.shape[0] != n_categories:
            raise RoutingValidationError(
                f"`{name}` must have {n_categories} entries (one per category), "
                f"got {arr.shape[0]}."
            )
        return np.tile(arr.reshape(1, -1), (n_records, 1))
    if arr.ndim == 2:
        if arr.shape != (n_records, n_categories):
            raise RoutingValidationError(
                f"`{name}` must have shape {(n_records, n_categories)}, " f"got {arr.shape}."
            )
        return arr
    raise RoutingValidationError(
        f"`{name}` must be a vector of size {n_categories} or a matrix of shape "
        f"{(n_records, n_categories)}."
    )


def normalize_review_costs(
    review_cost: Optional[Iterable],
    n_records: int,
    *,
    name: str = "review_cost",
) -> np.ndarray:
    """Normalize the manual review costs to a vector of ``n_records`` integers."""
    if review_cost is None:
        return np.zeros(n_records, dtype=np.int64)

    raw = np.asarray(review_cost)
    if raw.ndim == 0:
        arr = _integral_array([review_cost], name=name, ndim=1, length=1)
        return np.full(n_records, int(arr[0]), dtype=np.int64)
    arr = _integral_array(review_cost, name=name, ndim=1, length=n_records)
    return arr


def normalize_volumes(
    values: Optional[Iterable],
    n_categories: int,
    *,
    name: str,
    default: int,
) -> np.ndarray:
    """Normalize a vector of per-category volume bounds."""
    if values is None:
        return np.full(n_categories, default, dtype=np.int64)
    if isinstance(values, (int, np.integer)) and not isinstance(values, bool):
        raise RoutingValidationError(
            f"`{name}` must be a vector of {n_categories} integers, got a scalar. "
            "Use a list such as "
            f"[{int(values)}] * {n_categories} to share the same bound."
        )
    arr = _integral_array(values, name=name, ndim=1, length=n_categories)
    if np.any(arr < 0):
        raise RoutingValidationError(f"`{name}` must contain non-negative integers.")
    return arr


def normalize_review_capacity(review_capacity: Optional[int], n_records: int) -> int:
    """Normalize the manual review capacity."""
    if review_capacity is None:
        return n_records
    if isinstance(review_capacity, bool) or not isinstance(review_capacity, (int, np.integer)):
        raise RoutingValidationError("`review_capacity` must be a non-negative integer.")
    capacity = int(review_capacity)
    if capacity < 0:
        raise RoutingValidationError("`review_capacity` must be a non-negative integer.")
    return capacity


def resolve_business_labels(
    label_map: Optional[Union[Mapping, Sequence]],
    n_categories: int,
    *,
    review_label: Any = DEFAULT_REVIEW_LABEL,
) -> List[Any]:
    """Resolve the declared business labels of the category columns.

    :param label_map: ``None`` (labels default to the integer class indices), a
        mapping ``{class_index: business_label}`` covering ``0..n_categories-1``,
        or a sequence of ``n_categories`` business labels ordered by class index.
    :param int n_categories: number of category columns.
    :param review_label: label used for the manual review queue.

    :raises RoutingValidationError: incomplete, duplicated or colliding labels.
    """
    if label_map is None:
        labels = list(range(n_categories))
    elif isinstance(label_map, Mapping):
        keys = list(label_map.keys())
        if len(keys) != n_categories or set(keys) != set(range(n_categories)):
            raise RoutingValidationError(
                "`label_map` must map every class index in "
                f"0..{n_categories - 1} to a business label."
            )
        labels = [label_map[i] for i in range(n_categories)]
    elif isinstance(label_map, (str, bytes)):
        raise RoutingValidationError(
            "`label_map` must be a mapping or a sequence of labels, not a string."
        )
    else:
        try:
            labels = list(label_map)
        except TypeError as exc:
            raise RoutingValidationError(
                "`label_map` must be a mapping or a sequence of labels."
            ) from exc
        if len(labels) != n_categories:
            raise RoutingValidationError(
                f"`label_map` must declare {n_categories} labels, got {len(labels)}."
            )

    try:
        unique = set(labels)
        hashable = True
        _ = len(unique)
    except TypeError as exc:
        raise RoutingValidationError("`label_map` labels must be hashable.") from exc
    if not hashable or len(unique) != len(labels):
        raise RoutingValidationError("`label_map` labels must be unique.")
    if review_label in unique:
        raise RoutingValidationError(
            f"`review_label` {review_label!r} collides with a business label."
        )
    return list(labels)


def normalize_ids(ids: Optional[Iterable], n_records: int) -> List[Any]:
    """Normalize the original sample identifiers.

    The identifiers are kept as provided (any hashable object) and preserve the
    input order, so that routing details can be aligned with the original
    records. When ``ids`` is ``None``, the positional indices are used.
    """
    if ids is None:
        return list(range(n_records))
    if hasattr(ids, "tolist"):
        values = ids.tolist()
    else:
        values = list(ids)
    if len(values) != n_records:
        raise RoutingValidationError(
            f"`ids` must contain {n_records} identifiers, got {len(values)}."
        )
    try:
        unique = set(values)
    except TypeError as exc:
        raise RoutingValidationError("`ids` must contain hashable identifiers.") from exc
    if len(unique) != len(values):
        raise RoutingValidationError(
            "`ids` must be unique so that every routing detail can be aligned "
            "with its original sample."
        )
    return values


def sets_to_allowed(sets: Iterable[Iterable[int]], n_categories: int) -> np.ndarray:
    """Build the boolean ``(n_records, n_categories)`` allowed-assignment matrix.

    Duplicate labels inside a prediction set are ignored. Out-of-range labels
    are rejected because category columns must map to declared business labels.
    """
    sets = list(sets)
    allowed = np.zeros((len(sets), n_categories), dtype=bool)
    for i, pred_set in enumerate(sets):
        for label in pred_set:
            if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
                raise RoutingValidationError("Prediction sets must contain integer class indices.")
            label = int(label)
            if label < 0 or label >= n_categories:
                raise RoutingValidationError(
                    f"Prediction set of record {i} contains class {label}, "
                    f"outside the declared range 0..{n_categories - 1}."
                )
            allowed[i, label] = True
    return allowed


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Solver ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _diagnose_infeasibility(
    allowed: np.ndarray,
    review_capacity: int,
    min_volumes: np.ndarray,
    max_volumes: np.ndarray,
) -> Tuple[List[str], Dict[str, Any]]:
    """Collect simple, human readable reasons for an infeasible instance."""
    n_records, n_categories = allowed.shape
    available = allowed.sum(axis=0)
    no_candidate = int(np.sum(~allowed.any(axis=1)))

    reasons = []
    if no_candidate > review_capacity:
        reasons.append(
            f"{no_candidate} record(s) have an empty prediction set (manual review "
            f"only) but the review capacity is {review_capacity}."
        )
    for category in range(n_categories):
        if min_volumes[category] > available[category]:
            reasons.append(
                f"Category {category} requires at least {int(min_volumes[category])} "
                f"record(s) but only {int(available[category])} record(s) have it in "
                "their prediction set."
            )
    if int(max_volumes.sum()) + review_capacity < n_records:
        reasons.append(
            f"Maximum category volumes ({int(max_volumes.sum())}) plus the review "
            f"capacity ({review_capacity}) cannot absorb all {n_records} records."
        )

    diagnostics = {
        "n_records": n_records,
        "n_categories": n_categories,
        "records_with_empty_set": no_candidate,
        "available_per_category": [int(v) for v in available],
        "review_capacity": int(review_capacity),
        "min_volumes": [int(v) for v in min_volumes],
        "max_volumes": [int(v) for v in max_volumes],
    }
    return reasons, diagnostics


def _build_infeasible_error(
    allowed: np.ndarray,
    review_capacity: int,
    min_volumes: np.ndarray,
    max_volumes: np.ndarray,
) -> RoutingInfeasibleError:
    reasons, diagnostics = _diagnose_infeasibility(
        allowed, review_capacity, min_volumes, max_volumes
    )
    if reasons:
        detail = " ".join(reasons)
    else:
        detail = (
            "The per-category volume bounds and the review capacity cannot be "
            "satisfied simultaneously given the records' prediction sets."
        )
    message = (
        "No feasible routing plan exists under the requested constraints. "
        + detail
        + " Prediction sets are never enlarged and records are never forced to "
        "their most likely class to absorb capacity."
    )
    return RoutingInfeasibleError(message, reasons=reasons, diagnostics=diagnostics)


def _verify_plan(
    assignment: np.ndarray,
    costs: np.ndarray,
    review_costs: np.ndarray,
    min_volumes: np.ndarray,
    max_volumes: np.ndarray,
    review_capacity: int,
    objective: float,
) -> int:
    """Check that a solver plan is valid and return its integer total cost."""
    n_records, n_categories = costs.shape

    if assignment.shape != (n_records,):
        raise RoutingSolverError(
            "The routing solver returned an assignment with an unexpected shape."
        )

    review_mask = assignment == REVIEW
    category_mask = ~review_mask
    if np.any(assignment < REVIEW) or np.any(assignment >= n_categories):
        raise RoutingSolverError("The routing solver returned an out-of-range category index.")

    counts = np.bincount(assignment[category_mask], minlength=n_categories).astype(np.int64)
    n_review = int(review_mask.sum())

    if np.any(counts < min_volumes) or np.any(counts > max_volumes):
        raise RoutingSolverError("The routing solver violated the per-category volume bounds.")
    if n_review > review_capacity:
        raise RoutingSolverError("The routing solver violated the manual review capacity.")

    total_cost = int(np.sum(review_costs[review_mask]))
    if np.any(category_mask):
        rows = np.flatnonzero(category_mask)
        total_cost += int(np.sum(costs[rows, assignment[rows]]))

    if not np.isfinite(objective) or abs(float(total_cost) - float(objective)) > 1e-6:
        raise RoutingSolverError(
            "The routing solver objective does not match the recomputed cost of "
            "the returned plan."
        )
    return total_cost


def solve_batch_routing(  # pylint: disable=too-many-branches
    *,
    allowed: Iterable[Iterable[bool]],
    costs: Optional[Iterable] = None,
    review_costs: Optional[Iterable] = None,
    min_volumes: Optional[Iterable] = None,
    max_volumes: Optional[Iterable] = None,
    review_capacity: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Solve the capacity-constrained batch routing problem exactly.

    :param allowed: boolean matrix of shape ``(n_records, n_categories)``.
        ``allowed[i, j]`` is True if record ``i`` may be routed to category
        ``j`` (i.e. ``j`` belongs to its conformal prediction set).
    :param costs: integer routing costs. Either a vector of size
        ``n_categories`` shared by all records or a matrix of shape
        ``(n_records, n_categories)``. Defaults to zero costs.
    :param review_costs: non-negative integer cost of sending each record to
        manual review. Scalar or vector of size ``n_records``. Defaults to 0.
    :param min_volumes: non-negative integer minimum volume of each category.
        Defaults to 0.
    :param max_volumes: non-negative integer maximum volume of each category.
        Defaults to ``n_records``.
    :param review_capacity: maximum total number of records sent to manual
        review. Defaults to ``n_records``.

    :returns: ``(assignment, total_cost)`` where ``assignment[i]`` is the
        assigned category index, or ``-1`` for manual review, and
        ``total_cost`` is the integer sum of the costs of the chosen options.
    :rtype: Tuple[np.ndarray, int]

    :raises RoutingValidationError: malformed inputs.
    :raises RoutingInfeasibleError: no plan satisfies the constraints.
    :raises RoutingSolverError: the optimizer failed or returned an invalid plan.
    """
    allowed = np.asarray(allowed, dtype=bool)
    if allowed.ndim != 2:
        raise RoutingValidationError("`allowed` must be a 2-dimensional matrix.")
    n_records, n_categories = allowed.shape
    if n_records == 0:
        raise RoutingValidationError("At least one record is required to route.")
    if n_categories == 0:
        raise RoutingValidationError("At least one category is required.")

    costs = normalize_costs(costs, n_records, n_categories)
    review_costs = normalize_review_costs(review_costs, n_records)
    min_volumes = normalize_volumes(min_volumes, n_categories, name="min_volumes", default=0)
    max_volumes = normalize_volumes(
        max_volumes, n_categories, name="max_volumes", default=n_records
    )
    if np.any(min_volumes > max_volumes):
        raise RoutingValidationError("`min_volumes` must be lower than or equal to `max_volumes`.")
    review_capacity = normalize_review_capacity(review_capacity, n_records)

    # Build the variables: one per allowed (record, category) pair, plus one
    # per record for the manual review queue when the queue is usable.
    var_records: List[int] = []
    var_categories: List[int] = []
    var_costs: List[int] = []
    add_review = review_capacity > 0

    for i in range(n_records):
        for j in np.flatnonzero(allowed[i]):
            var_records.append(i)
            var_categories.append(int(j))
            var_costs.append(int(costs[i, j]))
        if add_review:
            var_records.append(i)
            var_categories.append(REVIEW)
            var_costs.append(int(review_costs[i]))

    n_vars = len(var_records)
    if n_vars == 0 or len(set(var_records)) != n_records:
        raise _build_infeasible_error(allowed, review_capacity, min_volumes, max_volumes)

    try:
        from scipy.optimize import (  # pylint: disable=import-outside-toplevel
            Bounds,
            LinearConstraint,
            milp,
        )
        from scipy.sparse import coo_matrix  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - scipy ships with sklearn
        raise RoutingSolverError(
            "The exact routing solver requires scipy.optimize.milp (SciPy >= 1.9)."
        ) from exc

    var_records_arr = np.asarray(var_records, dtype=np.int64)
    var_categories_arr = np.asarray(var_categories, dtype=np.int64)
    var_costs_arr = np.asarray(var_costs, dtype=np.float64)
    var_indices = np.arange(n_vars, dtype=np.int64)

    rows: List[np.ndarray] = [var_records_arr]
    cols: List[np.ndarray] = [var_indices]
    vals: List[np.ndarray] = [np.ones(n_vars, dtype=np.float64)]

    category_vars = var_categories_arr != REVIEW
    rows.append(n_records + var_categories_arr[category_vars])
    cols.append(var_indices[category_vars])
    vals.append(np.ones(int(category_vars.sum()), dtype=np.float64))

    rows.append(np.full(int((~category_vars).sum()), n_records + n_categories))
    cols.append(var_indices[~category_vars])
    vals.append(np.ones(int((~category_vars).sum()), dtype=np.float64))

    matrix = coo_matrix(
        (
            np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n_records + n_categories + 1, n_vars),
    ).tocsr()

    lower = np.concatenate(
        [
            np.ones(n_records, dtype=np.float64),
            min_volumes.astype(np.float64),
            np.zeros(1, dtype=np.float64),
        ]
    )
    upper = np.concatenate(
        [
            np.ones(n_records, dtype=np.float64),
            max_volumes.astype(np.float64),
            np.asarray([review_capacity], dtype=np.float64),
        ]
    )

    try:
        result = milp(
            c=var_costs_arr,
            integrality=np.ones(n_vars, dtype=np.uint8),
            bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"disp": False, "mip_rel_gap": 0.0},
        )
    except Exception as exc:
        raise RoutingSolverError(
            "The routing optimizer failed; no routing plan was produced."
        ) from exc

    status = int(getattr(result, "status", -1))
    if status == 2:
        raise _build_infeasible_error(allowed, review_capacity, min_volumes, max_volumes)
    if status != 0 or result.x is None:
        raise RoutingSolverError(
            "The routing optimizer did not find an optimal plan "
            f"(status={status}); no routing plan was produced."
        )

    values = np.asarray(result.x)
    if not np.allclose(values, np.rint(values), atol=1e-9):
        raise RoutingSolverError(
            "The routing solver returned a fractional plan; no routing plan was produced."
        )
    chosen = np.rint(values).astype(np.int64)

    assignment = np.empty(n_records, dtype=np.int64)
    filled = np.zeros(n_records, dtype=bool)
    for var, value in enumerate(chosen):
        if value == 0:
            continue
        if value != 1:
            raise RoutingSolverError("The routing solver returned an invalid variable value.")
        record = int(var_records_arr[var])
        if filled[record]:
            raise RoutingSolverError("The routing solver assigned a record more than once.")
        filled[record] = True
        assignment[record] = int(var_categories_arr[var])

    if not np.all(filled):
        raise RoutingSolverError(
            "The routing solver left some records unassigned; no routing plan was produced."
        )

    total_cost = _verify_plan(
        assignment,
        costs,
        review_costs,
        min_volumes,
        max_volumes,
        review_capacity,
        float(result.fun),
    )
    return assignment, total_cost
