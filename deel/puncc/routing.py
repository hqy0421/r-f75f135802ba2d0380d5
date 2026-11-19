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
Calibration-driven batch routing for conformal classification.

A batch of new records is first processed by a *fixed* probability predictor,
both on an independent calibration set and on the new records, and native
:class:`deel.puncc.classification.LAC` prediction sets are built. Each record
must then be routed to exactly one category contained in its LAC set, or to a
manual review queue. Per-category minimum/maximum processing volumes and a
total manual review cap are enforced, and the integer routing cost summed over
the whole batch is minimized globally (not record by record).

The prediction sets are never modified: a record whose LAC set is empty can
only be sent to manual review, and no record is ever forced to its most likely
class or given a larger set just to fill a queue. When no feasible routing
exists, :class:`deel.puncc.api.routing.RoutingInfeasibleError` is raised with
the reasons, and no partial plan is returned. The same holds when prediction,
calibration or optimization fails.

Example::

    from deel.puncc.routing import BatchRouter

    router = BatchRouter(pretrained_predictor, label_map={
        0: "billing", 1: "shipping", 2: "account"
    })
    router.fit(X_calib=X_calib, y_calib=y_calib)

    result = router.route(
        X_new,
        alpha=0.1,
        ids=new_ids,
        costs={0: 1, 1: 2, 2: 3},
        review_cost=10,
        min_volumes=[10, 0, 0],
        max_volumes=[100, 100, 100],
        review_capacity=25,
    )

    result.probabilities     # actual predicted probabilities (business labels)
    result.prediction_sets   # native LAC sets (class indices)
    result.details           # per-record routing details aligned on ids
    result.queue_volumes     # volumes counted from `details`
    result.total_cost        # integer cost counted from `details`
"""

import logging
from dataclasses import dataclass
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
import pandas as pd

from deel.puncc.api.routing import DEFAULT_REVIEW_LABEL
from deel.puncc.api.routing import REVIEW
from deel.puncc.api.routing import RoutingError
from deel.puncc.api.routing import RoutingInfeasibleError
from deel.puncc.api.routing import RoutingPredictionError
from deel.puncc.api.routing import RoutingSolverError
from deel.puncc.api.routing import RoutingValidationError
from deel.puncc.api.routing import normalize_costs
from deel.puncc.api.routing import normalize_ids
from deel.puncc.api.routing import normalize_review_capacity
from deel.puncc.api.routing import normalize_review_costs
from deel.puncc.api.routing import normalize_volumes
from deel.puncc.api.routing import resolve_business_labels
from deel.puncc.api.routing import sets_to_allowed
from deel.puncc.api.routing import solve_batch_routing
from deel.puncc.classification import LAC
from deel.puncc.api.conformalization import SplitConformalPredictor
from deel.puncc.api.prediction import BasePredictor

logger = logging.getLogger(__name__)

#: Columns of the routing details frame, in order.
DETAIL_COLUMNS = (
    "id",
    "prediction_set",
    "prediction_set_indices",
    "queue",
    "assigned_index",
    "assigned_probability",
    "is_review",
    "cost",
)


@dataclass(frozen=True)
class RoutingResult:
    """Outcome of a batch routing.

    :param ids: original sample identifiers, in the input order.
    :param probabilities: actual predicted probabilities of the new records.
        Rows are indexed by ``ids`` and columns are the declared business
        labels, in class-index order.
    :param prediction_sets: native LAC prediction sets, expressed as lists of
        class indices exactly as returned by
        :class:`deel.puncc.classification.LAC`.
    :param details: per-record routing details, aligned with ``ids`` and in the
        input order. Its ``queue`` column holds the business label of the
        assigned category, or the review label.
    :param queue_volumes: number of records per category queue, counted from
        ``details``. The manual review queue is excluded; see ``review_count``.
    :param queue_costs: routing cost per category queue, summed from
        ``details``. The manual review queue is excluded; the review cost is
        part of ``total_cost``.
    :param total_cost: integer total routing cost (categories and manual
        review), summed from ``details``.
    :param review_count: number of records sent to manual review, counted from
        ``details``.
    :param review_cost: total integer cost of the manual reviews, summed from
        ``details``.
    :param alpha: significance level used to build the LAC sets.
    :param label_map: resolved business label of each class index.
    :param review_label: business label of the manual review queue.
    """

    ids: List[Any]
    probabilities: pd.DataFrame
    prediction_sets: Tuple[List[int], ...]
    details: pd.DataFrame
    queue_volumes: Dict[Any, int]
    queue_costs: Dict[Any, int]
    total_cost: int
    review_count: int
    review_cost: int
    alpha: float
    label_map: Tuple[Any, ...]
    review_label: Any

    def summary(self) -> Dict[str, Any]:
        """Return the routing summary recomputed from the details frame."""
        details = self.details
        review_mask = details["is_review"].to_numpy(dtype=bool)
        category_mask = ~review_mask
        category_details = details.loc[category_mask]

        queue_volumes = {
            label: int((category_details["queue"] == label).sum()) for label in self.label_map
        }
        queue_costs = {
            label: int(category_details.loc[category_details["queue"] == label, "cost"].sum())
            for label in self.label_map
        }
        review_count = int(review_mask.sum())
        review_cost = int(details.loc[review_mask, "cost"].sum())
        total_cost = int(details["cost"].sum())
        return {
            "n_records": int(len(details)),
            "queue_volumes": queue_volumes,
            "queue_costs": queue_costs,
            "total_cost": total_cost,
            "review_count": review_count,
            "review_cost": review_cost,
            "alpha": float(self.alpha),
        }


class _ProbabilityPredictor(BasePredictor):
    """Adapter exposing a fixed probability matrix from a point predictor.

    Models exposing ``predict_proba`` (scikit-learn classifiers) are queried
    through it; other models are queried through ``predict`` and must return a
    2-dimensional probability matrix (e.g. a Keras ``softmax`` model). The
    output shape is validated, so a label-only predictor is rejected instead of
    being silently interpreted as probabilities.
    """

    def predict(self, X, **kwargs):
        try:
            if hasattr(self.model, "predict_proba"):
                y_pred = self.model.predict_proba(X, **kwargs)
            else:
                y_pred = self.model.predict(X, **kwargs)
        except Exception as exc:
            raise RoutingPredictionError(
                "The predictor failed to predict the provided records. No "
                "routing plan was produced."
            ) from exc
        probabilities = np.asarray(y_pred, dtype=float)
        if probabilities.ndim == 1:
            raise RoutingPredictionError(
                "The predictor returned 1-dimensional outputs (class labels or a "
                "single score) instead of a probability matrix. Use a predictor "
                "whose `predict` returns class probabilities or exposes "
                "`predict_proba`."
            )
        if probabilities.ndim != 2:
            probabilities = probabilities.reshape(len(probabilities), -1)
        return probabilities


def _as_probability_predictor(predictor: Any) -> Any:
    """Wrap a plain model so that it yields a probability matrix."""
    if isinstance(predictor, SplitConformalPredictor):
        return predictor
    if isinstance(predictor, BasePredictor):
        model = predictor.model
        if hasattr(model, "predict_proba"):
            return _ProbabilityPredictor(model, is_trained=True)
        return predictor
    if hasattr(predictor, "predict_proba") or hasattr(predictor, "predict"):
        return _ProbabilityPredictor(predictor, is_trained=True)
    raise RoutingValidationError(
        "The predictor must expose a `predict` (or `predict_proba`) method."
    )


class BatchRouter:
    """Calibration-driven, capacity-constrained batch router.

    The router wraps a *fixed* probability predictor and native
    :class:`deel.puncc.classification.LAC` sets:

    1. :meth:`fit` predicts the calibration records with the fixed predictor
       and calibrates LAC nonconformity scores on that independent set.
    2. :meth:`route` predicts the new records with the same predictor, builds
       native LAC sets, and solves the capacity-constrained routing problem to
       global optimality.

    :param predictor: a pre-trained model exposing ``predict`` (returning class
        probabilities), a :class:`deel.puncc.api.prediction.BasePredictor`, or
        an already fitted :class:`deel.puncc.classification.LAC`.
    :param label_map: declared business labels of the class columns. Either a
        mapping ``{class_index: label}``, a sequence of labels ordered by class
        index, or ``None`` to keep the integer class indices. The mapping is
        checked against the predictor's output at routing time.
    :param review_label: business label of the manual review queue.
    :param train: if True, the predictor is trained on the fit data passed to
        :meth:`fit`. Defaults to False, i.e. the predictor is fixed.
    :param random_state: seed used if the calibration data must be split.

    .. note::

        The predictor passed to the router is used as is: no probability is
        fabricated, and no calibration data is reused as new records.
    """

    def __init__(
        self,
        predictor: Any,
        *,
        label_map: Optional[Union[Mapping, Sequence]] = None,
        review_label: Any = DEFAULT_REVIEW_LABEL,
        train: bool = False,
        random_state: Optional[int] = None,
    ):
        if isinstance(predictor, SplitConformalPredictor):
            self._lac = predictor
            self._external_lac = True
        else:
            self._lac = None
            self._external_lac = False
        self.predictor = predictor
        self.label_map = label_map
        self.review_label = review_label
        self.train = bool(train)
        self.random_state = random_state
        self._n_categories: Optional[int] = None
        self._labels: Optional[Tuple[Any, ...]] = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        *,
        X_calib: Iterable,
        y_calib: Iterable,
        X_fit: Optional[Iterable] = None,
        y_fit: Optional[Iterable] = None,
        **kwargs: Optional[dict],
    ) -> "BatchRouter":
        """Predict the calibration set and calibrate native LAC scores.

        :param X_calib: calibration features, independent from the model's
            training data and from the records to route.
        :param y_calib: calibration labels.
        :param X_fit: optional fit features, required only when ``train=True``.
        :param y_fit: optional fit labels, required only when ``train=True``.
        :param dict kwargs: prediction configuration passed to the predictor.

        :returns: the fitted router.
        :rtype: BatchRouter

        :raises RoutingValidationError: missing or inconsistent calibration
            data.
        :raises RoutingPredictionError: the predictor failed on the calibration
            set. No router state is updated.
        """
        if X_calib is None or y_calib is None:
            raise RoutingValidationError(
                "An independent calibration set (`X_calib`, `y_calib`) is required."
            )
        if self.train and (X_fit is None or y_fit is None):
            raise RoutingValidationError(
                "`train=True` requires `X_fit` and `y_fit` to train the predictor."
            )

        if self._external_lac:
            lac = self._lac
            if self.train:
                raise RoutingValidationError(
                    "An already fitted conformal predictor cannot be trained again "
                    "through the router."
                )
            self._ensure_external_lac_fitted()
        else:
            lac = LAC(
                _as_probability_predictor(self.predictor),
                train=self.train,
                random_state=self.random_state,
            )
            fit_kwargs = dict(kwargs)
            try:
                if self.train:
                    lac.fit(
                        X_fit=X_fit,
                        y_fit=y_fit,
                        X_calib=X_calib,
                        y_calib=y_calib,
                        **fit_kwargs,
                    )
                else:
                    lac.fit(X_calib=X_calib, y_calib=y_calib, **fit_kwargs)
            except RoutingError:
                raise
            except Exception as exc:
                raise RoutingPredictionError(
                    "Calibration failed: the predictor could not be calibrated on "
                    "the provided calibration set. No routing plan was produced."
                ) from exc

        # Only commit the internal state once calibration fully succeeded, so
        # that a failed calibration never leaves the router half-configured.
        self._lac = lac
        self._n_categories = None
        self._labels = None
        return self

    def _ensure_external_lac_fitted(self) -> None:
        try:
            scores = self._lac.get_nonconformity_scores()
        except Exception as exc:
            raise RoutingPredictionError(
                "The provided conformal predictor is not calibrated. Call its "
                "`fit` method on an independent calibration set first."
            ) from exc
        if scores is None or len(scores) == 0:
            raise RoutingPredictionError(
                "The provided conformal predictor is not calibrated on any data."
            )

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def route(
        self,
        X: Iterable,
        *,
        alpha: float,
        costs: Optional[Iterable] = None,
        review_cost: Optional[Iterable] = None,
        min_volumes: Optional[Iterable] = None,
        max_volumes: Optional[Iterable] = None,
        review_capacity: Optional[int] = None,
        ids: Optional[Iterable] = None,
    ) -> RoutingResult:
        """Route new records to categories or manual review.

        Each record is predicted by the fixed probability predictor, assigned a
        native LAC set, then routed to exactly one category of its set or to
        manual review. Minimum/maximum category volumes and the review cap are
        enforced, and the total integer cost is minimized globally.

        :param X: features of the new records.
        :param float alpha: target miscoverage used by LAC.
        :param costs: integer routing cost per (record, category). Either a
            vector of size ``n_categories`` shared by all records, or a matrix
            of shape ``(n_records, n_categories)``. Defaults to zero.
        :param review_cost: non-negative integer cost of a manual review, either
            a scalar or one value per record. Defaults to zero.
        :param min_volumes: minimum number of records per category. Defaults to
            zero for every category.
        :param max_volumes: maximum number of records per category. Defaults to
            the number of records.
        :param review_capacity: maximum total number of manual reviews.
            Defaults to the number of records.
        :param ids: original sample identifiers, aligned with ``X``. Defaults to
            positional indices.

        :returns: the routing result, including actual probabilities, native
            LAC sets, per-record details aligned on ``ids``, and queue volumes
            and costs counted from the details.
        :rtype: RoutingResult

        :raises RoutingValidationError: malformed inputs.
        :raises RoutingPredictionError: prediction or calibration failed.
        :raises RoutingInfeasibleError: no feasible plan exists; the reasons are
            attached to the exception.
        :raises RoutingSolverError: the optimizer failed or returned an invalid
            plan.

        In every failure case, no partial routing plan is returned.
        """
        if self._lac is None:
            raise RoutingValidationError(
                "Call `fit` on an independent calibration set before routing."
            )
        self._ensure_calibrated()

        if X is None:
            raise RoutingValidationError("`X` must contain the records to route.")
        if isinstance(alpha, bool) or not isinstance(alpha, (float, int, np.floating)):
            raise RoutingValidationError("`alpha` must be a number in the open interval (0, 1).")
        if not 0.0 < float(alpha) < 1.0:
            raise RoutingValidationError(
                f"`alpha` must lie in the open interval (0, 1), got {alpha}."
            )
        n_records = self._n_records(X)
        if n_records == 0:
            raise RoutingValidationError("At least one record must be routed.")

        try:
            y_pred, set_pred = self._lac.predict(X, alpha=float(alpha))
        except RoutingError:
            raise
        except Exception as exc:
            raise RoutingPredictionError(
                "Prediction failed for the new records. No routing plan was produced."
            ) from exc

        probabilities = self._validated_probabilities(y_pred, n_records)
        n_categories = probabilities.shape[1]
        self._resolve_labels(n_categories)

        raw_sets = self._validated_sets(set_pred, n_records, n_categories)
        allowed = sets_to_allowed(raw_sets, n_categories)

        sample_ids = normalize_ids(ids, n_records)

        # Normalize every routing input before solving; validation errors are
        # raised here, before any plan exists.
        cost_matrix = normalize_costs(costs, n_records, n_categories)
        review_costs = normalize_review_costs(review_cost, n_records)
        min_volumes = normalize_volumes(min_volumes, n_categories, name="min_volumes", default=0)
        max_volumes = normalize_volumes(
            max_volumes, n_categories, name="max_volumes", default=n_records
        )
        review_capacity = normalize_review_capacity(review_capacity, n_records)

        assignment, solver_cost = solve_batch_routing(
            allowed=allowed,
            costs=cost_matrix,
            review_costs=review_costs,
            min_volumes=min_volumes,
            max_volumes=max_volumes,
            review_capacity=review_capacity,
        )

        result = self._build_result(
            sample_ids=sample_ids,
            probabilities=probabilities,
            raw_sets=raw_sets,
            assignment=assignment,
            cost_matrix=cost_matrix,
            review_costs=review_costs,
            solver_cost=solver_cost,
            alpha=alpha,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ensure_calibrated(self) -> None:
        try:
            scores = self._lac.get_nonconformity_scores()
        except Exception as exc:
            raise RoutingPredictionError(
                "The router is not calibrated. Call `fit` on an independent "
                "calibration set first."
            ) from exc
        if scores is None or len(scores) == 0:
            raise RoutingPredictionError("The router is not calibrated on any data.")

    @staticmethod
    def _n_records(X: Iterable) -> int:
        try:
            return len(X)
        except TypeError:
            try:
                return len(list(X))
            except TypeError as exc:
                raise RoutingValidationError("`X` must be a sized collection of records.") from exc

    @staticmethod
    def _validated_probabilities(y_pred: Iterable, n_records: int) -> np.ndarray:
        if hasattr(y_pred, "detach"):  # torch tensors
            y_pred = y_pred.detach()
        if hasattr(y_pred, "cpu"):
            y_pred = y_pred.cpu()
        if hasattr(y_pred, "numpy") and not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.numpy()
        try:
            probabilities = np.asarray(y_pred, dtype=float)
        except Exception as exc:
            raise RoutingPredictionError(
                "The predictor output cannot be interpreted as probabilities."
            ) from exc
        if probabilities.ndim != 2:
            raise RoutingPredictionError(
                "The predictor must return a 2-dimensional probability matrix "
                f"of shape (n_records, n_categories), got shape "
                f"{probabilities.shape}."
            )
        if probabilities.shape[0] != n_records:
            raise RoutingPredictionError(
                f"The predictor returned {probabilities.shape[0]} rows for " f"{n_records} records."
            )
        if probabilities.shape[1] == 0:
            raise RoutingPredictionError("The predictor returned no category column.")
        if not np.all(np.isfinite(probabilities)):
            raise RoutingPredictionError("The predictor returned NaN or infinite probabilities.")
        if np.any(probabilities < -1e-9) or np.any(probabilities > 1 + 1e-9):
            raise RoutingPredictionError(
                "The predictor output does not look like class probabilities."
            )
        return probabilities

    @staticmethod
    def _validated_sets(set_pred: Iterable, n_records: int, n_categories: int) -> List[List[int]]:
        if isinstance(set_pred, tuple):
            if len(set_pred) != 1:
                raise RoutingPredictionError(
                    "LAC prediction sets must be returned as a single collection."
                )
            set_pred = set_pred[0]
        try:
            raw_sets = [list(pred_set) for pred_set in set_pred]
        except TypeError as exc:
            raise RoutingPredictionError(
                "The prediction sets returned by LAC are malformed."
            ) from exc
        if len(raw_sets) != n_records:
            raise RoutingPredictionError(
                f"LAC returned {len(raw_sets)} prediction sets for " f"{n_records} records."
            )
        for i, pred_set in enumerate(raw_sets):
            for label in pred_set:
                if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
                    raise RoutingPredictionError(
                        "LAC prediction sets must contain integer class indices."
                    )
                if int(label) < 0 or int(label) >= n_categories:
                    raise RoutingPredictionError(
                        f"LAC prediction set of record {i} contains class "
                        f"{int(label)}, outside the predictor's "
                        f"0..{n_categories - 1} range."
                    )
            raw_sets[i] = [int(label) for label in pred_set]
        return raw_sets

    def _resolve_labels(self, n_categories: int) -> None:
        if self._labels is not None and self._n_categories == n_categories:
            return
        labels = resolve_business_labels(
            self.label_map, n_categories, review_label=self.review_label
        )
        self._labels = tuple(labels)
        self._n_categories = n_categories

    def _build_result(
        self,
        *,
        sample_ids: List[Any],
        probabilities: np.ndarray,
        raw_sets: List[List[int]],
        assignment: np.ndarray,
        cost_matrix: np.ndarray,
        review_costs: np.ndarray,
        solver_cost: int,
        alpha: float,
    ) -> RoutingResult:
        n_records = probabilities.shape[0]
        labels = self._labels
        assert labels is not None

        review_mask = assignment == REVIEW
        assigned_probability = np.full(n_records, np.nan)
        category_mask = ~review_mask
        assigned_probability[category_mask] = probabilities[
            np.flatnonzero(category_mask), assignment[category_mask]
        ]

        record_costs = np.where(review_mask, review_costs, 0)
        if np.any(category_mask):
            rows = np.flatnonzero(category_mask)
            record_costs[rows] = cost_matrix[rows, assignment[rows]]

        queue_labels = [
            self.review_label if is_review else labels[int(index)]
            for is_review, index in zip(review_mask, assignment)
        ]
        mapped_sets = [[labels[index] for index in pred_set] for pred_set in raw_sets]

        details = pd.DataFrame(
            {
                "id": sample_ids,
                "prediction_set": mapped_sets,
                "prediction_set_indices": [list(s) for s in raw_sets],
                "queue": queue_labels,
                "assigned_index": assignment.astype(np.int64),
                "assigned_probability": assigned_probability,
                "is_review": review_mask,
                "cost": record_costs.astype(np.int64),
            },
            columns=list(DETAIL_COLUMNS),
        )

        # Every statistic below is computed from the details frame, never from
        # the solver internals. Category queues are reported in label order; the
        # manual review queue is reported separately.
        category_details = details.loc[~review_mask]
        queue_volumes = {label: int((category_details["queue"] == label).sum()) for label in labels}
        queue_costs = {
            label: int(category_details.loc[category_details["queue"] == label, "cost"].sum())
            for label in labels
        }

        total_cost = int(details["cost"].sum())
        review_count = int(review_mask.sum())
        review_cost = int(details.loc[review_mask, "cost"].sum())

        if total_cost != int(solver_cost):
            raise RoutingValidationError(  # pragma: no cover - defensive
                "Internal inconsistency: details cost differs from the solver cost."
            )

        probability_frame = pd.DataFrame(
            probabilities, index=pd.Index(sample_ids), columns=list(labels)
        )

        return RoutingResult(
            ids=list(sample_ids),
            probabilities=probability_frame,
            prediction_sets=[list(pred_set) for pred_set in raw_sets],
            details=details,
            queue_volumes=queue_volumes,
            queue_costs=queue_costs,
            total_cost=total_cost,
            review_count=review_count,
            review_cost=review_cost,
            alpha=float(alpha),
            label_map=tuple(labels),
            review_label=self.review_label,
        )


def route_batch(  # pylint: disable=too-many-arguments
    predictor: Any,
    *,
    X_calib: Iterable,
    y_calib: Iterable,
    X: Iterable,
    alpha: float,
    costs: Optional[Iterable] = None,
    review_cost: Optional[Iterable] = None,
    min_volumes: Optional[Iterable] = None,
    max_volumes: Optional[Iterable] = None,
    review_capacity: Optional[int] = None,
    ids: Optional[Iterable] = None,
    label_map: Optional[Union[Mapping, Sequence]] = None,
    review_label: Any = DEFAULT_REVIEW_LABEL,
    train: bool = False,
    random_state: Optional[int] = None,
    X_fit: Optional[Iterable] = None,
    y_fit: Optional[Iterable] = None,
    **fit_kwargs: Optional[dict],
) -> RoutingResult:
    """One-shot calibration-driven batch routing.

    Calibrates native LAC on ``(X_calib, y_calib)``, predicts ``X`` with the
    fixed probability predictor, and solves the capacity-constrained routing
    problem. See :class:`BatchRouter` for the semantics of every argument.

    :returns: the routing result.
    :rtype: RoutingResult

    :raises RoutingError: on prediction, calibration, validation or solver
        failure. No partial routing plan is returned.
    """
    router = BatchRouter(
        predictor,
        label_map=label_map,
        review_label=review_label,
        train=train,
        random_state=random_state,
    )
    router.fit(
        X_calib=X_calib,
        y_calib=y_calib,
        X_fit=X_fit,
        y_fit=y_fit,
        **fit_kwargs,
    )
    return router.route(
        X,
        alpha=alpha,
        costs=costs,
        review_cost=review_cost,
        min_volumes=min_volumes,
        max_volumes=max_volumes,
        review_capacity=review_capacity,
        ids=ids,
    )


__all__ = [
    "BatchRouter",
    "RoutingResult",
    "route_batch",
    "RoutingError",
    "RoutingValidationError",
    "RoutingPredictionError",
    "RoutingInfeasibleError",
    "RoutingSolverError",
    "REVIEW",
    "DEFAULT_REVIEW_LABEL",
]
