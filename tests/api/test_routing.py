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
import itertools

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.routing import REVIEW
from deel.puncc.api.routing import RoutingInfeasibleError
from deel.puncc.api.routing import RoutingPredictionError
from deel.puncc.api.routing import RoutingSolverError
from deel.puncc.api.routing import RoutingValidationError
from deel.puncc.api.routing import solve_batch_routing
from deel.puncc.classification import LAC
from deel.puncc.routing import BatchRouter
from deel.puncc.routing import route_batch

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fixtures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture(scope="module")
def three_class_data():
    """A fixed 3-class problem with train/calibration/new splits."""
    X, y = make_classification(
        n_samples=700,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        flip_y=0.08,
        class_sep=0.9,
        random_state=7,
    )
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
    X_calib, X_new, y_calib, y_new = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
    model = RandomForestClassifier(n_estimators=40, random_state=0, min_samples_leaf=4).fit(
        X_train, y_train
    )
    return model, X_calib, y_calib, X_new, y_new


def _proba_predictor(model):
    """Wrap a scikit-learn classifier so that `predict` returns probabilities."""

    class _Proba(BasePredictor):
        def predict(self, X, **kwargs):
            return self.model.predict_proba(X, **kwargs)

    return _Proba(model, is_trained=True)


@pytest.fixture
def fitted_router(three_class_data):
    model, X_calib, y_calib, _, _ = three_class_data
    router = BatchRouter(
        model,
        label_map={0: "billing", 1: "shipping", 2: "account"},
        review_label="review",
    )
    router.fit(X_calib=X_calib, y_calib=y_calib)
    return router


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exact solver ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _brute_force(allowed, costs, review_costs, min_volumes, max_volumes, cap):
    n_records, n_categories = allowed.shape
    options = [
        [j for j in range(n_categories) if allowed[i, j]] + [REVIEW] for i in range(n_records)
    ]
    best = None
    for combo in itertools.product(*options):
        counts = np.zeros(n_categories, dtype=int)
        n_review = 0
        cost = 0
        for i, choice in enumerate(combo):
            if choice == REVIEW:
                n_review += 1
                cost += int(review_costs[i])
            else:
                counts[choice] += 1
                cost += int(costs[i, choice])
        if n_review > cap:
            continue
        if np.any(counts < min_volumes) or np.any(counts > max_volumes):
            continue
        if best is None or cost < best:
            best = cost
    return best


@pytest.mark.parametrize("seed", list(range(12)))
def test_solver_matches_brute_force(seed):
    rng = np.random.RandomState(seed)
    n_records = rng.randint(2, 6)
    n_categories = rng.randint(1, 4)
    allowed = rng.rand(n_records, n_categories) < 0.45
    costs = rng.randint(0, 7, size=(n_records, n_categories))
    review_costs = rng.randint(0, 7, size=n_records)
    min_volumes = rng.randint(0, 3, size=n_categories)
    max_volumes = min_volumes + rng.randint(0, 3, size=n_categories)
    cap = rng.randint(0, n_records + 1)

    reference = _brute_force(allowed, costs, review_costs, min_volumes, max_volumes, cap)
    try:
        assignment, cost = solve_batch_routing(
            allowed=allowed,
            costs=costs,
            review_costs=review_costs,
            min_volumes=min_volumes,
            max_volumes=max_volumes,
            review_capacity=cap,
        )
    except RoutingInfeasibleError:
        assert reference is None
        return

    assert reference is not None
    assert cost == reference

    # The returned plan is feasible and inside the native sets.
    counts = np.bincount(assignment[assignment != REVIEW], minlength=n_categories)
    assert np.all(counts >= min_volumes) and np.all(counts <= max_volumes)
    assert int(np.sum(assignment == REVIEW)) <= cap
    for record, choice in enumerate(assignment):
        assert choice == REVIEW or allowed[record, choice]
        if not allowed[record].any():
            assert choice == REVIEW


def test_solver_empty_sets_are_review_only():
    allowed = np.array(
        [
            [True, False],
            [False, False],
            [False, True],
        ]
    )
    assignment, cost = solve_batch_routing(allowed=allowed, review_capacity=3)
    assert assignment[1] == REVIEW
    assert cost == 0


def test_solver_global_optimum_can_differ_from_greedy():
    # Both records can go to category 0; only record 0 can go to category 1.
    # Per-row greedy sends both to category 0, which overflows its capacity.
    # The global optimum moves record 0 to category 1 instead.
    allowed = np.array([[True, True], [True, False]])
    costs = np.array([[0, 1], [0, 1]])
    assignment, cost = solve_batch_routing(
        allowed=allowed,
        costs=costs,
        max_volumes=[1, 1],
        review_capacity=0,
    )
    assert assignment.tolist() == [1, 0]
    assert cost == 1


def test_solver_lower_bounds_force_a_non_greedy_choice():
    # Category 1 must receive at least one record though it is never the
    # cheapest option, so a per-row cost minimum cannot satisfy the plan.
    allowed = np.array([[True, True], [True, False]])
    costs = np.array([[0, 4], [0, 4]])
    assignment, cost = solve_batch_routing(
        allowed=allowed,
        costs=costs,
        min_volumes=[0, 1],
        max_volumes=[2, 2],
        review_capacity=0,
    )
    assert assignment[0] == 1
    assert cost == 4


def test_solver_rejects_malformed_inputs():
    allowed = np.ones((2, 2), dtype=bool)
    with pytest.raises(RoutingValidationError):
        solve_batch_routing(allowed=allowed, costs=[1.5, 2])
    with pytest.raises(RoutingValidationError):
        solve_batch_routing(allowed=allowed, costs=[1, 2, 3])
    with pytest.raises(RoutingValidationError):
        solve_batch_routing(allowed=allowed, min_volumes=[3, 0], max_volumes=[1, 5])
    with pytest.raises(RoutingValidationError):
        solve_batch_routing(allowed=allowed, review_capacity=-1)
    with pytest.raises(RoutingValidationError):
        solve_batch_routing(allowed=allowed, min_volumes=2)


def test_solver_status_mapping(monkeypatch):
    allowed = np.ones((2, 1), dtype=bool)

    class _Result:
        def __init__(self, status, x=None, fun=0.0):
            self.status = status
            self.x = x
            self.fun = fun

    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "milp", lambda *a, **k: _Result(1, x=np.zeros(4)))
    with pytest.raises(RoutingSolverError):
        solve_batch_routing(allowed=allowed)

    monkeypatch.setattr(scipy.optimize, "milp", lambda *a, **k: _Result(2, x=None))
    with pytest.raises(RoutingInfeasibleError):
        solve_batch_routing(allowed=allowed)

    # A fractional plan must never be exposed.
    monkeypatch.setattr(
        scipy.optimize,
        "milp",
        lambda *a, **k: _Result(0, x=np.full(4, 0.5), fun=0.0),
    )
    with pytest.raises(RoutingSolverError):
        solve_batch_routing(allowed=allowed)

    # A plan whose objective does not match its cost is rejected.
    def _bad_plan(*args, **kwargs):
        x = np.zeros(4)
        x[0] = 1.0  # record 0 -> category 0
        x[2] = 1.0  # record 1 -> category 0
        return _Result(0, x=x, fun=42.0)

    monkeypatch.setattr(scipy.optimize, "milp", _bad_plan)
    with pytest.raises(RoutingSolverError):
        solve_batch_routing(allowed=allowed)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BatchRouter ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_router_returns_real_probabilities_and_native_sets(three_class_data, fitted_router):
    model, _, _, X_new, _ = three_class_data
    costs = {0: 1, 1: 3, 2: 2}
    result = fitted_router.route(
        X_new,
        alpha=0.2,
        costs=costs,
        review_cost=6,
        max_volumes=[len(X_new)] * 3,
        review_capacity=len(X_new),
    )

    # Actual probabilities come from the fixed predictor.
    expected = model.predict_proba(X_new)
    np.testing.assert_allclose(result.probabilities.to_numpy(), expected)
    assert list(result.probabilities.columns) == ["billing", "shipping", "account"]
    np.testing.assert_allclose(result.probabilities.sum(axis=1).to_numpy(), np.ones(len(X_new)))

    # The sets are the native LAC sets: compare with a standalone LAC fitted
    # on the exact same calibration data and the same fixed predictor.
    _, X_calib, y_calib, _, _ = three_class_data
    reference = LAC(_proba_predictor(model), train=False)
    reference.fit(X_calib=X_calib, y_calib=y_calib)
    _, expected_sets = reference.predict(X_new, alpha=0.2)
    assert [list(s) for s in result.prediction_sets] == [list(s) for s in expected_sets]


def test_details_alignment_and_statistics_from_details(fitted_router, three_class_data):
    _, _, _, X_new, _ = three_class_data
    ids = [f"REC-{i:04d}" for i in range(len(X_new))]
    result = fitted_router.route(
        X_new,
        alpha=0.25,
        ids=ids,
        costs=np.array([2, 5, 1]),
        review_cost=7,
        min_volumes=[5, 5, 5],
        max_volumes=[len(X_new)] * 3,
        review_capacity=len(X_new),
    )

    assert result.details["id"].tolist() == ids
    assert result.ids == ids
    assert result.probabilities.index.tolist() == ids

    # Each record is routed inside its native set, or to review.
    for row, pred_set in zip(result.details.itertuples(), result.prediction_sets):
        if row.is_review:
            assert row.assigned_index == REVIEW
            assert row.queue == "review"
            assert np.isnan(row.assigned_probability)
        else:
            assert row.assigned_index in pred_set

    # Empty sets can only be reviewed.
    for row, pred_set in zip(result.details.itertuples(), result.prediction_sets):
        if len(pred_set) == 0:
            assert row.is_review

    # Queue volumes and costs are exactly the ones counted from the details.
    details = result.details
    for label in ("billing", "shipping", "account"):
        mask = details["queue"] == label
        assert result.queue_volumes[label] == int(mask.sum())
        assert result.queue_costs[label] == int(details.loc[mask, "cost"].sum())
    assert result.review_count == int(details["is_review"].sum())
    assert result.review_cost == int(details.loc[details["is_review"], "cost"].sum())
    assert result.total_cost == int(details["cost"].sum())
    assert result.summary()["total_cost"] == result.total_cost


def test_min_volume_is_enforced_globally(fitted_router, three_class_data):
    _, _, _, X_new, _ = three_class_data
    n = len(X_new)
    result = fitted_router.route(
        X_new,
        alpha=0.2,
        costs=np.array([0, 0, 0]),
        review_cost=0,
        min_volumes=[10, 10, 10],
        max_volumes=[n, n, n],
        review_capacity=n,
    )
    assert all(result.queue_volumes[label] >= 10 for label in result.queue_volumes)


def test_infeasible_reports_reasons_and_returns_nothing(three_class_data, fitted_router):
    _, _, _, X_new, _ = three_class_data
    with pytest.raises(RoutingInfeasibleError) as excinfo:
        fitted_router.route(
            X_new,
            alpha=0.2,
            min_volumes=[len(X_new) + 1, 0, 0],
            max_volumes=[len(X_new) + 1] * 3,
            review_capacity=len(X_new),
        )
    error = excinfo.value
    assert error.reasons
    assert "No feasible routing plan" in str(error)


def test_review_capacity_cannot_hide_empty_sets(fitted_router, three_class_data):
    _, _, _, X_new, _ = three_class_data
    result = fitted_router.route(X_new, alpha=0.2, review_capacity=len(X_new))
    empty_sets = sum(len(s) == 0 for s in result.prediction_sets)
    if empty_sets == 0:
        pytest.skip("This split produced no empty LAC set.")
    with pytest.raises(RoutingInfeasibleError):
        fitted_router.route(X_new, alpha=0.2, review_capacity=empty_sets - 1)


def test_label_map_validation(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data
    for bad_labels in (
        ["a", "b"],
        {0: "a", 1: "b"},
        {0: "a", 1: "b", 2: "a"},
        ["a", "b", "review"],
        "abc",
    ):
        router = BatchRouter(model, label_map=bad_labels, review_label="review").fit(
            X_calib=X_calib, y_calib=y_calib
        )
        with pytest.raises(RoutingValidationError):
            router.route(X_new[:5], alpha=0.2)

    labels = (
        BatchRouter(model, label_map=["a", "b", "c"])
        .fit(X_calib=X_calib, y_calib=y_calib)
        .route(X_new[:5], alpha=0.2, review_capacity=5)
    )
    assert set(labels.details["queue"].unique()) <= {"a", "b", "c", "manual_review"}
    assert list(labels.probabilities.columns) == ["a", "b", "c"]


def test_business_labels_mapping_in_details(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data
    labels = {0: "billing", 1: "shipping", 2: "account"}
    result = (
        BatchRouter(model, label_map=labels)
        .fit(X_calib=X_calib, y_calib=y_calib)
        .route(X_new, alpha=0.2, review_capacity=len(X_new))
    )
    for _, row in result.details.iterrows():
        expected = [labels[index] for index in row["prediction_set_indices"]]
        assert row["prediction_set"] == expected
        if not row["is_review"]:
            assert row["queue"] == labels[row["assigned_index"]]


def test_prediction_failure_returns_no_plan(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data

    class _Exploding:
        def predict(self, X):
            raise RuntimeError("boom")

    router = BatchRouter(_Exploding())
    with pytest.raises(RoutingPredictionError):
        router.fit(X_calib=X_calib, y_calib=y_calib)

    class _LabelOnly:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    router = BatchRouter(_LabelOnly())
    with pytest.raises(RoutingPredictionError):
        router.fit(X_calib=X_calib, y_calib=y_calib)
    # State is not half-configured.
    with pytest.raises(RoutingValidationError):
        router.route(X_new, alpha=0.2)

    # A model that predicts fine but fails at routing time yields no plan.
    fitted = BatchRouter(model).fit(X_calib=X_calib, y_calib=y_calib)

    class _RouterSideFailure:
        pass

    with pytest.raises(RoutingValidationError):
        fitted.route(_RouterSideFailure(), alpha=0.2)


def test_bad_alpha_rejected(fitted_router, three_class_data):
    _, _, _, X_new, _ = three_class_data
    for alpha in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(RoutingValidationError):
            fitted_router.route(X_new, alpha=alpha)


def test_duplicate_ids_rejected(fitted_router, three_class_data):
    _, _, _, X_new, _ = three_class_data
    with pytest.raises(RoutingValidationError):
        fitted_router.route(X_new, alpha=0.2, ids=["x"] * len(X_new))


def test_route_batch_matches_router(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data
    kwargs = dict(
        alpha=0.2,
        costs=[1, 2, 3],
        review_cost=4,
        min_volumes=[2, 2, 2],
        max_volumes=[len(X_new)] * 3,
        review_capacity=len(X_new),
        ids=list(range(len(X_new))),
        label_map=["a", "b", "c"],
        review_label="review",
    )
    direct = route_batch(model, X_calib=X_calib, y_calib=y_calib, X=X_new, **kwargs)
    route_kwargs = dict(kwargs)
    router = BatchRouter(
        model,
        label_map=route_kwargs.pop("label_map"),
        review_label=route_kwargs.pop("review_label"),
    ).fit(X_calib=X_calib, y_calib=y_calib)
    via_router = router.route(X_new, **route_kwargs)
    assert direct.details.equals(via_router.details)
    assert direct.total_cost == via_router.total_cost


def test_router_accepts_external_lac(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data
    lac = LAC(_proba_predictor(model), train=False)
    lac.fit(X_calib=X_calib, y_calib=y_calib)
    router = BatchRouter(lac)
    result = router.route(X_new, alpha=0.2, review_capacity=len(X_new))
    assert result.total_cost == 0
    assert result.details["id"].tolist() == list(range(len(X_new)))


def test_router_accepts_base_predictor_wrapper(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data

    class _SkProba(BasePredictor):
        def predict(self, X, **kwargs):
            return self.model.predict_proba(X, **kwargs)

    predictor = _SkProba(model, is_trained=True)
    router = BatchRouter(predictor).fit(X_calib=X_calib, y_calib=y_calib)
    result = router.route(X_new[:20], alpha=0.2)
    np.testing.assert_allclose(result.probabilities.to_numpy(), model.predict_proba(X_new[:20]))


def test_training_mode_requires_fit_data(three_class_data):
    model, X_calib, y_calib, _, _ = three_class_data
    router = BatchRouter(RandomForestClassifier(n_estimators=10, random_state=0), train=True)
    with pytest.raises(RoutingValidationError):
        router.fit(X_calib=X_calib, y_calib=y_calib)


def test_missing_calibration_set_rejected(three_class_data):
    model, _, _, _, _ = three_class_data
    with pytest.raises(RoutingValidationError):
        BatchRouter(model).fit(X_calib=None, y_calib=None)
    with pytest.raises(RoutingValidationError):
        BatchRouter(model).route(np.zeros((2, 6)), alpha=0.2)


def test_legacy_lac_interface_unchanged(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data
    cp = LAC(_proba_predictor(model), train=False)
    cp.fit(X_calib=X_calib, y_calib=y_calib)
    y_pred, set_pred = cp.predict(X_new, alpha=0.2)
    assert isinstance(set_pred, list)
    assert len(set_pred) == len(X_new)
    np.testing.assert_allclose(np.asarray(y_pred), model.predict_proba(X_new))


def test_no_set_enlargement_when_capacity_tight(fitted_router, three_class_data):
    _, _, _, X_new, _ = three_class_data
    n = len(X_new)
    # Tight per-category maxima force most records to review; sets must stay
    # untouched and every assignment must remain inside the original set.
    result = fitted_router.route(
        X_new,
        alpha=0.2,
        max_volumes=[3, 3, 3],
        review_capacity=n,
    )
    for row, pred_set in zip(result.details.itertuples(), result.prediction_sets):
        if row.is_review:
            continue
        assert row.assigned_index in pred_set
    for volume in result.queue_volumes.values():
        assert volume <= 3


def test_prediction_and_solver_errors_surface_without_plan(three_class_data, monkeypatch):
    import deel.puncc.routing as routing_module

    model, X_calib, y_calib, X_new, _ = three_class_data

    class _BoomAtRouting:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def predict_proba(self, X):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("predict failed")
            return self.inner.predict_proba(X)

    boom = _BoomAtRouting(model)
    router = BatchRouter(boom).fit(X_calib=X_calib, y_calib=y_calib)
    with pytest.raises(RoutingPredictionError):
        router.route(X_new, alpha=0.2)

    router = BatchRouter(model).fit(X_calib=X_calib, y_calib=y_calib)

    def _raise_solver(**kwargs):
        del kwargs
        raise RoutingSolverError("optimizer unavailable")

    monkeypatch.setattr(routing_module, "solve_batch_routing", _raise_solver)
    with pytest.raises(RoutingSolverError):
        router.route(X_new, alpha=0.2)


def test_volumes_and_review_costs_validation(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data
    router = BatchRouter(model).fit(X_calib=X_calib, y_calib=y_calib)
    with pytest.raises(RoutingValidationError):
        router.route(X_new, alpha=0.2, review_capacity=-3)
    with pytest.raises(RoutingValidationError):
        router.route(X_new, alpha=0.2, review_capacity=2.5)
    with pytest.raises(RoutingValidationError):
        router.route(X_new, alpha=0.2, max_volumes=[1.2, 1, 1])
    with pytest.raises(RoutingValidationError):
        router.route(X_new, alpha=0.2, min_volumes=[5, 0, 0], max_volumes=[1, 1, 1])
    with pytest.raises(RoutingValidationError):
        router.route(X_new, alpha=0.2, costs=np.array([[1, 2], [3, 4]]))


def test_empty_new_batch_rejected(fitted_router):
    with pytest.raises(RoutingValidationError):
        fitted_router.route([], alpha=0.2)


def test_route_with_boolean_and_non_integral_cost_values(three_class_data):
    model, X_calib, y_calib, X_new, _ = three_class_data
    router = BatchRouter(model).fit(X_calib=X_calib, y_calib=y_calib)
    with pytest.raises(RoutingValidationError):
        router.route(X_new, alpha=0.2, costs=[True, False, True])
