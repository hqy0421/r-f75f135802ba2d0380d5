.. _routing:

🔀 Batch Routing
================

:mod:`deel.puncc.routing` implements calibration-driven batch routing for
conformal classification.

A batch of new records is predicted by a **fixed** probability predictor, both
on an independent calibration set and on the new records, and native
:class:`deel.puncc.classification.LAC` prediction sets are built. Every record
must then be routed to exactly **one** category contained in its LAC set, or to
a manual review queue.

The routing problem is solved **globally**: per-category minimum and maximum
processing volumes and a total manual review cap are enforced, and the integer
routing cost summed over the whole batch is minimized. This is not a per-record
"pick the cheapest category" rule, which may overflow queue capacities.

The following rules are enforced and never relaxed:

* A record is only assigned to a category contained in its native LAC set.
* A record with an empty LAC set can only be sent to manual review: the set is
  never enlarged and the most likely class is never forced in.
* Prediction sets are never modified to absorb capacity.
* If no feasible plan exists, :class:`deel.puncc.api.routing.RoutingInfeasibleError`
  is raised with the reasons; no partial plan is returned.
* Prediction, calibration or optimization failures raise an exception; no
  partial plan is returned.
* Category columns follow the declared business labels (``label_map``), and
  ``prediction_sets`` keep the native integer class indices.

The returned :class:`deel.puncc.routing.RoutingResult` exposes the actual
probability matrix, the native LAC sets, the per-record routing details aligned
on the original sample identifiers, as well as the queue volumes and costs
**counted from those details**.

.. code-block:: python

    from deel.puncc.routing import BatchRouter

    router = BatchRouter(
        pretrained_predictor,
        label_map={0: "billing", 1: "shipping", 2: "account"},
        review_label="manual_review",
    )
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

    result.probabilities     # actual predicted probabilities
    result.prediction_sets   # native LAC sets (class indices)
    result.details           # per-record routing details, aligned on ids
    result.queue_volumes     # queue volumes counted from `details`
    result.total_cost        # integer total cost counted from `details`

Driver
------

.. autoclass:: deel.puncc.routing.BatchRouter

.. autofunction:: deel.puncc.routing.route_batch

.. autoclass:: deel.puncc.routing.RoutingResult

Low-level exact solver
----------------------

.. autofunction:: deel.puncc.api.routing.solve_batch_routing

Errors
------

.. autoexception:: deel.puncc.api.routing.RoutingError
.. autoexception:: deel.puncc.api.routing.RoutingValidationError
.. autoexception:: deel.puncc.api.routing.RoutingPredictionError
.. autoexception:: deel.puncc.api.routing.RoutingInfeasibleError
.. autoexception:: deel.puncc.api.routing.RoutingSolverError
