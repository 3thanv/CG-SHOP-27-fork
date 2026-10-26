"""Compare the existing step model with edge_solver under matching solve budgets.

The existing solver's exporter repeats the starting point, which the official
schema rejects. This harness patches ONLY that exporter in memory, leaving its
optimization model unchanged. Worker counts and solver time limits are matched.
End-to-end timings include model construction. No existing source is edited.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from cgshop2027_pyutils.io import read_instance
from cgshop2027_pyutils.schemas import CutterTour
from ortools.sat.python import cp_model

import edge_solver
import solver as step_solver


def _reference_tour(solver, positions, centers, length):
    points = [centers[solver.value(p)] for p in positions[:max(1, length)]]
    return CutterTour(x=[x for x, _ in points], y=[y for _, y in points])


def compare(instance, *, time_limit, workers):
    rows = []
    if instance.number_of_cutters == 1:
        models, solvers = [], []
        original_model, original_solver = cp_model.CpModel, cp_model.CpSolver

        def make_model():
            model = original_model()
            models.append(model)
            return model

        def make_solver():
            solver = original_solver()
            solver.parameters.num_search_workers = workers
            solvers.append(solver)
            return solver

        started = perf_counter()
        with (
            patch.object(step_solver, "_make_tour", _reference_tour),
            patch.object(step_solver, "MAX_DURATION", time_limit),
            patch.object(cp_model, "CpModel", make_model),
            patch.object(cp_model, "CpSolver", make_solver),
        ):
            try:
                result = step_solver.solve(instance)
                longest = result.max_tour_length
            except TimeoutError:
                longest = None
        rows.append({
            "formulation": "step_reference_export_fixed",
            "seconds": perf_counter() - started,
            "longest": longest,
            "status": solvers[0].status_name(solvers[0].response_proto.status),
            "best_bound": solvers[0].best_objective_bound,
            "model_variables": len(models[0].proto.variables),
            "model_constraints": len(models[0].proto.constraints),
        })

    started = perf_counter()
    result = edge_solver.solve(instance, time_limit=time_limit, num_workers=workers)
    rows.append({
        "formulation": "edge_euler",
        "seconds": perf_counter() - started,
        "longest": result.max_tour_length,
        **{key: result.meta[key] for key in (
            "status", "optimal", "best_bound", "initial_max_tour_length", "used_fallback",
            "model_variables", "model_constraints", "preprocessing_seconds", "build_seconds",
        )},
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", type=Path, nargs="+")
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.instances:
        instance = read_instance(path)
        rows = compare(instance, time_limit=args.time_limit, workers=args.workers)
        results.append({"instance": str(path), "cutters": instance.number_of_cutters, "runs": rows})
        for row in rows:
            print(f"{path.name}: {row['formulation']}, length={row['longest']}, "
                  f"status={row['status']}, variables={row['model_variables']}, "
                  f"total={row['seconds']:.3f}s", flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
