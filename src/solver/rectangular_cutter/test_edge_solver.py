"""Spec regressions and independent exhaustive checks for edge_solver."""

import subprocess
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

from cgshop2027_pyutils.io import read_instance, read_solution
from cgshop2027_pyutils.schemas import CGSHOP2027Instance
from cgshop2027_pyutils.verify import SolutionValidator
from ortools.sat.python import cp_model

from edge_solver import Walk, _build_model, _greedy, _grid, _hint, _tour, solve


def rectangle(width, height, *, robots=1, cutter_width=1, center=(0, 0), offset=(0, 0)):
    x, y = offset
    return CGSHOP2027Instance.model_validate({
        "instance_uid": "edge-test",
        "region_to_cover": {
            "outer_boundary": {
                "x": [x, x + width, x + width, x],
                "y": [y, y, y + height, y + height],
            },
            "inner_boundaries": [],
        },
        "cutter": {"x": [0, cutter_width, cutter_width, 0], "y": [0, 0, 1, 1]},
        "cutter_center": center,
        "number_of_cutters": robots,
    })


def exhaustive_optimum(problem):
    """Enumerate short closed walks, then combine their coverage for k robots.

    This independent reference has no edge-count, parity, or flow variables.
    Restrict its use to tiny lawns: its state space is exponential in cells.
    """
    grid = _grid(problem)
    all_cells = (1 << len(grid.cells)) - 1
    costs = {}
    for start in range(len(grid.centers)):
        initial = start, grid.masks[start]
        distances = {initial: 0}
        queue = deque([initial])
        while queue:
            vertex, covered = queue.popleft()
            distance = distances[vertex, covered]
            if vertex == start:
                costs[covered] = min(costs.get(covered, float("inf")), distance)
            for neighbor, _ in grid.adjacency[vertex]:
                state = neighbor, covered | grid.masks[neighbor]
                if state not in distances:
                    distances[state] = distance + 1
                    queue.append(state)
    combined = {0: 0}
    for _ in range(problem.number_of_cutters):
        next_costs = {}
        for covered, longest in combined.items():
            for additional, length in costs.items():
                union = covered | additional
                value = max(longest, length)
                next_costs[union] = min(next_costs.get(union, float("inf")), value)
        combined = next_costs
    return combined[all_cells]


class EdgeSolverTests(unittest.TestCase):
    def assert_valid(self, problem, solution):
        self.assertEqual(len(solution.tours), problem.number_of_cutters)
        self.assertEqual(SolutionValidator(problem).check_for_errors(solution), [])
        self.assertLessEqual(solution.max_tour_length, solution.meta["initial_max_tour_length"])
        self.assertLessEqual(solution.meta["best_bound"], solution.max_tour_length)
        for tour in solution.tours:
            positions = list(tour.positions())
            self.assertEqual(positions[0], positions[-1])
            if len(tour.x) > 1:
                self.assertNotEqual((tour.x[0], tour.y[0]), (tour.x[-1], tour.y[-1]))

    def test_matches_exhaustive_walks_for_small_single_and_multi_cutter_cases(self):
        cases = [
            (1, 1, 1, 1), (2, 1, 1, 1), (3, 1, 2, 1),
            (2, 2, 1, 1), (2, 2, 2, 1), (3, 2, 1, 1),
            (3, 2, 2, 1), (3, 2, 1, 2), (3, 2, 2, 2),
        ]
        for width, height, robots, cutter_width in cases:
            with self.subTest(case=(width, height, robots, cutter_width)):
                problem = rectangle(width, height, robots=robots, cutter_width=cutter_width)
                solution = solve(problem, time_limit=5, num_workers=1)
                self.assert_valid(problem, solution)
                self.assertTrue(solution.meta["optimal"])
                self.assertEqual(solution.max_tour_length, exhaustive_optimum(problem))

    def test_minimizes_longest_tour_for_multiple_cutters(self):
        problem = rectangle(8, 1, robots=2)
        solution = solve(problem, time_limit=5, num_workers=1)
        self.assert_valid(problem, solution)
        self.assertTrue(solution.meta["optimal"])
        self.assertEqual(solution.max_tour_length, 6)
        self.assertEqual(sum(t.length for t in solution.tours), 12)

    def test_stationary_and_unused_cutters_are_legal(self):
        problem = rectangle(2, 1, robots=4)
        solution = solve(problem, time_limit=5, num_workers=1)
        self.assert_valid(problem, solution)
        self.assertTrue(solution.meta["optimal"])
        self.assertEqual(solution.max_tour_length, 0)
        self.assertTrue(all(len(tour.x) == 1 for tour in solution.tours))

    def test_no_fixed_depot_and_local_center_offset(self):
        problem = rectangle(3, 1, cutter_width=2, center=(1, 0), offset=(-8, 11))
        solution = solve(problem, time_limit=5, num_workers=1)
        self.assert_valid(problem, solution)
        self.assertTrue(solution.meta["optimal"])
        self.assertEqual(solution.max_tour_length, 2)

    def test_region_holes_are_legal_transit_positions(self):
        problem = rectangle(3, 3)
        data = problem.model_dump()
        data["region_to_cover"]["inner_boundaries"] = [
            {"x": [1, 1, 2, 2], "y": [1, 2, 2, 1]}
        ]
        problem = CGSHOP2027Instance.model_validate(data)
        grid = _grid(problem)
        hole = grid.index[1, 1]
        self.assertEqual(grid.masks[hole], 0)
        self.assertEqual(len(grid.adjacency[hole]), 4)
        solution = solve(problem, time_limit=5, num_workers=1)
        self.assert_valid(problem, solution)
        self.assertTrue(solution.meta["optimal"])
        self.assertEqual(solution.max_tour_length, 8)

    def test_nonrectangular_cutter(self):
        problem = rectangle(3, 2)
        data = problem.model_dump()
        data["cutter"] = {"x": [0, 2, 2, 1, 1, 0], "y": [0, 0, 1, 1, 2, 2]}
        problem = CGSHOP2027Instance.model_validate(data)
        solution = solve(problem, time_limit=5, num_workers=1)
        self.assert_valid(problem, solution)
        self.assertTrue(solution.meta["optimal"])
        self.assertEqual(solution.max_tour_length, exhaustive_optimum(problem))

    def test_connectivity_rejects_disconnected_cycles(self):
        grid = _grid(rectangle(4, 2))
        formulation = _build_model(grid, 1, 16)
        # Two disjoint squares have even degrees and cover every cell, but are
        # not one closed walk. Flow conservation must reject this assignment.
        cycles = [((0, 0), (1, 0), (1, 1), (0, 1)),
                  ((2, 0), (3, 0), (3, 1), (2, 1))]
        used = set()
        for cycle in cycles:
            for first, second in zip(cycle, (*cycle[1:], cycle[0])):
                used.add(tuple(sorted((grid.index[first], grid.index[second]))))
        for edge, count in zip(grid.edges, formulation.robots[0].counts):
            formulation.model.add(count == int(edge in used))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        self.assertEqual(solver.solve(formulation.model), cp_model.INFEASIBLE)

    def test_bridges_can_be_traversed_twice(self):
        problem = rectangle(20, 1)
        solution = solve(problem, time_limit=5, num_workers=1)
        self.assert_valid(problem, solution)
        self.assertTrue(solution.meta["optimal"])
        self.assertEqual(solution.max_tour_length, 38)

    def test_valid_greedy_fallback_when_no_search_time_is_available(self):
        problem = rectangle(5, 3, robots=2)
        for limit in (0, 0.000001):
            with self.subTest(limit=limit):
                solution = solve(problem, time_limit=limit, num_workers=1)
                self.assert_valid(problem, solution)
                self.assertTrue(solution.meta["used_fallback"])
                self.assertFalse(solution.meta["optimal"])

    def test_hints_are_optional(self):
        problem = rectangle(3, 2, robots=2)
        solution = solve(problem, time_limit=5, num_workers=1, use_hints=False)
        self.assert_valid(problem, solution)
        self.assertTrue(solution.meta["optimal"])
        self.assertEqual(solution.max_tour_length, 4)

    def test_complete_greedy_hint_is_a_feasible_assignment(self):
        grid = _grid(rectangle(5, 3, robots=2))
        walks = _greedy(grid, 2)
        formulation = _build_model(grid, 2, max(w.length for w in walks))
        _hint(formulation, grid, walks)
        self.assertEqual(len(formulation.model.proto.solution_hint.vars),
                         len(formulation.model.proto.variables))
        solver = cp_model.CpSolver()
        solver.parameters.fix_variables_to_their_hinted_value = True
        solver.parameters.max_time_in_seconds = 5
        self.assertEqual(solver.solve(formulation.model), cp_model.OPTIMAL)

    def test_model_size_does_not_depend_on_the_walk_length_bound(self):
        grid = _grid(rectangle(3, 2, robots=2))
        small = _build_model(grid, 2, 10).model.proto
        large = _build_model(grid, 2, 1000).model.proto
        self.assertEqual(len(small.variables), len(large.variables))
        self.assertEqual(len(small.constraints), len(large.constraints))

    def test_euler_reconstruction_rejects_disconnected_edges(self):
        grid = _grid(rectangle(4, 1))
        with self.assertRaisesRegex(RuntimeError, "connected Eulerian"):
            _tour(grid, Walk(0, [2, 0, 2]))

    def test_cli_writes_standard_solution_readable_by_existing_utils(self):
        directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.solution.json"
            source = directory / "examples/exact-small/rect-4x4.instance.json"
            subprocess.run([sys.executable, str(directory / "edge_solver.py"), str(source), str(output),
                            "--time-limit", "5", "--workers", "1"],
                           check=True, capture_output=True, text=True)
            solution = read_solution(output)
            self.assert_valid(read_instance(source), solution)
            self.assertEqual(solution.content_type, "CGSHOP2027_Solution")
            self.assertEqual(solution.max_tour_length, 10)
            result = subprocess.run([
                sys.executable, str(directory.parents[1] / "utils/verify.py"), str(source), str(output)
            ], check=True, capture_output=True, text=True)
            self.assertIn("got 0 errors", result.stdout)


if __name__ == "__main__":
    unittest.main()
