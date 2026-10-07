"""Exact CP-SAT solver for one cutter on cell-based polyomino instances."""

from time import perf_counter

from ortools.sat.python import cp_model

from cgshop2027_pyutils.grid import rasterize, rasterize_ring
from cgshop2027_pyutils.schemas import (
    CGSHOP2027Instance,
    CGSHOP2027Solution,
    CutterTour,
)
from cgshop2027_pyutils.verify import SolutionValidator

MAX_TIME_SECONDS = 50.0
MAX_CANDIDATE_CENTERS = 200


def _compress_closed_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove waits and straight-through positions from a closed unit-step path."""
    clean = [path[0]]
    for point in path[1:]:
        if point != clean[-1]:
            clean.append(point)
    if len(clean) == 1:
        return clean
    if clean[-1] != clean[0]:
        raise ValueError("The CP-SAT path is not closed.")

    vertices = clean[:-1]
    corners = []
    for index, current in enumerate(vertices):
        previous = vertices[index - 1]
        following = vertices[(index + 1) % len(vertices)]
        incoming = (current[0] - previous[0], current[1] - previous[1])
        outgoing = (following[0] - current[0], following[1] - current[1])
        if incoming != outgoing:
            corners.append(current)
    return corners or [vertices[0]]


def _build_candidate_centers(
    instance: CGSHOP2027Instance,
    region_cells: set[tuple[int, int]],
    cutter_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[int]]]:
    center_x, center_y = instance.cutter_center
    offsets = sorted((x - center_x, y - center_y) for x, y in cutter_cells)
    covering_positions = set()
    for cell_x, cell_y in region_cells:
        for offset_x, offset_y in offsets:
            covering_positions.add((cell_x - offset_x, cell_y - offset_y))
            if len(covering_positions) > MAX_CANDIDATE_CENTERS:
                raise ValueError(
                    "The exact solver supports at most "
                    f"{MAX_CANDIDATE_CENTERS} candidate cutter centers; "
                    "this instance has more."
                )

    min_x = min(x for x, _ in covering_positions)
    max_x = max(x for x, _ in covering_positions)
    min_y = min(y for _, y in covering_positions)
    max_y = max(y for _, y in covering_positions)
    centers = [
        (x, y)
        for x in range(min_x, max_x + 1)
        for y in range(min_y, max_y + 1)
    ]
    center_index = {center: index for index, center in enumerate(centers)}
    covering_centers = {
        (cell_x, cell_y): [
            center_index[(cell_x - offset_x, cell_y - offset_y)]
            for offset_x, offset_y in offsets
        ]
        for cell_x, cell_y in region_cells
    }
    return centers, covering_centers


def _transitions(centers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    center_index = {center: index for index, center in enumerate(centers)}
    allowed = []
    for start, (x, y) in enumerate(centers):
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            end = center_index.get(neighbor)
            if end is not None:
                allowed.append((start, end))
    return allowed


def _make_tour(
    solver: cp_model.CpSolver,
    positions: list[cp_model.IntVar],
    centers: list[tuple[int, int]],
    length: int,
) -> CutterTour:
    path = [centers[solver.Value(position)] for position in positions[: length + 1]]
    corners = _compress_closed_path(path)
    return CutterTour(x=[x for x, _ in corners], y=[y for _, y in corners])


def solve(instance: CGSHOP2027Instance) -> CGSHOP2027Solution:
    region_cells = set(rasterize(instance.region_to_cover).cells())
    cutter_cells = set(rasterize_ring(instance.cutter).cells())
    centers, covering_centers = _build_candidate_centers(
        instance, region_cells, cutter_cells
    )
    transitions = _transitions(centers)

    # Walking every edge of a spanning tree and returning to its root gives a
    # feasible closed tour through all candidate and transit positions.
    horizon = max(1, 2 * (len(centers) - 1))
    model = cp_model.CpModel()
    positions = [
        model.NewIntVar(0, len(centers) - 1, f"position_{step}")
        for step in range(horizon + 1)
    ]
    model.Add(positions[0] == positions[horizon])

    active = [
        model.NewBoolVar(f"active_{step}") for step in range(horizon)
    ]
    transition_states = [
        (start, end, 1) for start, end in transitions
    ] + [(center, center, 0) for center in range(len(centers))]
    for step in range(horizon):
        model.AddAllowedAssignments(
            [positions[step], positions[step + 1], active[step]],
            transition_states,
        )
        if step:
            model.Add(active[step - 1] >= active[step])

    tour_length = model.NewIntVar(0, horizon, "tour_length")
    model.Add(sum(active) == tour_length)
    model.AddElement(tour_length, positions, positions[0])

    for cell_index, (cell, covering_indices) in enumerate(covering_centers.items()):
        covered_steps = []
        valid_positions = [(center,) for center in covering_indices]
        for step, position in enumerate(positions):
            covered = model.NewBoolVar(f"covered_{cell_index}_{step}")
            model.AddAllowedAssignments([position], valid_positions).OnlyEnforceIf(
                covered
            )
            covered_steps.append(covered)
        model.Add(
            sum(covered_steps) >= 1
        )

    model.Minimize(tour_length)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_TIME_SECONDS
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True
    started_at = perf_counter()
    status = solver.Solve(model)
    solve_seconds = perf_counter() - started_at
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if status == cp_model.INFEASIBLE:
            raise ValueError("No feasible tour exists for this instance.")
        raise TimeoutError(
            f"CP-SAT found no tour within {MAX_TIME_SECONDS:g} seconds "
            f"(status: {solver.StatusName(status)})."
        )

    solution = CGSHOP2027Solution(
        instance_uid=instance.instance_uid,
        tours=[
            _make_tour(
                solver,
                positions,
                centers,
                solver.Value(tour_length),
            )
        ],
        meta={
            "algorithm": "rectangular_cutter_cp_sat",
            "status": solver.StatusName(status),
            "optimal": status == cp_model.OPTIMAL,
            "best_bound": solver.BestObjectiveBound(),
            "solve_seconds": solve_seconds,
        },
    )
    errors = SolutionValidator(instance).check_for_errors(solution)
    if errors:
        raise ValueError("Solver produced an invalid solution:\n" + "\n".join(errors))
    return solution
