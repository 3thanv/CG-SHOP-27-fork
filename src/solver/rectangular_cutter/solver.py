"""Exact CP-SAT solver for one cutter on cell-based polyomino instances."""

from collections import deque
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


def _check_supported(instance: CGSHOP2027Instance) -> tuple[
    set[tuple[int, int]], set[tuple[int, int]]
]:
    if instance.number_of_cutters != 1:
        raise ValueError("The exact solver supports exactly one cutter.")

    region_cells = set(rasterize(instance.region_to_cover).cells())
    if not region_cells:
        raise ValueError("The region contains no grid cells.")

    cutter_cells = set(rasterize_ring(instance.cutter).cells())
    if not cutter_cells:
        raise ValueError("The cutter contains no grid cells.")

    return region_cells, cutter_cells


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


def _shortest_paths_from(
    source: int, neighbors: list[list[int]]
) -> list[list[int] | None]:
    predecessor: list[int | None] = [None] * len(neighbors)
    predecessor[source] = source
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in neighbors[current]:
            if predecessor[neighbor] is None:
                predecessor[neighbor] = current
                queue.append(neighbor)

    paths: list[list[int] | None] = [None] * len(neighbors)
    for target, parent in enumerate(predecessor):
        if parent is None:
            continue
        path = []
        current = target
        while current != source:
            path.append(current)
            current = predecessor[current]
            if current is None:
                raise RuntimeError("Broken predecessor chain in shortest paths.")
        path.append(source)
        path.reverse()
        paths[target] = path
    return paths


def _check_connected(
    centers: list[tuple[int, int]], transitions: list[tuple[int, int]]
) -> None:
    neighbors: dict[int, list[int]] = {index: [] for index in range(len(centers))}
    for start, end in transitions:
        if start != end:
            neighbors[start].append(end)

    seen = {0}
    queue = deque([0])
    while queue:
        for neighbor in neighbors[queue.popleft()]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(seen) != len(centers):
        raise ValueError(
            "Candidate cutter centers are disconnected by unit moves; "
            "this instance is unsupported by the exact solver."
        )


def _transition_neighbors(
    center_count: int, transitions: list[tuple[int, int]]
) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(center_count)]
    for start, end in transitions:
        neighbors[start].append(end)
    return neighbors


def _greedy_closed_walk(
    centers: list[tuple[int, int]],
    covering_centers: dict[tuple[int, int], list[int]],
    neighbors: list[list[int]],
) -> list[int]:
    covered_by_center = [set() for _ in centers]
    for cell, indices in covering_centers.items():
        for index in indices:
            covered_by_center[index].add(cell)
    all_cells = set(covering_centers)

    def build_walk(start: int) -> list[int]:
        walk = [start]
        covered = set(covered_by_center[start])
        current = start
        while covered != all_cells:
            shortest_paths = _shortest_paths_from(current, neighbors)
            best: tuple[float, int, int, list[int]] | None = None
            for target, path in enumerate(shortest_paths):
                if path is None:
                    continue
                gain = len(covered_by_center[target] - covered)
                distance = len(path) - 1
                if not gain:
                    continue
                score = (gain / max(distance, 1), gain, -distance, path)
                if best is None or score[:3] > best[:3]:
                    best = score
            if best is None:
                raise ValueError("Could not construct a covering walk.")
            path = best[3]
            walk.extend(path[1:])
            for index in path[1:]:
                covered.update(covered_by_center[index])
            current = path[-1]

        shortest_paths = _shortest_paths_from(current, neighbors)
        path_home = shortest_paths[start]
        if path_home is None:
            raise ValueError("Could not close the covering walk.")
        walk.extend(path_home[1:])
        return walk

    start_candidates = sorted(
        range(len(centers)),
        key=lambda index: len(covered_by_center[index]),
        reverse=True,
    )[: min(24, len(centers))]
    walks = [build_walk(start) for start in start_candidates]
    return min(walks, key=len)


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
    """Solve a supported instance, raising ValueError if no tour is found."""
    region_cells, cutter_cells = _check_supported(instance)
    centers, covering_centers = _build_candidate_centers(
        instance, region_cells, cutter_cells
    )
    transitions = _transitions(centers)
    _check_connected(centers, transitions)
    neighbors = _transition_neighbors(len(centers), transitions)
    initial_walk = _greedy_closed_walk(centers, covering_centers, neighbors)

    # A feasible closed walk is an upper bound on the optimal tour length.
    horizon = max(1, len(initial_walk) - 1)
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
    initial_length = len(initial_walk) - 1
    for step, position in enumerate(positions):
        model.AddHint(
            position,
            initial_walk[step] if step <= initial_length else initial_walk[-1],
        )
    for step, is_active in enumerate(active):
        model.AddHint(is_active, int(step < initial_length))
    model.AddHint(tour_length, initial_length)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_TIME_SECONDS
    solver.parameters.num_search_workers = 8
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
            "initial_tour_length": initial_length,
            "best_bound": solver.BestObjectiveBound(),
            "solve_seconds": solve_seconds,
        },
    )
    errors = SolutionValidator(instance).check_for_errors(solution)
    if errors:
        raise ValueError("Solver produced an invalid solution:\n" + "\n".join(errors))
    return solution
