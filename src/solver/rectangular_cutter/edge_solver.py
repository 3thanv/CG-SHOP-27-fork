"""Horizon-free CP-SAT solver for the CG:SHOP 2027 lawn-mowing problem.

The supplied "Multi-Robot Lawn Mowing for Polyominoes" specs require k closed
rectilinear tours with integer turns and minimize the longest tour. There is
no depot, obstacle, collision, or restriction against leaving the lawn.

This model chooses a connected Eulerian multigraph per cutter. Edge counts
give route lengths directly: there are no position-at-time variables and no
max_steps. Single-commodity flows rule out disconnected cycles. A single
selected vertex represents a legal stationary cutter. Greedy spatial
partitions supply an incumbent, an objective bound, and optional hints.

Two reductions preserve an optimum under those specs:
* All useful integer center positions lie in a finite bounding rectangle.
  Clamping any trajectory to it preserves its useful positions and cannot
  increase its length. Include EVERY lattice vertex inside, including holes
  and positions that cut nothing: the specs allow transit outside the lawn.
* An edge needs multiplicity only 0, 1, or 2. Subtracting two traversals from
  any multiplicity >= 3 preserves positive support, even degrees, and hence
  a connected Eulerian walk, while decreasing its length.

These are exact reductions, not a claim that CP-SAT will prove optimality
within the time limit. Every returned result is checked by the official
SolutionValidator and uses the official CGSHOP2027Solution/CutterTour classes.
"""

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from cgshop2027_pyutils.grid import rasterize, rasterize_ring
from cgshop2027_pyutils.io import read_instance
from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution, CutterTour
from cgshop2027_pyutils.verify import SolutionValidator
from ortools.sat.python import cp_model

Point = tuple[int, int]


@dataclass
class Grid:
    cells: list[Point]
    centers: list[Point]
    index: dict[Point, int]
    edges: list[tuple[int, int]]
    adjacency: list[list[tuple[int, int]]]
    covering: list[list[int]]
    masks: list[int]


@dataclass
class Walk:
    root: int
    counts: list[int]

    @property
    def length(self) -> int:
        return sum(self.counts)


@dataclass
class RobotVariables:
    selected: list
    roots: list
    supply: list
    half_degree: list
    counts: list
    present: list
    flow: list
    root_index: cp_model.IntVar
    length: cp_model.IntVar


@dataclass
class EdgeModel:
    model: cp_model.CpModel
    robots: list[RobotVariables]
    longest: cp_model.IntVar
    half_longest: cp_model.IntVar


def _grid(instance: CGSHOP2027Instance) -> Grid:
    cells = sorted(set(rasterize(instance.region_to_cover).cells()))
    cutter = set(rasterize_ring(instance.cutter).cells())
    if not cells or not cutter:
        raise ValueError("Region and cutter must contain grid cells.")
    cx, cy = instance.cutter_center
    offsets = [(x - cx, y - cy) for x, y in sorted(cutter)]
    placements = [
        [(x - dx, y - dy) for dx, dy in offsets]
        for x, y in cells
    ]
    useful = {p for positions in placements for p in positions}
    left, right = min(x for x, _ in useful), max(x for x, _ in useful)
    bottom, top = min(y for _, y in useful), max(y for _, y in useful)
    centers = [(x, y) for x in range(left, right + 1) for y in range(bottom, top + 1)]
    index = {point: i for i, point in enumerate(centers)}
    covering = [[index[p] for p in positions] for positions in placements]
    masks = [0] * len(centers)
    for cell, indices in enumerate(covering):
        for i in indices:
            masks[i] |= 1 << cell
    edges = []
    adjacency: list[list[tuple[int, int]]] = [[] for _ in centers]
    for u, (x, y) in enumerate(centers):
        for point in ((x + 1, y), (x, y + 1)):
            v = index.get(point)
            if v is not None:
                edge = len(edges)
                edges.append((u, v))
                adjacency[u].append((v, edge))
                adjacency[v].append((u, edge))
    return Grid(cells, centers, index, edges, adjacency, covering, masks)


def _path(grid: Grid, source: int, target: int, axis: int) -> list[int]:
    """A shortest rectilinear connection; there are no obstacles in the specs."""
    point = list(grid.centers[source])
    destination = grid.centers[target]
    result = [source]
    for coordinate in (axis, 1 - axis):
        while point[coordinate] != destination[coordinate]:
            point[coordinate] += 1 if destination[coordinate] > point[coordinate] else -1
            result.append(grid.index[tuple(point)])
    return result


def _walk_from_path(grid: Grid, path: list[int]) -> Walk:
    counts = [0] * len(grid.edges)
    edge_index = {edge: i for i, edge in enumerate(grid.edges)}
    for u, v in zip(path, path[1:]):
        counts[edge_index[min(u, v), max(u, v)]] += 1
    # Removing pairs preserves support and degree parity, including on bridges.
    counts = [0 if n == 0 else 1 + (n - 1) % 2 for n in counts]
    return Walk(min(path), counts)


def _greedy_group(grid: Grid, cells: list[int], axis: int) -> Walk:
    wanted = sum(1 << cell for cell in cells)
    if not cells:
        return Walk(max(range(len(grid.centers)), key=lambda i: grid.masks[i].bit_count()),
                    [0] * len(grid.edges))
    candidates = [i for i, mask in enumerate(grid.masks) if mask & wanted]
    anchor = grid.cells[cells[len(cells) // 2]]
    start = max(candidates, key=lambda i: (
        (grid.masks[i] & wanted).bit_count(),
        -sum(abs(a - b) for a, b in zip(grid.centers[i], anchor)),
    ))
    path, current = [start], start
    uncovered = wanted & ~grid.masks[start]
    while uncovered:
        x, y = grid.centers[current]

        def score(i):
            gain = (grid.masks[i] & uncovered).bit_count()
            tx, ty = grid.centers[i]
            distance = abs(tx - x) + abs(ty - y)
            return gain / max(1, distance), gain, -distance

        target = max(candidates, key=score)
        segment = _path(grid, current, target, axis)
        path.extend(segment[1:])
        for i in segment:
            uncovered &= ~grid.masks[i]
        current = target
    path.extend(_path(grid, current, start, axis)[1:])
    return _walk_from_path(grid, path)


def _greedy(grid: Grid, robot_count: int) -> list[Walk]:
    """Try both spatial stripe directions and keep the smaller makespan."""
    solutions = []
    active_count = min(robot_count, len(grid.cells))
    for axis in (0, 1):
        order = sorted(range(len(grid.cells)), key=lambda i: (
            grid.cells[i][axis], grid.cells[i][1 - axis]
        ))
        walks = [
            _greedy_group(grid, order[i * len(order) // active_count:
                                      (i + 1) * len(order) // active_count], axis)
            for i in range(active_count)
        ]
        walks.extend(_greedy_group(grid, [], axis) for _ in range(robot_count - active_count))
        solutions.append(walks)
    # The doubled spanning tree supplies the old 2*(|V|-1) bound if greedy is worse.
    tree = [0] * len(grid.edges)
    seen, queue = {0}, deque([0])
    while queue:
        for neighbor, edge in grid.adjacency[queue.popleft()]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
                tree[edge] = 2
    solutions.append([Walk(0, tree), *[
        Walk(0, [0] * len(grid.edges)) for _ in range(robot_count - 1)
    ]])
    best = min(solutions, key=lambda walks: (max(w.length for w in walks), sum(w.length for w in walks)))
    return sorted(best, key=lambda walk: walk.root)


def _build_model(grid: Grid, robot_count: int, bound: int) -> EdgeModel:
    model = cp_model.CpModel()
    n = len(grid.centers)
    # A nonstationary closed walk of length L visits at most L distinct vertices.
    supply_cap = min(n, max(1, bound))
    flow_cap = supply_cap - 1
    robots = []
    for robot in range(robot_count):
        prefix = f"r{robot}"
        selected = [model.new_bool_var(f"{prefix}_visit_{v}") for v in range(n)]
        roots = [model.new_bool_var(f"{prefix}_root_{v}") for v in range(n)]
        supply = [model.new_int_var(0, supply_cap, f"{prefix}_supply_{v}") for v in range(n)]
        half_degree = [model.new_int_var(0, len(grid.adjacency[v]), f"{prefix}_halfdeg_{v}")
                       for v in range(n)]
        counts = [model.new_int_var(0, 2, f"{prefix}_count_{e}") for e in range(len(grid.edges))]
        present = [model.new_bool_var(f"{prefix}_edge_{e}") for e in range(len(grid.edges))]
        flow = [model.new_int_var(-flow_cap, flow_cap, f"{prefix}_flow_{e}")
                for e in range(len(grid.edges))]
        root_index = model.new_int_var(0, n - 1, f"{prefix}_root_index")
        length = model.new_int_var(0, bound, f"{prefix}_length")
        model.add(sum(roots) == 1)
        model.add(root_index == sum(v * roots[v] for v in range(n)))
        model.add(length == sum(counts))
        model.add(sum(selected) <= length + 1)

        for e, (u, v) in enumerate(grid.edges):
            model.add(counts[e] >= present[e])
            model.add(counts[e] <= 2 * present[e])
            model.add(present[e] <= selected[u])
            model.add(present[e] <= selected[v])
            model.add(flow[e] <= flow_cap * present[e])
            model.add(flow[e] >= -flow_cap * present[e])
        for v in range(n):
            incident = grid.adjacency[v]
            model.add(sum(counts[e] for _, e in incident) == 2 * half_degree[v])
            # Only a root may be selected without an incident edge (a stationary tour).
            model.add(half_degree[v] >= selected[v] - roots[v])
            model.add(roots[v] <= selected[v])
            model.add(supply[v] <= supply_cap * roots[v])
            balance = sum(flow[e] if grid.edges[e][1] == v else -flow[e] for _, e in incident)
            model.add(supply[v] + balance == selected[v])
            # Every tour can choose its smallest visited index as root.
            model.add(root_index <= v).only_enforce_if(selected[v])
        robots.append(RobotVariables(selected, roots, supply, half_degree, counts,
                                     present, flow, root_index, length))

    for covering in grid.covering:
        model.add_bool_or([robot.selected[v] for robot in robots for v in covering])
    # Robot identities are interchangeable; order their chosen roots, not their lengths.
    for first, second in zip(robots, robots[1:]):
        model.add(first.root_index <= second.root_index)
    longest = model.new_int_var(0, bound, "longest_tour")
    model.add_max_equality(longest, [robot.length for robot in robots])
    # Every closed unit-grid walk has even length.
    half_longest = model.new_int_var(0, bound // 2, "half_longest_tour")
    model.add(longest == 2 * half_longest)
    model.minimize(longest)
    return EdgeModel(model, robots, longest, half_longest)


def _hint(model: EdgeModel, grid: Grid, walks: list[Walk]) -> None:
    """Supply a complete feasible assignment, including connectivity flows."""
    for variables, walk in zip(model.robots, walks):
        n = len(grid.centers)
        parent, order = {walk.root: None}, [walk.root]
        degree = [0] * n
        for e, (u, v) in enumerate(grid.edges):
            degree[u] += walk.counts[e]
            degree[v] += walk.counts[e]
        for v in order:
            for neighbor, edge in grid.adjacency[v]:
                if walk.counts[edge] and neighbor not in parent:
                    parent[neighbor] = (v, edge)
                    order.append(neighbor)
        subtree = [1] * n
        flow = [0] * len(grid.edges)
        for v in reversed(order[1:]):
            previous, edge = parent[v]
            flow[edge] = subtree[v] if grid.edges[edge][0] == previous else -subtree[v]
            subtree[previous] += subtree[v]
        for v in range(n):
            model.model.add_hint(variables.selected[v], int(v in parent))
            model.model.add_hint(variables.roots[v], int(v == walk.root))
            model.model.add_hint(variables.supply[v], len(order) if v == walk.root else 0)
            model.model.add_hint(variables.half_degree[v], degree[v] // 2)
        for e, count in enumerate(walk.counts):
            model.model.add_hint(variables.counts[e], count)
            model.model.add_hint(variables.present[e], int(count > 0))
            model.model.add_hint(variables.flow[e], flow[e])
        model.model.add_hint(variables.root_index, walk.root)
        model.model.add_hint(variables.length, walk.length)
    longest = max(walk.length for walk in walks)
    model.model.add_hint(model.longest, longest)
    model.model.add_hint(model.half_longest, longest // 2)


def _tour(grid: Grid, walk: Walk) -> CutterTour:
    if not any(walk.counts):
        x, y = grid.centers[walk.root]
        return CutterTour(x=[x], y=[y])
    remaining = walk.counts.copy()
    cursor = [0] * len(grid.centers)
    stack, circuit = [walk.root], []
    while stack:
        v = stack[-1]
        incident = grid.adjacency[v]
        while cursor[v] < len(incident) and remaining[incident[cursor[v]][1]] == 0:
            cursor[v] += 1
        if cursor[v] == len(incident):
            circuit.append(stack.pop())
        else:
            neighbor, edge = incident[cursor[v]]
            remaining[edge] -= 1
            stack.append(neighbor)
    circuit.reverse()
    if any(remaining) or len(circuit) != walk.length + 1 or circuit[0] != circuit[-1]:
        raise RuntimeError("Edge solution is not a connected Eulerian walk.")
    # Remove straight-through vertices and the explicit repeat: CutterTour closes itself.
    points = [grid.centers[v] for v in circuit[:-1]]
    corners = []
    for i, current in enumerate(points):
        previous, following = points[i - 1], points[(i + 1) % len(points)]
        incoming = (current[0] - previous[0], current[1] - previous[1])
        outgoing = (following[0] - current[0], following[1] - current[1])
        if incoming != outgoing:
            corners.append(current)
    tour = CutterTour(x=[x for x, _ in corners], y=[y for _, y in corners])
    if tour.length != walk.length:
        raise RuntimeError("Euler reconstruction changed the route length.")
    return tour


def solve(
    instance: CGSHOP2027Instance,
    *,
    time_limit: float = 50.0,
    num_workers: int = 8,
    use_hints: bool = True,
    log_search: bool = False,
) -> CGSHOP2027Solution:
    """Find k closed tours minimizing their maximum length, with a valid fallback.

    time_limit covers the CP-SAT call; preprocessing/model-building times are
    reported separately. optimal=True means optimal for the full integer-grid
    contest model, not merely optimal for a selected set of greedy centers.
    """
    if not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be finite and nonnegative.")
    if num_workers < 1:
        raise ValueError("num_workers must be positive.")
    started = perf_counter()
    grid = _grid(instance)
    walks = _greedy(grid, instance.number_of_cutters)
    validator = SolutionValidator(instance)
    fallback_tours = [_tour(grid, walk) for walk in walks]
    fallback = CGSHOP2027Solution(instance_uid=instance.instance_uid, tours=fallback_tours)
    errors = validator.check_for_errors(fallback)
    if errors:
        raise RuntimeError("Invalid greedy upper bound:\n" + "\n".join(errors))
    initial_bound = max(walk.length for walk in walks)
    preprocessing_seconds = perf_counter() - started
    meta = {
        "algorithm": "edge_euler_cp_sat",
        "optimal": initial_bound == 0,
        "status": "OPTIMAL" if initial_bound == 0 else "GREEDY_FALLBACK",
        "initial_max_tour_length": initial_bound,
        "best_bound": 0.0,
        "region_cells": len(grid.cells),
        "grid_vertices": len(grid.centers),
        "grid_edges": len(grid.edges),
        "preprocessing_seconds": preprocessing_seconds,
        "build_seconds": 0.0,
        "solve_seconds": 0.0,
        "model_variables": 0,
        "model_constraints": 0,
        "used_fallback": initial_bound != 0,
        "hints": use_hints,
    }
    if initial_bound > 0 and time_limit > 0:
        built_at = perf_counter()
        formulation = _build_model(grid, instance.number_of_cutters, initial_bound)
        if use_hints:
            _hint(formulation, grid, walks)
        meta["build_seconds"] = perf_counter() - built_at
        meta["model_variables"] = len(formulation.model.proto.variables)
        meta["model_constraints"] = len(formulation.model.proto.constraints)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.num_search_workers = num_workers
        solver.parameters.log_search_progress = log_search
        solved_at = perf_counter()
        status = solver.solve(formulation.model)
        meta["solve_seconds"] = perf_counter() - solved_at
        meta["status"] = solver.status_name(status)
        meta["best_bound"] = solver.best_objective_bound
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            walks = [Walk(solver.value(robot.root_index), [solver.value(v) for v in robot.counts])
                     for robot in formulation.robots]
            meta["optimal"] = status == cp_model.OPTIMAL
            meta["used_fallback"] = False
        elif status != cp_model.UNKNOWN:
            raise RuntimeError("A model with a validated feasible construction failed: " +
                               solver.response_stats())
    solution = CGSHOP2027Solution(
        instance_uid=instance.instance_uid,
        tours=fallback_tours if meta["used_fallback"] or initial_bound == 0 else
              [_tour(grid, walk) for walk in walks],
        meta=meta,
    )
    errors = validator.check_for_errors(solution)
    if errors:
        raise RuntimeError("Invalid edge solution:\n" + "\n".join(errors))
    meta["total_seconds"] = perf_counter() - started
    # Pydantic copies the metadata dictionary at construction.
    solution.meta["total_seconds"] = meta["total_seconds"]
    return solution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--time-limit", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-hints", action="store_true")
    parser.add_argument("--log-search", action="store_true")
    args = parser.parse_args()
    try:
        solution = solve(read_instance(args.input), time_limit=args.time_limit,
                         num_workers=args.workers, use_hints=not args.no_hints,
                         log_search=args.log_search)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    args.output.write_text(solution.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved solution to {args.output}")
    print("Longest route:", solution.max_tour_length)
    print("Optimal:", solution.meta["optimal"])
    print("Initial greedy bound:", solution.meta["initial_max_tour_length"])
    print("Best lower bound:", solution.meta["best_bound"])
    print("Model variables:", solution.meta["model_variables"])


if __name__ == "__main__":
    main()
