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
    edge_index: dict[tuple[int, int], int]
    adjacency: list[list[tuple[int, int]]]
    covering: list[list[int]]
    minimal_covering: list[list[int]]
    masks: list[int]
    all_cells_mask: int


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
    doubled: list
    flow: list
    moving: cp_model.IntVar
    root_index: cp_model.IntVar
    half_length: cp_model.IntVar
    length: cp_model.IntVar


@dataclass
class EdgeModel:
    model: cp_model.CpModel
    robots: list[RobotVariables]
    longest: cp_model.IntVar
    half_longest: cp_model.IntVar


def _minimal_covering_sets(covering: list[list[int]]) -> list[list[int]]:
    """Keep only inclusion-minimal cell coverage sets to remove dominated clauses."""
    unique_sets = sorted({frozenset(indices) for indices in covering}, key=len)
    minimal: list[frozenset[int]] = []
    for candidate in unique_sets:
        if not any(existing.issubset(candidate) for existing in minimal):
            minimal.append(candidate)
    return [sorted(s) for s in minimal]


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
    covering = [sorted(index[p] for p in positions) for positions in placements]
    minimal_covering = _minimal_covering_sets(covering)
    masks = [0] * len(centers)
    for cell, indices in enumerate(covering):
        bit = 1 << cell
        for i in indices:
            masks[i] |= bit
    edges: list[tuple[int, int]] = []
    edge_index: dict[tuple[int, int], int] = {}
    adjacency: list[list[tuple[int, int]]] = [[] for _ in centers]
    for u, (x, y) in enumerate(centers):
        for point in ((x + 1, y), (x, y + 1)):
            v = index.get(point)
            if v is not None:
                edge = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = edge
                adjacency[u].append((v, edge))
                adjacency[v].append((u, edge))
    all_cells_mask = (1 << len(cells)) - 1
    return Grid(
        cells=cells,
        centers=centers,
        index=index,
        edges=edges,
        edge_index=edge_index,
        adjacency=adjacency,
        covering=covering,
        minimal_covering=minimal_covering,
        masks=masks,
        all_cells_mask=all_cells_mask,
    )


def _mandatory_span_bounds(grid: Grid) -> tuple[int, int, int, int]:
    """Return (X_L, X_R, Y_B, Y_T) that any full cover of the region must span."""
    x_left = min(max(grid.centers[v][0] for v in cov) for cov in grid.minimal_covering)
    x_right = max(min(grid.centers[v][0] for v in cov) for cov in grid.minimal_covering)
    y_bottom = min(max(grid.centers[v][1] for v in cov) for cov in grid.minimal_covering)
    y_top = max(min(grid.centers[v][1] for v in cov) for cov in grid.minimal_covering)
    return x_left, x_right, y_bottom, y_top


def _coverage_lower_bound(grid: Grid, robot_count: int) -> int:
    """Bound route length using marginal cell gain and mandatory rectilinear span."""
    largest_start_coverage = max(mask.bit_count() for mask in grid.masks)
    largest_move_gain = max(
        (
            (grid.masks[v] & ~grid.masks[u]).bit_count()
            for a, b in grid.edges
            for u, v in ((a, b), (b, a))
        ),
        default=0,
    )
    gain_bound = 0
    remaining_cells = len(grid.cells) - robot_count * largest_start_coverage
    if remaining_cells > 0 and largest_move_gain > 0:
        divisor = robot_count * largest_move_gain
        # The closing step back to the start vertex covers 0 new cells, so a
        # closed walk of length L >= 2 has at most L - 1 productive transitions.
        productive_steps = (remaining_cells + divisor - 1) // divisor
        raw_steps = productive_steps + 1
        gain_bound = raw_steps + (raw_steps % 2)

    span_bound = 0
    if robot_count == 1:
        x_left, x_right, y_bottom, y_top = _mandatory_span_bounds(grid)
        dx = max(0, x_right - x_left)
        dy = max(0, y_top - y_bottom)
        span_bound = 2 * (dx + dy)

    return max(gain_bound, span_bound)


def _reflection_symmetry_axes(grid: Grid) -> tuple[bool, bool]:
    """Return axes whose reflections preserve every center's coverage mask."""
    cell_index = {point: i for i, point in enumerate(grid.cells)}
    min_x = min(x for x, _ in grid.cells)
    max_x = max(x for x, _ in grid.cells)
    min_y = min(y for _, y in grid.cells)
    max_y = max(y for _, y in grid.cells)
    center_bounds = (
        min(x for x, _ in grid.centers),
        max(x for x, _ in grid.centers),
        min(y for _, y in grid.centers),
        max(y for _, y in grid.centers),
    )
    axes = []
    for axis, reflection in (
        (0, lambda x, y: (min_x + max_x - x, y)),
        (1, lambda x, y: (x, min_y + max_y - y)),
    ):
        symmetric = True
        for center, mask in zip(grid.centers, grid.masks):
            reflected_center = (
                center_bounds[axis * 2]
                + center_bounds[axis * 2 + 1]
                - center[axis]
            )
            reflected_point = (
                (reflected_center, center[1])
                if axis == 0
                else (center[0], reflected_center)
            )
            reflected_index = grid.index.get(reflected_point)
            if reflected_index is None:
                symmetric = False
                break
            reflected_mask = 0
            remaining = mask
            while remaining:
                bit = remaining & -remaining
                cell = grid.cells[bit.bit_length() - 1]
                reflected_cell = reflection(*cell)
                cell_id = cell_index.get(reflected_cell)
                if cell_id is None:
                    symmetric = False
                    break
                reflected_mask |= 1 << cell_id
                remaining ^= bit
            if not symmetric or grid.masks[reflected_index] != reflected_mask:
                symmetric = False
                break
        axes.append(symmetric)
    return axes[0], axes[1]


def _reflect_walk(grid: Grid, walk: Walk, axis: int) -> Walk:
    """Reflect a walk across axis 0 (x) or axis 1 (y)."""
    min_c = min(p[axis] for p in grid.centers)
    max_c = max(p[axis] for p in grid.centers)

    def reflect_vertex(v: int) -> int:
        x, y = grid.centers[v]
        pt = (min_c + max_c - x, y) if axis == 0 else (x, min_c + max_c - y)
        return grid.index[pt]

    if not any(walk.counts):
        return Walk(reflect_vertex(walk.root), [0] * len(grid.edges))

    new_counts = [0] * len(grid.edges)
    visited = set()
    for e, count in enumerate(walk.counts):
        if count:
            u, v = grid.edges[e]
            ru, rv = reflect_vertex(u), reflect_vertex(v)
            visited.add(ru)
            visited.add(rv)
            new_edge = grid.edge_index[(min(ru, rv), max(ru, rv))]
            new_counts[new_edge] = count
    return Walk(min(visited), new_counts)


def _canonicalize_walk_symmetry(
    grid: Grid,
    walks: list[Walk],
    symmetry_axes: tuple[bool, bool],
) -> list[Walk]:
    """Ensure a single-robot warm-start walk respects root symmetry breaking."""
    if len(walks) != 1:
        return walks
    walk = walks[0]
    symmetric_x, symmetric_y = symmetry_axes
    min_x = min(x for x, _ in grid.centers)
    max_x = max(x for x, _ in grid.centers)
    min_y = min(y for _, y in grid.centers)
    max_y = max(y for _, y in grid.centers)
    if symmetric_x and 2 * grid.centers[walk.root][0] > min_x + max_x:
        walk = _reflect_walk(grid, walk, 0)
    if symmetric_y and 2 * grid.centers[walk.root][1] > min_y + max_y:
        walk = _reflect_walk(grid, walk, 1)
    return [walk]


def _best_manhattan_path(
    grid: Grid,
    source: int,
    target: int,
    uncovered: int,
    prefer_axis: int,
) -> list[int]:
    """Greedy shortest rectilinear path that maximizes newly covered cells."""
    x, y = grid.centers[source]
    tx, ty = grid.centers[target]
    result = [source]
    rem = uncovered & ~grid.masks[source]
    while x != tx or y != ty:
        options = []
        if x != tx:
            nx = x + (1 if tx > x else -1)
            nv = grid.index[(nx, y)]
            gain = (grid.masks[nv] & rem).bit_count()
            options.append((gain, int(prefer_axis == 0), nv, nx, y))
        if y != ty:
            ny = y + (1 if ty > y else -1)
            nv = grid.index[(x, ny)]
            gain = (grid.masks[nv] & rem).bit_count()
            options.append((gain, int(prefer_axis == 1), nv, x, ny))
        _, _, best_v, x, y = max(options)
        result.append(best_v)
        rem &= ~grid.masks[best_v]
    return result


def _walk_from_path(grid: Grid, path: list[int], wanted: int) -> Walk:
    counts = [0] * len(grid.edges)
    for u, v in zip(path, path[1:]):
        if u != v:
            counts[grid.edge_index[(min(u, v), max(u, v))]] += 1
    # Removing pairs preserves support and degree parity, including on bridges.
    counts = [0 if n == 0 else 1 + (n - 1) % 2 for n in counts]

    # Iteratively peel degree-1 doubled leaf spikes whose tip is redundant.
    active_edges = [0] * len(grid.centers)
    visited = set()
    for e, c in enumerate(counts):
        if c:
            u, v = grid.edges[e]
            active_edges[u] += 1
            active_edges[v] += 1
            visited.add(u)
            visited.add(v)

    if len(visited) > 1:
        leaves = deque(v for v in visited if active_edges[v] == 1)
        while leaves and len(visited) > 1:
            leaf = leaves.popleft()
            if leaf not in visited or active_edges[leaf] != 1:
                continue
            # Check if remaining visited vertices still cover all wanted cells.
            covered_without_leaf = 0
            for v in visited:
                if v != leaf:
                    covered_without_leaf |= grid.masks[v]
            if (covered_without_leaf & wanted) != wanted:
                continue
            for neighbor, edge in grid.adjacency[leaf]:
                if counts[edge] == 2:
                    counts[edge] = 0
                    visited.remove(leaf)
                    active_edges[leaf] = 0
                    active_edges[neighbor] -= 1
                    if active_edges[neighbor] == 1:
                        leaves.append(neighbor)
                    break

    root = min(visited) if visited else min(path)
    return Walk(root, counts)


def _two_opt_waypoints(grid: Grid, waypoints: list[int]) -> list[int]:
    """Untangle waypoint visiting order using 2-opt under L1 distance."""
    n = len(waypoints)
    if n <= 3:
        return waypoints
    pts = [grid.centers[w] for w in waypoints]

    def dist(i: int, j: int) -> int:
        return abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a, b = i, i + 1
            dab = dist(a, b)
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, d = j, (j + 1) % n
                if dab + dist(c, d) > dist(a, c) + dist(b, d):
                    waypoints[i + 1 : j + 1] = reversed(waypoints[i + 1 : j + 1])
                    pts[i + 1 : j + 1] = reversed(pts[i + 1 : j + 1])
                    dab = dist(a, b)
                    improved = True
    return waypoints


def _build_greedy_walk_from_start(
    grid: Grid,
    wanted: int,
    candidates: list[int],
    start: int,
    axis: int,
) -> Walk:
    waypoints = [start]
    path = [start]
    current = start
    uncovered = wanted & ~grid.masks[start]
    while uncovered:
        x, y = grid.centers[current]

        def score(i: int) -> tuple[float, int, int]:
            gain = (grid.masks[i] & uncovered).bit_count()
            tx, ty = grid.centers[i]
            distance = abs(tx - x) + abs(ty - y)
            return gain / max(1, distance), gain, -distance

        target = max(candidates, key=score)
        waypoints.append(target)
        segment = _best_manhattan_path(grid, current, target, uncovered, axis)
        path.extend(segment[1:])
        for i in segment:
            uncovered &= ~grid.masks[i]
        current = target

    path.extend(_best_manhattan_path(grid, current, start, 0, axis)[1:])
    raw_walk = _walk_from_path(grid, path, wanted)

    # Also try pruning redundant waypoints + 2-opt reordering.
    pruned = list(dict.fromkeys(waypoints))
    if len(pruned) > 1:
        idx = 0
        while idx < len(pruned) and len(pruned) > 1:
            rem_mask = 0
            for j, w in enumerate(pruned):
                if j != idx:
                    rem_mask |= grid.masks[w]
            if (rem_mask & wanted) == wanted:
                pruned.pop(idx)
            else:
                idx += 1
        pruned = _two_opt_waypoints(grid, pruned)
        opt_path = [pruned[0]]
        rem = wanted & ~grid.masks[pruned[0]]
        for nxt in pruned[1:]:
            seg = _best_manhattan_path(grid, opt_path[-1], nxt, rem, axis)
            opt_path.extend(seg[1:])
            for v in seg:
                rem &= ~grid.masks[v]
        opt_path.extend(
            _best_manhattan_path(grid, opt_path[-1], pruned[0], rem, axis)[1:]
        )
        opt_walk = _walk_from_path(grid, opt_path, wanted)
        if opt_walk.length < raw_walk.length:
            return opt_walk

    return raw_walk


def _greedy_group(grid: Grid, cells: list[int], axis: int) -> Walk:
    wanted = sum(1 << cell for cell in cells)
    if not cells:
        return Walk(
            max(range(len(grid.centers)), key=lambda i: grid.masks[i].bit_count()),
            [0] * len(grid.edges),
        )
    candidates = [i for i, mask in enumerate(grid.masks) if mask & wanted]
    anchor_mid = grid.cells[cells[len(cells) // 2]]
    anchor_first = grid.cells[cells[0]]
    anchor_last = grid.cells[cells[-1]]
    starts = {
        max(
            candidates,
            key=lambda i, anc=anc: (
                (grid.masks[i] & wanted).bit_count(),
                -sum(abs(a - b) for a, b in zip(grid.centers[i], anc)),
            ),
        )
        for anc in (anchor_mid, anchor_first, anchor_last)
    }
    return min(
        (_build_greedy_walk_from_start(grid, wanted, candidates, s, axis) for s in starts),
        key=lambda w: w.length,
    )


def _greedy(grid: Grid, robot_count: int) -> list[Walk]:
    """Try spatial stripe directions with 2-opt/leaf-pruned walks and keep best makespan."""
    solutions = []
    active_count = min(robot_count, len(grid.cells))
    for axis in (0, 1):
        order = sorted(
            range(len(grid.cells)),
            key=lambda i: (grid.cells[i][axis], grid.cells[i][1 - axis]),
        )
        walks = [
            _greedy_group(
                grid,
                order[i * len(order) // active_count : (i + 1) * len(order) // active_count],
                axis,
            )
            for i in range(active_count)
        ]
        walks.extend(_greedy_group(grid, [], axis) for _ in range(robot_count - active_count))
        solutions.append(walks)

    # The doubled spanning tree supplies the 2*(|V|-1) bound if greedy is worse.
    tree = [0] * len(grid.edges)
    seen, queue = {0}, deque([0])
    while queue:
        for neighbor, edge in grid.adjacency[queue.popleft()]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
                tree[edge] = 2
    solutions.append(
        [Walk(0, tree), *[Walk(0, [0] * len(grid.edges)) for _ in range(robot_count - 1)]]
    )
    best = min(
        solutions,
        key=lambda walks: (max(w.length for w in walks), sum(w.length for w in walks)),
    )
    return sorted(best, key=lambda walk: walk.root)


def _build_model(grid: Grid, robot_count: int, bound: int) -> EdgeModel:
    model = cp_model.CpModel()
    n = len(grid.centers)
    m = len(grid.edges)
    # A nonstationary closed walk of length L visits at most L distinct vertices.
    supply_cap = min(n, max(1, bound))
    flow_cap = supply_cap - 1
    half_bound = bound // 2

    # The smallest root across all robots cannot exceed the max covering vertex
    # of any cell, because some robot must cover that cell and pick root <= v.
    first_robot_max_root = min(max(cov) for cov in grid.minimal_covering)

    robots = []
    for robot in range(robot_count):
        prefix = f"r{robot}"
        selected = [model.new_bool_var(f"{prefix}_visit_{v}") for v in range(n)]
        roots = [model.new_bool_var(f"{prefix}_root_{v}") for v in range(n)]
        supply = [model.new_int_var(0, supply_cap, f"{prefix}_supply_{v}") for v in range(n)]
        half_degree = [
            model.new_int_var(0, len(grid.adjacency[v]), f"{prefix}_halfdeg_{v}")
            for v in range(n)
        ]
        present = [model.new_bool_var(f"{prefix}_edge_{e}") for e in range(m)]
        doubled = [model.new_bool_var(f"{prefix}_doubled_{e}") for e in range(m)]
        counts = [model.new_int_var(0, 2, f"{prefix}_count_{e}") for e in range(m)]
        flow = [model.new_int_var(-flow_cap, flow_cap, f"{prefix}_flow_{e}") for e in range(m)]

        moving = model.new_bool_var(f"{prefix}_moving")
        max_root = first_robot_max_root if robot == 0 else n - 1
        root_index = model.new_int_var(0, max_root, f"{prefix}_root_index")
        half_length = model.new_int_var(0, half_bound, f"{prefix}_half_length")
        length = model.new_int_var(0, bound, f"{prefix}_length")

        if robot == 0:
            for v in range(max_root + 1, n):
                model.add(roots[v] == 0)
                model.add(supply[v] == 0)

        model.add(sum(roots) == 1)
        model.add(root_index == sum(v * roots[v] for v in range(max_root + 1)))
        model.add(length == 2 * half_length)
        model.add(length == sum(counts))
        model.add(sum(half_degree) == length)

        # Link moving indicator (whether length >= 2).
        model.add(half_length >= moving)
        model.add(half_length <= half_bound * moving)

        # Connected graph inequality: active undirected edges >= selected vertices - 1,
        # and a moving tour visits at most `length` vertices (stationary visits 1).
        total_selected = sum(selected)
        model.add(total_selected <= length + 1 - moving)
        model.add(sum(present) >= total_selected - 1)
        model.add(sum(supply) == total_selected)

        for e, (u, v) in enumerate(grid.edges):
            model.add(counts[e] == present[e] + doubled[e])
            model.add_implication(doubled[e], present[e])
            model.add_implication(present[e], selected[u])
            model.add_implication(present[e], selected[v])
            model.add(flow[e] <= flow_cap * present[e])
            model.add(flow[e] >= -flow_cap * present[e])

        for v in range(n):
            incident = grid.adjacency[v]
            model.add(sum(counts[e] for _, e in incident) == 2 * half_degree[v])
            model.add(half_degree[v] <= len(incident) * selected[v])
            # Non-root selected vertices always have half_degree >= 1;
            # the root also has half_degree >= 1 whenever the robot moves.
            model.add(half_degree[v] >= selected[v] - roots[v])
            model.add(half_degree[v] >= roots[v] + moving - 1)
            model.add_implication(roots[v], selected[v])
            model.add(supply[v] <= supply_cap * roots[v])
            balance = sum(
                flow[e] if grid.edges[e][1] == v else -flow[e]
                for _, e in incident
            )
            model.add(supply[v] + balance == selected[v])
            # Every tour chooses its smallest visited index as root.
            model.add(root_index <= v).only_enforce_if(selected[v])

        robots.append(
            RobotVariables(
                selected=selected,
                roots=roots,
                supply=supply,
                half_degree=half_degree,
                counts=counts,
                present=present,
                doubled=doubled,
                flow=flow,
                moving=moving,
                root_index=root_index,
                half_length=half_length,
                length=length,
            )
        )

    # Enforce coverage over inclusion-minimal cell coverage sets.
    for covering in grid.minimal_covering:
        if robot_count == 1 and len(covering) == 1:
            model.add(robots[0].selected[covering[0]] == 1)
        else:
            model.add_bool_or([robot.selected[v] for robot in robots for v in covering])

    # Mandatory Eulerian slice-cut inequalities for k = 1.
    if robot_count == 1:
        x_left, x_right, y_bottom, y_top = _mandatory_span_bounds(grid)
        horiz_cuts: dict[int, list[int]] = {x: [] for x in range(x_left, x_right)}
        vert_cuts: dict[int, list[int]] = {y: [] for y in range(y_bottom, y_top)}
        for e, (u, v) in enumerate(grid.edges):
            ux, uy = grid.centers[u]
            vx, vy = grid.centers[v]
            if ux != vx and ux in horiz_cuts:
                horiz_cuts[ux].append(e)
            elif uy != vy and uy in vert_cuts:
                vert_cuts[uy].append(e)
        for cut_edges in (*horiz_cuts.values(), *vert_cuts.values()):
            if cut_edges:
                model.add_bool_or([robots[0].present[e] for e in cut_edges])
                model.add(sum(robots[0].counts[e] for e in cut_edges) >= 2)

    # Robot identities are interchangeable; order their chosen roots.
    for first, second in zip(robots, robots[1:]):
        model.add(first.root_index <= second.root_index)

    half_longest = model.new_int_var(0, half_bound, "half_longest_tour")
    model.add_max_equality(half_longest, [robot.half_length for robot in robots])
    longest = model.new_int_var(0, bound, "longest_tour")
    model.add(longest == 2 * half_longest)
    model.add(longest >= _coverage_lower_bound(grid, robot_count))

    # Reflections preserve feasibility when they preserve the full coverage
    # masks. Canonicalize the smallest selected vertex to one side per axis.
    symmetric_x, symmetric_y = _reflection_symmetry_axes(grid)
    min_x = min(x for x, _ in grid.centers)
    max_x = max(x for x, _ in grid.centers)
    min_y = min(y for _, y in grid.centers)
    max_y = max(y for _, y in grid.centers)
    if robot_count == 1:
        for v, (x, y) in enumerate(grid.centers):
            if symmetric_x and 2 * x > min_x + max_x:
                model.add(robots[0].roots[v] == 0)
            if symmetric_y and 2 * y > min_y + max_y:
                model.add(robots[0].roots[v] == 0)

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
            model.model.add_hint(variables.present[e], int(count >= 1))
            model.model.add_hint(variables.doubled[e], int(count == 2))
            model.model.add_hint(variables.flow[e], flow[e])
        model.model.add_hint(variables.moving, int(walk.length > 0))
        model.model.add_hint(variables.root_index, walk.root)
        model.model.add_hint(variables.half_length, walk.length // 2)
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
    coverage_lower_bound = _coverage_lower_bound(grid, instance.number_of_cutters)
    symmetry_axes = _reflection_symmetry_axes(grid)
    walks = _canonicalize_walk_symmetry(
        grid,
        _greedy(grid, instance.number_of_cutters),
        symmetry_axes,
    )
    validator = SolutionValidator(instance)
    fallback_tours = [_tour(grid, walk) for walk in walks]
    fallback = CGSHOP2027Solution(instance_uid=instance.instance_uid, tours=fallback_tours)
    errors = validator.check_for_errors(fallback)
    if errors:
        raise RuntimeError("Invalid greedy upper bound:\n" + "\n".join(errors))
    initial_bound = max(walk.length for walk in walks)
    already_optimal = initial_bound <= coverage_lower_bound
    preprocessing_seconds = perf_counter() - started
    meta = {
        "algorithm": "edge_euler_cp_sat",
        "optimal": already_optimal,
        "status": "OPTIMAL" if already_optimal else "GREEDY_FALLBACK",
        "initial_max_tour_length": initial_bound,
        "coverage_lower_bound": coverage_lower_bound,
        "reflection_symmetry_axes": symmetry_axes,
        "best_bound": float(initial_bound if already_optimal else coverage_lower_bound),
        "region_cells": len(grid.cells),
        "minimal_cells": len(grid.minimal_covering),
        "grid_vertices": len(grid.centers),
        "grid_edges": len(grid.edges),
        "preprocessing_seconds": preprocessing_seconds,
        "build_seconds": 0.0,
        "solve_seconds": 0.0,
        "model_variables": 0,
        "model_constraints": 0,
        "used_fallback": not already_optimal,
        "hints": use_hints,
    }
    if not already_optimal and time_limit > 0:
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
            walks = [
                Walk(solver.value(robot.root_index), [solver.value(v) for v in robot.counts])
                for robot in formulation.robots
            ]
            meta["optimal"] = status == cp_model.OPTIMAL
            meta["used_fallback"] = False
        elif status != cp_model.UNKNOWN:
            raise RuntimeError(
                "A model with a validated feasible construction failed: "
                + solver.response_stats()
            )
    solution = CGSHOP2027Solution(
        instance_uid=instance.instance_uid,
        tours=(
            fallback_tours
            if meta["used_fallback"] or already_optimal
            else [_tour(grid, walk) for walk in walks]
        ),
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
        solution = solve(
            read_instance(args.input),
            time_limit=args.time_limit,
            num_workers=args.workers,
            use_hints=not args.no_hints,
            log_search=args.log_search,
        )
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