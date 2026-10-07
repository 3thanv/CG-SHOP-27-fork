# Exact single-cutter solver

This solver uses CP-SAT to find a minimum-length closed tour for one cutter.
Region and cutter polygons are rasterized to grid cells. Candidate positions
that cover region cells are used for coverage constraints; the movement graph
also includes transit positions in their integer bounding box, so the route
isn't forced to move only between mowing locations. CP-SAT is given a
50-second time limit. A greedy closed walk gives the model a valid upper bound
and initial solution hint. The output metadata reports that initial tour
length, the best objective bound, solve time, and whether optimality was
proved. Larger instances can require substantial memory and may time out.

Run it from this directory:

```sh
uv run main.py input.instance.json output.solution.json
```

The output is a standard CG:SHOP solution JSON. Its `meta` object includes the
CP-SAT status and whether optimality was proved. Unsupported instances and
instances for which CP-SAT finds no tour exit with an error instead of writing
a solution file.

Run the focused tests with:

```sh
uv run python -m unittest
```
