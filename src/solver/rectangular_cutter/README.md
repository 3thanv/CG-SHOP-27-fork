# Exact single-cutter solver

This solver uses CP-SAT to find a minimum-length closed tour for one cutter.
It supports exactly one cutter, requires the cutter to rasterize to a filled
axis-aligned rectangle, and accepts at most 36 region cells and 20 distinct
cutter-center positions that cover region cells. Region and cutter polygons
are rasterized to grid cells. The movement graph also includes transit
positions in its integer bounding box, so the tour can travel between mowing
positions without cutting. The model uses a closed-walk bound of twice the
number of movement-grid edges and no heuristic starting route or solution
hint. CP-SAT has a fixed 50-second time limit. The output metadata reports the
best objective bound, solve time, and whether optimality was proved.

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

The testcase generator can create larger cases for experiments; the exact
solver rejects generated cases that exceed its supported region or candidate
position limits. For example:

```sh
uv run python generate_testcases.py examples/generated-large \
  --seed 2030 \
  --region-cells 100 150 200 \
  --cutter-cells 12 18 24 \
  --region-grid-size 50 \
  --cutter-grid-size 10
```
