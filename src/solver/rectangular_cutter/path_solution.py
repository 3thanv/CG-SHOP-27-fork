from collections.abc import Iterator
from typing import Literal

from cgshop2027_pyutils.schemas import CGSHOP2027Solution, CutterTour


class CutterPath(CutterTour):
    def edges(self) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
        points = list(self.points())
        yield from zip(points, points[1:])
        

class PathSolution(CGSHOP2027Solution):
    tours: list[CutterPath]
