
from pymoo.core.duplicate     import ElementwiseDuplicateElimination
class RouteDuplicateElim(ElementwiseDuplicateElimination):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def is_equal(self, a, b):
        idx1, idx2 = int(a.X[0]), int(b.X[0])

        # 1) Same chromosome value → duplicate
        if idx1 == idx2:
            return True

        # 2) Try deep comparison only if BOTH routes have been cached
        try:
            route1 = self.problem.get_route(idx1)
            route2 = self.problem.get_route(idx2)
        except KeyError:
            # At least one route not stored yet (still in initialisation)
            return False

        return route1 == route2
