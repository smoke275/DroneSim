"""Fleet routing and energy-replenishment planning shared by every strategy.

Every strategy in the benchmark uses the *same* routes and the *same*
replenishment planner; strategies differ only in which replenishment options
the planner may use (aerial swap, depot detour, fixed-station detour) and, for
the reactive baseline, in dispatching without prediction. No Qt here.

1. ``distance_matrix``     road-graph shortest-path distances between the
   depot, every task and every base station (canvas units).
2. ``plan_routes``         min-max multi-vehicle TSP: every truck leaves and
   returns to the depot, minimise the longest route (makespan proxy), total
   distance as tie-break. OR-Tools when installed, otherwise a deterministic
   pure-Python construction + local search. Both are seed-deterministic so
   paired scenarios get identical routes across strategies.
3. ``plan_replenishment``  resource-constrained shortest path on the route
   DAG: given a truck's remaining route and battery, pick the cheapest set of
   replenishment events such that no segment between two replenishments
   exceeds the usable capacity. Exact (dynamic programme), O(m^2) states.
4. ``assign_drones``       Hungarian assignment of idle drones to pending swap
   requests, minimising total expected truck waiting time.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

INF = float('inf')


# ---------------------------------------------------------------- distances
def distance_matrix(world, cells: Sequence[Tuple[int, int]]) -> List[List[float]]:
    """Shortest road distance (canvas units) between every pair of cells."""
    cells = [tuple(c) for c in cells]
    n = len(cells)
    dist = [[INF] * n for _ in range(n)]
    for i, src in enumerate(cells):
        lengths = nx.single_source_shortest_path_length(world.graph, src)
        for j, dst in enumerate(cells):
            hops = lengths.get(dst)
            if hops is not None:
                dist[i][j] = hops * world.cell_size
    return dist


def cell_distances(world, src: Tuple[int, int], cells: Sequence[Tuple[int, int]]) -> List[float]:
    """Shortest road distance from one cell to each of ``cells`` (one BFS)."""
    lengths = nx.single_source_shortest_path_length(world.graph, tuple(src))
    return [lengths.get(tuple(c), INF) * world.cell_size if lengths.get(tuple(c)) is not None
            else INF for c in cells]


# ------------------------------------------------------------------ routing
def route_length(dist, depot: int, route: Sequence[int]) -> float:
    if not route:
        return 0.0
    total = dist[depot][route[0]] + dist[route[-1]][depot]
    for a, b in zip(route, route[1:]):
        total += dist[a][b]
    return total


def _objective(dist, depot, routes):
    lengths = [route_length(dist, depot, r) for r in routes]
    return (max(lengths) if lengths else 0.0, sum(lengths))


def plan_routes(dist, depot: int, customers: Sequence[int], num_vehicles: int,
                seed: int = 0, solver: str = 'auto') -> List[List[int]]:
    """Min-max mTSP over ``customers`` (node indices into ``dist``).

    Returns ``num_vehicles`` routes (lists of customer indices, depot implied
    at both ends). Routes may be empty when there are fewer customers than
    vehicles.
    """
    customers = list(customers)
    if not customers:
        return [[] for _ in range(num_vehicles)]
    if solver in ('auto', 'ortools'):
        routes = _ortools_routes(dist, depot, customers, num_vehicles, seed)
        if routes is not None:
            # Polish with the local search (cheap, deterministic, never worse).
            return _local_search(dist, depot, routes, random.Random(seed))
        if solver == 'ortools':
            raise RuntimeError('OR-Tools requested but not importable')
    return _heuristic_routes(dist, depot, customers, num_vehicles, seed)


def _ortools_routes(dist, depot, customers, num_vehicles, seed):
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return None

    nodes = [depot] + customers
    n = len(nodes)
    scale = 10  # integer costs, 0.1 canvas-unit resolution
    big = 10 ** 9
    matrix = [[big if dist[a][b] == INF else int(round(dist[a][b] * scale)) for b in nodes]
              for a in nodes]

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit(i, j):
        return matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]

    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimension(cb, 0, big, True, 'Distance')
    # Global span coefficient >> arc costs: minimise the longest route first,
    # total distance as tie-break (min-max mTSP).
    routing.GetDimensionOrDie('Distance').SetGlobalSpanCostCoefficient(1000)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # Solution-count limit (not wall-clock) keeps the search deterministic.
    params.solution_limit = 600
    params.time_limit.seconds = 60
    params.log_search = False

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None
    routes = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(nodes[node])
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)
    return routes


def _heuristic_routes(dist, depot, customers, num_vehicles, seed):
    """Deterministic min-max construction + local search with restarts."""
    rng = random.Random(seed)
    best = None
    best_obj = (INF, INF)
    restarts = 8
    for r in range(restarts):
        order = list(customers)
        if r > 0:
            rng.shuffle(order)
        routes = _construct(dist, depot, order, num_vehicles, seeded=(r == 0))
        routes = _local_search(dist, depot, routes, rng)
        obj = _objective(dist, depot, routes)
        if obj < best_obj:
            best, best_obj = routes, obj
    return best


def _construct(dist, depot, customers, num_vehicles, seeded=True):
    """Spread seeds (k-centre on road distance), then min-max cheapest insertion."""
    routes = [[] for _ in range(num_vehicles)]
    unassigned = list(customers)
    if seeded and len(unassigned) >= num_vehicles:
        first = max(unassigned, key=lambda c: dist[depot][c])
        seeds = [first]
        unassigned.remove(first)
        while len(seeds) < num_vehicles:
            nxt = max(unassigned, key=lambda c: min(dist[s][c] for s in seeds))
            seeds.append(nxt)
            unassigned.remove(nxt)
        for i, s in enumerate(seeds):
            routes[i].append(s)
        unassigned.sort(key=lambda c: -dist[depot][c])
    lengths = [route_length(dist, depot, r) for r in routes]
    for c in unassigned:
        best = None
        for ri, route in enumerate(routes):
            for pos in range(len(route) + 1):
                prev = depot if pos == 0 else route[pos - 1]
                nxt = depot if pos == len(route) else route[pos]
                delta = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                new_len = lengths[ri] + delta
                new_max = max(new_len, max(l for i, l in enumerate(lengths) if i != ri) if len(lengths) > 1 else 0)
                key = (new_max, delta)
                if best is None or key < best[0]:
                    best = (key, ri, pos, new_len)
        _, ri, pos, new_len = best
        routes[ri].insert(pos, c)
        lengths[ri] = new_len
    return routes


def _two_opt(dist, depot, route):
    """2-opt on an open route with the depot fixed at both ends."""
    if len(route) < 3:
        return route
    improved = True
    path = [depot] + list(route) + [depot]
    while improved:
        improved = False
        n = len(path)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = path[i - 1], path[i]
                c, d = path[j], path[j + 1]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < -1e-9:
                    path[i:j + 1] = reversed(path[i:j + 1])
                    improved = True
    return path[1:-1]


def _or_opt(dist, depot, route):
    """Move segments of 1..3 nodes within a route while it improves."""
    if len(route) < 2:
        return route
    improved = True
    while improved:
        improved = False
        path = [depot] + list(route) + [depot]
        n = len(path)
        for seg in (1, 2, 3):
            for i in range(1, n - seg):
                a = path[i - 1]
                segment = path[i:i + seg]
                b = path[i + seg]
                remove_gain = dist[a][segment[0]] + dist[segment[-1]][b] - dist[a][b]
                rest = path[:i] + path[i + seg:]
                best = None
                for k in range(1, len(rest)):
                    p, q = rest[k - 1], rest[k]
                    for s in (segment, segment[::-1]):
                        add = dist[p][s[0]] + dist[s[-1]][q] - dist[p][q]
                        if add < remove_gain - 1e-9 and (best is None or add < best[0]):
                            best = (add, k, s)
                if best is not None:
                    _, k, s = best
                    path = rest[:k] + list(s) + rest[k:]
                    route = path[1:-1]
                    improved = True
                    break
            if improved:
                break
    return route


def _local_search(dist, depot, routes, rng):
    routes = [list(r) for r in routes]
    routes = [_or_opt(dist, depot, _two_opt(dist, depot, r)) for r in routes]
    obj = _objective(dist, depot, routes)
    improved = True
    while improved:
        improved = False
        # Inter-route relocate: move one customer from route A to route B.
        for a in range(len(routes)):
            for i in range(len(routes[a])):
                for b in range(len(routes)):
                    if a == b:
                        continue
                    node = routes[a][i]
                    ra = routes[a][:i] + routes[a][i + 1:]
                    for pos in range(len(routes[b]) + 1):
                        rb = routes[b][:pos] + [node] + routes[b][pos:]
                        cand = list(routes)
                        cand[a] = _two_opt(dist, depot, ra)
                        cand[b] = _two_opt(dist, depot, rb)
                        cobj = _objective(dist, depot, cand)
                        if cobj < obj:
                            routes, obj, improved = cand, cobj, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Inter-route exchange: swap one customer between routes A and B.
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        cand = list(routes)
                        cand[a] = _two_opt(dist, depot, ra)
                        cand[b] = _two_opt(dist, depot, rb)
                        cobj = _objective(dist, depot, cand)
                        if cobj < obj:
                            routes, obj, improved = cand, cobj, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return [_or_opt(dist, depot, _two_opt(dist, depot, r)) for r in routes]


# ------------------------------------------------------- replenishment DP
class ReplenishmentOptions:
    """Which replenishment moves a strategy may plan, and what they cost.

    Costs are in frames (truck time). ``swap_cost(node)`` returns None when
    no drone base can reach that node.
    """

    def __init__(self, allow_depot=False, allow_station=False, swap_cost=None,
                 station_nodes=(), service_frames=0, truck_speed=1.0):
        self.allow_depot = allow_depot
        self.allow_station = allow_station
        self.swap_cost = swap_cost  # callable node -> frames or None
        self.station_nodes = list(station_nodes)
        self.service_frames = service_frames
        self.truck_speed = truck_speed


def plan_replenishment(dist, route: Sequence[int], depot: int, start_dists: Sequence[float],
                       fuel_now: float, capacity: float, margin: float,
                       opts: ReplenishmentOptions,
                       allow_start_refill: bool = True) -> Optional[List[Tuple[int, str, Optional[int]]]]:
    """Exact resource-constrained shortest path on the remaining route.

    Positions: 0 = current location (distances given by ``start_dists``,
    indexed like ``dist``), 1..m-1 = ``route[0..m-2]`` (remaining task nodes),
    m = final depot (``route[-1]`` must equal ``depot``).

    A plan is a list of events ``(position, kind, station)`` with kind in
    {'swap', 'depot', 'station'} meaning: after serving that position, refill
    by that means before continuing. Between refills (and from the start) the
    driven distance must not exceed the usable budget (``capacity - margin``
    or ``fuel_now - margin`` for the first segment). Returns None if no
    feasible plan exists. ``allow_start_refill`` permits an event at position
    0 (refill right now, before moving); ``opts.swap_cost(None)`` must then
    return the cost of a swap at the current location.
    """
    m = len(route)
    if m == 0:
        return []
    nodes = [None] + list(route)  # position -> node index (position 0: current cell)

    def d(p, q):
        """Road distance between positions p<q travelling along the route."""
        if p == 0:
            total = start_dists[nodes[1]]
            for k in range(1, q):
                total += dist[nodes[k]][nodes[k + 1]]
            return total
        total = 0.0
        for k in range(p, q):
            total += dist[nodes[k]][nodes[k + 1]]
        return total

    def to_node(p, node):
        return start_dists[node] if p == 0 else dist[nodes[p]][node]

    def next_leg(p):
        return d(p, p + 1)

    budget_full = capacity - margin
    # State: (position p, origin) where origin is the point the truck leaves
    # from with a full battery after serving p: ('here', fuel) at p=0,
    # ('node', -) after a swap at p, ('depot', -) after a depot detour,
    # ('station', s) after a station detour. Value: (cost, events).
    # dp[p] maps origin -> (cost, events).
    dp: List[Dict[Tuple, Tuple[float, list]]] = [dict() for _ in range(m + 1)]
    dp[0][('here', None)] = (0.0, [])
    if allow_start_refill:
        start_budget = fuel_now - margin
        nxt = nodes[1]
        if opts.swap_cost is not None:
            c = opts.swap_cost(None)
            if c is not None:
                _relax(dp[0], ('node', None), c, [(0, 'swap', None)])
        if opts.allow_depot and start_dists[depot] <= start_budget:
            extra = start_dists[depot] + dist[depot][nxt] - start_dists[nxt]
            _relax(dp[0], ('depot', None), max(0.0, extra) / opts.truck_speed, [(0, 'depot', None)])
        if opts.allow_station:
            for s in opts.station_nodes:
                if start_dists[s] > start_budget:
                    continue
                extra = start_dists[s] + dist[s][nxt] - start_dists[nxt]
                c = max(0.0, extra) / opts.truck_speed + opts.service_frames
                _relax(dp[0], ('station', s), c, [(0, 'station', s)])

    def origin_budget(origin):
        return fuel_now - margin if origin[0] == 'here' else budget_full

    def origin_to(p, origin, node):
        """Distance from origin point after position p to ``node``."""
        kind, s = origin
        if kind in ('here', 'node'):
            return to_node(p, node)
        if kind == 'depot':
            return dist[depot][node]
        return dist[s][node]

    best_final = None
    for p in range(m):
        for origin, (cost, events) in dp[p].items():
            budget = origin_budget(origin)
            # travel origin -> position p+1 ... q without refilling
            run = origin_to(p, origin, nodes[p + 1])
            for q in range(p + 1, m + 1):
                if q > p + 1:
                    run += dist[nodes[q - 1]][nodes[q]]
                if run > budget:
                    break
                if q == m:  # reached final depot
                    cand = (cost, events)
                    if best_final is None or cand[0] < best_final[0]:
                        best_final = cand
                    break
                node = nodes[q]
                nxt = nodes[q + 1]
                # Option: aerial swap at q (battery full when leaving q).
                if opts.swap_cost is not None:
                    c = opts.swap_cost(node)
                    if c is not None:
                        _relax(dp[q], ('node', None), cost + c, events + [(q, 'swap', None)])
                # Option: depot detour after q.
                if opts.allow_depot and run + dist[node][depot] <= budget:
                    extra = dist[node][depot] + dist[depot][nxt] - dist[node][nxt]
                    c = max(0.0, extra) / opts.truck_speed
                    _relax(dp[q], ('depot', None), cost + c, events + [(q, 'depot', None)])
                # Option: fixed-station detour after q (best reachable station).
                if opts.allow_station:
                    for s in opts.station_nodes:
                        if run + dist[node][s] > budget:
                            continue
                        extra = dist[node][s] + dist[s][nxt] - dist[node][nxt]
                        c = max(0.0, extra) / opts.truck_speed + opts.service_frames
                        _relax(dp[q], ('station', s), cost + c, events + [(q, 'station', s)])
    if best_final is None:
        return None
    return best_final[1]


def _relax(table, key, cost, events):
    cur = table.get(key)
    if cur is None or cost < cur[0]:
        table[key] = (cost, events)


# -------------------------------------------------------------- assignment
def assign_drones(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """Min-cost one-to-one assignment (rows = requests, cols = drones).

    Entries of ``inf`` are infeasible. Uses the Hungarian algorithm from
    scipy when available, otherwise a greedy fallback. Returns (row, col)
    pairs with finite cost.
    """
    if not cost or not cost[0]:
        return []
    rows, cols = len(cost), len(cost[0])
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        big = 1e12
        c = np.array([[big if v == INF else v for v in row] for row in cost], dtype=float)
        r_idx, c_idx = linear_sum_assignment(c)
        return [(int(r), int(k)) for r, k in zip(r_idx, c_idx) if cost[r][k] != INF]
    except ImportError:
        pairs = []
        used_r, used_c = set(), set()
        cand = sorted((cost[r][k], r, k) for r in range(rows) for k in range(cols)
                      if cost[r][k] != INF)
        for v, r, k in cand:
            if r not in used_r and k not in used_c:
                pairs.append((r, k))
                used_r.add(r)
                used_c.add(k)
        return pairs
