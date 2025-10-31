"""
Hybrid Q-learning + Ant Colony System (ACS) for UAV area scanning assignment
Optimized version: precompute matrices, epsilon-greedy, rank-based pheromone update,
local search (2-opt + cross-swap), and per-UAV normalized reward Q-update.

Input assumptions:
- uav dicts: {"id": 1..n, "max_velocity": ..., "scan_width": ...}
- region dicts: {"id": 1..m, "coords": [x,y], "area": ...}
- V_matrix: list-of-lists shape (n, m) indexed by 0-based uav_id-1 and region_id-1
"""

import math
import random
import numpy as np
from copy import deepcopy
import json
from typing import List

# ---------------------------
# Data classes / helpers
# ---------------------------
class UAV:
    def __init__(self, id, max_velocity, scan_width):
        # incoming id assumed 1..n; store zero-based index
        self.id = int(id) - 1
        self.max_velocity = float(max_velocity)
        self.scan_width = float(scan_width)

class Region:
    def __init__(self, id, coords, area):
        self.id = int(id) - 1
        self.coords = tuple(coords)
        self.area = float(area)

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ---------------------------
# Optimized Hybrid ACS + Q class
# ---------------------------
class HybridACS_Q_Optimized:
    def __init__(self, uavs: List[UAV], regions: List[Region], V_matrix,
                 n_ants=None, n_iters=300,
                 alpha=1.0, beta=2.0, zeta=1.0,    # pheromone, Q influence (via exp), heuristic
                 rho=0.03, phi=1.0,               # evaporation, pheromone deposit scale
                 alpha_q=0.05, gamma_q=0.0,       # Q params
                 epsilon_start=0.3, epsilon_end=0.05, epsilon_decay=0.995,
                 top_k=5,                         # rank-based pheromone top-k
                 seed=0):
        self.uavs = uavs
        self.regions = regions
        self.V = np.array(V_matrix, dtype=float)
        self.n_uav = len(uavs)
        self.n_region = len(regions)

        # ants
        self.n_ants = n_ants if n_ants is not None else max(40, self.n_region * 2)
        self.n_iters = n_iters

        # ACS / Q params
        self.alpha = alpha
        self.beta = beta
        self.zeta = zeta
        self.rho = rho
        self.phi = phi
        self.alpha_q = alpha_q
        self.gamma_q = gamma_q

        # epsilon-greedy schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.top_k = max(1, top_k)

        # RNG
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # pheromone & Q table
        init_tau = 1.0
        self.tau = np.ones((self.n_uav, self.n_region)) * init_tau
        self.Q = np.zeros((self.n_uav, self.n_region))

        # Precompute scan_time[u,i] and travel_time[u,i,j]
        self.scan_time = np.zeros((self.n_uav, self.n_region))
        for i, u in enumerate(self.uavs):
            for j, r in enumerate(self.regions):
                v = max(1e-8, self.V[i, j])
                eff = max(1e-8, u.scan_width * v)
                self.scan_time[i, j] = r.area / eff

        # distances between regions (euclidean)
        self.dist = np.zeros((self.n_region, self.n_region))
        for a in range(self.n_region):
            for b in range(self.n_region):
                self.dist[a,b] = euclid(self.regions[a].coords, self.regions[b].coords)

        # travel_time per UAV between any two regions
        self.travel_time = np.zeros((self.n_uav, self.n_region, self.n_region))
        for i, u in enumerate(self.uavs):
            for a in range(self.n_region):
                va = max(1e-8, self.V[i, a])
                for b in range(self.n_region):
                    vb = max(1e-8, self.V[i, b])
                    vmove = min(u.max_velocity, va, vb)
                    vmove = max(vmove, 1e-8)
                    self.travel_time[i, a, b] = self.dist[a,b] / vmove

        # heuristic: eta = 1 / scan_time (can incorporate travel distance later)
        self.eta = np.zeros((self.n_uav, self.n_region))
        for i in range(self.n_uav):
            for j in range(self.n_region):
                self.eta[i,j] = 1.0 / max(self.scan_time[i,j], 1e-8)

        # best
        self.best_solution = None
        self.best_makespan = float('inf')

    # compute route time quickly with precomputed matrices (route given as list of region indices)
    def compute_route_time_idx(self, u_idx, route_idx_list):
        if not route_idx_list:
            return 0.0
        total = 0.0
        # scan first region
        total += self.scan_time[u_idx, route_idx_list[0]]
        for k in range(1, len(route_idx_list)):
            a = route_idx_list[k-1]
            b = route_idx_list[k]
            total += self.travel_time[u_idx, a, b]
            total += self.scan_time[u_idx, b]
        return total

    # nearest-neighbor ordering (start from largest area to bias heavy tasks)
    def order_route_nn(self, u_idx, region_idx_list):
        if not region_idx_list:
            return []
        remaining = region_idx_list.copy()
        # start at largest area
        start = max(remaining, key=lambda idx: self.regions[idx].area)
        route = [start]
        remaining.remove(start)
        while remaining:
            last = route[-1]
            next_reg = min(remaining, key=lambda c: self.travel_time[u_idx, last, c])
            route.append(next_reg)
            remaining.remove(next_reg)
        return route

    # 2-opt for single route (indices)
    def two_opt_route(self, u_idx, route):
        improved = True
        best = route
        best_time = self.compute_route_time_idx(u_idx, best)
        while improved:
            improved = False
            n = len(best)
            for i in range(0, n-1):
                for j in range(i+2, n):
                    if j - i == 1:  # adjacent, no effect
                        continue
                    new_route = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                    new_time = self.compute_route_time_idx(u_idx, new_route)
                    if new_time + 1e-9 < best_time:
                        best = new_route
                        best_time = new_time
                        improved = True
                        break
                if improved:
                    break
        return best, best_time

    # cross-swap between UAVs: try swapping single region assignments to reduce makespan
    def cross_swap_between_uavs(self, uav_routes_idx, times):
        improved = True
        while improved:
            improved = False
            # find uav with max time and uav with min time
            max_i = int(np.argmax(times))
            min_i = int(np.argmin(times))
            if times[max_i] - times[min_i] < 1e-6:
                break
            # try swapping one region from max_i with one region from min_i (or empty)
            route_max = uav_routes_idx[max_i]
            route_min = uav_routes_idx[min_i]
            best_gain = 0.0
            best_pair = None
            # consider removing an element from max and moving to min (one-way), and vice versa
            # We'll try single-region move (max -> min) and single-swap (max region <-> min region)
            # Single move:
            for r_idx in route_max:
                new_max_route = [x for x in route_max if x != r_idx]
                new_min_route = route_min + [r_idx]
                t_max_new = self.compute_route_time_idx(max_i, new_max_route)
                t_min_new = self.compute_route_time_idx(min_i, new_min_route)
                new_makespan = max([t_max_new if idx==max_i else times[idx] for idx in range(self.n_uav)])
                new_makespan = max(new_makespan, t_min_new)
                gain = times[max_i] - max(t_max_new, t_min_new, *(times[k] for k in range(self.n_uav) if k not in (max_i, min_i)))
                if gain > best_gain + 1e-9:
                    best_gain = gain
                    best_pair = ("move", max_i, min_i, r_idx, new_max_route, new_min_route, t_max_new, t_min_new)
            # Single-swap:
            for r1 in route_max:
                for r2 in route_min:
                    new_max = [x if x!=r1 else r2 for x in route_max]
                    new_min = [x if x!=r2 else r1 for x in route_min]
                    t1 = self.compute_route_time_idx(max_i, new_max)
                    t2 = self.compute_route_time_idx(min_i, new_min)
                    new_makespan = max([t1 if idx==max_i else times[idx] for idx in range(self.n_uav)])
                    new_makespan = max(new_makespan, t2)
                    gain = times[max_i] - max(t1, t2, *(times[k] for k in range(self.n_uav) if k not in (max_i, min_i)))
                    if gain > best_gain + 1e-9:
                        best_gain = gain
                        best_pair = ("swap", max_i, min_i, r1, r2, new_max, new_min, t1, t2)
            if best_pair is not None:
                improved = True
                if best_pair[0] == "move":
                    _, i_max, i_min, r_idx, new_max_route, new_min_route, t_max_new, t_min_new = best_pair
                    uav_routes_idx[i_max] = new_max_route
                    uav_routes_idx[i_min] = new_min_route
                    times[i_max] = t_max_new
                    times[i_min] = t_min_new
                else:
                    (_, i_max, i_min, r1, r2, new_max, new_min, t1, t2) = best_pair
                    uav_routes_idx[i_max] = new_max
                    uav_routes_idx[i_min] = new_min
                    times[i_max] = t1
                    times[i_min] = t2
        return uav_routes_idx, times

    # construct one ant's assignment using epsilon-greedy + tau/eta/Q combination
    def construct_solution(self):
        assign = [-1] * self.n_region
        for j in range(self.n_region):
            # epsilon-greedy exploration
            if self.rng.random() < self.epsilon:
                i_choice = self.rng.randrange(self.n_uav)
                assign[j] = i_choice
                continue
            # compute weights
            weights = np.zeros(self.n_uav)
            for i in range(self.n_uav):
                tau_ij = (self.tau[i,j] ** self.alpha)
                qterm = math.exp(self.beta * self.Q[i,j])
                etaterm = (self.eta[i,j] ** self.zeta)
                weights[i] = tau_ij * qterm * etaterm
            s = weights.sum()
            if s <= 0:
                i_choice = self.rng.randrange(self.n_uav)
            else:
                probs = weights / s
                # sample using rng.choices
                i_choice = self.rng.choices(range(self.n_uav), weights=probs, k=1)[0]
            assign[j] = i_choice
        return assign

    # given assignment list (region -> uav), build ordered routes, apply local search, compute times and makespan
    def build_routes_and_makespan(self, assignment):
        # collect indices per UAV
        uav_routes_idx = [[] for _ in range(self.n_uav)]
        for r_idx, u_idx in enumerate(assignment):
            uav_routes_idx[u_idx].append(r_idx)

        # order each route via NN then 2-opt refine
        ordered_routes_idx = []
        times = np.zeros(self.n_uav)
        for i in range(self.n_uav):
            regs = uav_routes_idx[i]
            if not regs:
                ordered_routes_idx.append([])
                times[i] = 0.0
                continue
            route = self.order_route_nn(i, regs)
            route, t = self.two_opt_route(i, route)
            ordered_routes_idx.append(route)
            times[i] = t

        # cross-swap refinement across UAVs
        ordered_routes_idx, times = self.cross_swap_between_uavs(ordered_routes_idx, times.tolist())
        makespan = float(np.max(times)) if len(times)>0 else 0.0
        return ordered_routes_idx, times, makespan

    # rank-based pheromone update: top_k ants deposit proportional to quality
    def update_pheromones(self, ant_solutions_sorted):
        # evaporate global
        self.tau *= (1.0 - self.rho)
        # deposit from top_k solutions
        k = min(self.top_k, len(ant_solutions_sorted))
        for rank in range(k):
            sol = ant_solutions_sorted[rank]
            assign = sol['assignment']
            makespan = sol['makespan']
            # quality weight: better solutions deposit more; use inverse makespan
            weight = (self.phi * (k - rank)) / max(makespan, 1e-8)
            for j, i in enumerate(assign):
                self.tau[i,j] += weight

    # update Q using per-UAV normalized reward (balance load)
    def update_Q(self, assignment, times):
        times = np.array(times, dtype=float)
        mean_t = float(np.mean(times)) if len(times)>0 else 1.0
        # normalize by mean_t to keep rewards stable across problem sizes
        for i in range(self.n_uav):
            reward_i = -abs(times[i] - mean_t) / max(mean_t, 1e-8)
            # update Q for pairs (i, regions assigned to i)
            for j, a in enumerate(assignment):
                if a == i:
                    old = self.Q[i,j]
                    td_target = reward_i
                    self.Q[i,j] = old + self.alpha_q * (td_target - old)

    def run(self, verbose=False):
        for it in range(self.n_iters):
            ant_solutions = []
            for a in range(self.n_ants):
                assign = self.construct_solution()
                routes, times, makespan = self.build_routes_and_makespan(assign)
                ant_solutions.append({'assignment': assign, 'routes': routes, 'times': times, 'makespan': makespan})

            # sort by makespan ascending (lower is better)
            ant_solutions.sort(key=lambda s: s['makespan'])
            iter_best = ant_solutions[0]
            if iter_best['makespan'] < self.best_makespan:
                self.best_makespan = iter_best['makespan']
                self.best_solution = deepcopy(iter_best)

            # pheromone update using top-k (rank-based)
            self.update_pheromones(ant_solutions)

            # Q-learning update using iteration best solution (you can also aggregate top-k)
            self.update_Q(iter_best['assignment'], iter_best['times'])

            # epsilon decay
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
                if self.epsilon < self.epsilon_end:
                    self.epsilon = self.epsilon_end

            if verbose and ((it+1) % max(1, self.n_iters//10) == 0 or it==0):
                print(f"Iter {it+1}/{self.n_iters} best_so_far: {self.best_makespan:.4f} eps={self.epsilon:.3f}")

        return self.best_solution

# ---------------------------
# Helper wrapper to run with JSON-like data
# ---------------------------
def q_learning_acs_run_optimized(data, verbose=False):
    uavs = [UAV(**u) for u in data['uavs_list']]
    regions = [Region(**r) for r in data['regions_list']]
    V = data['V_matrix']

    solver = HybridACS_Q_Optimized(uavs, regions, V,
                                   n_ants=max(40, len(regions)*2),
                                   n_iters=250,
                                   alpha=1.0, beta=2.0, zeta=1.0,
                                   rho=0.03, phi=1.0,
                                   alpha_q=0.05, gamma_q=0.0,
                                   epsilon_start=0.3, epsilon_end=0.05, epsilon_decay=0.995,
                                   top_k=8,
                                   seed=42)
    best = solver.run(verbose=verbose)
    return best

# ---------------------------
# Example usage (synthetic or from sample.json)
# ---------------------------
def example_run_from_file(path="./sample.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    best = q_learning_acs_run_optimized(data, verbose=True)
    print("\nBEST makespan:", best['makespan'])
    assign = best['assignment']
    # print summary
    per_uav = {}
    for i in range(len(data['uavs_list'])):
        assigned = [j for j,a in enumerate(assign) if a==i]
        per_uav[i] = assigned
        print(f"UAV {i}: {len(assigned)} regions -> {assigned}")
    return best, per_uav

if __name__ == "__main__":
    # run example (expects sample.json in same folder)
    try:
        best, per_uav = example_run_from_file("./sample.json")
    except Exception as e:
        print("Error running example. Ensure sample.json present or call q_learning_acs_run_optimized with your data.")
        raise
