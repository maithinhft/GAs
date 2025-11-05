import numpy as np
import random
from typing import List, Dict, Tuple
import math
from utils.config import *
class APPAAlgorithm:
    def __init__(
        self,
        uavs_list: List[UAV],
        regions_list: List[Region],
        V_matrix: List[List[float]],
        num_ants: int = 20,
        max_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        epsilon: float = 0.1,
        q0: float = 0.9
    ):
        self.uavs_list = uavs_list
        self.regions_list = regions_list
        self.V_matrix = np.array(V_matrix)
        
        self.num_uavs = len(uavs_list)
        self.num_regions = len(regions_list)
        
        # ACS parameters
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        self.q0 = q0
        
        # Precompute distance matrix
        self.D_matrix = self._compute_distance_matrix()
        
        # Compute TS and TF matrices
        self.TS_matrix = self._compute_scan_time_matrix()
        self.TF_matrix = self._compute_flight_time_matrix()
        
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute distance matrix between all regions"""
        D = np.zeros((self.num_regions, self.num_regions))
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if i != j:
                    x1, y1 = self.regions_list[i].coords
                    x2, y2 = self.regions_list[j].coords
                    D[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return D
    
    def _compute_scan_time_matrix(self) -> np.ndarray:
        """Compute scan time matrix TS[i,j] for UAV i scanning region j"""
        TS = np.full((self.num_uavs, self.num_regions), np.inf)
        for i in range(self.num_uavs):
            for j in range(self.num_regions):
                if self.V_matrix[i][j] > 0:
                    area = self.regions_list[j].area
                    scan_width = self.uavs_list[i].scan_width
                    TS[i][j] = area / (self.V_matrix[i][j] * scan_width)
        return TS
    
    def _compute_flight_time_matrix(self) -> np.ndarray:
        """Compute flight time matrix TF[i,j,k] for UAV i flying from region j to k"""
        TF = np.zeros((self.num_uavs, self.num_regions, self.num_regions))
        for i in range(self.num_uavs):
            for j in range(self.num_regions):
                for k in range(self.num_regions):
                    if j != k:
                        TF[i][j][k] = self.D_matrix[j][k] / self.uavs_list[i].max_velocity
        return TF
    
    def _compute_effective_time_ratio(
        self, 
        uav_idx: int, 
        last_region_idx: int, 
        target_region_idx: int
    ) -> float:
        """
        Compute effective time ratio ETR according to Eq. (9)
        ETR[i,j,k] = TS[i,k] / (TF[i,j,k] + TS[i,k])
        """
        ts = self.TS_matrix[uav_idx][target_region_idx]
        
        if ts == np.inf:
            return -np.inf
        
        if last_region_idx == -1:
            # Flying from base (assume base at origin)
            base_coords = (0, 0)
            target_coords = self.regions_list[target_region_idx].coords
            distance = math.sqrt(
                (target_coords[0] - base_coords[0])**2 + 
                (target_coords[1] - base_coords[1])**2
            )
            tf = distance / self.uavs_list[uav_idx].max_velocity
        else:
            tf = self.TF_matrix[uav_idx][last_region_idx][target_region_idx]
        
        etr = ts / (tf + ts)
        return etr
    
    def region_allocation_phase(self) -> Dict[int, List[int]]:
        """
        Phase 1: Region Allocation (Algorithm 1)
        Allocate regions to UAVs based on effective time ratio
        """
        left_regions = set(range(self.num_regions))
        region_assignment = {i: [] for i in range(self.num_uavs)}
        finish_time = [0.0] * self.num_uavs
        last_region = [-1] * self.num_uavs  # -1 means at base
        
        while left_regions:
            # Find UAV that would finish earliest
            uav_idx = np.argmin(finish_time)
            
            # Compute ETR for all remaining regions
            best_region = None
            best_etr = -np.inf
            
            for region_idx in left_regions:
                etr = self._compute_effective_time_ratio(
                    uav_idx, 
                    last_region[uav_idx], 
                    region_idx
                )
                if etr > best_etr:
                    best_etr = etr
                    best_region = region_idx
            
            # Assign best region to UAV
            if best_region is not None:
                left_regions.remove(best_region)
                region_assignment[uav_idx].append(best_region)
                
                # Update finish time
                ts = self.TS_matrix[uav_idx][best_region]
                if last_region[uav_idx] == -1:
                    # From base
                    base_coords = (0, 0)
                    target_coords = self.regions_list[best_region].coords
                    distance = math.sqrt(
                        (target_coords[0] - base_coords[0])**2 + 
                        (target_coords[1] - base_coords[1])**2
                    )
                    tf = distance / self.uavs_list[uav_idx].max_velocity
                else:
                    tf = self.TF_matrix[uav_idx][last_region[uav_idx]][best_region]
                
                finish_time[uav_idx] += tf + ts
                last_region[uav_idx] = best_region
        
        return region_assignment
    
    def _nearest_neighbor_tour(self, regions: List[int]) -> Tuple[List[int], float]:
        """Get initial tour using nearest neighbor heuristic"""
        if len(regions) <= 1:
            return regions, 0.0
        
        unvisited = set(regions)
        current = regions[0]
        unvisited.remove(current)
        tour = [current]
        total_length = 0.0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.D_matrix[current][x])
            total_length += self.D_matrix[current][nearest]
            tour.append(nearest)
            current = nearest
            unvisited.remove(nearest)
        
        return tour, total_length
    
    def order_optimization_phase(
        self, 
        uav_idx: int, 
        assigned_regions: List[int]
    ) -> List[int]:
        """
        Phase 2: Order Optimization (Algorithm 2)
        Use ACS to optimize visiting order of regions for a single UAV
        """
        if len(assigned_regions) <= 1:
            return assigned_regions
        
        num_regions_assigned = len(assigned_regions)
        
        # Initialize pheromone matrix
        initial_tour, L = self._nearest_neighbor_tour(assigned_regions)
        tau0 = 1.0 / (num_regions_assigned * L) if L > 0 else 1.0
        
        # Pheromone matrix (only between assigned regions)
        tau = np.full((num_regions_assigned, num_regions_assigned), tau0)
        
        # Heuristic information matrix
        eta = np.zeros((num_regions_assigned, num_regions_assigned))
        for i in range(num_regions_assigned):
            for j in range(num_regions_assigned):
                if i != j:
                    dist = self.D_matrix[assigned_regions[i]][assigned_regions[j]]
                    eta[i][j] = 1.0 / dist if dist > 0 else 0.0
        
        # Best solution tracking
        best_tour = initial_tour.copy()
        best_length = self._compute_tour_length(uav_idx, best_tour)
        
        # Stagnation counter for early stopping (optional optimization)
        stagnation_count = 0
        max_stagnation = 10  # Tăng lên để cho phép explore nhiều hơn
        
        # ACS iterations
        for iteration in range(self.max_iterations):
            # Each ant constructs a solution
            ant_tours = []
            ant_lengths = []
            
            for ant in range(self.num_ants):
                # Random starting position
                current_idx = random.randint(0, num_regions_assigned - 1)
                unvisited = set(range(num_regions_assigned))
                unvisited.remove(current_idx)
                tour = [current_idx]
                
                # Construct tour
                while unvisited:
                    next_idx = self._select_next_region(
                        current_idx, unvisited, tau, eta
                    )
                    
                    # Local pheromone update
                    tau[current_idx][next_idx] = (1 - self.rho) * tau[current_idx][next_idx] + self.rho * tau0
                    
                    tour.append(next_idx)
                    unvisited.remove(next_idx)
                    current_idx = next_idx
                
                # Convert indices back to region IDs
                actual_tour = [assigned_regions[idx] for idx in tour]
                tour_length = self._compute_tour_length(uav_idx, actual_tour)
                
                ant_tours.append(tour)
                ant_lengths.append(tour_length)
            
            # Find best ant in this iteration
            iteration_best_idx = np.argmin(ant_lengths)
            iteration_best_tour = ant_tours[iteration_best_idx]
            iteration_best_length = ant_lengths[iteration_best_idx]
            
            # Update global best
            if iteration_best_length < best_length:
                best_tour = [assigned_regions[idx] for idx in iteration_best_tour]
                best_length = iteration_best_length
                stagnation_count = 0  # Reset khi có cải thiện
            else:
                stagnation_count += 1
            
            # Early stopping nếu không cải thiện trong nhiều iterations
            if stagnation_count >= max_stagnation:
                break
            
            # Global pheromone update
            for i in range(len(iteration_best_tour)):
                j = (i + 1) % len(iteration_best_tour)
                idx_i = iteration_best_tour[i]
                idx_j = iteration_best_tour[j]
                tau[idx_i][idx_j] = (1 - self.epsilon) * tau[idx_i][idx_j] + \
                                     self.epsilon * (1.0 / best_length)
        
        return best_tour
    
    def _select_next_region(
        self, 
        current_idx: int, 
        unvisited: set, 
        tau: np.ndarray, 
        eta: np.ndarray
    ) -> int:
        """Select next region using ACS rule (Eq. 10, 11)"""
        q = random.random()
        
        if q <= self.q0:
            # Exploitation: choose best
            best_value = -np.inf
            best_idx = None
            for j in unvisited:
                value = (tau[current_idx][j] ** self.alpha) * (eta[current_idx][j] ** self.beta)
                if value > best_value:
                    best_value = value
                    best_idx = j
            return best_idx
        else:
            # Exploration: roulette wheel selection
            probabilities = []
            candidates = list(unvisited)
            
            for j in candidates:
                prob = (tau[current_idx][j] ** self.alpha) * (eta[current_idx][j] ** self.beta)
                probabilities.append(prob)
            
            total = sum(probabilities)
            if total == 0:
                return random.choice(candidates)
            
            probabilities = [p / total for p in probabilities]
            return np.random.choice(candidates, p=probabilities)
    
    def _compute_tour_length(self, uav_idx: int, tour: List[int]) -> float:
        """Compute total time for UAV to complete tour"""
        if not tour:
            return 0.0
        
        total_time = 0.0
        
        # From base to first region
        base_coords = (0, 0)
        first_coords = self.regions_list[tour[0]].coords
        distance = math.sqrt(
            (first_coords[0] - base_coords[0])**2 + 
            (first_coords[1] - base_coords[1])**2
        )
        total_time += distance / self.uavs_list[uav_idx].max_velocity
        total_time += self.TS_matrix[uav_idx][tour[0]]
        
        # Between regions
        for i in range(len(tour) - 1):
            total_time += self.TF_matrix[uav_idx][tour[i]][tour[i+1]]
            total_time += self.TS_matrix[uav_idx][tour[i+1]]
        
        # Return to base from last region
        last_coords = self.regions_list[tour[-1]].coords
        distance_back = math.sqrt(
            (base_coords[0] - last_coords[0])**2 + 
            (base_coords[1] - last_coords[1])**2
        )
        total_time += distance_back / self.uavs_list[uav_idx].max_velocity
                
        return total_time
    
    def solve(self) -> Dict:
        """
        Main solving method combining both phases
        Returns optimized paths and completion times
        """
        region_assignment = self.region_allocation_phase()
        
        optimized_paths = {}
        completion_times = {}
        
        for uav_idx in range(self.num_uavs):
            if region_assignment[uav_idx]:
                optimized_tour = self.order_optimization_phase(
                    uav_idx, 
                    region_assignment[uav_idx]
                )
                optimized_paths[uav_idx] = optimized_tour
                completion_times[uav_idx] = self._compute_tour_length(uav_idx, optimized_tour)
            else:
                optimized_paths[uav_idx] = []
                completion_times[uav_idx] = 0.0
        
        max_completion_time = max(completion_times.values()) if completion_times else 0.0
        
        return {
            'paths': optimized_paths,
            'completion_times': completion_times,
            'max_completion_time': max_completion_time
        }


# Example usage
if __name__ == "__main__":
    # Sample data structure (you would load this from create_sample)
    data = {
        'uavs_list': [
            {'id': 1, 'max_velocity': 20.0, 'scan_width': 10.0},
            {'id': 2, 'max_velocity': 25.0, 'scan_width': 15.0},
        ],
        'regions_list': [
            {'id': 1, 'coords': (100, 200), 'area': 5000},
            {'id': 2, 'coords': (300, 400), 'area': 6000},
            {'id': 3, 'coords': (500, 100), 'area': 4500},
        ],
        'V_matrix': [
            [18.0, 19.0, 17.5],
            [22.5, 23.0, 21.0]
        ]
    }
    
    # Initialize
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    
    # Run APPA
    appa = APPAAlgorithm(uavs_list, regions_list, V_matrix)
    result = appa.solve()
    
    print("\n=== APPA Results ===")
    print(f"Max Completion Time: {result['max_completion_time']:.2f}")
    for uav_idx, path in result['paths'].items():
        print(f"UAV {uav_idx}: {path} (Time: {result['completion_times'][uav_idx]:.2f})")