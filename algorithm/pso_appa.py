"""
PSO-APPA: Particle Swarm Optimization Phase 1 + Ant Colony System Phase 2
Combined algorithm for UAV coverage path planning
"""
import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from utils.config import UAV, Region


@dataclass
class PSOConfig:
    """Configuration for PSO parameters"""
    swarm_size: int = 50
    max_iterations: int = 100
    w: float = 0.7298  # Inertia weight
    c1: float = 1.49618  # Cognitive coefficient
    c2: float = 1.49618  # Social coefficient
    w_max: float = 0.9  # Maximum inertia weight
    w_min: float = 0.4  # Minimum inertia weight
    random_seed: Optional[int] = None


class PSOPhase1:
    """
    Particle Swarm Optimization for Phase 1: Region Allocation
    
    Particle representation:
    - Position: Continuous values in [0, num_uavs) for each region
    - Velocity: Rate of change for each dimension
    - Decoded: Round position to get UAV index for each region
    
    Fitness: Minimize max completion time across all UAVs
    """
    
    def __init__(
        self,
        num_uavs: int,
        num_regions: int,
        TS_matrix: np.ndarray,
        TF_matrix: np.ndarray,
        D_matrix: np.ndarray,
        V_matrix: np.ndarray,
        uavs_max_velocity: List[float],
        regions_coords: List[Tuple[float, float]],
        config: PSOConfig = None
    ):
        self.num_uavs = num_uavs
        self.num_regions = num_regions
        self.TS_matrix = TS_matrix
        self.TF_matrix = TF_matrix
        self.D_matrix = D_matrix
        self.V_matrix = V_matrix
        self.uavs_max_velocity = uavs_max_velocity
        self.regions_coords = regions_coords
        
        self.config = config if config else PSOConfig()
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Build feasibility matrix
        self.feasibility_matrix = (self.V_matrix > 0).astype(int)
        
        # Track best solution
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        self.fitness_history = []
        
    def _get_feasible_uavs(self, region_idx: int) -> List[int]:
        """Get list of UAVs that can scan a given region"""
        return [i for i in range(self.num_uavs) if self.feasibility_matrix[i, region_idx] > 0]
    
    def _initialize_swarm(self):
        """Initialize particle swarm"""
        positions = np.zeros((self.config.swarm_size, self.num_regions))
        velocities = np.zeros((self.config.swarm_size, self.num_regions))
        
        for p in range(self.config.swarm_size):
            for r in range(self.num_regions):
                feasible_uavs = self._get_feasible_uavs(r)
                if feasible_uavs:
                    # Initialize position within feasible range
                    positions[p, r] = random.choice(feasible_uavs) + random.random()
                else:
                    positions[p, r] = random.random() * self.num_uavs
                
                # Initialize velocity
                velocities[p, r] = random.uniform(-1, 1)
        
        return positions, velocities
    
    def _decode_position(self, position: np.ndarray) -> List[int]:
        """Convert continuous position to discrete UAV assignment"""
        chromosome = []
        for r in range(self.num_regions):
            feasible_uavs = self._get_feasible_uavs(r)
            if feasible_uavs:
                # Map continuous value to feasible UAV
                idx = int(position[r]) % len(feasible_uavs)
                chromosome.append(feasible_uavs[idx])
            else:
                chromosome.append(int(position[r]) % self.num_uavs)
        return chromosome
    
    def _calculate_fitness(self, position: np.ndarray) -> float:
        """Calculate fitness for a particle position"""
        chromosome = self._decode_position(position)
        completion_times = self._calculate_completion_times(chromosome)
        max_time = max(completion_times.values()) if completion_times else float('inf')
        
        # Fitness is inverse of max completion time (we want to minimize)
        return 1.0 / (max_time + 1e-6)
    
    def _calculate_completion_times(self, chromosome: List[int]) -> Dict[int, float]:
        """Calculate completion time for each UAV given a chromosome"""
        uav_regions = {i: [] for i in range(self.num_uavs)}
        for region_idx, uav_idx in enumerate(chromosome):
            uav_regions[uav_idx].append(region_idx)
        
        completion_times = {}
        
        for uav_idx in range(self.num_uavs):
            regions = uav_regions[uav_idx]
            
            if not regions:
                completion_times[uav_idx] = 0.0
                continue
            
            ordered_regions = self._nearest_neighbor_order(uav_idx, regions)
            total_time = self._calculate_tour_time(uav_idx, ordered_regions)
            completion_times[uav_idx] = total_time
        
        return completion_times
    
    def _nearest_neighbor_order(self, uav_idx: int, regions: List[int]) -> List[int]:
        """Order regions using nearest neighbor heuristic"""
        if len(regions) <= 1:
            return regions
        
        ordered = []
        remaining = set(regions)
        current_pos = (0, 0)
        
        while remaining:
            min_dist = np.inf
            nearest = None
            
            for region_idx in remaining:
                region_coords = self.regions_coords[region_idx]
                dist = math.sqrt(
                    (region_coords[0] - current_pos[0])**2 + 
                    (region_coords[1] - current_pos[1])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest = region_idx
            
            ordered.append(nearest)
            remaining.remove(nearest)
            current_pos = self.regions_coords[nearest]
        
        return ordered
    
    def _calculate_tour_time(self, uav_idx: int, tour: List[int]) -> float:
        """Calculate total time for UAV to complete a tour"""
        if not tour:
            return 0.0
        
        total_time = 0.0
        
        # From base to first region
        first_coords = self.regions_coords[tour[0]]
        distance = math.sqrt(first_coords[0]**2 + first_coords[1]**2)
        total_time += distance / self.uavs_max_velocity[uav_idx]
        total_time += self.TS_matrix[uav_idx, tour[0]]
        
        # Between regions
        for i in range(len(tour) - 1):
            total_time += self.TF_matrix[uav_idx, tour[i], tour[i+1]]
            total_time += self.TS_matrix[uav_idx, tour[i+1]]
        
        # Return to base
        last_coords = self.regions_coords[tour[-1]]
        distance_back = math.sqrt(last_coords[0]**2 + last_coords[1]**2)
        total_time += distance_back / self.uavs_max_velocity[uav_idx]
        
        return total_time
    
    def solve(self) -> Dict[int, List[int]]:
        """Run PSO algorithm to find optimal region allocation"""
        # Initialize swarm
        positions, velocities = self._initialize_swarm()
        
        # Personal best for each particle
        personal_best_positions = positions.copy()
        personal_best_fitness = np.array([self._calculate_fitness(p) for p in positions])
        
        # Global best
        best_idx = np.argmax(personal_best_fitness)
        self.global_best_position = positions[best_idx].copy()
        self.global_best_fitness = personal_best_fitness[best_idx]
        
        self.fitness_history = [self.global_best_fitness]
        
        # Main PSO loop
        for iteration in range(self.config.max_iterations):
            # Adaptive inertia weight
            w = self.config.w_max - (self.config.w_max - self.config.w_min) * iteration / self.config.max_iterations
            
            for p in range(self.config.swarm_size):
                # Update velocity
                r1 = np.random.random(self.num_regions)
                r2 = np.random.random(self.num_regions)
                
                cognitive = self.config.c1 * r1 * (personal_best_positions[p] - positions[p])
                social = self.config.c2 * r2 * (self.global_best_position - positions[p])
                
                velocities[p] = w * velocities[p] + cognitive + social
                
                # Clamp velocity
                v_max = self.num_uavs * 0.5
                velocities[p] = np.clip(velocities[p], -v_max, v_max)
                
                # Update position
                positions[p] = positions[p] + velocities[p]
                
                # Ensure positions are within bounds
                positions[p] = np.clip(positions[p], 0, self.num_uavs - 0.01)
                
                # Calculate fitness
                fitness = self._calculate_fitness(positions[p])
                
                # Update personal best
                if fitness > personal_best_fitness[p]:
                    personal_best_fitness[p] = fitness
                    personal_best_positions[p] = positions[p].copy()
                    
                    # Update global best
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = positions[p].copy()
            
            self.fitness_history.append(self.global_best_fitness)
        
        # Convert best position to region assignment
        best_chromosome = self._decode_position(self.global_best_position)
        
        # Group regions by UAV
        region_assignment = {i: [] for i in range(self.num_uavs)}
        for region_idx, uav_idx in enumerate(best_chromosome):
            region_assignment[uav_idx].append(region_idx)
        
        return region_assignment
    
    def get_fitness_history(self) -> List[float]:
        return self.fitness_history


class PSOAPPAAlgorithm:
    """
    PSO-APPA Algorithm: Uses PSO for Phase 1 (region allocation) and ACS for Phase 2 (order optimization)
    """
    
    def __init__(
        self,
        uavs_list: List[UAV],
        regions_list: List[Region],
        V_matrix: List[List[float]],
        # PSO parameters
        pso_swarm_size: int = 50,
        pso_max_iterations: int = 100,
        pso_w: float = 0.7298,
        pso_c1: float = 1.49618,
        pso_c2: float = 1.49618,
        # ACS parameters
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
        
        # PSO config
        self.pso_config = PSOConfig(
            swarm_size=pso_swarm_size,
            max_iterations=pso_max_iterations,
            w=pso_w,
            c1=pso_c1,
            c2=pso_c2
        )
        
        # ACS parameters
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        self.q0 = q0
        
        # Precompute matrices
        self.D_matrix = self._compute_distance_matrix()
        self.TS_matrix = self._compute_scan_time_matrix()
        self.TF_matrix = self._compute_flight_time_matrix()
        
        self.pso_fitness_history = []
        
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
        """Compute scan time matrix"""
        TS = np.full((self.num_uavs, self.num_regions), np.inf)
        for i in range(self.num_uavs):
            for j in range(self.num_regions):
                if self.V_matrix[i][j] > 0:
                    area = self.regions_list[j].area
                    scan_width = self.uavs_list[i].scan_width
                    TS[i][j] = area / (self.V_matrix[i][j] * scan_width)
        return TS
    
    def _compute_flight_time_matrix(self) -> np.ndarray:
        """Compute flight time matrix"""
        TF = np.zeros((self.num_uavs, self.num_regions, self.num_regions))
        for i in range(self.num_uavs):
            for j in range(self.num_regions):
                for k in range(self.num_regions):
                    if j != k:
                        TF[i][j][k] = self.D_matrix[j][k] / self.uavs_list[i].max_velocity
        return TF
    
    def region_allocation_phase(self) -> Dict[int, List[int]]:
        """Phase 1: Region Allocation using PSO"""
        uavs_max_velocity = [uav.max_velocity for uav in self.uavs_list]
        regions_coords = [region.coords for region in self.regions_list]
        
        pso = PSOPhase1(
            num_uavs=self.num_uavs,
            num_regions=self.num_regions,
            TS_matrix=self.TS_matrix,
            TF_matrix=self.TF_matrix,
            D_matrix=self.D_matrix,
            V_matrix=self.V_matrix,
            uavs_max_velocity=uavs_max_velocity,
            regions_coords=regions_coords,
            config=self.pso_config
        )
        
        region_assignment = pso.solve()
        self.pso_fitness_history = pso.get_fitness_history()
        
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
    
    def order_optimization_phase(self, uav_idx: int, assigned_regions: List[int]) -> List[int]:
        """Phase 2: Order Optimization using ACS"""
        if len(assigned_regions) <= 1:
            return assigned_regions
        
        num_regions_assigned = len(assigned_regions)
        
        # Initialize pheromone matrix
        initial_tour, L = self._nearest_neighbor_tour(assigned_regions)
        tau0 = 1.0 / (num_regions_assigned * L) if L > 0 else 1.0
        
        tau = np.full((num_regions_assigned, num_regions_assigned), tau0)
        
        # Heuristic information matrix
        eta = np.zeros((num_regions_assigned, num_regions_assigned))
        for i in range(num_regions_assigned):
            for j in range(num_regions_assigned):
                if i != j:
                    dist = self.D_matrix[assigned_regions[i]][assigned_regions[j]]
                    eta[i][j] = 1.0 / dist if dist > 0 else 0.0
        
        best_tour = initial_tour.copy()
        best_length = self._compute_tour_length(uav_idx, best_tour)
        
        stagnation_count = 0
        max_stagnation = 10
        
        for iteration in range(self.max_iterations):
            ant_tours = []
            ant_lengths = []
            
            for ant in range(self.num_ants):
                current_idx = random.randint(0, num_regions_assigned - 1)
                unvisited = set(range(num_regions_assigned))
                unvisited.remove(current_idx)
                tour = [current_idx]
                
                while unvisited:
                    next_idx = self._select_next_region(current_idx, unvisited, tau, eta)
                    tau[current_idx][next_idx] = (1 - self.rho) * tau[current_idx][next_idx] + self.rho * tau0
                    tour.append(next_idx)
                    unvisited.remove(next_idx)
                    current_idx = next_idx
                
                actual_tour = [assigned_regions[idx] for idx in tour]
                tour_length = self._compute_tour_length(uav_idx, actual_tour)
                
                ant_tours.append(tour)
                ant_lengths.append(tour_length)
            
            iteration_best_idx = np.argmin(ant_lengths)
            iteration_best_tour = ant_tours[iteration_best_idx]
            iteration_best_length = ant_lengths[iteration_best_idx]
            
            if iteration_best_length < best_length:
                best_tour = [assigned_regions[idx] for idx in iteration_best_tour]
                best_length = iteration_best_length
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            if stagnation_count >= max_stagnation:
                break
            
            for i in range(len(iteration_best_tour)):
                j = (i + 1) % len(iteration_best_tour)
                idx_i = iteration_best_tour[i]
                idx_j = iteration_best_tour[j]
                tau[idx_i][idx_j] = (1 - self.epsilon) * tau[idx_i][idx_j] + \
                                     self.epsilon * (1.0 / best_length)
        
        return best_tour
    
    def _select_next_region(self, current_idx: int, unvisited: set, tau: np.ndarray, eta: np.ndarray) -> int:
        """Select next region using ACS rule"""
        q = random.random()
        
        if q <= self.q0:
            best_value = -np.inf
            best_idx = None
            for j in unvisited:
                value = (tau[current_idx][j] ** self.alpha) * (eta[current_idx][j] ** self.beta)
                if value > best_value:
                    best_value = value
                    best_idx = j
            return best_idx
        else:
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
        
        base_coords = (0, 0)
        first_coords = self.regions_list[tour[0]].coords
        distance = math.sqrt(
            (first_coords[0] - base_coords[0])**2 + 
            (first_coords[1] - base_coords[1])**2
        )
        total_time += distance / self.uavs_list[uav_idx].max_velocity
        total_time += self.TS_matrix[uav_idx][tour[0]]
        
        for i in range(len(tour) - 1):
            total_time += self.TF_matrix[uav_idx][tour[i]][tour[i+1]]
            total_time += self.TS_matrix[uav_idx][tour[i+1]]
        
        last_coords = self.regions_list[tour[-1]].coords
        distance_back = math.sqrt(
            (base_coords[0] - last_coords[0])**2 + 
            (base_coords[1] - last_coords[1])**2
        )
        total_time += distance_back / self.uavs_list[uav_idx].max_velocity
                
        return total_time
    
    def solve(self) -> Dict:
        """Main solving method"""
        region_assignment = self.region_allocation_phase()
        
        optimized_paths = {}
        completion_times = {}
        
        for uav_idx in range(self.num_uavs):
            if region_assignment[uav_idx]:
                optimized_tour = self.order_optimization_phase(uav_idx, region_assignment[uav_idx])
                optimized_paths[uav_idx] = optimized_tour
                completion_times[uav_idx] = self._compute_tour_length(uav_idx, optimized_tour)
            else:
                optimized_paths[uav_idx] = []
                completion_times[uav_idx] = 0.0
        
        max_completion_time = max(completion_times.values()) if completion_times else 0.0
        
        return {
            'paths': optimized_paths,
            'completion_times': completion_times,
            'max_completion_time': max_completion_time,
            'pso_fitness_history': self.pso_fitness_history
        }


if __name__ == "__main__":
    from utils.create_sample import create_sample
    
    data = create_sample(NUM_UAVS=4, NUM_REGIONS=20)
    
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    
    pso_appa = PSOAPPAAlgorithm(uavs_list, regions_list, V_matrix)
    result = pso_appa.solve()
    
    print("\n=== PSO-APPA Results ===")
    print(f"Max Completion Time: {result['max_completion_time']:.2f}")
    for uav_idx, path in result['paths'].items():
        print(f"UAV {uav_idx}: {path} (Time: {result['completion_times'][uav_idx]:.2f})")
