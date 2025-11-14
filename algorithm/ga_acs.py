import random
import numpy as np
from typing import List, Tuple
import math

# Giả định có các class và hàm utils
# from utils.utils import UAV, Region, BASE_COORDS, get_distance, calculate_path_time

def get_distance(coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
    """Tính khoảng cách Euclidean giữa 2 điểm"""
    return math.sqrt((coords2[0] - coords1[0])**2 + (coords2[1] - coords1[1])**2)

class ACSOptimizer:
    """Ant Colony System optimizer cho TSP cục bộ của mỗi UAV"""
    
    def __init__(
        self,
        uav,
        regions: List,
        v_matrix: List[List[float]],
        uav_idx: int,
        all_regions_list: List,
        num_ants: int = 10,
        max_iterations: int = 50,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        epsilon: float = 0.1,
        q0: float = 0.9
    ):
        self.uav = uav
        self.regions = regions
        self.v_matrix = v_matrix
        self.uav_idx = uav_idx
        self.all_regions_list = all_regions_list
        
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        self.q0 = q0
        
        self.num_regions = len(regions)
        
        # Precompute distance matrix
        self.dist_matrix = np.zeros((self.num_regions, self.num_regions))
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if i != j:
                    self.dist_matrix[i][j] = get_distance(
                        regions[i].coords, 
                        regions[j].coords
                    )
    
    def _nearest_neighbor_tour(self) -> Tuple[List[int], float]:
        """Get initial tour using nearest neighbor"""
        if self.num_regions <= 1:
            return list(range(self.num_regions)), 0.0
        
        unvisited = set(range(self.num_regions))
        current = 0
        unvisited.remove(current)
        tour = [current]
        total_length = 0.0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.dist_matrix[current][x])
            total_length += self.dist_matrix[current][nearest]
            tour.append(nearest)
            current = nearest
            unvisited.remove(nearest)
        
        return tour, total_length
    
    def _compute_tour_time(self, tour_indices: List[int]) -> float:
        """Tính tổng thời gian cho tour (từ base -> regions -> cuối)"""
        if not tour_indices:
            return 0.0
        
        BASE_COORDS = (0, 0)
        total_time = 0.0
        
        # From base to first region
        first_region = self.regions[tour_indices[0]]
        dist = get_distance(BASE_COORDS, first_region.coords)
        total_time += dist / self.uav.max_velocity
        
        # Scan first region
        region_idx_in_all = self.all_regions_list.index(first_region)
        scan_velocity = self.v_matrix[self.uav_idx][region_idx_in_all]
        if scan_velocity > 0:
            total_time += first_region.area / (scan_velocity * self.uav.scan_width)
        
        # Between regions
        for i in range(len(tour_indices) - 1):
            curr_region = self.regions[tour_indices[i]]
            next_region = self.regions[tour_indices[i + 1]]
            
            # Flight time
            dist = self.dist_matrix[tour_indices[i]][tour_indices[i + 1]]
            total_time += dist / self.uav.max_velocity
            
            # Scan time
            region_idx_in_all = self.all_regions_list.index(next_region)
            scan_velocity = self.v_matrix[self.uav_idx][region_idx_in_all]
            if scan_velocity > 0:
                total_time += next_region.area / (scan_velocity * self.uav.scan_width)
        
        return total_time
    
    def optimize(self) -> List:
        """Tối ưu tour bằng ACS, trả về danh sách Region objects"""
        if self.num_regions <= 1:
            return self.regions
        
        # Initialize pheromone
        initial_tour, L = self._nearest_neighbor_tour()
        tau0 = 1.0 / (self.num_regions * L) if L > 0 else 1.0
        tau = np.full((self.num_regions, self.num_regions), tau0)
        
        # Heuristic information
        eta = np.zeros((self.num_regions, self.num_regions))
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if i != j and self.dist_matrix[i][j] > 0:
                    eta[i][j] = 1.0 / self.dist_matrix[i][j]
        
        # Best solution tracking
        best_tour = initial_tour.copy()
        best_time = self._compute_tour_time(best_tour)
        
        # ACS iterations
        for iteration in range(self.max_iterations):
            ant_tours = []
            ant_times = []
            
            for ant in range(self.num_ants):
                # Random start
                current_idx = random.randint(0, self.num_regions - 1)
                unvisited = set(range(self.num_regions))
                unvisited.remove(current_idx)
                tour = [current_idx]
                
                # Construct tour
                while unvisited:
                    next_idx = self._select_next(current_idx, unvisited, tau, eta)
                    
                    # Local pheromone update
                    tau[current_idx][next_idx] = (1 - self.rho) * tau[current_idx][next_idx] + self.rho * tau0
                    
                    tour.append(next_idx)
                    unvisited.remove(next_idx)
                    current_idx = next_idx
                
                tour_time = self._compute_tour_time(tour)
                ant_tours.append(tour)
                ant_times.append(tour_time)
            
            # Find best ant
            iteration_best_idx = np.argmin(ant_times)
            iteration_best_tour = ant_tours[iteration_best_idx]
            iteration_best_time = ant_times[iteration_best_idx]
            
            # Update global best
            if iteration_best_time < best_time:
                best_tour = iteration_best_tour
                best_time = iteration_best_time
            
            # Global pheromone update
            for i in range(len(iteration_best_tour)):
                j = (i + 1) % len(iteration_best_tour)
                idx_i = iteration_best_tour[i]
                idx_j = iteration_best_tour[j]
                tau[idx_i][idx_j] = (1 - self.epsilon) * tau[idx_i][idx_j] + \
                                    self.epsilon * (1.0 / best_time if best_time > 0 else 1.0)
        
        # Convert indices back to Region objects
        optimized_regions = [self.regions[idx] for idx in best_tour]
        return optimized_regions
    
    def _select_next(self, current_idx: int, unvisited: set, tau: np.ndarray, eta: np.ndarray) -> int:
        """Select next region using ACS rule"""
        q = random.random()
        
        if q <= self.q0:
            # Exploitation
            best_value = -np.inf
            best_idx = None
            for j in unvisited:
                value = (tau[current_idx][j] ** self.alpha) * (eta[current_idx][j] ** self.beta)
                if value > best_value:
                    best_value = value
                    best_idx = j
            return best_idx
        else:
            # Exploration: roulette wheel
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


def _ga_acs_solve_path(
    uav,
    regions_for_uav: List,
    v_matrix: List[List[float]],
    uav_idx: int,
    all_regions_list: List,
    use_acs: bool = True
) -> List:
    """
    Sắp xếp lộ trình cho UAV bằng ACS hoặc Nearest Neighbor
    
    Args:
        use_acs: True = dùng ACS, False = dùng NN (như GA gốc)
    """
    if not regions_for_uav:
        return []
    
    if not use_acs:
        # Fallback to Nearest Neighbor (GA original)
        BASE_COORDS = (0, 0)
        path = []
        current_coords = BASE_COORDS
        remaining = list(regions_for_uav)
        
        while remaining:
            best_region = None
            min_dist = float('inf')
            
            for region in remaining:
                dist = get_distance(current_coords, region.coords)
                if dist < min_dist:
                    region_idx = all_regions_list.index(region)
                    if v_matrix[uav_idx][region_idx] > 0:
                        min_dist = dist
                        best_region = region
            
            if best_region is None:
                break
                
            path.append(best_region)
            remaining.remove(best_region)
            current_coords = best_region.coords
        
        return path
    
    # Use ACS optimizer
    acs = ACSOptimizer(
        uav=uav,
        regions=regions_for_uav,
        v_matrix=v_matrix,
        uav_idx=uav_idx,
        all_regions_list=all_regions_list,
        num_ants=10,
        max_iterations=50
    )
    
    optimized_path = acs.optimize()
    return optimized_path


def _ga_acs_calculate_fitness(
    chromosome: List[int],
    uavs_list: List,
    regions_list: List,
    v_matrix: List[List[float]],
    use_acs: bool = True
) -> float:
    """
    Tính toán độ thích nghi (max_completion_time)
    
    Args:
        use_acs: True = dùng ACS optimize, False = dùng NN
    """
    num_uavs = len(uavs_list)
    uav_times = [0.0] * num_uavs
    
    # Phân bổ regions cho từng UAV
    uav_assignments: List[List] = [[] for _ in range(num_uavs)]
    for region_idx, uav_idx in enumerate(chromosome):
        uav_assignments[uav_idx].append(regions_list[region_idx])
    
    # Tính thời gian cho từng UAV
    for uav_idx in range(num_uavs):
        regions_for_uav = uav_assignments[uav_idx]
        if not regions_for_uav:
            continue
        
        uav = uavs_list[uav_idx]
        
        # Optimize path với ACS hoặc NN
        ordered_path = _ga_acs_solve_path(
            uav, 
            regions_for_uav, 
            v_matrix, 
            uav_idx, 
            regions_list,
            use_acs=use_acs
        )
        
        # Tính thời gian
        path_time = calculate_path_time(uav, ordered_path, v_matrix, uav_idx, regions_list)
        uav_times[uav_idx] = path_time
    
    return max(uav_times)


def solve_ga_acs(
    uavs_list: List,
    regions_list: List,
    v_matrix: List[List[float]],
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.05,
    crossover_rate: float = 0.8,
    use_acs: bool = True,
    local_search_interval: int = 10
) -> Tuple[float, List[List]]:
    """
    Giải bài toán bằng Hybrid GA-ACS Algorithm
    
    Args:
        uavs_list: Danh sách UAV
        regions_list: Danh sách Region
        v_matrix: Ma trận vận tốc quét
        population_size: Kích thước quần thể
        generations: Số thế hệ
        mutation_rate: Tỷ lệ đột biến
        crossover_rate: Tỷ lệ lai ghép
        use_acs: True = dùng ACS, False = dùng NN (GA gốc)
        local_search_interval: Cứ mỗi X thế hệ thì chạy local search
    
    Returns:
        (best_fitness, final_paths_ordered): Thời gian tốt nhất và đường đi cho từng UAV
    """
    num_uavs = len(uavs_list)
    num_regions = len(regions_list)
    
    # Initialize population
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, num_uavs - 1) for _ in range(num_regions)]
        population.append(chromosome)
    
    best_chromosome = None
    best_fitness = float('inf')
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [
            _ga_acs_calculate_fitness(c, uavs_list, regions_list, v_matrix, use_acs)
            for c in population
        ]
        
        # Track best
        for i in range(population_size):
            if fitness_scores[i] < best_fitness:
                best_fitness = fitness_scores[i]
                best_chromosome = population[i][:]
        
        # Selection: Tournament selection (k=2)
        new_population = []
        for _ in range(population_size):
            p1_idx, p2_idx = random.sample(range(population_size), 2)
            winner_idx = p1_idx if fitness_scores[p1_idx] < fitness_scores[p2_idx] else p2_idx
            new_population.append(population[winner_idx][:])
        
        population = new_population
        
        # Crossover: Uniform crossover
        for i in range(0, population_size, 2):
            if i + 1 < population_size and random.random() < crossover_rate:
                p1 = population[i]
                p2 = population[i + 1]
                c1, c2 = list(p1), list(p2)
                for j in range(num_regions):
                    if random.random() < 0.5:
                        c1[j], c2[j] = c2[j], c1[j]
                population[i], population[i + 1] = c1, c2
        
        # Mutation: Random reassignment
        for i in range(population_size):
            for j in range(num_regions):
                if random.random() < mutation_rate:
                    population[i][j] = random.randint(0, num_uavs - 1)
        
        # Local search on best chromosome (optional enhancement)
        if local_search_interval > 0 and (gen + 1) % local_search_interval == 0:
            improved_chromosome = _local_search_2opt(
                best_chromosome,
                uavs_list,
                regions_list,
                v_matrix,
                use_acs
            )
            improved_fitness = _ga_acs_calculate_fitness(
                improved_chromosome,
                uavs_list,
                regions_list,
                v_matrix,
                use_acs
            )
            if improved_fitness < best_fitness:
                best_fitness = improved_fitness
                best_chromosome = improved_chromosome
        
    # Reconstruct final paths
    best_paths: List[List] = [[] for _ in range(num_uavs)]
    if best_chromosome:
        for region_idx, uav_idx in enumerate(best_chromosome):
            best_paths[uav_idx].append(regions_list[region_idx])
    
    # Optimize order with ACS/NN
    final_paths_ordered = []
    for uav_idx in range(num_uavs):
        ordered = _ga_acs_solve_path(
            uavs_list[uav_idx],
            best_paths[uav_idx],
            v_matrix,
            uav_idx,
            regions_list,
            use_acs=use_acs
        )
        final_paths_ordered.append(ordered)
    
    return best_fitness, final_paths_ordered


def _local_search_2opt(
    chromosome: List[int],
    uavs_list: List,
    regions_list: List,
    v_matrix: List[List[float]],
    use_acs: bool
) -> List[int]:
    """
    Local search: Thử swap 2 regions giữa các UAV để cải thiện fitness
    """
    best_chromosome = chromosome[:]
    best_fitness = _ga_acs_calculate_fitness(best_chromosome, uavs_list, regions_list, v_matrix, use_acs)
    
    improved = True
    max_attempts = 20
    attempts = 0
    
    while improved and attempts < max_attempts:
        improved = False
        attempts += 1
        
        # Try swapping regions between UAVs
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                # Swap
                new_chromosome = best_chromosome[:]
                new_chromosome[i], new_chromosome[j] = new_chromosome[j], new_chromosome[i]
                
                # Evaluate
                new_fitness = _ga_acs_calculate_fitness(new_chromosome, uavs_list, regions_list, v_matrix, use_acs)
                
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_chromosome = new_chromosome
                    improved = True
                    break
            
            if improved:
                break
    
    return best_chromosome


def calculate_path_time(uav, ordered_path: List, v_matrix: List[List[float]], uav_idx: int, all_regions_list: List) -> float:
    """
    Tính tổng thời gian cho đường đi (từ base -> regions -> end)
    """
    if not ordered_path:
        return 0.0
    
    BASE_COORDS = (0, 0)
    total_time = 0.0
    
    # From base to first region
    first_region = ordered_path[0]
    dist = get_distance(BASE_COORDS, first_region.coords)
    total_time += dist / uav.max_velocity
    
    # Scan first region
    region_idx = all_regions_list.index(first_region)
    scan_velocity = v_matrix[uav_idx][region_idx]
    if scan_velocity > 0:
        total_time += first_region.area / (scan_velocity * uav.scan_width)
    
    # Between regions
    for i in range(len(ordered_path) - 1):
        curr_region = ordered_path[i]
        next_region = ordered_path[i + 1]
        
        # Flight time
        dist = get_distance(curr_region.coords, next_region.coords)
        total_time += dist / uav.max_velocity
        
        # Scan time
        region_idx = all_regions_list.index(next_region)
        scan_velocity = v_matrix[uav_idx][region_idx]
        if scan_velocity > 0:
            total_time += next_region.area / (scan_velocity * uav.scan_width)
    
    return total_time


# Example usage
if __name__ == "__main__":
    # Mock classes for testing
    class UAV:
        def __init__(self, id, max_velocity, scan_width):
            self.id = id
            self.max_velocity = max_velocity
            self.scan_width = scan_width
    
    class Region:
        def __init__(self, id, coords, area):
            self.id = id
            self.coords = coords
            self.area = area
    
    # Sample data
    uavs_list = [
        UAV(1, 20.0, 10.0),
        UAV(2, 25.0, 15.0),
    ]
    
    regions_list = [
        Region(1, (100, 200), 5000),
        Region(2, (300, 400), 6000),
        Region(3, (500, 100), 4500),
        Region(4, (200, 500), 5500),
    ]
    
    v_matrix = [
        [18.0, 19.0, 17.5, 18.5],
        [22.5, 23.0, 21.0, 22.0]
    ]
    
    # Run GA-ACS
    best_fitness, final_paths = solve_ga_acs(
        uavs_list,
        regions_list,
        v_matrix,
        population_size=30,
        generations=50,
        use_acs=True
    )
    
    print(f"\n=== GA-ACS Results ===")
    print(f"Best completion time: {best_fitness:.2f}")
    for uav_idx, path in enumerate(final_paths):
        region_ids = [r.id for r in path]
        print(f"UAV {uav_idx}: {region_ids}")