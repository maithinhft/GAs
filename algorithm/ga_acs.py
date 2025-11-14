import random
import numpy as np
from typing import List, Tuple, Dict
import math

def get_distance(coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
    """Tính khoảng cách Euclidean giữa 2 điểm"""
    return math.sqrt((coords2[0] - coords1[0])**2 + (coords2[1] - coords1[1])**2)

class ACSOptimizer:
    """Ant Colony System optimizer cho TSP cục bộ của mỗi UAV - OPTIMIZED VERSION"""
    
    def __init__(
        self,
        uav,
        regions: List,
        v_matrix: List[List[float]],
        uav_idx: int,
        all_regions_list: List,
        num_ants: int = 5,  # GIẢM: 10 -> 5
        max_iterations: int = 20,  # GIẢM: 50 -> 20
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
        
        # CACHE: Precompute region indices in all_regions_list
        self.region_idx_cache = {
            id(region): all_regions_list.index(region) 
            for region in regions
        }
    
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
        """Tính tổng thời gian cho tour (từ base -> regions -> cuối) - CACHED"""
        if not tour_indices:
            return 0.0
        
        BASE_COORDS = (0, 0)
        total_time = 0.0
        
        # From base to first region
        first_region = self.regions[tour_indices[0]]
        dist = get_distance(BASE_COORDS, first_region.coords)
        total_time += dist / self.uav.max_velocity
        
        # Scan first region - USE CACHE
        region_idx_in_all = self.region_idx_cache[id(first_region)]
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
            
            # Scan time - USE CACHE
            region_idx_in_all = self.region_idx_cache[id(next_region)]
            scan_velocity = self.v_matrix[self.uav_idx][region_idx_in_all]
            if scan_velocity > 0:
                total_time += next_region.area / (scan_velocity * self.uav.scan_width)
        
        return total_time
    
    def optimize(self) -> List:
        """Tối ưu tour bằng ACS, trả về danh sách Region objects"""
        if self.num_regions <= 1:
            return self.regions
        
        # OPTIMIZATION: Nếu quá ít regions, dùng NN thay vì ACS
        if self.num_regions <= 3:
            return self._nn_fallback()
        
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
        
        # OPTIMIZATION: Early stopping if no improvement
        no_improvement_count = 0
        max_no_improvement = 5
        
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
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # EARLY STOPPING
            if no_improvement_count >= max_no_improvement:
                break
            
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
    
    def _nn_fallback(self) -> List:
        """Fallback to NN for small problem sizes"""
        tour_indices, _ = self._nearest_neighbor_tour()
        return [self.regions[idx] for idx in tour_indices]
    
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
    """
    if not regions_for_uav:
        return []
    
    # OPTIMIZATION: Chỉ dùng ACS nếu có đủ regions
    if len(regions_for_uav) <= 3:
        use_acs = False
    
    if not use_acs:
        # Fallback to Nearest Neighbor
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
        num_ants=5,  # GIẢM
        max_iterations=20  # GIẢM
    )
    
    optimized_path = acs.optimize()
    return optimized_path


# CACHE để tránh tính lại fitness cho chromosome giống nhau
_fitness_cache: Dict[tuple, float] = {}

def _ga_acs_calculate_fitness(
    chromosome: List[int],
    uavs_list: List,
    regions_list: List,
    v_matrix: List[List[float]],
    use_acs: bool = True
) -> float:
    """
    Tính toán độ thích nghi với CACHING
    """
    # OPTIMIZATION: Cache fitness for identical chromosomes
    chromosome_key = tuple(chromosome)
    if chromosome_key in _fitness_cache:
        return _fitness_cache[chromosome_key]
    
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
    
    fitness = max(uav_times)
    
    # CACHE result
    _fitness_cache[chromosome_key] = fitness
    
    # OPTIMIZATION: Giới hạn cache size
    if len(_fitness_cache) > 200:
        _fitness_cache.clear()
    
    return fitness


def solve_ga_acs(
    uavs_list: List,
    regions_list: List,
    v_matrix: List[List[float]],
    population_size: int = 30,  # GIẢM: 50 -> 30
    generations: int = 50,  # GIẢM: 100 -> 50
    mutation_rate: float = 0.1,  # TĂNG để explore nhanh
    crossover_rate: float = 0.8,
    use_acs: bool = True,
    local_search_interval: int = 0,  # TẮT local search mặc định
    elite_size: int = 5  # THÊM: Elitism
) -> Tuple[float, List[List]]:
    """
    Giải bài toán bằng OPTIMIZED Hybrid GA-ACS Algorithm
    
    MAJOR OPTIMIZATIONS:
    - Giảm population_size: 50 -> 30
    - Giảm generations: 100 -> 50  
    - Giảm ACS iterations: 50 -> 20
    - Giảm num_ants: 10 -> 5
    - Thêm fitness caching
    - Thêm early stopping
    - Thêm elitism
    - Tắt local search mặc định
    """
    num_uavs = len(uavs_list)
    num_regions = len(regions_list)
    
    # Clear cache
    global _fitness_cache
    _fitness_cache.clear()
    
    # Initialize population
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, num_uavs - 1) for _ in range(num_regions)]
        population.append(chromosome)
    
    best_chromosome = None
    best_fitness = float('inf')
    
    # Track best solutions (ELITISM)
    elite_chromosomes = []
    
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
        
        # ELITISM: Keep top solutions
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        elite_chromosomes = [population[i][:] for i in sorted_indices[:elite_size]]
        
        # Selection: Tournament selection (k=2)
        new_population = elite_chromosomes[:]  # Start with elites
        
        while len(new_population) < population_size:
            p1_idx, p2_idx = random.sample(range(population_size), 2)
            winner_idx = p1_idx if fitness_scores[p1_idx] < fitness_scores[p2_idx] else p2_idx
            new_population.append(population[winner_idx][:])
        
        population = new_population[:population_size]
        
        # Crossover: Uniform crossover
        for i in range(elite_size, population_size, 2):  # Skip elites
            if i + 1 < population_size and random.random() < crossover_rate:
                p1 = population[i]
                p2 = population[i + 1]
                c1, c2 = list(p1), list(p2)
                for j in range(num_regions):
                    if random.random() < 0.5:
                        c1[j], c2[j] = c2[j], c1[j]
                population[i], population[i + 1] = c1, c2
        
        # Mutation: Random reassignment
        for i in range(elite_size, population_size):  # Skip elites
            for j in range(num_regions):
                if random.random() < mutation_rate:
                    population[i][j] = random.randint(0, num_uavs - 1)
        
        # OPTIONAL: Local search (disabled by default)
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
    Local search: Swap 2 regions - GIỚI HẠN attempts
    """
    best_chromosome = chromosome[:]
    best_fitness = _ga_acs_calculate_fitness(best_chromosome, uavs_list, regions_list, v_matrix, use_acs)
    
    improved = True
    max_attempts = 5  # GIẢM: 20 -> 5
    attempts = 0
    
    while improved and attempts < max_attempts:
        improved = False
        attempts += 1
        
        # OPTIMIZATION: Chỉ thử một số swap ngẫu nhiên
        num_swaps = min(20, len(chromosome) * (len(chromosome) - 1) // 2)
        
        for _ in range(num_swaps):
            i = random.randint(0, len(chromosome) - 1)
            j = random.randint(0, len(chromosome) - 1)
            if i == j:
                continue
            
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
    
    return best_chromosome


def calculate_path_time(uav, ordered_path: List, v_matrix: List[List[float]], uav_idx: int, all_regions_list: List) -> float:
    """
    Tính tổng thời gian cho đường đi
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