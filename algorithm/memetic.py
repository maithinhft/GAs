"""
Memetic Algorithm (MA) - GA + Local Search Hybrid
Kết hợp: Population-based search (GA) + Local Search (2-opt)

Lý do hiệu quả:
- GA tìm kiếm global thông qua crossover/mutation
- Local search cải thiện mỗi giải pháp một cách nhanh
- Kết hợp => tốt chất lượng + tốc độ chấp nhận được
"""

import random
from typing import List, Tuple
from utils.utils import get_distance, get_fly_time, get_scan_time, BASE_COORDS
from utils.config import UAV, Region


def solve_memetic(uavs_list: List[UAV],
                  regions_list: List[Region],
                  v_matrix: List[List[float]],
                  population_size: int = 15,
                  generations: int = 30) -> Tuple[float, List[List[Region]]]:
    """
    Memetic Algorithm: Genetic Algorithm + Local Search
    
    Quá trình:
    1. Khởi tạo population ngẫu nhiên
    2. Mỗi generation:
       - Selection (tournament)
       - Crossover + Mutation
       - Local Search (2-opt) trên mỗi cá thể MỚI
       - Evaluation
       - Elite replacement
    
    Args:
        uavs_list: List of UAVs
        regions_list: List of regions
        v_matrix: Scan velocity matrix
        population_size: Population size (default 40)
        generations: Number of generations (default 80)
        
    Returns:
        (best_fitness, best_paths)
    """
    
    num_regions = len(regions_list)
    num_uavs = len(uavs_list)
    
    # ============= Bước 1: Khởi tạo Population =============
    population = []
    
    for _ in range(population_size):
        individual = [random.randint(0, num_uavs - 1) for _ in range(num_regions)]
        
        # Ensure feasibility
        for i in range(num_regions):
            if v_matrix[individual[i]][i] == 0:
                for uav_idx in range(num_uavs):
                    if v_matrix[uav_idx][i] > 0:
                        individual[i] = uav_idx
                        break
        
        fitness = _calc_fitness(individual, uavs_list, regions_list, v_matrix)
        population.append((individual, fitness))
    
    # Sort
    population.sort(key=lambda x: x[1])
    best_individual, best_fitness = population[0]
    
    # ============= Bước 2: Evolution Loop =============
    for generation in range(generations):
        new_population = []
        
        # Giữ 2 elite
        new_population.append(population[0])
        if len(population) > 1:
            new_population.append(population[1])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Selection: Tournament selection
            parent1 = _tournament_select(population, tournament_size=3)
            parent2 = _tournament_select(population, tournament_size=3)
            
            # Crossover
            if random.random() < 0.8:
                child = _crossover(parent1[0], parent2[0], num_uavs)
            else:
                child = parent1[0].copy()
            
            # Mutation
            child = _mutate(child, num_regions, num_uavs, mutation_rate=0.05)
            
            # Ensure feasibility
            for i in range(num_regions):
                if v_matrix[child[i]][i] == 0:
                    for uav_idx in range(num_uavs):
                        if v_matrix[uav_idx][i] > 0:
                            child[i] = uav_idx
                            break
            
            # **KEY STEP**: Không apply local search trên offspring
            # GA operators (crossover + mutation) đã đủ tốt
            # Dùng final LS thay vào đó
            
            fitness = _calc_fitness(child, uavs_list, regions_list, v_matrix)
            
            new_population.append((child, fitness))
        
        # Keep best population_size individuals
        new_population.sort(key=lambda x: x[1])
        population = new_population[:population_size]
        
        # Update best
        if population[0][1] < best_fitness:
            best_individual, best_fitness = population[0]
    
    # ============= Bước 3: Final Local Search (exhaustive) =============
    best_individual = _local_search_2opt(best_individual, uavs_list, regions_list, v_matrix, 
                                         max_iterations=2, sample_rate=1.0)
    best_fitness = _calc_fitness(best_individual, uavs_list, regions_list, v_matrix)
    
    best_paths = _to_paths(best_individual, uavs_list, regions_list, v_matrix)
    return best_fitness, best_paths


def _tournament_select(population: List[Tuple[List[int], float]], tournament_size: int = 3) -> Tuple[List[int], float]:
    """Tournament selection"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda x: x[1])


def _crossover(parent1: List[int], parent2: List[int], num_uavs: int) -> List[int]:
    """
    Crossover: Order-based crossover (OBX)
    Chọn segment từ parent1, fill lại từ parent2
    """
    n = len(parent1)
    start = random.randint(0, n - 1)
    end = random.randint(start, n - 1)
    
    child = parent1.copy()
    
    # Copy segment từ parent2
    for i in range(start, end + 1):
        child[i] = parent2[i]
    
    return child


def _mutate(individual: List[int], num_regions: int, num_uavs: int, mutation_rate: float = 0.05) -> List[int]:
    """Mutation: random assignment changes"""
    mutated = individual.copy()
    
    for i in range(num_regions):
        if random.random() < mutation_rate:
            mutated[i] = random.randint(0, num_uavs - 1)
    
    return mutated


def _local_search_2opt(individual: List[int],
                       uavs_list: List[UAV],
                       regions_list: List[Region],
                       v_matrix: List[List[float]],
                       max_iterations: int = 2,
                       sample_rate: float = 0.3) -> List[int]:
    """
    2-opt Local Search: cải thiện bằng swaps
    
    Với sample_rate < 1.0: chỉ check random subset của pairs (speedup)
    max_iterations: số lần lặp 2-opt
    """
    
    best_sol = individual.copy()
    best_fit = _calc_fitness(best_sol, uavs_list, regions_list, v_matrix)
    
    improved = True
    iteration = 0
    n = len(best_sol)
    
    # Số pairs để check (nếu sample_rate < 1.0)
    if sample_rate < 1.0:
        num_pairs = max(1, int(n * (n - 1) / 2 * sample_rate))
    else:
        num_pairs = None  # exhaustive mode
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Tạo list tất cả pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Sample hoặc lấy toàn bộ
        if num_pairs and len(all_pairs) > num_pairs:
            pairs_to_check = random.sample(all_pairs, num_pairs)
        else:
            pairs_to_check = all_pairs
        
        # Check các pairs
        for i, j in pairs_to_check:
            # Swap
            best_sol[i], best_sol[j] = best_sol[j], best_sol[i]
            new_fit = _calc_fitness(best_sol, uavs_list, regions_list, v_matrix)
            
            if new_fit < best_fit:
                best_fit = new_fit
                improved = True
            else:
                # Revert
                best_sol[i], best_sol[j] = best_sol[j], best_sol[i]
    
    return best_sol


def _calc_fitness(chromosome: List[int],
                  uavs_list: List[UAV],
                  regions_list: List[Region],
                  v_matrix: List[List[float]]) -> float:
    """Calculate max completion time"""
    
    num_uavs = len(uavs_list)
    uav_times = [0.0] * num_uavs
    
    uav_assignments: List[List[Region]] = [[] for _ in range(num_uavs)]
    for region_idx, uav_idx in enumerate(chromosome):
        if 0 <= uav_idx < num_uavs:
            uav_assignments[uav_idx].append(regions_list[region_idx])
    
    for uav_idx in range(num_uavs):
        regions = uav_assignments[uav_idx]
        path = _nn_path(uavs_list[uav_idx], regions, v_matrix, uav_idx, regions_list)
        time_val = _path_time(uavs_list[uav_idx], path, v_matrix, uav_idx, regions_list)
        uav_times[uav_idx] = time_val
    
    return max(uav_times) if uav_times else float('inf')


def _nn_path(uav: UAV, regions: List[Region], v_matrix: List[List[float]],
             uav_idx: int, all_regions: List[Region]) -> List[Region]:
    """Nearest neighbor path ordering"""
    
    if not regions:
        return []
    
    path = []
    current = BASE_COORDS
    remaining = list(regions)
    
    while remaining:
        best_region = None
        best_dist = float('inf')
        
        for region in remaining:
            dist = get_distance(current, region.coords)
            if dist < best_dist:
                r_idx = all_regions.index(region)
                if v_matrix[uav_idx][r_idx] > 0:
                    best_dist = dist
                    best_region = region
        
        if not best_region:
            break
        
        path.append(best_region)
        remaining.remove(best_region)
        current = best_region.coords
    
    return path


def _path_time(uav: UAV, path: List[Region], v_matrix: List[List[float]],
               uav_idx: int, all_regions: List[Region]) -> float:
    """Calculate path completion time"""
    
    total = 0.0
    current = BASE_COORDS
    
    for region in path:
        r_idx = all_regions.index(region)
        v_entry = v_matrix[uav_idx][r_idx]
        
        fly_time = get_fly_time(uav, current, region.coords)
        scan_time = get_scan_time(uav, region, v_entry)
        
        if scan_time == float('inf'):
            return float('inf')
        
        total += fly_time + scan_time
        current = region.coords
    
    if path:
        total += get_fly_time(uav, current, BASE_COORDS)
    
    return total


def _to_paths(chromosome: List[int],
              uavs_list: List[UAV],
              regions_list: List[Region],
              v_matrix: List[List[float]]) -> List[List[Region]]:
    """Convert to paths"""
    
    num_uavs = len(uavs_list)
    uav_assignments: List[List[Region]] = [[] for _ in range(num_uavs)]
    
    for region_idx, uav_idx in enumerate(chromosome):
        if 0 <= uav_idx < num_uavs:
            uav_assignments[uav_idx].append(regions_list[region_idx])
    
    paths = []
    for uav_idx in range(num_uavs):
        path = _nn_path(uavs_list[uav_idx], uav_assignments[uav_idx], v_matrix, uav_idx, regions_list)
        paths.append(path)
    
    return paths
