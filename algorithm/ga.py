from utils.utils import *

def _ga_solve_nn_path(uav: UAV, 
                      regions_for_uav: List[Region], 
                      v_matrix: List[List[float]],
                      uav_idx: int,
                      all_regions_list: List[Region]) -> List[Region]:
    """Hàm nội bộ GA: Sắp xếp lộ trình cho UAV bằng Nearest Neighbor."""
    if not regions_for_uav:
        return []
    
    path = []
    current_coords = BASE_COORDS
    remaining = list(regions_for_uav)
    
    while remaining:
        best_region = None
        min_dist = float('inf')
        
        for region in remaining: # region là đối tượng Region
            dist = get_distance(current_coords, region.coords) # SỬA: Dùng region.coords
            if dist < min_dist:
                region_idx = all_regions_list.index(region)
                if v_matrix[uav_idx][region_idx] > 0:
                    min_dist = dist
                    best_region = region
        
        if best_region is None:
            return path 
            
        path.append(best_region)
        remaining.remove(best_region)
        current_coords = best_region.coords # SỬA: Dùng best_region.coords
        
    return path

def _ga_calculate_fitness(chromosome: List[int], 
                          uavs_list: List[UAV], 
                          regions_list: List[Region], 
                          v_matrix: List[List[float]]) -> float:
    """Hàm nội bộ GA: Tính toán độ thích nghi (max_completion_time)."""
    num_uavs = len(uavs_list)
    uav_times = [0.0] * num_uavs
    
    uav_assignments: List[List[Region]] = [[] for _ in range(num_uavs)]
    for region_idx, uav_idx in enumerate(chromosome):
        uav_assignments[uav_idx].append(regions_list[region_idx])

    for uav_idx in range(num_uavs):
        regions_for_uav = uav_assignments[uav_idx]
        if not regions_for_uav:
            continue
            
        uav = uavs_list[uav_idx] # uav là đối tượng UAV
        
        ordered_path = _ga_solve_nn_path(uav, regions_for_uav, v_matrix, uav_idx, regions_list)
        
        path_time = calculate_path_time(uav, ordered_path, v_matrix, uav_idx, regions_list)
        uav_times[uav_idx] = path_time

    return max(uav_times)

def solve_ga(uavs_list: List[UAV], 
             regions_list: List[Region], 
             v_matrix: List[List[float]],
             population_size: int = 50,
             generations: int = 100,
             mutation_rate: float = 0.05,
             crossover_rate: float = 0.8) -> Tuple[float, List[List[Region]]]:
    """
    Giải bài toán bằng Thuật toán Di truyền (GA).
    """
    num_uavs = len(uavs_list)
    num_regions = len(regions_list)

    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, num_uavs - 1) for _ in range(num_regions)]
        population.append(chromosome)

    best_chromosome = None
    best_fitness = float('inf')

    for gen in range(generations):
        fitness_scores = [_ga_calculate_fitness(c, uavs_list, regions_list, v_matrix) for c in population]
        
        for i in range(population_size):
            if fitness_scores[i] < best_fitness:
                best_fitness = fitness_scores[i]
                best_chromosome = population[i]
        
        new_population = []
        for _ in range(population_size):
            p1_idx, p2_idx = random.sample(range(population_size), 2)
            winner_idx = p1_idx if fitness_scores[p1_idx] < fitness_scores[p2_idx] else p2_idx
            new_population.append(population[winner_idx])
        
        population = new_population
        
        for i in range(0, population_size, 2):
            if i + 1 < population_size and random.random() < crossover_rate:
                p1 = population[i]
                p2 = population[i+1]
                c1, c2 = list(p1), list(p2)
                for j in range(num_regions):
                    if random.random() < 0.5:
                        c1[j], c2[j] = c2[j], c1[j]
                population[i], population[i+1] = c1, c2

        for i in range(population_size):
            for j in range(num_regions):
                if random.random() < mutation_rate:
                    population[i][j] = random.randint(0, num_uavs - 1)
                    
    best_paths: List[List[Region]] = [[] for _ in range(num_uavs)]
    if best_chromosome: # Đảm bảo best_chromosome không phải là None
        for region_idx, uav_idx in enumerate(best_chromosome):
            best_paths[uav_idx].append(regions_list[region_idx])
    
    final_paths_ordered = []
    for uav_idx in range(num_uavs):
        ordered = _ga_solve_nn_path(uavs_list[uav_idx], best_paths[uav_idx], v_matrix, uav_idx, regions_list)
        final_paths_ordered.append(ordered)

    return best_fitness, final_paths_ordered