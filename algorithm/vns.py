"""
Variable Neighborhood Search (VNS) for Multi-UAV Area Coverage

VNS là thuật toán đơn giản nhưng mạnh mẽ:
- Sử dụng nhiều neighborhood operators (2-opt, 3-opt, random restart)
- Khi local search bị stuck, chuyển sang neighborhood khác
- Dễ implement, dễ tune, thường tốt hơn các thuật toán phức tạp
"""

import random
from typing import List, Tuple
from utils.utils import get_distance, get_fly_time, get_scan_time, BASE_COORDS
from utils.config import UAV, Region


def solve_vns(uavs_list: List[UAV],
              regions_list: List[Region],
              v_matrix: List[List[float]],
              max_iterations: int = 100) -> Tuple[float, List[List[Region]]]:
    """
    Variable Neighborhood Search (VNS) Algorithm
    
    Đơn giản: Greedy init + Local Search + Change Neighborhood
    
    Args:
        uavs_list: Danh sách UAV
        regions_list: Danh sách region  
        v_matrix: Ma trận V[uav_idx][region_idx]
        max_iterations: Max iterations
        
    Returns:
        (best_fitness, best_paths)
    """
    
    num_regions = len(regions_list)
    num_uavs = len(uavs_list)
    
    # ============= Bước 1: Khởi tạo bằng Greedy =============
    best_solution = _greedy_init(uavs_list, regions_list, v_matrix)
    best_fitness = _calc_fitness(best_solution, uavs_list, regions_list, v_matrix)
    
    no_improve = 0
    
    # ============= Bước 2: VNS Main Loop =============
    for iteration in range(max_iterations):
        k = 1  # Neighborhood index
        
        while k <= 3:  # 3 neighborhoods
            # Chọn neighborhood k
            if k == 1:
                # Neighborhood 1: 2-opt
                candidate = _2opt_random_one(best_solution, uavs_list, regions_list, v_matrix)
            elif k == 2:
                # Neighborhood 2: 3-opt (rotate 3 regions)
                candidate = _3opt_random_one(best_solution, num_regions, num_uavs)
            else:
                # Neighborhood 3: Swap assignments của 2 UAVs
                candidate = _swap_uav_assignments(best_solution, num_uavs)
            
            cand_fitness = _calc_fitness(candidate, uavs_list, regions_list, v_matrix)
            
            # Nếu tốt hơn: update + reset neighborhood
            if cand_fitness < best_fitness:
                best_solution = candidate
                best_fitness = cand_fitness
                k = 1  # Reset neighborhood
                no_improve = 0
            else:
                k += 1  # Thử neighborhood tiếp theo
        
        no_improve += 1
        
        # Early stopping
        if no_improve >= 20:
            break
    
    # ============= Bước 3: Final intensification =============
    # 1 vòng 2-opt mạnh
    for _ in range(min(5, num_regions)):
        candidate = _2opt_random_one(best_solution, uavs_list, regions_list, v_matrix)
        cand_fitness = _calc_fitness(candidate, uavs_list, regions_list, v_matrix)
        
        if cand_fitness < best_fitness:
            best_solution = candidate
            best_fitness = cand_fitness
    
    best_paths = _to_paths(best_solution, uavs_list, regions_list, v_matrix)
    return best_fitness, best_paths


def _greedy_init(uavs_list: List[UAV],
                 regions_list: List[Region],
                 v_matrix: List[List[float]]) -> List[int]:
    """Greedy initialization: assign each region to best UAV"""
    
    num_regions = len(regions_list)
    num_uavs = len(uavs_list)
    chromosome = [-1] * num_regions
    
    for region_idx in range(num_regions):
        best_uav = -1
        best_score = float('inf')
        
        for uav_idx in range(num_uavs):
            if v_matrix[uav_idx][region_idx] > 0:
                scan_time = get_scan_time(uavs_list[uav_idx], regions_list[region_idx],
                                         v_matrix[uav_idx][region_idx])
                if scan_time < best_score:
                    best_score = scan_time
                    best_uav = uav_idx
        
        if best_uav >= 0:
            chromosome[region_idx] = best_uav
        else:
            for uav_idx in range(num_uavs):
                if v_matrix[uav_idx][region_idx] > 0:
                    chromosome[region_idx] = uav_idx
                    break
    
    for i in range(num_regions):
        if chromosome[i] == -1:
            chromosome[i] = i % num_uavs
    
    return chromosome


def _2opt_random_one(chromosome: List[int],
                     uavs_list: List[UAV],
                     regions_list: List[Region],
                     v_matrix: List[List[float]]) -> List[int]:
    """2-opt: thử 1 random swap, keep nếu tốt hơn"""
    
    solution = chromosome.copy()
    n = len(solution)
    
    i = random.randint(0, n - 1)
    j = random.randint(0, n - 1)
    
    if i != j:
        solution[i], solution[j] = solution[j], solution[i]
        new_fit = _calc_fitness(solution, uavs_list, regions_list, v_matrix)
        old_fit = _calc_fitness(chromosome, uavs_list, regions_list, v_matrix)
        
        if new_fit >= old_fit:  # Không tốt hơn
            solution[i], solution[j] = solution[j], solution[i]
    
    return solution


def _3opt_random_one(chromosome: List[int], num_regions: int, num_uavs: int) -> List[int]:
    """3-opt: rotate 3 random regions"""
    
    solution = chromosome.copy()
    
    i = random.randint(0, num_regions - 1)
    j = random.randint(0, num_regions - 1)
    k = random.randint(0, num_regions - 1)
    
    if len(set([i, j, k])) == 3:
        temp = solution[i]
        solution[i] = solution[j]
        solution[j] = solution[k]
        solution[k] = temp
    
    return solution


def _swap_uav_assignments(chromosome: List[int], num_uavs: int) -> List[int]:
    """Swap: chọn 2 UAVs, hoán đổi một số assignments"""
    
    solution = chromosome.copy()
    uav1 = random.randint(0, num_uavs - 1)
    uav2 = random.randint(0, num_uavs - 1)
    
    if uav1 != uav2:
        # Hoán đổi 1 region từ mỗi UAV
        indices_uav1 = [i for i, u in enumerate(solution) if u == uav1]
        indices_uav2 = [i for i, u in enumerate(solution) if u == uav2]
        
        if indices_uav1 and indices_uav2:
            i = random.choice(indices_uav1)
            j = random.choice(indices_uav2)
            solution[i], solution[j] = solution[j], solution[i]
    
    return solution


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
    """Nearest neighbor path"""
    
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
