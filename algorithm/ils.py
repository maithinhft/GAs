"""
Iterated Local Search (ILS) Algorithm for Multi-UAV Area Coverage Optimization

ILS là một metaheuristic mạnh mẽ cho discrete optimization problems.
Nguyên lý:
1. Khởi tạo giải pháp tốt (bằng greedy hoặc random)
2. Local Search: Cải thiện giải pháp với 2-opt
3. Perturbation: Thay đổi giải pháp để thoát local optima
4. Lặp lại cho đến khi không có cải thiện

ILS thường tốt hơn GA nhờ:
- Local search mạnh: 2-opt, 3-opt
- Perturbation thông minh: không phá hủy hoàn toàn giải pháp
- Acceptance criteria: cho phép chấp nhận giải pháp tệ hơn (để thoát local optima)
"""

import random
import math
from typing import List, Tuple
from utils.utils import get_distance, get_fly_time, get_scan_time, BASE_COORDS
from utils.config import UAV, Region


def solve_ils(uavs_list: List[UAV],
              regions_list: List[Region],
              v_matrix: List[List[float]],
              max_iterations: int = 80,
              max_no_improve: int = 20,
              perturbation_strength: int = 3) -> Tuple[float, List[List[Region]]]:
    """
    Iterated Local Search (ILS) Algorithm - BALANCED VERSION
    
    Strategy:
    1. Single strong initialization with good local search
    2. Balanced local search (not too fast, not too slow)
    3. Intelligent perturbation to escape local optima
    
    Args:
        uavs_list: Danh sách các UAV
        regions_list: Danh sách các region
        v_matrix: Ma trận scan velocity V[uav_idx][region_idx]
        max_iterations: Số lần iteration trong ILS main loop
        max_no_improve: Số lần không cải thiện cho phép
        perturbation_strength: Cường độ perturbation
        
    Returns:
        (best_fitness, best_paths)
    """
    
    num_regions = len(regions_list)
    
    # Khởi tạo tốt: Thử 2-3 random starts, lấy tốt nhất
    best_solution = None
    best_fitness = float('inf')
    
    for _ in range(2):  # Quick 2 random starts
        init_solution = _random_initialization(uavs_list, regions_list, v_matrix)
        # Quick local search on initialization
        init_solution = _local_search_smart(init_solution, uavs_list, regions_list, v_matrix, iterations=3)
        init_fitness = _calculate_fitness(init_solution, uavs_list, regions_list, v_matrix)
        
        if init_fitness < best_fitness:
            best_solution = init_solution.copy()
            best_fitness = init_fitness
    
    current_solution = best_solution.copy()
    current_fitness = best_fitness
    no_improve_count = 0
    
    # ===== ILS Main Loop =====
    for iteration in range(max_iterations):
        # Local Search - Balanced
        improved_solution = _local_search_smart(
            current_solution,
            uavs_list,
            regions_list,
            v_matrix,
            iterations=8  # Balanced
        )
        improved_fitness = _calculate_fitness(improved_solution, uavs_list, regions_list, v_matrix)
        
        if improved_fitness < current_fitness:
            current_solution = improved_solution
            current_fitness = improved_fitness
            no_improve_count = 0
            
            # Update global best
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        else:
            no_improve_count += 1
        
        # Perturbation & Acceptance
        if no_improve_count > 0:
            # Simulated Annealing-style acceptance
            temperature = max(0.01, 1.0 - (iteration / max_iterations))
            acceptance_prob = math.exp(-no_improve_count / 3.0) * temperature
            
            perturbed = _perturbation(current_solution, num_regions, strength=perturbation_strength)
            perturbed_fitness = _calculate_fitness(perturbed, uavs_list, regions_list, v_matrix)
            
            if perturbed_fitness < current_fitness or random.random() < acceptance_prob:
                current_solution = perturbed
                current_fitness = perturbed_fitness
                no_improve_count = max(0, no_improve_count - 1)
        
        # Early stopping
        if no_improve_count >= max_no_improve:
            break
    
    best_paths = _convert_chromosome_to_paths(best_solution, uavs_list, regions_list, v_matrix)
    return best_fitness, best_paths


def _random_initialization(uavs_list: List[UAV],
                          regions_list: List[Region],
                          v_matrix: List[List[float]]) -> List[int]:
    """Random initialization với feasibility check"""
    num_regions = len(regions_list)
    num_uavs = len(uavs_list)
    chromosome = [random.randint(0, num_uavs - 1) for _ in range(num_regions)]
    
    # Ensure feasibility
    for i in range(num_regions):
        if v_matrix[chromosome[i]][i] == 0:
            for uav_idx in range(num_uavs):
                if v_matrix[uav_idx][i] > 0:
                    chromosome[i] = uav_idx
                    break
    
    return chromosome


def _local_search_smart(chromosome: List[int],
                       uavs_list: List[UAV],
                       regions_list: List[Region],
                       v_matrix: List[List[float]],
                       iterations: int = 5) -> List[int]:
    """
    Smart Local Search: Không check tất cả cặp, chỉ check promising ones
    
    Idea: Những regions được assign cho cùng UAV thường không cần swap.
    Focus vào regions assigned to different UAVs.
    """
    
    best_solution = chromosome.copy()
    best_fitness = _calculate_fitness(best_solution, uavs_list, regions_list, v_matrix)
    
    for _ in range(iterations):
        improved = False
        
        # Tìm các region được gán cho các UAV khác nhau
        num_regions = len(chromosome)
        indices_by_uav = {}
        for idx, uav_idx in enumerate(best_solution):
            if uav_idx not in indices_by_uav:
                indices_by_uav[uav_idx] = []
            indices_by_uav[uav_idx].append(idx)
        
        # Chỉ check swap giữa regions của các UAV khác nhau
        uav_list = list(indices_by_uav.keys())
        for i in range(len(uav_list)):
            for j in range(i + 1, len(uav_list)):
                # Sample random regions từ 2 UAV này
                if len(indices_by_uav[uav_list[i]]) > 0 and len(indices_by_uav[uav_list[j]]) > 0:
                    idx1 = random.choice(indices_by_uav[uav_list[i]])
                    idx2 = random.choice(indices_by_uav[uav_list[j]])
                    
                    # Thử swap
                    best_solution[idx1], best_solution[idx2] = best_solution[idx2], best_solution[idx1]
                    new_fitness = _calculate_fitness(best_solution, uavs_list, regions_list, v_matrix)
                    
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        improved = True
                    else:
                        best_solution[idx1], best_solution[idx2] = best_solution[idx2], best_solution[idx1]
        
        if not improved:
            break
    
    return best_solution


def _greedy_construction(uavs_list: List[UAV],
                        regions_list: List[Region],
                        v_matrix: List[List[float]]) -> List[int]:
    """
    Hybrid Initialization: Kết hợp greedy + random để tránh local optima
    
    Thay vì pure greedy (có thể bị stuck), ta làm:
    1. Pure greedy assignment
    2. Một số random swaps để add diversity
    """
    num_regions = len(regions_list)
    num_uavs = len(uavs_list)
    
    # Start with random assignment (simpler, often better than pure greedy)
    chromosome = [random.randint(0, num_uavs - 1) for _ in range(num_regions)]
    
    # Ensure each UAV can actually scan their assigned regions
    for region_idx in range(num_regions):
        uav_idx = chromosome[region_idx]
        
        # If this UAV can't scan this region, find one that can
        if v_matrix[uav_idx][region_idx] == 0:
            for alt_uav in range(num_uavs):
                if v_matrix[alt_uav][region_idx] > 0:
                    chromosome[region_idx] = alt_uav
                    break
    
    return chromosome


def _local_search_2opt(chromosome: List[int],
                       uavs_list: List[UAV],
                       regions_list: List[Region],
                       v_matrix: List[List[float]],
                       max_iterations: int = 50,
                       sample_rate: float = 1.0) -> List[int]:
    """
    2-opt Local Search - DEPRECATED, use _local_search_smart instead
    Kept for backward compatibility
    """
    return _local_search_smart(chromosome, uavs_list, regions_list, v_matrix, iterations=max_iterations)


def _perturbation(chromosome: List[int],
                  num_regions: int,
                  strength: int = 4) -> List[int]:
    """
    Perturbation: Thay đổi ngẫu nhiên `strength` regions của chromosome
    để thoát khỏi local optima.
    
    Không phá hủy hoàn toàn giải pháp, chỉ thay đổi một phần nhỏ.
    """
    
    perturbed = chromosome.copy()
    num_uavs = max(chromosome) + 1
    
    # Chọn ngẫu nhiên `strength` regions để thay đổi assignment
    indices_to_change = random.sample(range(num_regions), min(strength, num_regions))
    
    for idx in indices_to_change:
        # Gán cho UAV khác ngẫu nhiên
        new_uav = random.randint(0, num_uavs - 1)
        perturbed[idx] = new_uav
    
    return perturbed


def _calculate_fitness(chromosome: List[int],
                       uavs_list: List[UAV],
                       regions_list: List[Region],
                       v_matrix: List[List[float]]) -> float:
    """
    Tính toán độ thích nghi: thời gian hoàn thành tối đa (max completion time).
    
    Formula: fitness = max(F(U_i)) where F(U_i) là thời gian hoàn thành UAV i
    """
    
    num_uavs = len(uavs_list)
    uav_times = [0.0] * num_uavs
    
    # Gán regions cho từng UAV
    uav_assignments: List[List[Region]] = [[] for _ in range(num_uavs)]
    for region_idx, uav_idx in enumerate(chromosome):
        if 0 <= uav_idx < num_uavs:
            uav_assignments[uav_idx].append(regions_list[region_idx])
    
    # Tính thời gian hoàn thành cho mỗi UAV
    for uav_idx in range(num_uavs):
        regions_for_uav = uav_assignments[uav_idx]
        
        # Sắp xếp lộ trình bằng Nearest Neighbor
        path = _solve_nn_path(uavs_list[uav_idx], regions_for_uav, v_matrix, uav_idx, regions_list)
        
        # Tính thời gian cho lộ trình này
        completion_time = _calculate_path_time(uavs_list[uav_idx], path, v_matrix, uav_idx, regions_list)
        uav_times[uav_idx] = completion_time
    
    # Fitness là max completion time
    return max(uav_times) if uav_times else float('inf')


def _solve_nn_path(uav: UAV,
                   regions_for_uav: List[Region],
                   v_matrix: List[List[float]],
                   uav_idx: int,
                   all_regions_list: List[Region]) -> List[Region]:
    """
    Nearest Neighbor: Sắp xếp lộ trình bằng tìm region gần nhất chưa được thăm.
    """
    
    if not regions_for_uav:
        return []
    
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


def _calculate_path_time(uav: UAV,
                         region_path: List[Region],
                         v_matrix: List[List[float]],
                         uav_idx: int,
                         regions_list: List[Region]) -> float:
    """
    Tính thời gian hoàn thành của một UAV với lộ trình đã cho.
    """
    
    total_time = 0.0
    current_coords = BASE_COORDS
    
    for region in region_path:
        region_idx = regions_list.index(region)
        v_entry = v_matrix[uav_idx][region_idx]
        
        # Thời gian bay
        fly_time = get_fly_time(uav, current_coords, region.coords)
        
        # Thời gian quét
        scan_time = get_scan_time(uav, region, v_entry)
        
        if scan_time == float('inf'):
            return float('inf')
        
        total_time += fly_time + scan_time
        current_coords = region.coords
    
    # Bay về base
    if region_path:
        fly_back_time = get_fly_time(uav, current_coords, BASE_COORDS)
        total_time += fly_back_time
    
    return total_time


def _convert_chromosome_to_paths(chromosome: List[int],
                                 uavs_list: List[UAV],
                                 regions_list: List[Region],
                                 v_matrix: List[List[float]]) -> List[List[Region]]:
    """
    Convert chromosome (assignment) thành paths (lộ trình cho mỗi UAV).
    """
    
    num_uavs = len(uavs_list)
    uav_assignments: List[List[Region]] = [[] for _ in range(num_uavs)]
    
    for region_idx, uav_idx in enumerate(chromosome):
        if 0 <= uav_idx < num_uavs:
            uav_assignments[uav_idx].append(regions_list[region_idx])
    
    # Sắp xếp lộ trình cho mỗi UAV
    best_paths = []
    for uav_idx in range(num_uavs):
        regions_for_uav = uav_assignments[uav_idx]
        path = _solve_nn_path(uavs_list[uav_idx], regions_for_uav, v_matrix, uav_idx, regions_list)
        best_paths.append(path)
    
    return best_paths
