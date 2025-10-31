# -*- coding: utf-8 -*-
import math
import copy
import numpy as np
from typing import List, Dict, Tuple, Optional
from .appa import Region, UAV, calculate_tf, calculate_ts, calculate_distance, region_allocation, OrderOptimizerACS
from utils.create_sample import create_sample
# ==============================
# Gradient-Based Refinement
# ==============================

def calculate_path_completion_time(path: List[Region], uav: UAV, V_matrix: List[List[float]], 
                                   base_coords: Tuple[float, float] = (0.0, 0.0)) -> float:
    """
    Tính tổng thời gian hoàn thành một đường đi của UAV
    
    Args:
        path: Danh sách các vùng theo thứ tự viếng thăm
        uav: UAV thực hiện đường đi
        V_matrix: Ma trận vận tốc quét
        base_coords: Tọa độ base
    
    Returns:
        Tổng thời gian hoàn thành (phút hoặc giây)
    """
    if not path:
        return 0.0
    
    total_time = 0.0
    current_pos = base_coords
    
    for region in path:
        # Thời gian bay từ vị trí hiện tại đến vùng
        tf = calculate_tf(uav, current_pos, region.coords)
        total_time += tf
        
        # Thời gian quét vùng
        ts = calculate_ts(uav, region, V_matrix)
        if math.isinf(ts):
            return float('inf')  # UAV không thể quét vùng này
        total_time += ts
        
        # Cập nhật vị trí hiện tại
        current_pos = region.coords
    
    # Thêm thời gian quay trở lại base từ vùng cuối cùng
    if path:
        tf_back_to_base = calculate_tf(uav, current_pos, base_coords)
        total_time += tf_back_to_base
    
    return total_time


def calculate_system_completion_time(assignment: Dict[int, List[Region]], 
                                     uavs: List[UAV], 
                                     V_matrix: List[List[float]],
                                     base_coords: Tuple[float, float] = (0.0, 0.0)) -> float:
    """
    Tính thời gian hoàn thành của cả hệ thống (max của các UAV)
    """
    max_time = 0.0
    for uav in uavs:
        path = assignment.get(uav.id, [])
        completion_time = calculate_path_completion_time(path, uav, V_matrix, base_coords)
        max_time = max(max_time, completion_time)
    return max_time


def find_best_insertion_position(region: Region, target_path: List[Region], 
                                  target_uav: UAV, V_matrix: List[List[float]],
                                  base_coords: Tuple[float, float] = (0.0, 0.0)) -> Tuple[int, float]:
    """
    Tìm vị trí tốt nhất để chèn region vào target_path
    
    Returns:
        (best_position, min_completion_time)
    """
    if not target_path:
        # Nếu path rỗng, chỉ có 1 vị trí
        new_path = [region]
        time = calculate_path_completion_time(new_path, target_uav, V_matrix, base_coords)
        return 0, time
    
    min_time = float('inf')
    best_pos = 0
    
    # Thử chèn vào từng vị trí
    for pos in range(len(target_path) + 1):
        new_path = target_path[:pos] + [region] + target_path[pos:]
        completion_time = calculate_path_completion_time(new_path, target_uav, V_matrix, base_coords)
        
        if completion_time < min_time:
            min_time = completion_time
            best_pos = pos
    
    return best_pos, min_time


def quick_path_optimization_nearest_neighbor(path: List[Region], uav: UAV, 
                                             V_matrix: List[List[float]],
                                             base_coords: Tuple[float, float] = (0.0, 0.0)) -> List[Region]:
    """
    Tối ưu nhanh thứ tự các vùng bằng thuật toán nearest neighbor
    """
    if len(path) <= 1:
        return path
    
    # Bắt đầu từ vùng gần base nhất
    remaining = path.copy()  # Sử dụng list thay vì set
    optimized = []
    current_pos = base_coords
    
    while remaining:
        # Tìm vùng gần nhất với vị trí hiện tại
        nearest = min(remaining, key=lambda r: calculate_distance(current_pos, r.coords))
        optimized.append(nearest)
        current_pos = nearest.coords
        remaining.remove(nearest)
    
    return optimized


def estimate_time_change(region: Region, source_uav: UAV, target_uav: UAV,
                        current_assignment: Dict[int, List[Region]],
                        uavs: List[UAV], V_matrix: List[List[float]],
                        base_coords: Tuple[float, float] = (0.0, 0.0),
                        use_quick_optimization: bool = True) -> float:
    """
    Ước lượng thay đổi thời gian hoàn thành khi chuyển region từ source_uav sang target_uav
    
    Args:
        region: Vùng cần chuyển
        source_uav: UAV hiện đang phụ trách vùng này
        target_uav: UAV sẽ nhận vùng này
        current_assignment: Phân bổ hiện tại
        uavs: Danh sách tất cả UAV
        V_matrix: Ma trận vận tốc quét
        base_coords: Tọa độ base
        use_quick_optimization: Có sử dụng tối ưu nhanh cho source path không
    
    Returns:
        Thay đổi thời gian (âm = cải thiện, dương = xấu đi)
    """
    # 1. Tính thời gian hiện tại của hệ thống
    current_max_time = calculate_system_completion_time(current_assignment, uavs, V_matrix, base_coords)
    
    # 2. Tạo assignment mới
    new_assignment = {uav_id: path.copy() for uav_id, path in current_assignment.items()}
    
    # 3. Xóa region khỏi source_uav
    source_path = new_assignment[source_uav.id]
    if region not in source_path:
        return float('inf')  # Region không thuộc source_uav
    
    source_path.remove(region)
    
    # 4. Tối ưu lại đường đi của source_uav nếu cần
    if use_quick_optimization and len(source_path) > 1:
        source_path = quick_path_optimization_nearest_neighbor(
            source_path, source_uav, V_matrix, base_coords
        )
    
    new_assignment[source_uav.id] = source_path
    
    # 5. Tìm vị trí tốt nhất để chèn region vào target_uav
    target_path = new_assignment[target_uav.id]
    best_pos, _ = find_best_insertion_position(
        region, target_path, target_uav, V_matrix, base_coords
    )
    
    # 6. Chèn region vào target_path
    target_path.insert(best_pos, region)
    new_assignment[target_uav.id] = target_path
    
    # 7. Tính thời gian mới của hệ thống
    new_max_time = calculate_system_completion_time(new_assignment, uavs, V_matrix, base_coords)
    
    # 8. Trả về sự thay đổi
    return new_max_time - current_max_time


def gradient_based_refinement(current_assignment: Dict[int, List[Region]],
                              uavs: List[UAV], V_matrix: List[List[float]],
                              base_coords: Tuple[float, float] = (0.0, 0.0),
                              verbose: bool = False) -> Tuple[Dict[int, List[Region]], bool]:
    """
    Tối ưu phân bổ bằng gradient information
    
    Returns:
        (new_assignment, improved): Assignment mới và có cải thiện hay không
    """
    if verbose:
        print("  → Bắt đầu gradient-based refinement...")
    
    # Tính độ nhạy (sensitivity) cho tất cả các moves có thể
    sensitivity = {}
    
    for source_uav in uavs:
        source_path = current_assignment.get(source_uav.id, [])
        
        for region in source_path:
            for target_uav in uavs:
                if target_uav.id == source_uav.id:
                    continue
                
                # Tính thay đổi thời gian khi chuyển region
                time_change = estimate_time_change(
                    region, source_uav, target_uav,
                    current_assignment, uavs, V_matrix, base_coords
                )
                
                sensitivity[(region.id, source_uav.id, target_uav.id)] = time_change
    
    if not sensitivity:
        if verbose:
            print("  → Không có move nào khả thi")
        return current_assignment, False
    
    # Tìm move tốt nhất (giảm thời gian nhiều nhất)
    best_move = min(sensitivity, key=sensitivity.get)
    best_time_change = sensitivity[best_move]
    
    if verbose:
        region_id, source_id, target_id = best_move
        print(f"  → Best move: Region {region_id}: UAV {source_id} → UAV {target_id}")
        print(f"  → Time change: {best_time_change:.2f}")
    
    # Nếu có cải thiện, thực hiện move
    if best_time_change < -1e-6:  # Ngưỡng nhỏ để tránh sai số số học
        region_id, source_id, target_id = best_move
        
        # Tìm region object
        region_to_move = None
        for r in current_assignment[source_id]:
            if r.id == region_id:
                region_to_move = r
                break
        
        if region_to_move is None:
            return current_assignment, False
        
        # Tạo assignment mới
        new_assignment = {uav_id: path.copy() for uav_id, path in current_assignment.items()}
        
        # Xóa khỏi source
        new_assignment[source_id].remove(region_to_move)
        
        # Tối ưu lại source path
        if len(new_assignment[source_id]) > 1:
            source_uav = next(u for u in uavs if u.id == source_id)
            new_assignment[source_id] = quick_path_optimization_nearest_neighbor(
                new_assignment[source_id], source_uav, V_matrix, base_coords
            )
        
        # Chèn vào target
        target_uav = next(u for u in uavs if u.id == target_id)
        best_pos, _ = find_best_insertion_position(
            region_to_move, new_assignment[target_id], 
            target_uav, V_matrix, base_coords
        )
        new_assignment[target_id].insert(best_pos, region_to_move)
        
        if verbose:
            print(f"  → Move applied! Improvement: {-best_time_change:.2f}")
        
        return new_assignment, True
    else:
        if verbose:
            print("  → Không có cải thiện, giữ nguyên assignment")
        return current_assignment, False


# ==============================
# Iterative APPA với Gradient Refinement
# ==============================

def iterative_appa_with_gradient(uavs: List[UAV], regions: List[Region], 
                                 V_matrix: List[List[float]],
                                 base_coords: Tuple[float, float] = (0.0, 0.0),
                                 max_iterations: int = 10,
                                 convergence_threshold: float = 1.0,
                                 acs_params: Optional[Dict] = None,
                                 visualize_iterations: bool = False,
                                 verbose: bool = True) -> Tuple[Dict[int, List[Region]], List[float], List[Dict[int, List[Region]]]]:
    """
    APPA với iterative refinement sử dụng gradient information
    
    Args:
        uavs: Danh sách UAV
        regions: Danh sách vùng
        V_matrix: Ma trận vận tốc quét
        base_coords: Tọa độ base
        max_iterations: Số vòng lặp tối đa
        convergence_threshold: Ngưỡng hội tụ (giây)
        acs_params: Tham số cho ACS (n_ants, n_generations, etc.)
        visualize_iterations: Nếu True, lưu lại assignments qua các iteration để visualize
        verbose: In log
    
    Returns:
        (best_assignment, history, all_assignments): Phân bổ tốt nhất, lịch sử thời gian và assignments qua các iteration
    """
    if verbose:
        print("=" * 60)
        print("ITERATIVE APPA WITH GRADIENT-BASED REFINEMENT")
        print("=" * 60)
    
    # Tham số mặc định cho ACS
    if acs_params is None:
        acs_params = {
            'n_ants': 10,
            'n_generations': 50,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.1,
            'epsilon': 0.1,
            'q0': 0.9
        }
    
    # Khởi tạo: Giai đoạn 1
    if verbose:
        print("\n[Iteration 0] Phân bổ ban đầu...")
    
    assignment = region_allocation(uavs, regions, V_matrix, base_coords, verbose=False)
    
    # Giai đoạn 2: Tối ưu thứ tự
    if verbose:
        print("[Iteration 0] Tối ưu thứ tự với ACS...")
    
    optimized_assignment = {}
    for uav in uavs:
        regions_for_uav = assignment[uav.id]
        if len(regions_for_uav) == 0:
            optimized_assignment[uav.id] = []
            continue
        
        optimizer = OrderOptimizerACS(
            uav=uav,
            regions=regions_for_uav,
            V_matrix=V_matrix,
            base_coords=base_coords,
            **acs_params
        )
        
        best_tour_ids, best_time = optimizer.run()
        
        # Chuyển từ IDs về objects (bỏ base id=-1)
        optimized_path = [r for r in regions_for_uav 
                         if r.id in best_tour_ids and r.id != -1]
        
        # Sắp xếp theo thứ tự trong best_tour_ids
        id_to_region = {r.id: r for r in optimized_path}
        ordered_path = [id_to_region[rid] for rid in best_tour_ids if rid != -1]
        
        optimized_assignment[uav.id] = ordered_path
    
    assignment = optimized_assignment
    
    # Tính thời gian ban đầu
    current_time = calculate_system_completion_time(assignment, uavs, V_matrix, base_coords)
    history = [current_time]
    
    if verbose:
        print(f"[Iteration 0] Thời gian hoàn thành: {current_time:.2f}")
        print()
    
    best_assignment = copy.deepcopy(assignment)
    best_time = current_time
    stagnation_count = 0
    
    # Lưu tất cả assignment qua các iteration nếu cần visualize
    all_assignments = [copy.deepcopy(assignment)] if visualize_iterations else []
    
    # Vòng lặp iterative refinement
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"[Iteration {iteration}] Gradient refinement...")
        
        # Gradient-based refinement
        new_assignment, improved = gradient_based_refinement(
            assignment, uavs, V_matrix, base_coords, verbose=verbose
        )
        
        if not improved:
            stagnation_count += 1
            if verbose:
                print(f"  → Stagnation count: {stagnation_count}")
            
            # Nếu stagnate quá nhiều, dừng
            if stagnation_count >= 10:
                if verbose:
                    print("  → Đã hội tụ (stagnation)")
                break
            
            assignment = new_assignment
            history.append(history[-1])
            continue
        
        assignment = new_assignment
        stagnation_count = 0
        
        # Giai đoạn 2: Tối ưu lại thứ tự với ACS
        if verbose:
            print(f"[Iteration {iteration}] Tối ưu lại thứ tự với ACS...")
        
        optimized_assignment = {}
        for uav in uavs:
            regions_for_uav = assignment[uav.id]
            if len(regions_for_uav) == 0:
                optimized_assignment[uav.id] = []
                continue
            
            optimizer = OrderOptimizerACS(
                uav=uav,
                regions=regions_for_uav,
                V_matrix=V_matrix,
                base_coords=base_coords,
                **acs_params
            )
            
            best_tour_ids, best_time_uav = optimizer.run()
            
            id_to_region = {r.id: r for r in regions_for_uav}
            ordered_path = [id_to_region[rid] for rid in best_tour_ids if rid != -1]
            
            optimized_assignment[uav.id] = ordered_path
        
        assignment = optimized_assignment
        
        # Lưu assignment ở iteration hiện tại nếu cần visualize
        if visualize_iterations:
            all_assignments.append(copy.deepcopy(assignment))
        
        # Tính thời gian mới
        new_time = calculate_system_completion_time(assignment, uavs, V_matrix, base_coords)
        history.append(new_time)
        
        improvement = current_time - new_time
        
        if verbose:
            print(f"[Iteration {iteration}] Thời gian: {new_time:.2f} (Cải thiện: {improvement:.2f})")
            print()
        
        # Cập nhật best
        if new_time < best_time:
            best_time = new_time
            best_assignment = copy.deepcopy(assignment)
        
        # Kiểm tra hội tụ
        if abs(improvement) < convergence_threshold:
            if verbose:
                print("  → Đã hội tụ (convergence threshold)")
            break
        
        current_time = new_time
    
    if verbose:
        print("=" * 60)
        print(f"KẾT QUẢ CUỐI CÙNG")
        print(f"Thời gian tốt nhất: {best_time:.2f}")
        print(f"Số iteration: {len(history)}")
        print(f"Cải thiện so với ban đầu: {history[0] - best_time:.2f} ({(history[0] - best_time) / history[0] * 100:.1f}%)")
        print("=" * 60)
    
    return best_assignment, history, all_assignments


# ==============================
# Visualization Functions
# ==============================

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches

def visualize_uav_coverage_paths(assignment: Dict[int, List[Region]], 
                                uavs: List[UAV], 
                                base_coords: Tuple[float, float] = (0.0, 0.0),
                                save_path: str = None):
    """
    Visualize UAV coverage paths with distinct colors for each UAV and its assigned regions
    
    Args:
        assignment: Dictionary mapping UAV ids to their assigned regions
        uavs: List of UAV objects
        base_coords: Coordinates of the base (default: (0.0, 0.0))
        save_path: Path to save the figure (if None, just display)
    """
    # Define distinct colors for UAVs (up to 10 UAVs)
    colors = ['#8A2BE2', '#20B2AA', '#FFA500', '#FF6347', '#32CD32', 
              '#4682B4', '#DC143C', '#9932CC', '#FF8C00', '#008080']
    
    plt.figure(figsize=(12, 8))
    plt.grid(True, alpha=0.3)
    
    # Plot base station
    plt.scatter(base_coords[0], base_coords[1], s=150, color='black', marker='*', 
                edgecolor='white', linewidth=1.5, zorder=100, label='Base')
    
    # Generate legend handles
    legend_handles = []
    
    # Plot each UAV's path
    for i, uav in enumerate(uavs):
        uav_color = colors[i % len(colors)]
        legend_handles.append(mpatches.Patch(color=uav_color, label=f'UAV {uav.id}'))
        
        regions = assignment.get(uav.id, [])
        if not regions:
            continue
        
        # Extract coordinates và thêm base vào đầu và cuối để tạo đường khép kín
        coords = [base_coords] + [r.coords for r in regions] + [base_coords]
        x_coords = [x for x, y in coords]
        y_coords = [y for x, y in coords]
        
        # Plot path
        plt.plot(x_coords, y_coords, color=uav_color, linewidth=2, alpha=0.7)
        
        # Plot regions
        for r in regions:
            plt.scatter(r.coords[0], r.coords[1], s=100, color=uav_color, edgecolor='white', 
                        linewidth=0.8, alpha=0.9, zorder=10)
            plt.annotate(f'R{r.id}', (r.coords[0], r.coords[1]), 
                        xytext=(3, 3), textcoords='offset points', 
                        fontsize=8, color='black', fontweight='bold')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('UAV Coverage Paths', fontsize=14)
    plt.legend(handles=legend_handles, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

def visualize_convergence_curve(history: List[float], save_path: str = None):
    """
    Visualize the convergence curve of the algorithm
    
    Args:
        history: List of completion times for each iteration
        save_path: Path to save the figure (if None, just display)
    """
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(history)), history, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Cost (Time)', fontsize=12)
    plt.title('Convergence Curve', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence curve to {save_path}")
    else:
        plt.show()

# ==============================
# Hàm test
# ==============================

if __name__ == "__main__":
    # Tạo sample data
    sample = create_sample()

    uavs = [UAV(**uav_data) for uav_data in sample['uavs_list']]
    regions = [Region(**region_data) for region_data in sample['regions_list']]
    V_matrix = np.array(sample['V_matrix'])
    
    # Chạy iterative APPA
    best_assignment, history, all_assignments = iterative_appa_with_gradient(
        uavs,
        regions,
        V_matrix,
        base_coords=(0, 0),
        max_iterations=100,
        convergence_threshold=1.0,
        visualize_iterations=True,
        acs_params={
            'n_ants': 10,
            'n_generations': 30,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.1,
            'epsilon': 0.1,
            'q0': 0.9
        },
        verbose=False
    )
    
    # In kết quả chi tiết
    print("\n" + "=" * 60)
    print("PHÂN BỔ CUỐI CÙNG:")
    print("=" * 60)
    for uav_id, path in best_assignment.items():
        region_ids = [r.id for r in path]
        print(f"UAV {uav_id}: {region_ids}")
    
    print("\n" + "=" * 60)
    print("LỊCH SỬ THỜI GIAN:")
    print("=" * 60)
    for i, time in enumerate(history):
        improvement = ""
        if i > 0:
            diff = history[i-1] - time
            improvement = f" (Δ={diff:+.2f})"
        print(f"Iteration {i}: {time:.2f}{improvement}")
    
    # Visualize kết quả
    import os
    
    # Tạo thư mục fig nếu chưa tồn tại
    os.makedirs('fig', exist_ok=True)
    
    # Visualize đường đi của UAV trong phân bổ cuối cùng
    visualize_uav_coverage_paths(
        best_assignment, 
        uavs, 
        base_coords=(0, 0),
        save_path='fig/uav_coverage_paths_final.png'
    )
    
    # Visualize đường cong hội tụ
    visualize_convergence_curve(
        history,
        save_path='fig/convergence_curve.png'
    )
    
    # Visualize quá trình biến đổi phân vùng qua các iteration
    if all_assignments:
        print("\nVisualizing iterations...")
        for i, assignment in enumerate(all_assignments):
            visualize_uav_coverage_paths(
                assignment,
                uavs,
                base_coords=(0, 0),
                save_path=f'fig/uav_coverage_paths_iter_{i}.png'
            )
        print(f"Đã tạo {len(all_assignments)} hình visualize cho các iteration")