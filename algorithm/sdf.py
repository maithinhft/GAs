from utils.utils import *

def solve_sdf(uavs_list: List[UAV], 
              regions_list: List[Region], 
              v_matrix: List[List[float]]) -> Tuple[float, List[List[Region]]]:
    """
    Giải bài toán bằng thuật toán Short Distance First (SDF).
    """
    num_uavs = len(uavs_list)
    
    uav_finish_times = [0.0] * num_uavs
    uav_last_coords = [BASE_COORDS] * num_uavs
    uav_paths: List[List[Region]] = [[] for _ in range(num_uavs)]
    
    unassigned_regions = list(regions_list)
    
    while unassigned_regions:
        earliest_uav_idx = np.argmin(uav_finish_times)
        current_time = uav_finish_times[earliest_uav_idx]
        current_coords = uav_last_coords[earliest_uav_idx]
        uav = uavs_list[earliest_uav_idx] # uav giờ là đối tượng UAV

        best_region = None
        min_dist = float('inf')

        for region in unassigned_regions: # region giờ là đối tượng Region
            dist = get_distance(current_coords, region.coords) # SỬA: Dùng region.coords
            if dist < min_dist:
                region_idx = regions_list.index(region)
                v_entry = v_matrix[earliest_uav_idx][region_idx]
                if v_entry > 0:
                    min_dist = dist
                    best_region = region
        
        if best_region is None:
            uav_finish_times[earliest_uav_idx] = float('inf')
            if all(t == float('inf') for t in uav_finish_times):
                print("Lỗi: Không thể gán các khu vực còn lại")
                break
            continue

        best_region_idx = regions_list.index(best_region)
        fly_t = get_fly_time(uav, current_coords, best_region.coords) # SỬA: Dùng best_region.coords
        scan_t = get_scan_time(uav, best_region, v_matrix[earliest_uav_idx][best_region_idx])
        
        uav_finish_times[earliest_uav_idx] = current_time + fly_t + scan_t
        uav_last_coords[earliest_uav_idx] = best_region.coords # SỬA: Dùng best_region.coords
        uav_paths[earliest_uav_idx].append(best_region)
        unassigned_regions.remove(best_region)

    if unassigned_regions:
         print("Lỗi: Không gán được tất cả các vùng, trả về inf")
         return float('inf'), uav_paths

    # Thêm thời gian bay về base cho mỗi UAV
    for uav_idx in range(num_uavs):
        if uav_paths[uav_idx]:  # Nếu UAV có ít nhất 1 region
            uav = uavs_list[uav_idx]
            last_coords = uav_last_coords[uav_idx]
            fly_back_time = get_fly_time(uav, last_coords, BASE_COORDS)
            uav_finish_times[uav_idx] += fly_back_time

    valid_finish_times = [t for t in uav_finish_times if t != float('inf')]

    if not valid_finish_times:
        return 0.0, uav_paths

    max_completion_time = max(valid_finish_times)
    return max_completion_time, uav_paths