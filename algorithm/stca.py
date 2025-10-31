from utils.utils import *

def solve_stca_ne(uavs_list: List[UAV], 
                  regions_list: List[Region], 
                  v_matrix: List[List[float]]) -> Tuple[float, List[List[Region]]]:
    """
    Giải bằng STCA-NE (Nearest-to-End, diễn giải là "Minimum Insertion Cost").
    """
    num_uavs = len(uavs_list)
    uav_finish_times = [0.0] * num_uavs
    uav_last_coords = [BASE_COORDS] * num_uavs
    uav_paths: List[List[Region]] = [[] for _ in range(num_uavs)]
    
    unassigned_regions = list(regions_list)
    
    while unassigned_regions:
        best_region = None
        best_uav_idx = -1
        min_new_finish_time = float('inf')

        for uav_idx in range(num_uavs):
            uav = uavs_list[uav_idx] # uav giờ là đối tượng UAV
            current_time = uav_finish_times[uav_idx]
            current_coords = uav_last_coords[uav_idx]
            
            if current_time == float('inf'):
                continue

            for region in unassigned_regions: # region giờ là đối tượng Region
                region_idx = regions_list.index(region)
                v_entry = v_matrix[uav_idx][region_idx]
                
                if v_entry == 0:
                    continue

                fly_t = get_fly_time(uav, current_coords, region.coords) # SỬA: Dùng region.coords
                scan_t = get_scan_time(uav, region, v_entry)
                new_finish_time = current_time + fly_t + scan_t
                
                if new_finish_time < min_new_finish_time:
                    min_new_finish_time = new_finish_time
                    best_region = region
                    best_uav_idx = uav_idx

        if best_uav_idx == -1 or best_region is None:
            print("Lỗi: Không thể gán các khu vực còn lại")
            break
            
        uav_paths[best_uav_idx].append(best_region)
        uav_finish_times[best_uav_idx] = min_new_finish_time
        uav_last_coords[best_uav_idx] = best_region.coords # SỬA: Dùng best_region.coords
        unassigned_regions.remove(best_region)

    max_completion_time = max(t for t in uav_finish_times if t != float('inf'))
    return max_completion_time, uav_paths