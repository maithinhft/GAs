import random
import numpy as np
from .config import Region, UAV, RANDOM_SEED, MAP_BOUNDARY

def uunifast(total_sum, num_items):
    """
    Triển khai thuật toán UUniFast để tạo ra một danh sách `num_items` số
    có tổng bằng `total_sum`.
    """
    if num_items == 0:
        return []
    if num_items == 1:
        return [total_sum]
        
    split_points = sorted([random.uniform(0, total_sum) for _ in range(num_items - 1)])
    
    full_points = [0] + split_points + [total_sum]
    
    return [full_points[i+1] - full_points[i] for i in range(num_items)]

def generate_problem_instance(
    num_uavs: int,
    num_regions: int,
    system_area_ratio: float,
    system_drag_factor: float,
    map_boundary: float = MAP_BOUNDARY
):
    """
    Sinh ra một bộ dữ liệu thử nghiệm theo phương pháp trong bài báo.

    Args:
        num_uavs (int): Số lượng UAV.
        num_regions (int): Số lượng khu vực (m).
        system_area_ratio (float): Tỷ lệ diện tích hệ thống (u).
        system_drag_factor (float): Hệ số cản hệ thống (d).
        map_boundary (float): Biên của bản đồ, dùng để tính tổng diện tích.
    """
    uavs_list = []
    for i in range(1, num_uavs + 1):
        uav = UAV(
            id=i,
            max_velocity=random.uniform(15, 30),
            scan_width=random.uniform(5, 20)
        )
        uavs_list.append(uav)
        
    regions_list = []
    
    total_map_area = (2 * map_boundary) ** 2
    area_ratios = uunifast(system_area_ratio, num_regions)
    
    for i in range(1, num_regions + 1):
        region_area = area_ratios[i-1] * total_map_area
        
        region_coords = (
            random.uniform(-map_boundary, map_boundary),
            random.uniform(-map_boundary, map_boundary)
        )
        
        region = Region(
            id=i,
            coords=region_coords,
            area=region_area
        )
        regions_list.append(region)
        
    V_matrix_np = np.zeros((num_uavs, num_regions))
    
    delta = min(1 - system_drag_factor, system_drag_factor)
    drag_factor_min = system_drag_factor - delta
    drag_factor_max = system_drag_factor + delta

    for i in range(num_uavs):
        for j in range(num_regions):
            if random.random() < 0.05:
                V_matrix_np[i, j] = 0
            else:
                drag_factor = random.uniform(drag_factor_min, drag_factor_max)
                scan_velocity = uavs_list[i].max_velocity * drag_factor
                V_matrix_np[i, j] = scan_velocity
    
    for j in range(num_regions):
        if all(V_matrix_np[i, j] == 0 for i in range(num_uavs)):
            random_uav = random.randint(0, num_uavs - 1)
            drag_factor = random.uniform(drag_factor_min, drag_factor_max)
            V_matrix_np[random_uav, j] = uavs_list[random_uav].max_velocity * drag_factor

    return uavs_list, regions_list, V_matrix_np.tolist()

def create_sample(NUM_UAVS = 4, NUM_REGIONS = 50,SYSTEM_AREA_RATIO = 0.05, SYSTEM_DRAG_FACTOR = 0.9):     
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    uavs, regions, v_matrix = generate_problem_instance(
        NUM_UAVS, 
        NUM_REGIONS,
        SYSTEM_AREA_RATIO,
        SYSTEM_DRAG_FACTOR
    )
    
    uavs_dict_list = [uav.__dict__ for uav in uavs]
    regions_dict_list = [region.__dict__ for region in regions]
    
    data_to_write = {
        "uavs_list": uavs_dict_list,
        "regions_list": regions_dict_list,
        "V_matrix": v_matrix
    }

    # file_path = "sample.json"
    # with open(file_path, 'w', encoding='utf-8') as f:
    #     json.dump(data_to_write, f, ensure_ascii=False, indent=4)
    # print(f"Dữ liệu đã được ghi thành công vào file '{file_path}'.")
    
    return data_to_write

if __name__ == "__main__":
    create_sample()