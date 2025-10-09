import json
import random
from dataclasses import dataclass
import numpy as np

@dataclass
class Region:
    id: int
    coords: tuple[float, float]
    area: float

@dataclass
class UAV:
    id: int
    max_velocity: float
    scan_width: float

def generate_problem_instance(num_uavs: int, num_regions: int):
    uavs_list = []
    for i in range(1, num_uavs + 1):
        uav = UAV(
            id=i,
            max_velocity=random.uniform(15, 30),
            scan_width=random.uniform(5, 20)
        )
        uavs_list.append(uav)
        
    regions_list = []
    map_boundary = 5000 
    for i in range(1, num_regions + 1):
        region = Region(
            id=i,
            coords=(
                random.uniform(-map_boundary, map_boundary),
                random.uniform(-map_boundary, map_boundary)
            ),
            area=random.uniform(500, 5000)
        )
        regions_list.append(region)
        
    V_matrix_np = np.zeros((num_uavs, num_regions))
    for i in range(num_uavs):
        for j in range(num_regions):
            if random.random() < 0.05:
                V_matrix_np[i, j] = 0
            else:
                drag_factor = random.uniform(0.1, 0.9)
                scan_velocity = uavs_list[i].max_velocity * drag_factor
                V_matrix_np[i, j] = scan_velocity
    
    V_matrix = V_matrix_np.tolist()
    
    return uavs_list, regions_list, V_matrix

if __name__ == "__main__":
    NUM_UAVS = 10
    NUM_REGIONS = 1000
    
    uavs, regions, v_matrix = generate_problem_instance(NUM_UAVS, NUM_REGIONS)
    
    uavs_dict_list = [uav.__dict__ for uav in uavs]
    regions_dict_list = [region.__dict__ for region in regions]
    
    data_to_write = {
        "uavs_list": uavs_dict_list,
        "regions_list": regions_dict_list,
        "V_matrix": v_matrix
    }
    
    file_path = "sample.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_write, f, ensure_ascii=False, indent=4)
        
    print(f"Dữ liệu đã được ghi thành công vào file '{file_path}'.")