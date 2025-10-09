import math
import random
import numpy as np
from dataclasses import dataclass
import json

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

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ts(uav: UAV, region: Region, V_matrix: list[list[float]]):
    uav_idx = uav.id - 1
    region_idx = region.id - 1
    
    scan_velocity = V_matrix[uav_idx][region_idx]
    
    if scan_velocity == 0 or uav.scan_width == 0:
        return float('inf')
        
    return region.area / (scan_velocity * uav.scan_width)

def calculate_tf(uav: UAV, p1, p2):
    """Tính Thời gian Bay (TF)"""
    distance = calculate_distance(p1, p2)
    if uav.max_velocity == 0:
        return float('inf')
    return distance / uav.max_velocity


def region_allocation(uavs: list[UAV], regions: list[Region], V_matrix: list[list[float]], base_coords=(0,0)):
    print("--- Bắt đầu Giai đoạn 1: Phân bổ khu vực ---")
    
    unassigned_regions = regions.copy()
    assigned_regions = {uav.id: [] for uav in uavs}
    finish_times = {uav.id: 0.0 for uav in uavs}
    last_coords = {uav.id: base_coords for uav in uavs}

    while unassigned_regions:
        earliest_uav_id = min(finish_times, key=finish_times.get)
        current_uav = next(u for u in uavs if u.id == earliest_uav_id)
        
        best_region = None
        max_etr = -1

        for region in unassigned_regions:
            ts = calculate_ts(current_uav, region, V_matrix)
            tf = calculate_tf(current_uav, last_coords[current_uav.id], region.coords)
            
            if ts == float('inf'): continue

            etr = ts / (ts + tf)
            
            if etr > max_etr:
                max_etr = etr
                best_region = region
        
        if best_region:
            assigned_regions[current_uav.id].append(best_region)
            unassigned_regions.remove(best_region)
            
            ts = calculate_ts(current_uav, best_region, V_matrix)
            tf = calculate_tf(current_uav, last_coords[current_uav.id], best_region.coords)
            finish_times[current_uav.id] += (ts + tf)
            last_coords[current_uav.id] = best_region.coords
            
            print(f"Gán khu vực {best_region.id} cho UAV {current_uav.id}. ETR = {max_etr:.2f}")

    print("--- Kết thúc Giai đoạn 1 ---\n")
    return assigned_regions


class OrderOptimizerACS:
    def __init__(self, uav: UAV, regions: list[Region], V_matrix: list[list[float]], base_coords=(0,0),
                 n_ants=10, n_generations=50, alpha=1.0, beta=2.0,
                 rho=0.1, epsilon=0.1, q0=0.9):
        self.uav = uav
        self.points = [Region(id=-1, coords=base_coords, area=0)] + regions
        self.V_matrix = V_matrix 
        self.n_points = len(self.points)
        
        self.n_ants, self.n_generations, self.alpha, self.beta, self.rho, self.epsilon, self.q0 = \
            n_ants, n_generations, alpha, beta, rho, epsilon, q0

        self._initialize()
    
    def _calculate_nearest_neighbour_tour(self):
        current_idx = 0 
        tour = [current_idx]
        unvisited = list(range(1, self.n_points))
        tour_length = 0
        
        while unvisited:
            nearest_dist = float('inf')
            nearest_idx = -1
            for next_idx in unvisited:
                dist = self.dist_matrix[current_idx, next_idx]
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = next_idx
            
            tour_length += nearest_dist
            current_idx = nearest_idx
            tour.append(current_idx)
            unvisited.remove(current_idx)
            
        return tour, tour_length

    def _initialize(self):
        self.dist_matrix = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_points):
            for j in range(self.n_points):
                self.dist_matrix[i, j] = calculate_distance(self.points[i].coords, self.points[j].coords)
        self.eta = 1.0 / (self.dist_matrix + 1e-10) # heuristic information

        _, L_nn = self._calculate_nearest_neighbour_tour()
        initial_pheromone = 1.0 / (self.n_points * L_nn)
        self.tau0 = initial_pheromone 
        self.tau = np.full((self.n_points, self.n_points), initial_pheromone) # phenome

    def _calculate_tour_time(self, tour):
        total_time = 0
        for i in range(len(tour) - 1):
            p1_idx, p2_idx = tour[i], tour[i+1]
            p1 = self.points[p1_idx]
            p2 = self.points[p2_idx]
            
            tf = calculate_tf(self.uav, p1.coords, p2.coords)
            ts = calculate_ts(self.uav, p2, self.V_matrix) 
            total_time += tf + ts
        return total_time

    def _select_next_region(self, current_region_idx, unvisited):
        q = random.random()
        attractiveness = (self.tau[current_region_idx, unvisited] ** self.alpha) * (self.eta[current_region_idx, unvisited] ** self.beta)
        if q <= self.q0:
            next_region_idx = unvisited[np.argmax(attractiveness)]
        else:
            probs = attractiveness / np.sum(attractiveness)
            next_region_idx = np.random.choice(unvisited, p=probs)
        return next_region_idx

    def run(self):
        best_tour, best_tour_time = None, float('inf')

        for gen in range(self.n_generations):
            for ant in range(self.n_ants):
                
                start_idx = random.choice(list(range(1, self.n_points)))
                
                tour = [start_idx]
                current_region_idx = start_idx
                
                unvisited = list(range(1, self.n_points))
                unvisited.remove(start_idx)

                while unvisited:
                    next_region_idx = self._select_next_region(current_region_idx, unvisited)
                    
                    self.tau[current_region_idx, next_region_idx] = (1 - self.rho) * self.tau[current_region_idx, next_region_idx] + self.rho * self.tau0
                    
                    tour.append(next_region_idx)
                    unvisited.remove(next_region_idx)
                    current_region_idx = next_region_idx
                
                full_tour_for_timing = [0] + tour 
                tour_time = self._calculate_tour_time(full_tour_for_timing)
                
                if tour_time < best_tour_time:
                    best_tour_time = tour_time
                    best_tour = full_tour_for_timing
            
            if best_tour:
                self.tau *= (1 - self.epsilon)
                for i in range(len(best_tour) - 1):
                    p1, p2 = best_tour[i], best_tour[i+1]
                    self.tau[p1, p2] += self.epsilon * (1.0 / best_tour_time)
                    
        best_tour_ids = [self.points[i].id for i in best_tour]
        return best_tour_ids, best_tour_time

if __name__ == "__main__":
    file_path = 'sample.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
        regions_list = [Region(**region) for region in data['regions_list']]
        V_matrix = data['V_matrix']
        
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{file_path}'")
    except json.JSONDecodeError:
        print(f"Lỗi: Nội dung trong file '{file_path}' không phải là định dạng JSON hợp lệ.")

    # uavs_list = [
    #     UAV(id=1, max_velocity=20, scan_width=10),
    #     UAV(id=2, max_velocity=15, scan_width=15)
    # ]
    # regions_list = [
    #     Region(id=1, coords=(100, 50), area=1000), 
    #     Region(id=2, coords=(20, 150), area=1500),
    #     Region(id=3, coords=(-50, 80), area=1200), 
    #     Region(id=4, coords=(120, -30), area=2000),
    #     Region(id=5, coords=(-80, -60), area=800), 
    #     Region(id=6, coords=(0, -100), area=1800)
    # ]
    
    # V_matrix = [
    #     [12, 10, 11, 9, 12, 0],   
    #     [8,  9,  0,  7, 6,  8]
    # ]

    assignments = region_allocation(uavs_list, regions_list, V_matrix)
    
    print("Kết quả phân bổ:")
    for uav_id, regions in assignments.items():
        region_ids = [r.id for r in regions]
        print(f"  - UAV {uav_id} được gán các khu vực: {region_ids}")
    print("-" * 30)

    for uav in uavs_list:
        assigned_r = assignments[uav.id]
        if assigned_r:
            print(f"\n--- Bắt đầu Giai đoạn 2: Tối ưu hóa thứ tự cho UAV {uav.id} ---")
            optimizer = OrderOptimizerACS(uav, assigned_r, V_matrix)
            best_path, best_time = optimizer.run()
            
            print(f"Lộ trình tối ưu cho UAV {uav.id}: {best_path}")
            print(f"Ước tính thời gian hoàn thành: {best_time:.2f} giây")
            print(f"--- Kết thúc Giai đoạn 2 cho UAV {uav.id} ---")