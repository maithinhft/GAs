import json
import numpy as np
import random
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class Region:
    id: int
    coords: Tuple[float, float]
    area: float

@dataclass
class UAV:
    id: int
    max_velocity: float
    scan_width: float

class MultiColonyACS:
    """
    Thuật toán Đàn kiến đa bầy đàn để giải quyết bài toán phân vùng và
    tối ưu đường đi cho UAV đồng thời trong một quá trình thống nhất.
    
    Mỗi bầy đàn tương ứng với một UAV và cạnh tranh để phân vùng các region.
    """
    
    def __init__(
        self,
        uavs: List[UAV],
        regions: List[Region],
        v_matrix: np.ndarray,
        num_ants_per_colony: int = 10,
        max_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.1,
        epsilon: float = 0.1,
        q0: float = 0.9,
        omega: float = 0.5  # Trọng số cân bằng giữa thời gian bay và thời gian quét
    ):
        self.uavs = uavs
        self.regions = regions
        self.num_uavs = len(uavs)
        self.num_regions = len(regions)
        self.v_matrix = v_matrix
        
        # Tham số ACS
        self.num_ants_per_colony = num_ants_per_colony
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        self.q0 = q0
        self.omega = omega
        
        # Khởi tạo ma trận khoảng cách
        self.distance_matrix = self._compute_distance_matrix()
        
        # Khởi tạo ma trận pheromone cho mỗi bầy đàn (UAV)
        # pheromone[uav_id][i][j]: pheromone của UAV uav_id từ region i đến region j
        self.pheromone = {}
        for uav_id in range(self.num_uavs):
            self.pheromone[uav_id] = self._initialize_pheromone()
        
        # Lưu giải pháp tốt nhất
        self.best_solution = None
        self.best_cost = float('inf')
        self.convergence_curve = []
        
    def _compute_distance_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách giữa các region"""
        D = np.zeros((self.num_regions, self.num_regions))
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if i != j:
                    dx = self.regions[i].coords[0] - self.regions[j].coords[0]
                    dy = self.regions[i].coords[1] - self.regions[j].coords[1]
                    D[i][j] = math.sqrt(dx**2 + dy**2)
        return D
    
    def _initialize_pheromone(self) -> np.ndarray:
        """Khởi tạo ma trận pheromone ban đầu"""
        tau0 = 1.0 / (self.num_regions * np.mean(self.distance_matrix[self.distance_matrix > 0]))
        return np.full((self.num_regions, self.num_regions), tau0)
    
    def _compute_scan_time(self, uav_id: int, region_id: int) -> float:
        """Tính thời gian quét region của UAV"""
        if self.v_matrix[uav_id][region_id] == 0:
            return float('inf')
        scan_time = self.regions[region_id].area / (
            self.v_matrix[uav_id][region_id] * self.uavs[uav_id].scan_width
        )
        return scan_time
    
    def _compute_flight_time(self, uav_id: int, from_region: int, to_region: int) -> float:
        """Tính thời gian bay từ region này sang region khác"""
        distance = self.distance_matrix[from_region][to_region]
        return distance / self.uavs[uav_id].max_velocity
    
    def _compute_heuristic(self, uav_id: int, current_region: int, next_region: int) -> float:
        """
        Tính giá trị heuristic kết hợp:
        - Khoảng cách ngắn
        - Effective Time Ratio cao (ưu tiên region có diện tích lớn và gần)
        """
        if current_region == next_region:
            return 0
        
        # Tính ETR (Effective Time Ratio)
        scan_time = self._compute_scan_time(uav_id, next_region)
        flight_time = self._compute_flight_time(uav_id, current_region, next_region)
        
        if scan_time == float('inf') or flight_time == 0:
            return 0
        
        etr = scan_time / (scan_time + flight_time)
        distance = self.distance_matrix[current_region][next_region]
        
        # Kết hợp ETR và khoảng cách
        if distance > 0:
            heuristic = (self.omega * etr + (1 - self.omega) / distance)
        else:
            heuristic = 0
            
        return heuristic
    
    def _select_next_region(
        self, 
        uav_id: int, 
        current_region: int, 
        unvisited: List[int],
        pheromone: np.ndarray
    ) -> int:
        """Chọn region tiếp theo dựa trên pheromone và heuristic"""
        if not unvisited:
            return None
        
        q = random.random()
        
        if q <= self.q0:
            # Exploitation: chọn region tốt nhất
            best_value = -1
            best_region = unvisited[0]
            
            for region in unvisited:
                tau = pheromone[current_region][region] ** self.alpha
                eta = self._compute_heuristic(uav_id, current_region, region) ** self.beta
                value = tau * eta
                
                if value > best_value:
                    best_value = value
                    best_region = region
            
            return best_region
        else:
            # Exploration: chọn theo xác suất
            probabilities = []
            total = 0
            
            for region in unvisited:
                tau = pheromone[current_region][region] ** self.alpha
                eta = self._compute_heuristic(uav_id, current_region, region) ** self.beta
                prob = tau * eta
                probabilities.append(prob)
                total += prob
            
            if total == 0:
                return random.choice(unvisited)
            
            probabilities = [p / total for p in probabilities]
            return np.random.choice(unvisited, p=probabilities)
    
    def _construct_solution(self, iteration: int) -> Tuple[Dict, float]:
        """
        Xây dựng giải pháp bằng cách cho tất cả các bầy đàn (UAVs) cạnh tranh
        để phân vùng và tìm đường đi đồng thời.
        """
        # Mỗi UAV có một tập kiến
        colony_solutions = {}
        for uav_id in range(self.num_uavs):
            colony_solutions[uav_id] = []
        
        # Mỗi kiến trong mỗi bầy đàn xây dựng một giải pháp
        all_ants_solutions = []
        
        for uav_id in range(self.num_uavs):
            for ant_id in range(self.num_ants_per_colony):
                # Khởi tạo
                unvisited_global = list(range(self.num_regions))
                path = []
                
                # Chọn region bắt đầu ngẫu nhiên
                if unvisited_global:
                    current = random.choice(unvisited_global)
                    path.append(current)
                    unvisited_global.remove(current)
                
                # Xây dựng đường đi
                while unvisited_global:
                    next_region = self._select_next_region(
                        uav_id, 
                        current, 
                        unvisited_global,
                        self.pheromone[uav_id]
                    )
                    
                    if next_region is None:
                        break
                    
                    path.append(next_region)
                    unvisited_global.remove(next_region)
                    
                    # Local pheromone update
                    self._local_pheromone_update(uav_id, current, next_region)
                    current = next_region
                
                # Lưu giải pháp của kiến này
                all_ants_solutions.append({
                    'uav_id': uav_id,
                    'ant_id': ant_id,
                    'path': path
                })
        
        # Phân vùng regions cho các UAVs dựa trên giải pháp tốt nhất
        # Sử dụng cơ chế đấu giá: mỗi region được gán cho UAV có chi phí thấp nhất
        region_assignment = {}
        uav_paths = {uav_id: [] for uav_id in range(self.num_uavs)}
        
        # Đánh giá chi phí của mỗi UAV cho từng region
        region_costs = {region_id: {} for region_id in range(self.num_regions)}
        
        for solution in all_ants_solutions:
            uav_id = solution['uav_id']
            path = solution['path']
            
            for idx, region_id in enumerate(path):
                # Tính chi phí nếu UAV này đến region này
                if idx == 0:
                    cost = self._compute_scan_time(uav_id, region_id)
                else:
                    prev_region = path[idx - 1]
                    cost = (self._compute_flight_time(uav_id, prev_region, region_id) +
                           self._compute_scan_time(uav_id, region_id))
                
                if uav_id not in region_costs[region_id]:
                    region_costs[region_id][uav_id] = []
                region_costs[region_id][uav_id].append(cost)
        
        # Gán region cho UAV có chi phí trung bình thấp nhất
        for region_id in range(self.num_regions):
            best_uav = None
            best_cost = float('inf')
            
            for uav_id in range(self.num_uavs):
                if uav_id in region_costs[region_id] and region_costs[region_id][uav_id]:
                    avg_cost = np.mean(region_costs[region_id][uav_id])
                    if avg_cost < best_cost:
                        best_cost = avg_cost
                        best_uav = uav_id
            
            if best_uav is not None:
                region_assignment[region_id] = best_uav
                if region_id not in uav_paths[best_uav]:
                    uav_paths[best_uav].append(region_id)
        
        # Tối ưu thứ tự trong mỗi path của UAV bằng nearest neighbor
        for uav_id in range(self.num_uavs):
            if len(uav_paths[uav_id]) > 1:
                uav_paths[uav_id] = self._optimize_path_order(uav_id, uav_paths[uav_id])
        
        # Tính tổng chi phí
        total_cost = self._evaluate_solution(uav_paths)
        
        return uav_paths, total_cost
    
    def _optimize_path_order(self, uav_id: int, regions_to_visit: List[int]) -> List[int]:
        """Tối ưu thứ tự các region trong path bằng nearest neighbor"""
        if len(regions_to_visit) <= 1:
            return regions_to_visit
        
        unvisited = regions_to_visit.copy()
        path = [unvisited.pop(random.randint(0, len(unvisited) - 1))]
        
        while unvisited:
            current = path[-1]
            nearest = min(unvisited, 
                         key=lambda r: self.distance_matrix[current][r])
            path.append(nearest)
            unvisited.remove(nearest)
        
        return path
    
    def _evaluate_solution(self, uav_paths: Dict[int, List[int]]) -> float:
        """Đánh giá chi phí của giải pháp (thời gian hoàn thành lớn nhất)"""
        max_time = 0
        
        for uav_id, path in uav_paths.items():
            if not path:
                continue
            
            total_time = 0
            for idx, region_id in enumerate(path):
                # Thời gian quét
                total_time += self._compute_scan_time(uav_id, region_id)
                
                # Thời gian bay đến region tiếp theo
                if idx < len(path) - 1:
                    next_region = path[idx + 1]
                    total_time += self._compute_flight_time(uav_id, region_id, next_region)
            
            max_time = max(max_time, total_time)
        
        return max_time
    
    def _local_pheromone_update(self, uav_id: int, i: int, j: int):
        """Cập nhật pheromone cục bộ"""
        tau0 = 1.0 / (self.num_regions * np.mean(self.distance_matrix[self.distance_matrix > 0]))
        self.pheromone[uav_id][i][j] = (1 - self.rho) * self.pheromone[uav_id][i][j] + self.rho * tau0
    
    def _global_pheromone_update(self, best_paths: Dict[int, List[int]], best_cost: float):
        """Cập nhật pheromone toàn cục"""
        # Làm bay hơi tất cả pheromone
        for uav_id in range(self.num_uavs):
            self.pheromone[uav_id] *= (1 - self.epsilon)
        
        # Tăng cường pheromone trên đường đi tốt nhất
        if best_cost > 0:
            delta_tau = 1.0 / best_cost
            
            for uav_id, path in best_paths.items():
                for idx in range(len(path) - 1):
                    i = path[idx]
                    j = path[idx + 1]
                    self.pheromone[uav_id][i][j] += self.epsilon * delta_tau
    
    def solve(self) -> Tuple[Dict[int, List[int]], float]:
        """Giải bài toán bằng Multi-Colony ACS"""
        print("Bắt đầu tối ưu bằng Multi-Colony Ant System...")
        
        for iteration in range(self.max_iterations):
            # Xây dựng giải pháp
            current_solution, current_cost = self._construct_solution(iteration)
            
            # Cập nhật giải pháp tốt nhất
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_solution = current_solution
                print(f"Iteration {iteration + 1}: Tìm thấy giải pháp tốt hơn - Cost = {self.best_cost:.2f}")
            
            # Cập nhật pheromone toàn cục
            self._global_pheromone_update(self.best_solution, self.best_cost)
            
            # Lưu để vẽ đồ thị hội tụ
            self.convergence_curve.append(self.best_cost)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}: Best Cost = {self.best_cost:.2f}")
        
        print(f"\nHoàn thành! Chi phí tốt nhất: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost
    
    def visualize_solution(self, save_path: str = None):
        """Visualize the solution"""
        if self.best_solution is None:
            print("Chưa có giải pháp để visualize!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Paths của các UAVs
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_uavs))
        
        for uav_id, path in self.best_solution.items():
            if not path:
                continue
            
            color = colors[uav_id]
            
            # Vẽ các region
            for region_id in path:
                region = self.regions[region_id]
                ax1.scatter(region.coords[0], region.coords[1], 
                           c=[color], s=100, marker='o', alpha=0.6)
                ax1.text(region.coords[0], region.coords[1], 
                        f'R{region_id}', fontsize=8)
            
            # Vẽ đường đi
            if len(path) > 1:
                for i in range(len(path) - 1):
                    r1 = self.regions[path[i]]
                    r2 = self.regions[path[i + 1]]
                    ax1.plot([r1.coords[0], r2.coords[0]], 
                            [r1.coords[1], r2.coords[1]], 
                            c=color, alpha=0.5, linewidth=2,
                            label=f'UAV {uav_id + 1}' if i == 0 else '')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('UAV Coverage Paths')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Convergence curve
        ax2.plot(self.convergence_curve, linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Best Cost (Time)')
        ax2.set_title('Convergence Curve')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu hình ảnh vào {save_path}")
        
        plt.show()
    
    def print_solution_details(self):
        """In chi tiết giải pháp"""
        if self.best_solution is None:
            print("Chưa có giải pháp!")
            return
        
        print("\n" + "="*60)
        print("CHI TIẾT GIẢI PHÁP")
        print("="*60)
        
        for uav_id, path in self.best_solution.items():
            if not path:
                print(f"\nUAV {uav_id + 1}: Không được gán region nào")
                continue
            
            print(f"\nUAV {uav_id + 1}:")
            print(f"  - Số region: {len(path)}")
            print(f"  - Thứ tự: {[r + 1 for r in path]}")
            
            total_time = 0
            total_distance = 0
            
            for idx, region_id in enumerate(path):
                scan_time = self._compute_scan_time(uav_id, region_id)
                total_time += scan_time
                
                if idx < len(path) - 1:
                    next_region = path[idx + 1]
                    flight_time = self._compute_flight_time(uav_id, region_id, next_region)
                    distance = self.distance_matrix[region_id][next_region]
                    total_time += flight_time
                    total_distance += distance
            
            print(f"  - Tổng thời gian: {total_time:.2f}")
            print(f"  - Tổng quãng đường: {total_distance:.2f}")
        
        print(f"\nThời gian hoàn thành tổng thể: {self.best_cost:.2f}")
        print("="*60)


def main():
    # Load dữ liệu từ file sample.json
    with open('sample.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse dữ liệu
    uavs = [UAV(**uav_data) for uav_data in data['uavs_list']]
    regions = [Region(**region_data) for region_data in data['regions_list']]
    v_matrix = np.array(data['V_matrix'])
    
    print(f"Số UAVs: {len(uavs)}")
    print(f"Số Regions: {len(regions)}")
    
    # Khởi tạo và chạy Multi-Colony ACS
    mcacs = MultiColonyACS(
        uavs=uavs,
        regions=regions,
        v_matrix=v_matrix,
        num_ants_per_colony=10,
        max_iterations=100,
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        epsilon=0.1,
        q0=0.9,
        omega=0.5
    )
    
    # Giải bài toán
    solution, cost = mcacs.solve()
    
    # In chi tiết
    mcacs.print_solution_details()
    
    # Visualize
    mcacs.visualize_solution('solution_visualization.png')
    
    # Lưu kết quả
    result = {
        'best_cost': cost,
        'solution': {f'UAV_{k+1}': [r+1 for r in v] for k, v in solution.items()}
    }
    
    with open('solution.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print("\nĐã lưu kết quả vào solution.json")


if __name__ == "__main__":
    main()