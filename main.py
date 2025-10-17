# -*- coding: utf-8 -*-
import math
import random
import numpy as np
from dataclasses import dataclass
import json
from typing import List, Tuple, Dict
import time
from create_sample import create_sample
import matplotlib.pyplot as plt
import tqdm

# ==============================
# Cấu trúc dữ liệu
# ==============================

@dataclass
class Region:
    id: int
    coords: Tuple[float, float]
    area: float

@dataclass
class UAV:
    id: int
    max_velocity: float  # V_max_i
    scan_width: float    # W_i


# ==============================
# Hàm tiện ích
# ==============================

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ts(uav: UAV, region: Region, V_matrix: List[List[float]]) -> float:
    """
    TS_i,j = A_j / (V_i,j * W_i)  (Eq. (1) của paper: nếu V_i,j = 0 thì TS = +inf)
    """
    if region.id == -1:  # base (không có area)
        return 0.0
    uav_idx = uav.id - 1
    region_idx = region.id - 1
    scan_velocity = V_matrix[uav_idx][region_idx]  # V_i,j
    if scan_velocity <= 0 or uav.scan_width <= 0:
        return float('inf')
    return region.area / (scan_velocity * uav.scan_width)

def calculate_tf(uav: UAV, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    TF_i,j,k = D_j,k / V_max_i  (Eq. (2))
    """
    if uav.max_velocity <= 0:
        return float('inf')
    distance = calculate_distance(p1, p2)
    return distance / uav.max_velocity


# ==============================
# Giai đoạn 1: Phân bổ vùng (Algorithm 1)
# ==============================

def region_allocation(uavs: List[UAV], regions: List[Region], V_matrix: List[List[float]],
                      base_coords: Tuple[float, float] = (0.0, 0.0),
                      verbose: bool = True) -> Dict[int, List[Region]]:
    """
    Phân bổ từng vùng cho UAV theo 'tỷ lệ thời gian hiệu quả' (ETR = TS / (TS + TF)) – Eq. (9)
    và luôn gán cho UAV có 'finish_time' nhỏ nhất tại thời điểm đó – Algorithm 1 (bước 4–10).
    """
    # if verbose:
    #     print("--- Bắt đầu Giai đoạn 1: Phân bổ khu vực ---")

    unassigned = regions.copy()
    assigned: Dict[int, List[Region]] = {u.id: [] for u in uavs}
    finish_times: Dict[int, float] = {u.id: 0.0 for u in uavs}
    last_coords: Dict[int, Tuple[float, float]] = {u.id: base_coords for u in uavs}

    while unassigned:
        # UAV dự kiến hoàn tất sớm nhất
        earliest_uav_id = min(finish_times, key=finish_times.get)
        current_uav = next(u for u in uavs if u.id == earliest_uav_id)

        best_region = None
        best_etr = -1.0

        for r in unassigned:
            ts = calculate_ts(current_uav, r, V_matrix)
            if math.isinf(ts):
                continue  # UAV này không quét được vùng r
            tf = calculate_tf(current_uav, last_coords[current_uav.id], r.coords)
            etr = ts / (ts + tf) if (ts + tf) > 0 else 0.0  # Eq. (9) dạng TR = TS/(TF+TS)
            if etr > best_etr:
                best_etr = etr
                best_region = r

        # Nếu UAV hiện tại không quét được vùng nào còn lại, chuyển cho UAV khác
        if best_region is None:
            # Tìm UAV khác có thể quét được ít nhất một vùng còn lại
            assigned_flag = False
            for u in sorted(uavs, key=lambda x: finish_times[x.id]):
                cand = None
                cand_etr = -1.0
                for r in unassigned:
                    ts = calculate_ts(u, r, V_matrix)
                    if math.isinf(ts):
                        continue
                    tf = calculate_tf(u, last_coords[u.id], r.coords)
                    etr = ts / (ts + tf) if (ts + tf) > 0 else 0.0
                    if etr > cand_etr:
                        cand_etr, cand = etr, r
                if cand is not None:
                    # gán cho UAV u
                    ts = calculate_ts(u, cand, V_matrix)
                    tf = calculate_tf(u, last_coords[u.id], cand.coords)
                    finish_times[u.id] += (ts + tf)
                    last_coords[u.id] = cand.coords
                    assigned[u.id].append(cand)
                    unassigned.remove(cand)
                    assigned_flag = True
                    # if verbose:
                    #     print(f"Gán khu vực {cand.id} cho UAV {u.id}. ETR = {cand_etr:.4f}")
                    break
            if not assigned_flag:
                raise RuntimeError("Không còn UAV nào có thể quét các vùng còn lại (V_matrix không khả thi).")
        else:
            # gán cho UAV 'earliest'
            ts = calculate_ts(current_uav, best_region, V_matrix)
            tf = calculate_tf(current_uav, last_coords[current_uav.id], best_region.coords)
            finish_times[current_uav.id] += (ts + tf)
            last_coords[current_uav.id] = best_region.coords
            assigned[current_uav.id].append(best_region)
            unassigned.remove(best_region)
            # if verbose:
            #     print(f"Gán khu vực {best_region.id} cho UAV {current_uav.id}. ETR = {best_etr:.4f}")

    # if verbose:
    #     print("--- Kết thúc Giai đoạn 1 ---\n")
    return assigned


# ==============================
# Giai đoạn 2: Tối ưu thứ tự bằng ACS (Algorithm 2)
# ==============================

class OrderOptimizerACS:
    """
    Hiện thực ACS đúng theo paper:
    - Heuristic: eta(i,j) = 1/D(i,j) (Eq. (13))
    - Khởi tạo pheromone: tau0 = 1 / (m * L_nn) với m = số vùng (exclude base) (Eq. (12))
    - Quy tắc chọn: nếu q <= q0 => argmax tau^alpha * eta^beta; else => Roulette (Eq. (10)-(11))
    - Cập nhật cục bộ: tau = (1 - rho) * tau + rho * tau0 (Eq. (14))
    - Cập nhật toàn cục trên best path: tau = (1 - eps) * tau + eps * (1/L_best) (Eq. (15)-(16))
    - Thời gian tour: sum(TF + TS) theo Eq. (7), mặc định KHÔNG cộng thời gian quay về base
      (theo mô tả thực nghiệm trong paper).
    """

    def __init__(self,
                 uav: UAV,
                 regions: List[Region],
                 V_matrix: List[List[float]],
                 base_coords: Tuple[float, float] = (0.0, 0.0),
                 n_ants: int = 10,
                 n_generations: int = 50,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 epsilon: float = 0.1,
                 q0: float = 0.9,
                 include_return_to_base: bool = False,
                 rng_seed: int = None):
        self.uav = uav
        # điểm 0 là base (id=-1); sau đó là các vùng được gán cho UAV
        self.points: List[Region] = [Region(id=-1, coords=base_coords, area=0.0)] + regions
        self.V_matrix = V_matrix
        self.n_points = len(self.points)  # = m + 1 (tính cả base)

        self.n_ants = n_ants
        self.n_generations = n_generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        self.q0 = q0
        self.include_return_to_base = include_return_to_base

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        self._initialize()

    def _calculate_nearest_neighbour_tour(self) -> Tuple[List[int], float]:
        """
        L_nn dùng cho Eq. (12). Tính trên đồ thị khoảng cách Euclid (như paper),
        chỉ trên các NODES gồm base + vùng (đúng định nghĩa).
        """
        current_idx = 0  # bắt đầu từ base
        tour = [current_idx]
        unvisited = list(range(1, self.n_points))
        tour_length = 0.0

        while unvisited:
            nearest = min(unvisited, key=lambda j: self.dist_matrix[current_idx, j])
            tour_length += self.dist_matrix[current_idx, nearest]
            current_idx = nearest
            tour.append(current_idx)
            unvisited.remove(current_idx)

        # không cần quay về base cho L_nn theo paper (dùng như hằng số khởi tạo)
        return tour, tour_length

    def _initialize(self):
        # Ma trận khoảng cách + heuristic
        self.dist_matrix = np.zeros((self.n_points, self.n_points), dtype=float)
        for i in range(self.n_points):
            for j in range(self.n_points):
                self.dist_matrix[i, j] = calculate_distance(self.points[i].coords, self.points[j].coords)

        self.eta = 1.0 / (self.dist_matrix + 1e-12)  # heuristic (Eq. (13)); tránh chia 0 ở đường chéo

        # Khởi tạo pheromone theo Eq. (12) với m = số vùng (exclude base)
        _, L_nn = self._calculate_nearest_neighbour_tour()
        m = self.n_points - 1  # số vùng (không tính base)
        if m <= 0 or L_nn <= 0:
            # trường hợp biên: 0–1 vùng => đặt giá trị an toàn
            initial_pheromone = 1.0
        else:
            initial_pheromone = 1.0 / (m * L_nn)

        self.tau0 = initial_pheromone
        self.tau = np.full((self.n_points, self.n_points), initial_pheromone, dtype=float)

    def _calculate_tour_time(self, tour: List[int]) -> float:
        """
        Thời gian tour theo Eq. (7): sum_{edges} (TF + TS(điểm đến)).
        Mặc định KHÔNG cộng TF quay về base (theo phần thực nghiệm của paper).
        """
        total_time = 0.0
        # nếu cần tính quay về base, thêm cạnh cuối -> base
        path = list(tour)
        if self.include_return_to_base and (tour[-1] != 0):
            path = tour + [0]

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            p1, p2 = self.points[a], self.points[b]
            tf = calculate_tf(self.uav, p1.coords, p2.coords)
            ts = calculate_ts(self.uav, p2, self.V_matrix)  # TS của điểm đến
            total_time += tf + ts
        return total_time

    def _select_next_region(self, current_idx: int, unvisited: List[int]) -> int:
        """
        Quy tắc chọn theo Eq. (10)–(11)
        """
        q = random.random()
        attractiveness = (self.tau[current_idx, unvisited] ** self.alpha) * (self.eta[current_idx, unvisited] ** self.beta)
        if q <= self.q0:
            return unvisited[int(np.argmax(attractiveness))]
        else:
            probs = attractiveness / np.sum(attractiveness)
            return int(np.random.choice(unvisited, p=probs))

    def run(self) -> Tuple[List[int], float]:
        best_tour, best_time = None, float('inf')

        for gen in range(self.n_generations):
            for _ in range(self.n_ants):
                # Đặt kiến ngẫu nhiên trong các vùng (không phải base) – Algorithm 2 (bước 3)
                start_idx = random.choice(list(range(1, self.n_points)))
                tour = [start_idx]
                unvisited = list(range(1, self.n_points))
                unvisited.remove(start_idx)
                current = start_idx

                while unvisited:
                    nxt = self._select_next_region(current, unvisited)

                    # Cập nhật cục bộ – Eq. (14)
                    self.tau[current, nxt] = (1 - self.rho) * self.tau[current, nxt] + self.rho * self.tau0

                    tour.append(nxt)
                    unvisited.remove(nxt)
                    current = nxt

                # Thời gian (không cần thêm base ở đầu – ta sẽ thêm khi tính nếu cần)
                full_tour_for_timing = [0] + tour  # bắt đầu từ base theo định nghĩa thời gian
                tour_time = self._calculate_tour_time(full_tour_for_timing)

                if tour_time < best_time:
                    best_time = tour_time
                    best_tour = full_tour_for_timing

            # Cập nhật toàn cục – Eq. (15)–(16) (chỉ trên đường tốt nhất hiện tại)
            if best_tour is not None and best_time > 0:
                self.tau *= (1 - self.epsilon)
                delta = self.epsilon * (1.0 / best_time)
                for i in range(len(best_tour) - 1):
                    a, b = best_tour[i], best_tour[i + 1]
                    self.tau[a, b] += delta

        # Trả về dãy id các điểm theo tour tốt nhất (bao gồm base id=-1 ở đầu)
        best_ids = [self.points[i].id for i in best_tour] if best_tour is not None else []
        return best_ids, best_time

def create_table_image(headers, data, filename='academic_table.png', figsize=(12, 5)):
    """
    Tạo bảng với định dạng học thuật và lưu thành file ảnh.
    
    Args:
        headers (list): Danh sách các tiêu đề cột
        data (list): Danh sách các hàng dữ liệu
        filename (str): Tên file ảnh đầu ra
        figsize (tuple): Kích thước hình (chiều rộng, chiều cao) tính bằng inch
    """
    import os
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Chuyển dữ liệu thành chuỗi và định dạng số
    string_data = []
    for row in data:
        string_row = []
        for item in row:
            if isinstance(item, (int, float)):
                if abs(item) < 0.01:  # Cho số rất nhỏ
                    string_row.append(f"{item:.5f}")
                elif abs(item) < 0.1:  # Cho số nhỏ
                    string_row.append(f"{item:.3f}")
                else:  # Cho số thông thường
                    string_row.append(f"{item:.2f}")
            else:
                string_row.append(str(item))
        string_data.append(string_row)
    
    # Tạo bảng
    table = ax.table(
        cellText=string_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.1] + [0.09] * (len(headers)-1),  # Chiều rộng cột đầu lớn hơn
        bbox=[0, 0, 1, 1]
    )
    
    # Điều chỉnh kích thước font và bảng
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)  # Tăng chiều cao hàng
    
    # Định dạng toàn bộ bảng
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(0.5)  # Độ dày đường viền
        cell.set_text_props(wrap=True)  # Cho phép text wrap
        
        # Thêm đường viền ngang đậm ở đầu và cuối bảng
        if i == 0:  # Header
            cell.set_linewidth(1.0)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')  # Màu xám nhạt cho header
        
        # Đảm bảo các viền ngang đều hiển thị
        if i in [0, len(data)]:
            cell.visible_edges = 'BTRL'  # Tất cả các viền
        else:
            cell.visible_edges = 'BTRL'  # Tất cả các viền
    
    # Đường viền ngang đậm giữa header và nội dung
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_linewidth(1.0)
        
        # Đảm bảo hiển thị đường viền dưới header
        cell = table[(1, j)]
        cell.visible_edges = 'BTRL'
        cell.set_linewidth(0.8)
    
    # Lưu bảng
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
def run_sample(data):
    # file_path = 'sample.json'
    # try:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)

    #     uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    #     regions_list = [Region(**region) for region in data['regions_list']]
    #     V_matrix = data['V_matrix']
    # except FileNotFoundError:
    #     print(f"Lỗi: Không tìm thấy file tại '{file_path}'.")
    #     exit(1)
    # except json.JSONDecodeError:
    #     print(f"Lỗi: Nội dung trong file '{file_path}' không phải JSON hợp lệ.")
    #     exit(1)
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    assignments = region_allocation(uavs_list, regions_list, V_matrix)

    # print("Kết quả phân bổ:")
    # for uav_id, regs in assignments.items():
    #     print(f"  - UAV {uav_id}: {[r.id for r in regs]}")
    # print("-" * 50)

    last_finish_time = 0
    for uav in uavs_list:
        regs = assignments[uav.id]
        if regs:
            # print(f"\n--- Giai đoạn 2: Tối ưu hóa thứ tự cho UAV {uav.id} ---")
            optimizer = OrderOptimizerACS(
                uav=uav,
                regions=regs,
                V_matrix=V_matrix,
                base_coords=(0.0, 0.0),
                n_ants=10,
                n_generations=50,
                alpha=1.0,
                beta=2.0,
                rho=0.1,
                epsilon=0.1,
                q0=0.9,
                include_return_to_base=False,
                rng_seed=42
            )
            best_path, best_time = optimizer.run()
            last_finish_time = max(last_finish_time, best_time)
            # print(f"Lộ trình tối ưu (id): {best_path}")
            # print(f"Ước tính thời gian hoàn thành: {best_time:.4f} (đv: giây)")
            # print(f"--- Kết thúc tối ưu cho UAV {uav.id} ---")
    return last_finish_time
def estimate_run_time(NUM_UAVS = 4, NUM_REGIONS = 50):
    random.seed()
    np.random.seed()
    sample_data = create_sample(NUM_UAVS=NUM_UAVS, NUM_REGIONS=NUM_REGIONS)
    start_time = time.perf_counter()
    run_sample(sample_data)
    end_time = time.perf_counter()
    return end_time - start_time

def time_statistic_base_on_minutes(num_uavs=4, num_regions = 50, u = 0.02, d = 0.9):
    random.seed()
    np.random.seed()
    sample_data = create_sample(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions, SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)
    return run_sample(sample_data) / 60

def main():
    headers = ['Number', 'APPA']
    rows = []
    x_points = []
    y_points = []
    for try_time in tqdm.tqdm(range(0, 100), desc="Fig 2", position=0):
        for num_regions in range(5, 55, 5):
            if try_time == 0: 
                x_points.append(num_regions)
                y_points.append(estimate_run_time(NUM_REGIONS=num_regions))
            else:
                y_points[num_regions // 5 - 1] += estimate_run_time(NUM_REGIONS=num_regions)
    
    for index, value in enumerate(y_points):
        y_points[index] /= 100
        rows.append([index * 5 + 5, value])
    plt.plot(x_points, y_points, marker = 'o', linestyle='-', label='APPA')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.xlabel("Number of regions", fontsize=12)
    plt.ylabel("Execution time (s)", fontsize=12)
    plt.savefig('./fig/fig_2.png')
    create_table_image(headers=headers, data=rows, filename='./table/table_1.png')

    markers = ['o', 'D', '^', 'x', '+']
    record = [[] for _ in range(10)]
    
    for num_uavs in range(2, 12, 2):
        x_points = []
        y_points = []
        
        for try_time in tqdm.tqdm(range(0, 100), desc=f"Fig 3, num uavs={num_uavs}", position=0):
            for num_regions in range(5, 55, 5):
                tmp = estimate_run_time(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions)
                if try_time == 0: 
                    x_points.append(num_regions)
                    y_points.append(tmp)
                else:
                    y_points[num_regions // 5 - 1] += tmp
                    
                if num_uavs == 4:
                    record[num_regions // 5 -1].append(tmp)
        
        for index, _ in enumerate(y_points):
            y_points[index] /= 100
            
        plt.plot(x_points, y_points, marker=markers[num_uavs // 2 -1], linestyle='-', label=f'n={num_uavs}')
    
    plt.xlabel("Number of regions", fontsize=12)
    plt.ylabel("Execution time (s)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.savefig('./fig/fig_3.png')
    
    headers = ['Region number'] + [i for i in range(5, 55, 5)]
    rows = [
        ['Minimum time (s)'] + [np.min(record[i]) for i in range(len(record))],
        ['Average time (s)'] + [np.mean(record[i]) for i in range(len(record))],
        ['Maximum time (s)'] + [np.max(record[i]) for i in range(len(record))],
        ['Standard deviation'] + [np.std(record[i]) for i in range(len(record))]
    ]
    create_table_image(headers=headers, data=rows, filename='./table/table_2.png')
    
    headers = ['Number', 'APPA']
    rows = []
    x_points = []
    y_points = []
    for try_time in tqdm.tqdm(range(0, 100), desc="Table 3", position=0):
        for num_regions in range(5, 55, 5):
            if try_time == 0: 
                x_points.append(num_regions)
                y_points.append(time_statistic_base_on_minutes(num_regions=num_regions, u=0.02, d = 0.9))
            else:
                y_points[num_regions // 5 - 1] += time_statistic_base_on_minutes(num_regions=num_regions, u=0.02, d = 0.9)
    
    for index, value in enumerate(y_points):
        y_points[index] /= 100
        rows.append([index * 5 + 5, value])
    plt.plot(x_points, y_points, marker = 'o', linestyle='-', label='APPA')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.xlabel("Number of regions", fontsize=12)
    plt.ylabel("Task completion time (s)", fontsize=12)
    plt.savefig('./fig/fig_4.png')
    create_table_image(headers=headers, data=rows, filename='./table/table_3.png')
    
    x_points = []
    y_points = []
    for try_time in tqdm.tqdm(range(0, 100), desc="Fig 5", position=0):
        for system_drag_factor in range(1, 10):
            tmp = time_statistic_base_on_minutes(num_regions=15, u = 0.01, d = system_drag_factor / 10)
            if try_time == 0:
                x_points.append(system_drag_factor / 10)
                y_points.append(tmp)
            else:
                y_points[system_drag_factor - 1] += tmp
    
    for index, _ in enumerate(y_points):
        y_points[index] /= 100
    
    plt.plot(x_points, y_points, marker = 'o', linestyle='-', label='APPA')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.xlabel("System drag factor", fontsize=12)
    plt.ylabel("Task completion time (s)", fontsize=12)
    plt.savefig('./fig/fig_5.png')
if __name__ == "__main__":
    main()