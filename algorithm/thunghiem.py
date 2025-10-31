from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
import random
from utils.create_sample import create_sample 


# =============================
# Dataclasses
# =============================

@dataclass
class Region:
    id: int
    coords: Tuple[float, float]
    area: float

@dataclass
class UAV:
    id: int
    max_velocity: float     # m/s
    scan_width: float       # W_i (m)


# =============================
# Các hàm tiện ích
# =============================

def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def travel_time(uav: UAV, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Thời gian bay TF = distance / velocity."""
    if uav.max_velocity <= 0:
        return float("inf")
    return euclidean(p1, p2) / uav.max_velocity

def scan_time(uav: UAV, region: Region, V_matrix: List[List[float]]) -> float:
    """TS_{i,k} = A_k / (V_{i,k} * W_i)."""
    if region.id <= 0:
        return 0.0
    i = uav.id - 1
    j = region.id - 1
    if i < 0 or j < 0 or i >= len(V_matrix) or j >= len(V_matrix[0]):
        return float("inf")
    v_scan = V_matrix[i][j]
    if v_scan <= 0 or uav.scan_width <= 0:
        return float("inf")
    return region.area / (v_scan * uav.scan_width)

def route_cost_with_scan(
    uav: UAV,
    route: List[Region],
    base: Tuple[float, float],
    V_matrix: List[List[float]],
    return_to_base: bool = False,
) -> float:
    """Tổng thời gian = (bay giữa các điểm) + (quét mỗi vùng)."""
    time_sum = 0.0
    cur = base
    for r in route:
        time_sum += travel_time(uav, cur, r.coords)
        time_sum += scan_time(uav, r, V_matrix)
        cur = r.coords
    if return_to_base and route:
        time_sum += travel_time(uav, cur, base)
    return time_sum


# =============================
# Pha 2: Tối ưu thứ tự (NN + 2-Opt)
# =============================

def nearest_neighbor_order(points: List[Region], start: Tuple[float, float]) -> List[Region]:
    if not points:
        return []
    unused = points[:]
    first = min(unused, key=lambda r: euclidean(start, r.coords))
    route = [first]
    unused.remove(first)
    cur = first
    while unused:
        nxt = min(unused, key=lambda r: euclidean(cur.coords, r.coords))
        route.append(nxt)
        unused.remove(nxt)
        cur = nxt
    return route

def two_opt_once(route: List[Region]) -> Optional[List[Region]]:
    """Thực hiện 1 phép đảo 2-opt (tạo ứng viên)."""
    n = len(route)
    if n < 4:
        return None
    for i in range(0, n - 3):
        for k in range(i + 2, n - 1):
            new_route = route[:i+1] + list(reversed(route[i+1:k+1])) + route[k+1:]
            return new_route
    return None

def optimize_order_for_uav(
    uav: UAV,
    regions: List[Region],
    base: Tuple[float, float],
    V_matrix: List[List[float]],
    max_2opt_iters: int = 100,
) -> List[Region]:
    """Route = NN -> cải tiến bằng 2-opt nếu giảm cost."""
    route = nearest_neighbor_order(regions, base)
    if len(route) < 4:
        return route
    best_cost = route_cost_with_scan(uav, route, base, V_matrix, return_to_base=False)
    it = 0
    while it < max_2opt_iters:
        cand = two_opt_once(route)
        if cand is None:
            break
        cand_cost = route_cost_with_scan(uav, cand, base, V_matrix, return_to_base=False)
        if cand_cost + 1e-9 < best_cost:
            route, best_cost = cand, cand_cost
        else:
            # Không cải thiện -> xáo trộn nhẹ để thoát bẫy cục bộ
            random.shuffle(route)
        it += 1
    return route


# =============================
# Pha 1: Phân bổ vùng theo ETR
# =============================

def region_allocation_phase1(
    uavs: List[UAV],
    regions: List[Region],
    V_matrix: List[List[float]],
    base: Tuple[float, float],
) -> Dict[int, List[Region]]:
    """Greedy: luôn chọn UAV rảnh sớm nhất; gán vùng có ETR = TS/(TS+TF) cao nhất với UAV đó."""
    assigned: Dict[int, List[Region]] = {u.id: [] for u in uavs}
    finish_time: Dict[int, float] = {u.id: 0.0 for u in uavs}
    last_pos: Dict[int, Tuple[float, float]] = {u.id: base for u in uavs}

    remaining = regions[:]
    while remaining:
        u = min(uavs, key=lambda x: finish_time[x.id])
        cur_pos = last_pos[u.id]

        best_r = None
        best_score = -math.inf
        for r in remaining:
            ts = scan_time(u, r, V_matrix)
            tf = travel_time(u, cur_pos, r.coords)
            if math.isfinite(ts) and math.isfinite(tf) and (ts + tf) > 0:
                score = ts / (ts + tf)
                if score > best_score:
                    best_score = score
                    best_r = r

        if best_r is None:
            finish_time[u.id] = float("inf")
            continue

        assigned[u.id].append(best_r)
        finish_time[u.id] += travel_time(u, cur_pos, best_r.coords) + scan_time(u, best_r, V_matrix)
        last_pos[u.id] = best_r.coords
        remaining.remove(best_r)

        if all(math.isinf(ft) for ft in finish_time.values()) and remaining:
            raise RuntimeError("Không còn UAV nào có thể quét những vùng còn lại.")

    return assigned


# =============================
# Vòng lặp P1 <-> P2 + tái gán cục bộ
# =============================

def compute_uav_finish_times(
    uavs: List[UAV],
    assignment: Dict[int, List[Region]],
    V_matrix: List[List[float]],
    base: Tuple[float, float],
) -> Dict[int, float]:
    ft = {}
    for u in uavs:
        route = assignment.get(u.id, [])
        ft[u.id] = route_cost_with_scan(u, route, base, V_matrix, return_to_base=False)
    return ft

def clone_assignment(assignment: Dict[int, List[Region]]) -> Dict[int, List[Region]]:
    return {uid: lst[:] for uid, lst in assignment.items()}

def try_local_transfer(
    uavs: List[UAV],
    assignment: Dict[int, List[Region]],
    V_matrix: List[List[float]],
    base: Tuple[float, float],
    eps_gain: float,
) -> bool:
    """Thử chuyển 1 vùng từ UAV có makespan lớn nhất sang UAV khác nếu giảm makespan > eps_gain."""
    ft = compute_uav_finish_times(uavs, assignment, V_matrix, base)
    u_star_id = max(ft, key=lambda k: ft[k])
    u_star = next(u for u in uavs if u.id == u_star_id)

    if not assignment[u_star_id]:
        return False

    best_new_assignment = None
    best_new_makespan = max(ft.values())
    improved = False

    for r in assignment[u_star_id]:
        for u_other in uavs:
            if u_other.id == u_star_id:
                continue
            cand = {uid: lst[:] for uid, lst in assignment.items()}
            cand[u_star_id].remove(r)
            cand[u_other.id].append(r)

            # Tối ưu lại thứ tự cho 2 UAV bị ảnh hưởng
            cand[u_star_id] = optimize_order_for_uav(u_star, cand[u_star_id], base, V_matrix)
            cand[u_other.id] = optimize_order_for_uav(u_other, cand[u_other.id], base, V_matrix)

            # Đánh giá phương án
            ft_cand = compute_uav_finish_times(uavs, cand, V_matrix, base)
            ms = max(ft_cand.values())
            if ms + 1e-9 < best_new_makespan:
                best_new_makespan = ms
                best_new_assignment = cand
                improved = True

    if improved and (max(ft.values()) - best_new_makespan) > eps_gain:
        # Chấp nhận chuyển giao tốt nhất
        assignment.clear()
        assignment.update(best_new_assignment)  # type: ignore
        return True
    return False

def alternating_refinement(
    uavs: List[UAV],
    regions: List[Region],
    V_matrix: List[List[float]],
    base: Tuple[float, float] = (0.0, 0.0),
    max_iters: int = 20,
    eps_rel: float = 1e-3,
    local_gain_eps: float = 1e-6,
):
    """
    Ý tưởng mới: Sau khi thực hiện pha 2, quay lại tối ưu pha 1
    Trả về: (assignment, finish_times, makespan_seconds)
    """
    # Pha 1 ban đầu
    assignment = region_allocation_phase1(uavs, regions, V_matrix, base)

    # Pha 2 ban đầu
    for u in uavs:
        assignment[u.id] = optimize_order_for_uav(u, assignment[u.id], base, V_matrix)

    ft = compute_uav_finish_times(uavs, assignment, V_matrix, base)
    best_ms = max(ft.values())
    best_assignment = clone_assignment(assignment)
    best_ft = ft.copy()

    for _ in range(max_iters):
        # Thực hiện pha 2: Chuyển giao vùng giữa các UAV
        try_local_transfer(uavs, assignment, V_matrix, base, eps_gain=local_gain_eps)

        # Tối ưu lại thứ tự sau khi có chuyển giao
        for u in uavs:
            assignment[u.id] = optimize_order_for_uav(u, assignment[u.id], base, V_matrix)

        # Quay lại tối ưu pha 1: Phân bổ lại vùng dựa trên assignment hiện tại
        all_regions = [r for routes in assignment.values() for r in routes]
        # new_assignment = region_allocation_phase1(uavs, all_regions, V_matrix, base)
        new_assignment=assignment
        # Sau khi phân bổ lại, tối ưu thứ tự (pha 2)
        for u in uavs:
            new_assignment[u.id] = optimize_order_for_uav(u, new_assignment[u.id], base, V_matrix)

        new_ft = compute_uav_finish_times(uavs, new_assignment, V_matrix, base)
        ms = max(new_ft.values())
        improvement = best_ms - ms
        threshold = eps_rel * max(1.0, best_ms)

        if improvement < 0:
            break

        assignment = new_assignment
        ft = new_ft

        if improvement > threshold:
            best_assignment = clone_assignment(assignment)
            best_ft = ft.copy()
            best_ms = ms
            continue

        best_assignment = clone_assignment(assignment)
        best_ft = ft.copy()
        best_ms = ms
        break

    return best_assignment, best_ft, best_ms

def baseline_two_phase(
    uavs: List[UAV],
    regions: List[Region],
    V_matrix: List[List[float]],
    base: Tuple[float, float] = (0.0, 0.0),
):
    """
    'Code cũ': chỉ Pha 1 + Pha 2 (không tái gán cục bộ).
    Trả về: (assignment, finish_times, makespan_seconds)
    """
    assignment = region_allocation_phase1(uavs, regions, V_matrix, base)
    for u in uavs:
        assignment[u.id] = optimize_order_for_uav(u, assignment[u.id], base, V_matrix)
    ft = compute_uav_finish_times(uavs, assignment, V_matrix, base)
    return assignment, ft, max(ft.values())


# =============================
# Helpers chuyển đổi từ create_sample.py
# =============================

def _as_tuple_xy(x) -> Tuple[float, float]:
    return (float(x[0]), float(x[1]))

def build_from_sample_dict(sample: Dict) -> Tuple[List[UAV], List[Region], List[List[float]]]:
    """
    Chuyển dict do create_sample.create_sample(...) trả về
    thành (uavs_list, regions_list, V_matrix) theo dataclass hiện tại.
    """
    uavs_dict_list = sample["uavs_list"]
    regions_dict_list = sample["regions_list"]
    V_matrix = sample["V_matrix"]

    uavs = [UAV(int(d["id"]), float(d["max_velocity"]), float(d["scan_width"])) for d in uavs_dict_list]
    regions = [Region(int(d["id"]), _as_tuple_xy(d["coords"]), float(d["area"])) for d in regions_dict_list]
    return uavs, regions, V_matrix


# =============================
# Demo + Line chart (Oy = thời gian hoàn thành tác vụ - PHÚT)
# =============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import csv
    import statistics as stats

    # Cấu hình chung
    NUM_UAVS = 4
    SYSTEM_AREA_RATIO = 0.02  # giống thiết lập hình 4 trong paper (u = 0.02)
    SYSTEM_DRAG_FACTOR = 0.9  # d = 0.9
    REGION_GRID = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # số vùng
    NUM_TRIALS = 10  # số lần lặp để lấy trung bình (bạn có thể tăng lên 100 cho sát paper)
    SYSTEM_DRAG_FACTOR_GRID = [i / 10 for i in range(1, 10)]
    NUM_UAV_GRID =[i for i in range(2, 10)]
    old_means_min = []
    new_means_min = []

    # Lưu thêm độ lệch chuẩn nếu cần
    old_stds_min = []
    new_stds_min = []

    # for m in REGION_GRID:
    for uavs in NUM_UAV_GRID:
        print(uavs)
        old_trials_min = []
        new_trials_min = []

        for _ in range(NUM_TRIALS):
            sample = create_sample(
                NUM_UAVS=uavs,
                NUM_REGIONS=50,
                SYSTEM_AREA_RATIO=SYSTEM_AREA_RATIO,
                SYSTEM_DRAG_FACTOR=0.9,
            )
            uavs, regions, V_matrix = build_from_sample_dict(sample)

            # Code cũ (seconds)
            _, _, base_ms_sec = baseline_two_phase(uavs, regions, V_matrix, base=(0.0, 0.0))
            # Code mới (seconds)
            _, _, new_ms_sec  = alternating_refinement(uavs, regions, V_matrix, base=(0.0, 0.0),
                                                       max_iters=10, eps_rel=1e-3, local_gain_eps=1e-6)
            # Đổi sang phút cho line chart
            old_trials_min.append(base_ms_sec / 60.0)
            new_trials_min.append(new_ms_sec  / 60.0)

        old_means_min.append(sum(old_trials_min) / len(old_trials_min))
        new_means_min.append(sum(new_trials_min) / len(new_trials_min))


    # ===== Vẽ LINE CHART (Oy = phút, 1 plot, không set màu) =====
    plt.figure(figsize=(7, 4.5))
    plt.plot(NUM_UAV_GRID, old_means_min, marker="o", label="Cũ ")
    plt.plot(NUM_UAV_GRID, new_means_min, marker="s", label="Mới ")
    plt.title("Thời gian hoàn thành tác vụ theo số vùng (phút)")
    plt.xlabel("System drag factor")
    plt.ylabel("Thời gian hoàn thành tác vụ (phút)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./fig/thunghiem.png")
