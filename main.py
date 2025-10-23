# -*- coding: utf-8 -*-
import random
import numpy as np
import time
from create_sample import create_sample
import matplotlib.pyplot as plt
import tqdm
from appa import region_allocation, OrderOptimizerACS
from dataclasses import dataclass
from typing import Tuple
from mcaco import MultiColonyACS
from optimization_appa import iterative_appa_with_gradient


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
        # Chiều rộng cột đầu lớn hơn
        colWidths=[0.1] + [0.09] * (len(headers)-1),
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


def appa_run_sample(data):
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    assignments = region_allocation(uavs_list, regions_list, V_matrix)

    last_finish_time = 0
    for uav in uavs_list:
        regs = assignments[uav.id]
        if regs:
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
    return last_finish_time


def mcaco_run_sample(data):
    uavs = [UAV(**uav_data) for uav_data in data['uavs_list']]
    regions = [Region(**region_data) for region_data in data['regions_list']]
    v_matrix = np.array(data['V_matrix'])
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
    solution, cost = mcacs.solve()
    return cost


def optimization_run_sample(data):
    uavs = [UAV(**uav_data) for uav_data in data['uavs_list']]
    regions = [Region(**region_data) for region_data in data['regions_list']]
    V_matrix = np.array(data['V_matrix'])

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

    return float(history[-1])


def estimate_run_time(NUM_UAVS=4, NUM_REGIONS=50):
    random.seed()
    np.random.seed()
    sample_data = create_sample(NUM_UAVS=NUM_UAVS, NUM_REGIONS=NUM_REGIONS)
    start_time = time.perf_counter()
    appa_run_sample(sample_data)
    end_time = time.perf_counter()
    return end_time - start_time


def time_statistic_base_on_minutes(num_uavs=4, num_regions=50, u=0.02, d=0.9):
    random.seed()
    np.random.seed()
    sample_data = create_sample(
        NUM_UAVS=num_uavs, NUM_REGIONS=num_regions, SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)
    return appa_run_sample(sample_data) / 60


def benchmark_run_time(num_uavs=4, num_regions=50):
    random.seed()
    np.random.seed()
    sample_data = create_sample(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions)
    appa_start_time = time.perf_counter()
    appa_run_sample(sample_data)
    appa_end_time = time.perf_counter()

    mcaco_start_time = time.perf_counter()
    mcaco_run_sample(sample_data)
    mcaco_end_time = time.perf_counter()
    return [appa_end_time - appa_start_time, mcaco_end_time - mcaco_start_time]


def benchmark_time_cost_base_on_minutes(num_uavs=4, num_regions=50, u=0.02, d=0.9):
    random.seed()
    np.random.seed()
    sample_data = create_sample(
        NUM_UAVS=num_uavs, NUM_REGIONS=num_regions, SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)
    return [appa_run_sample(sample_data) / 60, mcaco_run_sample(sample_data) / 60]


def benchmark_optimization(num_uavs=4, num_regions=50, u=0.02, d=0.9):
    random.seed()
    np.random.seed()
    sample = create_sample(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions,
                           SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)

    return [appa_run_sample(sample), optimization_run_sample(sample)]


def main():
    # headers = ['Number', 'APPA']
    # rows = []
    # x_points = []
    # y_points = []
    # for try_time in tqdm.tqdm(range(0, 100), desc="Fig 2", position=0):
    #     for num_regions in range(5, 55, 5):
    #         if try_time == 0:
    #             x_points.append(num_regions)
    #             y_points.append(estimate_run_time(NUM_REGIONS=num_regions))
    #         else:
    #             y_points[num_regions // 5 -
    #                      1] += estimate_run_time(NUM_REGIONS=num_regions)

    # for index, value in enumerate(y_points):
    #     y_points[index] /= 100
    #     rows.append([index * 5 + 5, value])
    # plt.plot(x_points, y_points, marker='o', linestyle='-', label='APPA')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.xlabel("Number of regions", fontsize=12)
    # plt.ylabel("Execution time (s)", fontsize=12)
    # plt.savefig('./fig/fig_2.png')
    # plt.close()
    # create_table_image(headers=headers, data=rows,
    #                    filename='./table/table_1.png')

    # markers = ['o', 'D', '^', 'x', '+']
    # record = [[] for _ in range(10)]

    # for num_uavs in range(2, 12, 2):
    #     x_points = []
    #     y_points = []

    #     for try_time in tqdm.tqdm(range(0, 100), desc=f"Fig 3, num uavs={num_uavs}", position=0):
    #         for num_regions in range(5, 55, 5):
    #             tmp = estimate_run_time(
    #                 NUM_UAVS=num_uavs, NUM_REGIONS=num_regions)
    #             if try_time == 0:
    #                 x_points.append(num_regions)
    #                 y_points.append(tmp)
    #             else:
    #                 y_points[num_regions // 5 - 1] += tmp

    #             if num_uavs == 4:
    #                 record[num_regions // 5 - 1].append(tmp)

    #     for index, _ in enumerate(y_points):
    #         y_points[index] /= 100

    #     plt.plot(x_points, y_points,
    #              marker=markers[num_uavs // 2 - 1], linestyle='-', label=f'n={num_uavs}')

    # plt.xlabel("Number of regions", fontsize=12)
    # plt.ylabel("Execution time (s)", fontsize=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/fig_3.png')
    # plt.close()

    # headers = ['Region number'] + [i for i in range(5, 55, 5)]
    # rows = [
    #     ['Minimum time (s)'] + [np.min(record[i]) for i in range(len(record))],
    #     ['Average time (s)'] + [np.mean(record[i])
    #                             for i in range(len(record))],
    #     ['Maximum time (s)'] + [np.max(record[i]) for i in range(len(record))],
    #     ['Standard deviation'] + [np.std(record[i])
    #                               for i in range(len(record))]
    # ]
    # create_table_image(headers=headers, data=rows,
    #                    filename='./table/table_2.png')

    # headers = ['Number', 'APPA']
    # rows = []
    # x_points = []
    # y_points = []
    # for try_time in tqdm.tqdm(range(0, 100), desc="Table 3", position=0):
    #     for num_regions in range(5, 55, 5):
    #         if try_time == 0:
    #             x_points.append(num_regions)
    #             y_points.append(time_statistic_base_on_minutes(
    #                 num_regions=num_regions, u=0.02, d=0.9))
    #         else:
    #             y_points[num_regions // 5 - 1] += time_statistic_base_on_minutes(
    #                 num_regions=num_regions, u=0.02, d=0.9)

    # for index, value in enumerate(y_points):
    #     y_points[index] /= 100
    #     rows.append([index * 5 + 5, value])
    # plt.plot(x_points, y_points, marker='o', linestyle='-', label='APPA')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.xlabel("Number of regions", fontsize=12)
    # plt.ylabel("Task completion time (s)", fontsize=12)
    # plt.savefig('./fig/fig_4.png')
    # plt.close()
    # create_table_image(headers=headers, data=rows,
    #                    filename='./table/table_3.png')

    # x_points = []
    # y_points = []
    # for try_time in tqdm.tqdm(range(0, 100), desc="Fig 5", position=0):
    #     for system_drag_factor in range(1, 10):
    #         tmp = time_statistic_base_on_minutes(
    #             num_regions=15, u=0.01, d=system_drag_factor / 10)
    #         if try_time == 0:
    #             x_points.append(system_drag_factor / 10)
    #             y_points.append(tmp)
    #         else:
    #             y_points[system_drag_factor - 1] += tmp

    # for index, _ in enumerate(y_points):
    #     y_points[index] /= 100

    # plt.plot(x_points, y_points, marker='o', linestyle='-', label='APPA')
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.xlabel("System drag factor", fontsize=12)
    # plt.ylabel("Task completion time (s)", fontsize=12)
    # plt.savefig('./fig/fig_5.png')
    # plt.close()

    # max_loop = 10
    # x_points = []
    # appa_y_points = []
    # mcaco_y_points = []
    # for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark run time", position=0):
    #     for num_regions in range(5, 55, 5):
    #         appa_y_point, mcaco_y_point = benchmark_run_time(num_regions=num_regions)
    #         if try_time == 0:
    #             x_points.append(num_regions)
    #             appa_y_points.append(appa_y_point)
    #             mcaco_y_points.append(mcaco_y_point)
    #         else:
    #             appa_y_points[num_regions // 5 -1] += appa_y_point
    #             mcaco_y_points[num_regions // 5 - 1] += mcaco_y_point
    # for index, _ in enumerate(appa_y_points):
    #     appa_y_points[index] /= max_loop
    # for index, _ in enumerate(mcaco_y_points):
    #     mcaco_y_points[index] /= max_loop
    # plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    # plt.plot(x_points, mcaco_y_points, marker='x', linestyle='-', label='mcaco')
    # plt.xlabel("Number of regions", fontsize=12)
    # plt.ylabel("Execution time (s)", fontsize=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/benchmark_runtime.png')
    # plt.close()

    # x_points = []
    # appa_y_points = []
    # mcaco_y_points = []
    # for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark time cost", position=0):
    #     for num_regions in range(5, 55, 5):
    #         appa_y_point, mcaco_y_point = benchmark_time_cost_base_on_minutes(num_regions=num_regions)
    #         if try_time == 0:
    #             x_points.append(num_regions)
    #             appa_y_points.append(appa_y_point)
    #             mcaco_y_points.append(mcaco_y_point)
    #         else:
    #             appa_y_points[num_regions // 5 -1] += appa_y_point
    #             mcaco_y_points[num_regions // 5 - 1] += mcaco_y_point
    # for index, _ in enumerate(appa_y_points):
    #     appa_y_points[index] /= max_loop
    # for index, _ in enumerate(mcaco_y_points):
    #     mcaco_y_points[index] /= max_loop
    # plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    # plt.plot(x_points, mcaco_y_points, marker='x', linestyle='-', label='mcaco')
    # plt.xlabel("Number of regions", fontsize=12)
    # plt.ylabel("Time cost (minutes)", fontsize=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/benchmark_time_cost.png')
    # plt.close()

    max_loop = 100
    
    x_points = []
    appa_y_points = []
    optimization_y_points = []
    for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - uav", position=0):
        for uav in range(2, 10):
            appa_y_point, optimizaion_y_point = benchmark_optimization(
                num_uavs=uav)
            if try_time == 0:
                x_points.append(uav)
                appa_y_points.append(appa_y_point)
                optimization_y_points.append(optimizaion_y_point)
            else:
                appa_y_points[uav- 2] += appa_y_point
                optimization_y_points[uav - 2] += optimizaion_y_point
    for index, _ in enumerate(appa_y_points):
        appa_y_points[index] /= max_loop
    for index, _ in enumerate(optimization_y_points):
        optimization_y_points[index] /= max_loop
    plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    plt.plot(x_points, optimization_y_points, marker='x',
             linestyle='-', label='optimization')
    plt.xlabel("Number of uavs", fontsize=12)
    plt.ylabel("Time cost (s)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.savefig('./fig/benchmark_optimization_uav.png')
    plt.close()

    x_points = []
    appa_y_points = []
    optimization_y_points = []
    for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - system drag factor", position=0):
        for system_drag_factor in range(1, 10):
            appa_y_point, optimizaion_y_point = benchmark_optimization(
                d=system_drag_factor/10)
            if try_time == 0:
                x_points.append(system_drag_factor / 10)
                appa_y_points.append(appa_y_point)
                optimization_y_points.append(optimizaion_y_point)
            else:
                appa_y_points[system_drag_factor - 1] += appa_y_point
                optimization_y_points[system_drag_factor - 1] += optimizaion_y_point
    for index, _ in enumerate(appa_y_points):
        appa_y_points[index] /= max_loop
    for index, _ in enumerate(optimization_y_points):
        optimization_y_points[index] /= max_loop
    plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    plt.plot(x_points, optimization_y_points, marker='x',
             linestyle='-', label='optimization')
    plt.xlabel("System drag factor", fontsize=12)
    plt.ylabel("Time cost (s)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.savefig('./fig/benchmark_optimization_sdf.png')
    plt.close()

    x_points = []
    appa_y_points = []
    optimization_y_points = []
    for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - region", position=0):
        for num_regions in range(5, 55, 5):
            appa_y_point, optimizaion_y_point = benchmark_optimization(
                num_regions=num_regions)
            if try_time == 0:
                x_points.append(num_regions)
                appa_y_points.append(appa_y_point)
                optimization_y_points.append(optimizaion_y_point)
            else:
                appa_y_points[num_regions // 5 - 1] += appa_y_point
                optimization_y_points[num_regions //
                                      5 - 1] += optimizaion_y_point
    for index, _ in enumerate(appa_y_points):
        appa_y_points[index] /= max_loop
    for index, _ in enumerate(optimization_y_points):
        optimization_y_points[index] /= max_loop
    plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    plt.plot(x_points, optimization_y_points, marker='x',
             linestyle='-', label='optimization')
    plt.xlabel("Number of regions", fontsize=12)
    plt.ylabel("Time cost (s)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.savefig('./fig/benchmark_optimization_region.png')
    plt.close()

if __name__ == "__main__":
    main()
