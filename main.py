import os
import random
import numpy as np
import time
from utils.create_sample import create_sample
import matplotlib.pyplot as plt
import tqdm
from algorithm.appa import APPAAlgorithm
from dataclasses import dataclass
from typing import Tuple
from algorithm.mcaco import MultiColonyACS
from algorithm.optimization_appa import iterative_appa_with_gradient
from utils.config import *
from algorithm.q_learning_acs import q_learning_acs_run_optimized
from algorithm.ga import solve_ga
from algorithm.sdf import solve_sdf
from algorithm.stca import solve_stca_ne
from algorithm.ils import solve_ils
from algorithm.vns import solve_vns
from algorithm.memetic import solve_memetic


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
    appa = APPAAlgorithm(uavs_list, regions_list, V_matrix)
    result = appa.solve()
    return result['max_completion_time']


def ga_run_sample(data):
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']

    best_fintness, _ = solve_ga(uavs_list, regions_list, V_matrix)

    return best_fintness


def sdf_run_sample(data):
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']

    max_completion_time, _ = solve_sdf(uavs_list, regions_list, V_matrix)
    return max_completion_time


def stca_run_sample(data):
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']

    max_completion_time, _ = solve_stca_ne(uavs_list, regions_list, V_matrix)
    return max_completion_time


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


def ils_run_sample(data):
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    
    best_fitness, _ = solve_ils(uavs_list, regions_list, V_matrix)
    
    return best_fitness


def ils_enhanced_run_sample(data):
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    
    best_fitness, _ = solve_ils_enhanced(uavs_list, regions_list, V_matrix)
    
    return best_fitness


def vns_run_sample(data):
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    
    best_fitness, _ = solve_vns(uavs_list, regions_list, V_matrix)
    
    return best_fitness


def memetic_run_sample(data):
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    
    best_fitness, _ = solve_memetic(uavs_list, regions_list, V_matrix)
    
    return best_fitness


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
    sample_data = create_sample(NUM_UAVS=NUM_UAVS, NUM_REGIONS=NUM_REGIONS)
    start_time = time.perf_counter()
    appa_run_sample(sample_data)
    end_time = time.perf_counter()
    return end_time - start_time


def time_statistic_base_on_minutes(num_uavs=4, num_regions=50, u=0.02, d=0.9):
    sample_data = create_sample(
        NUM_UAVS=num_uavs, NUM_REGIONS=num_regions, SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)
    return appa_run_sample(sample_data) / 60


def benchmark_run_time(num_uavs=4, num_regions=50):
    sample_data = create_sample(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions)
    appa_start_time = time.perf_counter()
    appa_run_sample(sample_data)
    appa_end_time = time.perf_counter()

    mcaco_start_time = time.perf_counter()
    mcaco_run_sample(sample_data)
    mcaco_end_time = time.perf_counter()
    return [appa_end_time - appa_start_time, mcaco_end_time - mcaco_start_time]


def benchmark_time_cost_base_on_minutes(num_uavs=4, num_regions=50, u=0.02, d=0.9):
    sample_data = create_sample(
        NUM_UAVS=num_uavs, NUM_REGIONS=num_regions, SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)
    return [appa_run_sample(sample_data) / 60, mcaco_run_sample(sample_data) / 60]


def benchmark_optimization(num_uavs=4, num_regions=50, u=0.02, d=0.9):
    sample = create_sample(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions,
                           SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)

    return [appa_run_sample(sample) / 60, optimization_run_sample(sample) / 60]


def benchmark_q_learning_acs(num_uavs=4, num_regions=50, u=0.02, d=0.9):
    sample = create_sample(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions,
                           SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)
    return [appa_run_sample(sample) / 60, q_learning_acs_run_optimized(sample)['makespan'] / 60]


def benchmark_all_run_time(num_uavs=4, num_regions=50, u=0.02, d=0.9):
    sample = create_sample(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions,
                           SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)

    ils_start_time = time.perf_counter()
    memetic_run_sample(sample)
    ils_end_time = time.perf_counter()

    appa_start_time = time.perf_counter()
    appa_run_sample(sample)
    appa_end_time = time.perf_counter()

    ga_start_time = time.perf_counter()
    ga_run_sample(sample)
    ga_end_time = time.perf_counter()

    return [ils_end_time - ils_start_time, appa_end_time - appa_start_time, ga_end_time - ga_start_time]


def benchmark_all(num_uavs=4, num_regions=50, u=0.05, d=0.9):
    sample = create_sample(NUM_UAVS=num_uavs, NUM_REGIONS=num_regions,
                           SYSTEM_AREA_RATIO=u, SYSTEM_DRAG_FACTOR=d)
    return [memetic_run_sample(sample), appa_run_sample(sample) / 60, ga_run_sample(sample)/60]


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

    # * benchmark optimize appa
    # x_points = []
    # appa_y_points = []
    # optimization_y_points = []
    # for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - uav", position=0):
    #     for uav in range(2, 10):
    #         appa_y_point, optimizaion_y_point = benchmark_optimization(
    #             num_uavs=uav)
    #         if try_time == 0:
    #             x_points.append(uav)
    #             appa_y_points.append(appa_y_point)
    #             optimization_y_points.append(optimizaion_y_point)
    #         else:
    #             appa_y_points[uav- 2] += appa_y_point
    #             optimization_y_points[uav - 2] += optimizaion_y_point
    # for index, _ in enumerate(appa_y_points):
    #     appa_y_points[index] /= max_loop
    # for index, _ in enumerate(optimization_y_points):
    #     optimization_y_points[index] /= max_loop
    # plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    # plt.plot(x_points, optimization_y_points, marker='x',
    #          linestyle='-', label='optimization')
    # plt.xlabel("Number of uavs", fontsize=12)
    # plt.ylabel("Time cost (m)", fontsize=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/benchmark_optimization_uav.png')
    # plt.close()

    # x_points = []
    # appa_y_points = []
    # optimization_y_points = []
    # for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - system drag factor", position=0):
    #     for system_drag_factor in range(1, 10):
    #         appa_y_point, optimizaion_y_point = benchmark_optimization(
    #             d=system_drag_factor/10)
    #         if try_time == 0:
    #             x_points.append(system_drag_factor / 10)
    #             appa_y_points.append(appa_y_point)
    #             optimization_y_points.append(optimizaion_y_point)
    #         else:
    #             appa_y_points[system_drag_factor - 1] += appa_y_point
    #             optimization_y_points[system_drag_factor - 1] += optimizaion_y_point
    # for index, _ in enumerate(appa_y_points):
    #     appa_y_points[index] /= max_loop
    # for index, _ in enumerate(optimization_y_points):
    #     optimization_y_points[index] /= max_loop
    # plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    # plt.plot(x_points, optimization_y_points, marker='x',
    #          linestyle='-', label='optimization')
    # plt.xlabel("System drag factor", fontsize=12)
    # plt.ylabel("Time cost (m)", fontsize=12)
    # plt.yscale('log')  # Sử dụng scale logarithmic cho trục Y
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/benchmark_optimization_sdf.png')
    # plt.close()

    # x_points = []
    # appa_y_points = []
    # optimization_y_points = []
    # for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - region", position=0):
    #     for num_regions in range(5, 55, 5):
    #         appa_y_point, optimizaion_y_point = benchmark_optimization(
    #             num_regions=num_regions)
    #         if try_time == 0:
    #             x_points.append(num_regions)
    #             appa_y_points.append(appa_y_point)
    #             optimization_y_points.append(optimizaion_y_point)
    #         else:
    #             appa_y_points[num_regions // 5 - 1] += appa_y_point
    #             optimization_y_points[num_regions //
    #                                   5 - 1] += optimizaion_y_point
    # for index, _ in enumerate(appa_y_points):
    #     appa_y_points[index] /= max_loop
    # for index, _ in enumerate(optimization_y_points):
    #     optimization_y_points[index] /= max_loop
    # plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    # plt.plot(x_points, optimization_y_points, marker='x',
    #          linestyle='-', label='optimization')
    # plt.xlabel("Number of regions", fontsize=12)
    # plt.ylabel("Time cost (m)", fontsize=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/benchmark_optimization_region.png')
    # plt.close()

    # * benchmark q learning acs
    # x_points = []
    # appa_y_points = []
    # optimization_y_points = []
    # for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - uav", position=0):
    #     for uav in range(2, 10):
    #         appa_y_point, optimizaion_y_point = benchmark_q_learning_acs(
    #             num_uavs=uav)
    #         if try_time == 0:
    #             x_points.append(uav)
    #             appa_y_points.append(appa_y_point)
    #             optimization_y_points.append(optimizaion_y_point)
    #         else:
    #             appa_y_points[uav- 2] += appa_y_point
    #             optimization_y_points[uav - 2] += optimizaion_y_point
    # for index, _ in enumerate(appa_y_points):
    #     appa_y_points[index] /= max_loop
    # for index, _ in enumerate(optimization_y_points):
    #     optimization_y_points[index] /= max_loop
    # plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    # plt.plot(x_points, optimization_y_points, marker='x',
    #          linestyle='-', label='optimization')
    # plt.xlabel("Number of uavs", fontsize=12)
    # plt.ylabel("Time cost (m)", fontsize=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/benchmark_optimization_uav.png')
    # plt.close()

    # x_points = []
    # appa_y_points = []
    # optimization_y_points = []
    # for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - system drag factor", position=0):
    #     for system_drag_factor in range(1, 10):
    #         appa_y_point, optimizaion_y_point = benchmark_q_learning_acs(
    #             d=system_drag_factor/10)
    #         if try_time == 0:
    #             x_points.append(system_drag_factor / 10)
    #             appa_y_points.append(appa_y_point)
    #             optimization_y_points.append(optimizaion_y_point)
    #         else:
    #             appa_y_points[system_drag_factor - 1] += appa_y_point
    #             optimization_y_points[system_drag_factor - 1] += optimizaion_y_point
    # for index, _ in enumerate(appa_y_points):
    #     appa_y_points[index] /= max_loop
    # for index, _ in enumerate(optimization_y_points):
    #     optimization_y_points[index] /= max_loop
    # plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    # plt.plot(x_points, optimization_y_points, marker='x',
    #          linestyle='-', label='optimization')
    # plt.xlabel("System drag factor", fontsize=12)
    # plt.ylabel("Time cost (m)", fontsize=12)
    # plt.yscale('log')  # Sử dụng scale logarithmic cho trục Y
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/benchmark_optimization_sdf.png')
    # plt.close()

    # x_points = []
    # appa_y_points = []
    # optimization_y_points = []
    # for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - region", position=0):
    #     for num_regions in range(5, 55, 5):
    #         appa_y_point, optimizaion_y_point = benchmark_q_learning_acs(
    #             num_regions=num_regions)
    #         if try_time == 0:
    #             x_points.append(num_regions)
    #             appa_y_points.append(appa_y_point)
    #             optimization_y_points.append(optimizaion_y_point)
    #         else:
    #             appa_y_points[num_regions // 5 - 1] += appa_y_point
    #             optimization_y_points[num_regions //
    #                                   5 - 1] += optimizaion_y_point
    # for index, _ in enumerate(appa_y_points):
    #     appa_y_points[index] /= max_loop
    # for index, _ in enumerate(optimization_y_points):
    #     optimization_y_points[index] /= max_loop
    # plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    # plt.plot(x_points, optimization_y_points, marker='x',
    #          linestyle='-', label='optimization')
    # plt.xlabel("Number of regions", fontsize=12)
    # plt.ylabel("Time cost (m)", fontsize=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend(loc='upper center', bbox_to_anchor=(
    #     0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    # plt.savefig('./fig/benchmark_optimization_region.png')
    # plt.close()

    # * overview all
    x_points = [_ for _ in range(5, 55, 5)]
    optimization_y_points = [0 for _ in range(5, 55, 5)]
    appa_y_points = [0 for _ in range(5, 55, 5)]
    ga_y_points = [0 for _ in range(5, 55, 5)]
    for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - region", position=0):
        for index, num_regions in enumerate(x_points):
            optimization_y_point, appa_y_point, ga_y_point = benchmark_all_run_time(
                num_regions=num_regions)

            optimization_y_points[index] += optimization_y_point
            appa_y_points[index] += appa_y_point
            ga_y_points[index] += ga_y_point

    for index, _ in enumerate(optimization_y_points):
        optimization_y_points[index] /= max_loop
    for index, _ in enumerate(appa_y_points):
        appa_y_points[index] /= max_loop
    for index, _ in enumerate(ga_y_points):
        ga_y_points[index] /= max_loop

    plt.plot(x_points, optimization_y_points, marker='*',
             linestyle='-', label='optimization')
    plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    plt.plot(x_points, ga_y_points, marker='x', linestyle='-', label='ga')

    plt.xlabel("Number of regions", fontsize=12)
    plt.ylabel("Execute time (m)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.savefig('./fig/fig2.png')
    plt.close()

    x_points = [_ for _ in range(5, 55, 5)]
    optimization_y_points = [0 for _ in range(5, 55, 5)]
    appa_y_points = [0 for _ in range(5, 55, 5)]
    ga_y_points = [0 for _ in range(5, 55, 5)]
    for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - region", position=0):
        for index, num_regions in enumerate(x_points):
            optimization_y_point, appa_y_point, ga_y_point = benchmark_all(
                num_regions=num_regions)

            optimization_y_points[index] += optimization_y_point
            appa_y_points[index] += appa_y_point
            ga_y_points[index] += ga_y_point

    for index, _ in enumerate(optimization_y_points):
        optimization_y_points[index] /= max_loop
    for index, _ in enumerate(appa_y_points):
        appa_y_points[index] /= max_loop
    for index, _ in enumerate(ga_y_points):
        ga_y_points[index] /= max_loop

    plt.plot(x_points, optimization_y_points, marker='*',
             linestyle='-', label='optimization')
    plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    plt.plot(x_points, ga_y_points, marker='x', linestyle='-', label='ga')

    plt.xlabel("Number of regions", fontsize=12)
    plt.ylabel("Task completion time (s)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.savefig('./fig/fig4.png')
    plt.close()

    x_points = [_/10 for _ in range(1, 10)]
    optimization_y_points = [0 for _ in range(1, 10)]
    appa_y_points = [0 for _ in range(1, 10)]
    ga_y_points = [0 for _ in range(1, 10)]
    for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - sdf", position=0):
        for index, sdf in enumerate(x_points):
            optimization_y_point, appa_y_point, ga_y_point = benchmark_all(
                d=sdf)

            optimization_y_points[index] += optimization_y_point
            appa_y_points[index] += appa_y_point
            ga_y_points[index] += ga_y_point

    for index, _ in enumerate(optimization_y_points):
        optimization_y_points[index] /= max_loop
    for index, _ in enumerate(appa_y_points):
        appa_y_points[index] /= max_loop
    for index, _ in enumerate(ga_y_points):
        ga_y_points[index] /= max_loop

    plt.plot(x_points, optimization_y_points, marker='*',
             linestyle='-', label='optimization')
    plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    plt.plot(x_points, ga_y_points, marker='x', linestyle='-', label='ga')

    plt.xlabel("System drag factor", fontsize=12)
    plt.ylabel("Task completion time (m)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.savefig('./fig/fig5.png')
    plt.close()

    x_points = [_ for _ in range(2, 12, 2)]
    optimization_y_points = [0 for _ in range(2, 12, 2)]
    appa_y_points = [0 for _ in range(2, 12, 2)]
    ga_y_points = [0 for _ in range(2, 12, 2)]
    for try_time in tqdm.tqdm(range(0, max_loop), desc="benchmark optimization - uav", position=0):
        for index, uav in enumerate(x_points):
            optimization_y_point, appa_y_point, ga_y_point = benchmark_all(
                num_uavs=uav)

            optimization_y_points[index] += optimization_y_point
            appa_y_points[index] += appa_y_point
            ga_y_points[index] += ga_y_point

    for index, _ in enumerate(optimization_y_points):
        optimization_y_points[index] /= max_loop
    for index, _ in enumerate(appa_y_points):
        appa_y_points[index] /= max_loop
    for index, _ in enumerate(ga_y_points):
        ga_y_points[index] /= max_loop

    plt.plot(x_points, optimization_y_points, marker='*',
             linestyle='-', label='optimization')
    plt.plot(x_points, appa_y_points, marker='o', linestyle='-', label='appa')
    plt.plot(x_points, ga_y_points, marker='x', linestyle='-', label='ga')

    plt.xlabel("Number of UAVs", fontsize=12)
    plt.ylabel("Task completion time (m)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.05), ncol=5, fancybox=True, shadow=False)
    plt.savefig('./fig/fig7.png')
    plt.close()
    pass


if __name__ == "__main__":
    os.makedirs('./fig', exist_ok=True)
    main()
