#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra optimization_appa với APPA mới
"""
import numpy as np
from utils.config import Region, UAV
from utils.create_sample import create_sample
from algorithm.optimization_appa import iterative_appa_with_gradient

def test_optimization_appa():
    print("Đang tạo sample data...")
    sample = create_sample()

    uavs = [UAV(**uav_data) for uav_data in sample['uavs_list']]
    regions = [Region(**region_data) for region_data in sample['regions_list']]
    V_matrix = np.array(sample['V_matrix'])
    
    print(f"Số UAVs: {len(uavs)}")
    print(f"Số Regions: {len(regions)}")
    print(f"V_matrix shape: {V_matrix.shape}")
    
    # Chạy iterative APPA với ít iterations để test nhanh
    print("\nĐang chạy optimization APPA...")
    best_assignment, history, all_assignments = iterative_appa_with_gradient(
        uavs,
        regions,
        V_matrix,
        base_coords=(0, 0),
        max_iterations=3,  # Chỉ 3 iterations để test
        convergence_threshold=1.0,
        visualize_iterations=False,
        acs_params={
            'n_ants': 5,
            'n_generations': 10,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.1,
            'epsilon': 0.1,
            'q0': 0.9
        },
        verbose=True
    )
    
    # In kết quả
    print("\n" + "=" * 60)
    print("PHÂN BỔ CUỐI CÙNG:")
    print("=" * 60)
    for uav_id, path in best_assignment.items():
        region_ids = [r.id for r in path]
        print(f"UAV {uav_id}: {region_ids} (số vùng: {len(path)})")
    
    print("\n" + "=" * 60)
    print("LỊCH SỬ THỜI GIAN:")
    print("=" * 60)
    for i, time in enumerate(history):
        improvement = ""
        if i > 0:
            diff = history[i-1] - time
            improvement = f" (Δ={diff:+.2f})"
        print(f"Iteration {i}: {time:.2f}{improvement}")
    
    print("\n✓ Test hoàn thành thành công!")

if __name__ == "__main__":
    test_optimization_appa()
