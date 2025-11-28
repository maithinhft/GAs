#!/usr/bin/env python3

import json
import time
from main import visualize_uav_paths, visualize_clusters
from algorithm.appa import APPAAlgorithm
from utils.config import UAV, Region
from utils.create_sample import create_sample

data = create_sample()
uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
regions_list = [Region(**region) for region in data['regions_list']]
V_matrix = data['V_matrix']

print("=" * 60)
print("Running APPA Algorithm...")
print(f"Problem: {len(uavs_list)} UAVs, {len(regions_list)} Regions")
print("=" * 60)

appa = APPAAlgorithm(uavs_list, regions_list, V_matrix)
start = time.time()
result = appa.solve()
elapsed = time.time() - start

paths = result['paths']
completion_times = result['completion_times']
max_completion_time = result['max_completion_time']

print(f"Max Completion Time: {max_completion_time:.4f}")
print(f"Time: {elapsed:.2f}s")

print("Path Summary:")
for uav_idx in paths:
    region_indices = paths[uav_idx]
    time_val = completion_times[uav_idx]
    print(f"   UAV {uav_idx + 1}: {len(region_indices)} regions -> {region_indices} (Time: {time_val:.2f})")

uav_paths_for_viz = []
for uav_idx in range(len(uavs_list)):
    if uav_idx in paths:
        region_path = [regions_list[r_idx] for r_idx in paths[uav_idx]]
        uav_paths_for_viz.append(region_path)
    else:
        uav_paths_for_viz.append([])

print("Generating visualization...")
visualize_uav_paths(
    uav_paths=uav_paths_for_viz,
    uavs_list=uavs_list,
    title=None,
    save_path="fig/uav_paths_appa.png",
    show=True
)

print("Generating cluster visualization...")
visualize_clusters(
    uav_paths=uav_paths_for_viz,
    uavs_list=uavs_list,
    title=None,
    save_path="fig/uav_clusters_appa.png",
    show=True
)

print("Done!")
