import numpy as np
import math
from typing import Dict, List, Any
from utils.config import UAV, Region

class Benchmark:
    def __init__(
        self, 
        uavs_list: List[UAV], 
        regions_list: List[Region],
        V_matrix: List[List[float]]
    ):
        self.uavs_list = uavs_list
        self.regions_list = regions_list
        self.V_matrix = np.array(V_matrix)
        self.num_uavs = len(uavs_list)
        self.num_regions = len(regions_list)
        
        # Precompute distance matrix (same as in APPA)
        self.D_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> np.ndarray:
        D = np.zeros((self.num_regions, self.num_regions))
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if i != j:
                    x1, y1 = self.regions_list[i].coords
                    x2, y2 = self.regions_list[j].coords
                    D[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return D

    def calculate_metrics(self, result: Dict[str, Any], execution_time: float, baseline_metrics: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Calculate all metrics for a single run result.
        If baseline_metrics is provided, also calculate Deviation Ratio.
        """
        paths = result['paths']
        completion_times = list(result['completion_times'].values())
        
        metrics = {}
        
        # 1. Time metrics
        metrics['Max Completion Time'] = max(completion_times) if completion_times else 0.0
        metrics['Avg Completion Time'] = np.mean(completion_times) if completion_times else 0.0
        metrics['Min Completion Time'] = min(completion_times) if completion_times else 0.0
        metrics['Execution Time'] = execution_time
        
        # 2. Workload balance metrics
        metrics['Workload Variance'] = np.var(completion_times) if completion_times else 0.0
        metrics['Workload Std Dev'] = np.std(completion_times) if completion_times else 0.0
        mean_time = metrics['Avg Completion Time']
        metrics['Workload Balance Index'] = (metrics['Workload Std Dev'] / mean_time) if mean_time > 0 else 0.0
        
        # 3. Distance metrics
        total_distance = 0.0
        active_uavs = 0
        
        for uav_idx, tour in paths.items():
            if not tour:
                continue
            active_uavs += 1
            
            # Base to first
            base_coords = (0, 0)
            first_coords = self.regions_list[tour[0]].coords
            dist = math.sqrt((first_coords[0] - base_coords[0])**2 + (first_coords[1] - base_coords[1])**2)
            total_distance += dist
            
            # Between regions
            for k in range(len(tour) - 1):
                total_distance += self.D_matrix[tour[k]][tour[k+1]]
                
            # Last to base
            last_coords = self.regions_list[tour[-1]].coords
            dist_back = math.sqrt((base_coords[0] - last_coords[0])**2 + (base_coords[1] - last_coords[1])**2)
            total_distance += dist_back
            
        metrics['Total Distance'] = total_distance
        metrics['Avg Distance per UAV'] = (total_distance / active_uavs) if active_uavs > 0 else 0.0
        
        # 4. Efficiency metrics
        total_scan_time = 0.0
        for uav_idx, tour in paths.items():
            for region_idx in tour:
                if self.V_matrix[uav_idx][region_idx] > 0:
                    area = self.regions_list[region_idx].area
                    scan_width = self.uavs_list[uav_idx].scan_width
                    total_scan_time += area / (self.V_matrix[uav_idx][region_idx] * scan_width)
        
        # Total Flight Time = Total Time (sum of completion times) - Total Scan Time ?? 
        # Wait, completion_times includes both flight and scan.
        # But sum(completion_times) is not necessarily "Total Flight + Total Scan" because they happen in parallel?
        # The user definition: Total Flight Time = Σ(Distance / MaxVelocity)
        
        total_flight_time = 0.0
        for uav_idx, tour in paths.items():
            if not tour:
                continue
            
            # Calculate distance for this UAV
            uav_dist = 0.0
            base_coords = (0, 0)
            first_coords = self.regions_list[tour[0]].coords
            uav_dist += math.sqrt((first_coords[0] - base_coords[0])**2 + (first_coords[1] - base_coords[1])**2)
            
            for k in range(len(tour) - 1):
                uav_dist += self.D_matrix[tour[k]][tour[k+1]]
                
            last_coords = self.regions_list[tour[-1]].coords
            uav_dist += math.sqrt((base_coords[0] - last_coords[0])**2 + (base_coords[1] - last_coords[1])**2)
            
            total_flight_time += uav_dist / self.uavs_list[uav_idx].max_velocity

        metrics['Total Scan Time'] = total_scan_time
        metrics['Total Flight Time'] = total_flight_time
        
        denom = total_scan_time + total_flight_time
        metrics['Efficiency Ratio'] = (total_scan_time / denom) if denom > 0 else 0.0
        
        # 5. Allocation metrics
        region_counts = [len(tour) for tour in paths.values()]
        metrics['Allocation Balance'] = (np.std(region_counts) / np.mean(region_counts)) if np.mean(region_counts) > 0 else 0.0
        
        max_comp = metrics['Max Completion Time']
        if max_comp > 0:
            utilizations = [t / max_comp for t in completion_times]
            metrics['Avg UAV Utilization'] = np.mean(utilizations)
        else:
            metrics['Avg UAV Utilization'] = 0.0
            
        # 6. Deviation ratio (if baseline provided)
        if baseline_metrics:
            # Calculate deviation for key metrics
            # Formula: (Solution - Baseline) / Baseline * 100
            # We can calculate this for Max Completion Time as the primary metric
            base_val = baseline_metrics.get('Max Completion Time', 0.0)
            curr_val = metrics['Max Completion Time']
            if base_val > 0:
                metrics['Deviation Ratio (Max Time)'] = ((curr_val - base_val) / base_val) * 100
            else:
                metrics['Deviation Ratio (Max Time)'] = 0.0
                
        return metrics
