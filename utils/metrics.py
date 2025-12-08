"""
Metrics calculation module for comparing APPA algorithms
Implements various performance metrics similar to those in the paper
"""
import numpy as np
import math
from typing import Dict, List, Tuple
from utils.config import UAV, Region, BASE_COORDS


class MetricsCalculator:
    """Calculate comprehensive metrics for algorithm comparison"""
    
    def __init__(self, uavs_list: List[UAV], regions_list: List[Region], V_matrix: np.ndarray):
        self.uavs_list = uavs_list
        self.regions_list = regions_list
        self.V_matrix = V_matrix
        self.num_uavs = len(uavs_list)
        self.num_regions = len(regions_list)
        
        # Precompute distance matrix
        self.D_matrix = self._compute_distance_matrix()
        self.TS_matrix = self._compute_scan_time_matrix()
        self.TF_matrix = self._compute_flight_time_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute distance matrix between all regions and base"""
        D = np.zeros((self.num_regions + 1, self.num_regions + 1))
        # Index 0 is base, indices 1 to num_regions are regions
        
        # Base to regions
        for j in range(1, self.num_regions + 1):
            region = self.regions_list[j - 1]
            dist = math.sqrt(region.coords[0]**2 + region.coords[1]**2)
            D[0][j] = dist
            D[j][0] = dist
        
        # Between regions
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if i != j:
                    x1, y1 = self.regions_list[i].coords
                    x2, y2 = self.regions_list[j].coords
                    D[i + 1][j + 1] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return D
    
    def _compute_scan_time_matrix(self) -> np.ndarray:
        """Compute scan time matrix TS[i,j] for UAV i scanning region j"""
        TS = np.full((self.num_uavs, self.num_regions), np.inf)
        for i in range(self.num_uavs):
            for j in range(self.num_regions):
                if self.V_matrix[i][j] > 0:
                    area = self.regions_list[j].area
                    scan_width = self.uavs_list[i].scan_width
                    TS[i][j] = area / (self.V_matrix[i][j] * scan_width)
        return TS
    
    def _compute_flight_time_matrix(self) -> np.ndarray:
        """Compute flight time matrix TF[i,j,k] for UAV i flying from region j to k"""
        TF = np.zeros((self.num_uavs, self.num_regions, self.num_regions))
        for i in range(self.num_uavs):
            for j in range(self.num_regions):
                for k in range(self.num_regions):
                    if j != k:
                        TF[i][j][k] = self.D_matrix[j + 1][k + 1] / self.uavs_list[i].max_velocity
        return TF
    
    def calculate_all_metrics(self, result: Dict, execution_time: float) -> Dict:
        """
        Calculate all metrics for a given algorithm result
        
        Args:
            result: Dictionary with keys 'paths', 'completion_times', 'max_completion_time'
            execution_time: Algorithm execution time in seconds
        
        Returns:
            Dictionary with all calculated metrics
        """
        paths = result['paths']
        completion_times = result['completion_times']
        max_completion_time = result['max_completion_time']
        
        metrics = {
            # Basic metrics
            'max_completion_time': max_completion_time,
            'execution_time': execution_time,
            
            # Completion time metrics
            'avg_completion_time': self._calculate_avg_completion_time(completion_times),
            'min_completion_time': min(completion_times.values()) if completion_times else 0.0,
            
            # Workload balance metrics
            'workload_variance': self._calculate_workload_variance(completion_times),
            'workload_std': self._calculate_workload_std(completion_times),
            'workload_balance_index': self._calculate_workload_balance_index(completion_times),
            
            # Distance metrics
            'total_distance': self._calculate_total_distance(paths),
            'avg_distance_per_uav': self._calculate_avg_distance_per_uav(paths),
            
            # Efficiency metrics
            'total_scan_time': self._calculate_total_scan_time(paths),
            'total_flight_time': self._calculate_total_flight_time(paths),
            'efficiency_ratio': self._calculate_efficiency_ratio(paths),
            
            # Region allocation metrics
            'regions_per_uav': self._calculate_regions_per_uav(paths),
            'allocation_balance': self._calculate_allocation_balance(paths),
            
            # Utilization metrics
            'uav_utilization': self._calculate_uav_utilization(completion_times, max_completion_time),
            'avg_uav_utilization': self._calculate_avg_uav_utilization(completion_times, max_completion_time),
        }
        
        return metrics
    
    def _calculate_avg_completion_time(self, completion_times: Dict[int, float]) -> float:
        """Calculate average completion time across all UAVs"""
        if not completion_times:
            return 0.0
        return np.mean(list(completion_times.values()))
    
    def _calculate_workload_variance(self, completion_times: Dict[int, float]) -> float:
        """Calculate variance of completion times (workload balance)"""
        if len(completion_times) < 2:
            return 0.0
        times = list(completion_times.values())
        return np.var(times)
    
    def _calculate_workload_std(self, completion_times: Dict[int, float]) -> float:
        """Calculate standard deviation of completion times"""
        if len(completion_times) < 2:
            return 0.0
        times = list(completion_times.values())
        return np.std(times)
    
    def _calculate_workload_balance_index(self, completion_times: Dict[int, float]) -> float:
        """
        Calculate workload balance index (0 = perfect balance, 1 = worst balance)
        Based on coefficient of variation
        """
        if not completion_times:
            return 0.0
        times = list(completion_times.values())
        if np.mean(times) == 0:
            return 0.0
        return np.std(times) / np.mean(times)
    
    def _calculate_total_distance(self, paths: Dict[int, List[int]]) -> float:
        """Calculate total distance traveled by all UAVs"""
        total = 0.0
        
        for uav_idx, path in paths.items():
            if not path:
                continue
            
            # Distance from base to first region
            first_region = self.regions_list[path[0]]
            dist = math.sqrt(first_region.coords[0]**2 + first_region.coords[1]**2)
            total += dist
            
            # Distance between regions
            for i in range(len(path) - 1):
                region1 = self.regions_list[path[i]]
                region2 = self.regions_list[path[i + 1]]
                dist = math.sqrt(
                    (region2.coords[0] - region1.coords[0])**2 + 
                    (region2.coords[1] - region1.coords[1])**2
                )
                total += dist
            
            # Distance from last region back to base
            last_region = self.regions_list[path[-1]]
            dist = math.sqrt(last_region.coords[0]**2 + last_region.coords[1]**2)
            total += dist
        
        return total
    
    def _calculate_avg_distance_per_uav(self, paths: Dict[int, List[int]]) -> float:
        """Calculate average distance traveled per UAV"""
        active_uavs = [uav_idx for uav_idx, path in paths.items() if path]
        if not active_uavs:
            return 0.0
        
        distances = []
        for uav_idx in active_uavs:
            path = paths[uav_idx]
            if not path:
                distances.append(0.0)
                continue
            
            dist = 0.0
            # From base to first
            first_region = self.regions_list[path[0]]
            dist += math.sqrt(first_region.coords[0]**2 + first_region.coords[1]**2)
            
            # Between regions
            for i in range(len(path) - 1):
                region1 = self.regions_list[path[i]]
                region2 = self.regions_list[path[i + 1]]
                dist += math.sqrt(
                    (region2.coords[0] - region1.coords[0])**2 + 
                    (region2.coords[1] - region1.coords[1])**2
                )
            
            # Back to base
            last_region = self.regions_list[path[-1]]
            dist += math.sqrt(last_region.coords[0]**2 + last_region.coords[1]**2)
            distances.append(dist)
        
        return np.mean(distances)
    
    def _calculate_total_scan_time(self, paths: Dict[int, List[int]]) -> float:
        """Calculate total scan time across all UAVs"""
        total = 0.0
        for uav_idx, path in paths.items():
            for region_idx in path:
                if self.TS_matrix[uav_idx][region_idx] != np.inf:
                    total += self.TS_matrix[uav_idx][region_idx]
        return total
    
    def _calculate_total_flight_time(self, paths: Dict[int, List[int]]) -> float:
        """Calculate total flight time across all UAVs"""
        total = 0.0
        
        for uav_idx, path in paths.items():
            if not path:
                continue
            
            # From base to first
            first_region = self.regions_list[path[0]]
            dist = math.sqrt(first_region.coords[0]**2 + first_region.coords[1]**2)
            total += dist / self.uavs_list[uav_idx].max_velocity
            
            # Between regions
            for i in range(len(path) - 1):
                total += self.TF_matrix[uav_idx][path[i]][path[i + 1]]
            
            # Back to base
            last_region = self.regions_list[path[-1]]
            dist = math.sqrt(last_region.coords[0]**2 + last_region.coords[1]**2)
            total += dist / self.uavs_list[uav_idx].max_velocity
        
        return total
    
    def _calculate_efficiency_ratio(self, paths: Dict[int, List[int]]) -> float:
        """
        Calculate efficiency ratio: scan_time / (scan_time + flight_time)
        Higher is better (more time spent scanning vs flying)
        """
        scan_time = self._calculate_total_scan_time(paths)
        flight_time = self._calculate_total_flight_time(paths)
        total_time = scan_time + flight_time
        
        if total_time == 0:
            return 0.0
        return scan_time / total_time
    
    def _calculate_regions_per_uav(self, paths: Dict[int, List[int]]) -> Dict[int, int]:
        """Calculate number of regions assigned to each UAV"""
        return {uav_idx: len(path) for uav_idx, path in paths.items()}
    
    def _calculate_allocation_balance(self, paths: Dict[int, List[int]]) -> float:
        """
        Calculate allocation balance (coefficient of variation of region counts)
        Lower is better (more balanced allocation)
        """
        region_counts = [len(path) for path in paths.values()]
        if not region_counts or np.mean(region_counts) == 0:
            return 0.0
        return np.std(region_counts) / np.mean(region_counts) if np.mean(region_counts) > 0 else 0.0
    
    def _calculate_uav_utilization(self, completion_times: Dict[int, float], max_time: float) -> Dict[int, float]:
        """
        Calculate utilization for each UAV (completion_time / max_completion_time)
        Higher utilization means UAV is working more
        """
        if max_time == 0:
            return {uav_idx: 0.0 for uav_idx in completion_times.keys()}
        return {uav_idx: time / max_time for uav_idx, time in completion_times.items()}
    
    def _calculate_avg_uav_utilization(self, completion_times: Dict[int, float], max_time: float) -> float:
        """Calculate average UAV utilization"""
        utilizations = self._calculate_uav_utilization(completion_times, max_time)
        if not utilizations:
            return 0.0
        return np.mean(list(utilizations.values()))
    
    def calculate_deviation_ratio(self, solution_value: float, baseline_value: float) -> float:
        """
        Calculate deviation ratio: (solution - baseline) / baseline * 100
        Used to compare solution quality relative to a baseline
        """
        if baseline_value == 0:
            return 0.0
        return ((solution_value - baseline_value) / baseline_value) * 100
    
    def compare_metrics(self, metrics1: Dict, metrics2: Dict, name1: str = "Method 1", name2: str = "Method 2") -> Dict:
        """
        Compare metrics between two methods and calculate improvements
        
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        # Metrics where lower is better
        lower_better = [
            'max_completion_time', 'execution_time', 'workload_variance', 
            'workload_std', 'workload_balance_index', 'total_distance',
            'avg_distance_per_uav', 'total_flight_time', 'allocation_balance'
        ]
        
        # Metrics where higher is better
        higher_better = [
            'avg_completion_time', 'efficiency_ratio', 'avg_uav_utilization'
        ]
        
        for metric in lower_better + higher_better:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                
                if metric in lower_better:
                    if val1 > 0:
                        improvement = ((val1 - val2) / val1) * 100
                    else:
                        improvement = 0.0
                    better = val2 < val1
                else:  # higher_better
                    if val1 > 0:
                        improvement = ((val2 - val1) / val1) * 100
                    else:
                        improvement = 0.0
                    better = val2 > val1
                
                comparison[metric] = {
                    name1: val1,
                    name2: val2,
                    'improvement_pct': improvement,
                    'better': name2 if better else name1
                }
        
        return comparison

