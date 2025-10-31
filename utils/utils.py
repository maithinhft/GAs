import math
import random
import numpy as np
from .config import *
from typing import List, Dict, Any, Tuple

def get_distance(coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
    """Tính khoảng cách Euclidean giữa hai điểm."""
    return math.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)

def get_fly_time(uav: UAV, 
                 coords_from: Tuple[float, float], 
                 coords_to: Tuple[float, float]) -> float:
    """
    Tính thời gian bay (TF) từ điểm A đến điểm B cho một UAV.
    Dựa trên công thức (2).
    """
    distance = get_distance(coords_from, coords_to)
    return distance / uav.max_velocity  # SỬA: Dùng uav.max_velocity

def get_scan_time(uav: UAV, 
                  region: Region, 
                  v_matrix_entry: float) -> float:
    """
    Tính thời gian quét (TS) của UAV tại một khu vực.
    Dựa trên công thức (1).
    """
    if v_matrix_entry == 0:
        return float('inf')  # UAV không thể quét khu vực này
    
    scan_velocity = v_matrix_entry
    scan_width = uav.scan_width  # SỬA: Dùng uav.scan_width
    area = region.area        # SỬA: Dùng region.area
    
    return area / (scan_velocity * scan_width)

def calculate_path_time(uav: UAV, 
                        region_path: List[Region], 
                        v_matrix: List[List[float]],
                        uav_idx: int,
                        regions_list: List[Region]) -> float:
    """
    Tính tổng thời gian hoàn thành F(Ui) cho một UAV với một lộ trình
    đã định sẵn (bao gồm bay từ căn cứ và quét).
    Dựa trên công thức (7).
    """
    total_time = 0.0
    current_coords = BASE_COORDS
    
    for region in region_path:
        # .index() sẽ hoạt động với các đối tượng Region
        region_idx = regions_list.index(region) 
        v_entry = v_matrix[uav_idx][region_idx]
        
        # Thời gian bay từ vị trí trước đó đến khu vực này
        fly_time = get_fly_time(uav, current_coords, region.coords) # SỬA: Dùng region.coords
        
        # Thời gian quét khu vực này
        scan_time = get_scan_time(uav, region, v_entry)
        
        if scan_time == float('inf'):
            return float('inf') # Lộ trình không hợp lệ

        total_time += fly_time + scan_time
        current_coords = region.coords # SỬA: Dùng region.coords
        
    return total_time