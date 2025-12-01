import os
import random
import numpy as np
import time
from utils.create_sample import create_sample
import matplotlib.pyplot as plt
import tqdm
from algorithm.appa import APPAAlgorithm
from dataclasses import dataclass
from typing import Tuple, List
from utils.config import *
import json


def visualize_uav_paths(uav_paths: List[List[Region]], 
                        uavs_list: List[UAV] = None,
                        title: str = "UAV Coverage Paths",
                        save_path: str = None,
                        show: bool = True):
    """
    Visualize lộ trình của từng UAV.
    
    Args:
        uav_paths: List các paths, mỗi path là List[Region] cho 1 UAV
        uavs_list: List các UAV (optional, để hiển thị thông tin)
        title: Tiêu đề của biểu đồ
        save_path: Đường dẫn lưu file (None = không lưu)
        show: Có hiển thị trực tiếp không
    """
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', 
              '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    base_x, base_y = BASE_COORDS
    ax.scatter(base_x, base_y, c='black', s=200, marker='s', zorder=5, label='Base')
    ax.annotate('BASE', (base_x, base_y), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    for uav_idx, path in enumerate(uav_paths):
        if not path:
            continue
            
        color = colors[uav_idx % len(colors)]
        uav_label = f'UAV {uav_idx + 1}'
        if uavs_list and uav_idx < len(uavs_list):
            uav_label = f'UAV {uavs_list[uav_idx].id}'
        
        path_coords = [(r.coords[0], r.coords[1]) for r in path]
        
        full_path = [(base_x, base_y)] + path_coords + [(base_x, base_y)]
        
        xs = [p[0] for p in full_path]
        ys = [p[1] for p in full_path]
        
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7, 
                linestyle='-', marker='o', markersize=4, label=uav_label)
        
        for i in range(len(full_path) - 1):
            x1, y1 = full_path[i]
            x2, y2 = full_path[i + 1]
            dx, dy = x2 - x1, y2 - y1
            
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            if abs(dx) > 1 or abs(dy) > 1: 
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                           xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        
        for idx, region in enumerate(path):
            rx, ry = region.coords
            ax.scatter(rx, ry, c=color, s=100, zorder=4, edgecolors='white', linewidth=1)
            ax.annotate(f'{idx+1}', (rx, ry), textcoords="offset points",
                       xytext=(5, 5), fontsize=8, color=color, fontweight='bold')
    
    all_regions_plotted = set()
    for path in uav_paths:
        for r in path:
            all_regions_plotted.add(r.id)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    total_regions = sum(len(p) for p in uav_paths)
    info_text = f'Total Regions: {total_regions}\nUAVs: {len(uav_paths)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved visualization to: {save_path}')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def visualize_clusters(uav_paths: List[List[Region]], 
                      uavs_list: List[UAV] = None,
                      title: str = "UAV Clusters",
                      save_path: str = None,
                      show: bool = True):
    """
    Visualize các cụm region được phân chia cho mỗi UAV.
    Sử dụng Convex Hull để bao quanh các cụm.
    """
    from scipy.spatial import ConvexHull
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', 
              '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    base_x, base_y = BASE_COORDS
    ax.scatter(base_x, base_y, c='black', s=200, marker='s', zorder=5, label='Base')
    ax.annotate('BASE', (base_x, base_y), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    for uav_idx, path in enumerate(uav_paths):
        if not path:
            continue
            
        color = colors[uav_idx % len(colors)]
        uav_label = f'UAV {uav_idx + 1}'
        if uavs_list and uav_idx < len(uavs_list):
            uav_label = f'UAV {uavs_list[uav_idx].id}'
            
        points = np.array([(r.coords[0], r.coords[1]) for r in path])
        
        ax.scatter(points[:, 0], points[:, 1], c=color, s=100, alpha=0.6, label=uav_label)
        
        for r in path:
            ax.annotate(f'{r.id}', (r.coords[0], r.coords[1]), 
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
        
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], color=color, linestyle='--', linewidth=2, alpha=0.8)
                
                ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.1)
            except Exception as e:
                print(f"Could not draw convex hull for UAV {uav_idx}: {e}")
        elif len(points) == 2:
            ax.plot(points[:, 0], points[:, 1], color=color, linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved cluster visualization to: {save_path}')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

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


def main():
    data = create_sample()
    #tạo file sample.json chứa data 
    with open('sample.json', 'w') as f:
        json.dump(data, f, indent=4)
    pass


if __name__ == "__main__":
    os.makedirs('./fig', exist_ok=True)
    main()
