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
from utils.config import *

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
    pass


if __name__ == "__main__":
    os.makedirs('./fig', exist_ok=True)
    main()
