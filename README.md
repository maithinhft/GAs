# UAV Coverage Path Planning - Optimization Algorithms Comparison

## Tổng quan

Dự án này triển khai và so sánh các thuật toán tối ưu hóa cho bài toán **Coverage Path Planning** của các UAV (Unmanned Aerial Vehicles) không đồng nhất.

### Các thuật toán được triển khai:

| Thuật toán | Mô tả |
|------------|-------|
| **APPA (Original)** | Sử dụng ETR heuristic cho Phase 1 + ACS cho Phase 2 |
| **GA-APPA** | Genetic Algorithm cho Phase 1 + ACS cho Phase 2 |
| **PSO-APPA** | Particle Swarm Optimization cho Phase 1 + ACS cho Phase 2 |
| **GWO-APPA** | Grey Wolf Optimizer cho Phase 1 + ACS cho Phase 2 |
| **Pure GA** | Genetic Algorithm cho cả allocation và ordering |

---

##  Yêu cầu hệ thống

- **Python**: >= 3.10
- **Hệ điều hành**: Windows, Linux, macOS

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/maithinhft/GAs.git
cd GAs
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Thư viện cần thiết:

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| `numpy` | >= 2.0.0 | Xử lý ma trận, tính toán số học |
| `matplotlib` | >= 3.8.0 | Vẽ biểu đồ so sánh |

---

## Cách chạy

### Chạy so sánh tất cả các thuật toán:

```bash
python compare_all_algorithms.py
```

### Kết quả đầu ra:

Chương trình sẽ tạo ra các biểu đồ so sánh trong thư mục `./fig/`:

- `all_algorithms_max_completion_time.png` - So sánh thời gian hoàn thành tối đa
- `all_algorithms_execution_time.png` - So sánh thời gian thực thi
- `all_algorithms_metrics_comparison.png` - So sánh đa metrics (normalized)
- `all_algorithms_scalability.png` - Test khả năng mở rộng

---

## Cấu trúc Project

```
prj/
├── algorithm/                    # Các thuật toán tối ưu hóa
│   ├── appa.py                   # APPA gốc (ETR + ACS)
│   ├── ga_appa.py                # GA-APPA (GA Phase 1 + ACS Phase 2)
│   ├── ga_phase1.py              # Genetic Algorithm cho Phase 1
│   ├── ga_pure.py                # Pure GA (GA cho cả 2 phase)
│   ├── pso_appa.py               # PSO-APPA (PSO Phase 1 + ACS Phase 2)
│   └── gwo_appa.py               # GWO-APPA (GWO Phase 1 + ACS Phase 2)
│
├── utils/                        # Các tiện ích
│   ├── config.py                 # Cấu hình (UAV, Region, constants)
│   ├── create_sample.py          # Tạo dữ liệu thử nghiệm
│   ├── metrics.py                # Tính toán các metrics đánh giá
│   └── utils.py                  # Các hàm tiện ích
│
├── fig/                          # Thư mục chứa biểu đồ đầu ra
│
├── compare_all_algorithms.py     # Script chính so sánh thuật toán
├── requirements.txt              # Danh sách thư viện cần thiết
└── README.md                     # Tài liệu hướng dẫn
```

---

## Metrics đánh giá

### Time Metrics
- **Max Completion Time**: Thời gian tối đa để hoàn thành (minimize)
- **Avg Completion Time**: Thời gian trung bình
- **Execution Time**: Thời gian chạy thuật toán

### Workload Balance
- **Workload Variance**: Phương sai tải công việc (lower = better)
- **Workload Balance Index**: Hệ số biến thiên (lower = better)

### Efficiency
- **Efficiency Ratio**: Tỷ lệ hiệu quả (scan time / total time)
- **Avg UAV Utilization**: Tỷ lệ sử dụng UAV

---

## Cấu hình tham số

Có thể điều chỉnh các tham số trong `compare_all_algorithms.py`:

```python
# Số lượng UAV và vùng
num_uavs = 4
num_regions = 30

# Số lần chạy để lấy thống kê
num_runs = 5

# Tham số GA
ga_population_size = 50
ga_max_generations = 100

# Tham số PSO
pso_swarm_size = 50
pso_max_iterations = 100

# Tham số GWO
gwo_pack_size = 50
gwo_max_iterations = 100
```

---

## License

MIT License
