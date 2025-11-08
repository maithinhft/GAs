# Optimization APPA: Algorithm Framework (UML)

## 📋 Tổng quan

**Iterative APPA with Gradient-Based Refinement** là một thuật toán lai hợp kết hợp:
- **APPA (Ant Colony System-based Phased Allocation)**: Phân bổ vùng + tối ưu thứ tự
- **Gradient-Based Refinement**: Di chuyển vùng giữa các UAV để giảm thời gian hoàn thành
- **Iterative Improvement**: Lặp cho đến khi hội tụ

---

## 🏗️ Cấu trúc file UML

### 1. **Activity Diagram** (`OPTIMIZATION_APPA_UML.puml`)
Mô tả luồng thực thi chính của thuật toán:

```
Start
  ↓
Initialize Parameters
  ↓
Run APPA (Phase 1 & 2)
  ├─ Phase 1: Allocate regions based on ETR
  └─ Phase 2: Optimize visiting order using ACS
  ↓
Loop [iterations = 1 to max_iterations]:
  ├─ Gradient-Based Refinement
  │  └─ Move regions between UAVs to minimize time
  ├─ ACS Reordering
  │  └─ Optimize visiting order for each UAV
  ├─ Calculate completion time
  ├─ Check convergence conditions
  └─ Update best solution
  ↓
Output Results
  └─ best_assignment, history, all_assignments
```

**Điểm hội tụ**:
- Số iteration đạt tối đa
- Stagnation: 10 vòng lặp không cải thiện
- Convergence threshold: cải thiện < 1s

---

### 2. **Class Diagram** (`OPTIMIZATION_APPA_CLASSES.puml`)

#### Main Classes:
```
┌─────────────────────┐
│  OptimizationAPPA   │  ← Main module
├─────────────────────┤
│ - Region            │
│ - UAV               │
│ - V_matrix          │
├─────────────────────┤
│ + gradient_based... │
│ + estimate_time...  │
│ + iterative_appa... │
│ + calculate_path... │
└─────────────────────┘
```

#### Core Functions:

| Hàm | Mục đích | Độ phức tạp |
|-----|---------|-----------|
| `calculate_path_completion_time()` | Tính thời gian hoàn thành tour của 1 UAV | O(path_length) |
| `calculate_system_completion_time()` | Tính thời gian hệ thống (max của tất cả UAV) | O(n_uavs × path_length) |
| `gradient_based_refinement()` | Đánh giá và di chuyển region để tối ưu | O(n_uavs² × n_regions × T_calc) |
| `estimate_time_change()` | Ước lượng thay đổi thời gian cho 1 move | O(n_uavs × path_length) |
| `quick_path_optimization_nearest_neighbor()` | Sắp xếp region bằng nearest neighbor | O(n²) |
| `iterative_appa_with_gradient()` | Main algorithm | O(iterations × refinement_cost) |

---

### 3. **Sequence Diagram** (`OPTIMIZATION_APPA_SEQUENCE.puml`)

Thứ tự thực thi chi tiết:

```
1. Main → OptimizationAPPA: Start iterative_appa_with_gradient()

2. OptimizationAPPA → APPA: Create instance + precompute matrices

3. APPA → APPA: Solve Phase 1 & 2
   - region_allocation_phase() → Assign regions to UAVs
   - order_optimization_phase() → Optimize order using ACS
   
4. OptimizationAPPA → Utils: Calculate initial completion time

5. LOOP [iterations]:
   a. OptimizationAPPA → GradientRefinement: Find best moves
      - Evaluate each (source, region, target) move
      - Calculate time change
      - Select move with best improvement
   
   b. OptimizationAPPA → APPA: Reorder optimization
      - For each UAV: optimize its route order
   
   c. OptimizationAPPA → Utils: Calculate new completion time
   
   d. OptimizationAPPA → OptimizationAPPA: Check convergence
   
   e. If converged → Break loop

6. OptimizationAPPA → Main: Return results
```

---

## 📊 Thuật toán chi tiết

### Phase 1: Region Allocation (APPA)
```
Input: UAVs, Regions, V_matrix
Output: assignment[uav_id] → List[Region]

Algorithm:
  while unassigned_regions:
    # Find UAV that finishes earliest
    uav = argmin(finish_time[i])
    
    # Find best region for this UAV (max ETR)
    best_region = argmax(ETR[uav][region])
    where ETR = TS / (TF + TS)
    
    # Assign region to UAV
    assignment[uav].append(best_region)
    finish_time[uav] += TF + TS
```

### Phase 2: Order Optimization (ACS)
```
Input: assignment[uav_id] → List[Region]
Output: optimized_assignment[uav_id] → List[Region]

For each UAV:
  - Initialize pheromone matrix τ
  - Loop n_generations times:
    * Each ant constructs a tour
    * Update pheromone based on best ant
  - Return best tour found
```

### Phase 3: Gradient-Based Refinement
```
Input: current_assignment, completion_time
Output: refined_assignment, improved_flag

Algorithm:
  best_move = None
  best_improvement = 0
  
  for source_uav in UAVs:
    for region in source_uav.regions:
      for target_uav in UAVs:
        if source_uav == target_uav:
          continue
        
        # Create new assignment with region moved
        new_assignment = move(current, source, region, target)
        
        # Calculate time change
        time_change = new_time - current_time
        improvement = -time_change  # Negative = better
        
        if improvement > best_improvement:
          best_improvement = improvement
          best_move = (source, region, target)
  
  if best_move and best_improvement > threshold:
    Apply best_move
    return new_assignment, True
  else:
    return current_assignment, False
```

---

## 📈 Độ phức tạp tính toán

### Time Complexity

| Thành phần | Độ phức tạp | Ghi chú |
|-----------|----------|--------|
| APPA Phase 1 | O(n_uavs × n_regions²) | Greedy allocation |
| APPA Phase 2 | O(n_uavs × n_ants × n_generations × n_regions²) | ACS per UAV |
| Gradient Refinement | O(n_uavs² × n_regions × T_calc) | Evaluate all moves |
| Total per iteration | O(refinement_cost) | ~10-20s for 50 regions |
| Iterative APPA | O(max_iterations × total_per_iteration) | Usually 5-10 iterations |

### Space Complexity
- Pheromone matrices: O(n_uavs × max_path_length²)
- Distance/Time matrices: O(n_regions²)
- Assignment storage: O(n_uavs × n_regions)

---

## 🎯 Các điểm tối ưu được áp dụng

1. **Set-based lookup** cho `quick_path_optimization_nearest_neighbor()`
   - O(1) vs O(n) khi remove region

2. **Early stopping** trong gradient refinement
   - Dừng sớm nếu không tìm được move tốt hơn

3. **Stagnation detection**
   - Dừng nếu không cải thiện trong 10 vòng lặp

4. **Lazy evaluation**
   - Chỉ tính các region thực sự được gán (không tính empty paths)

---

## 📝 Convergence Criteria

Thuật toán dừng khi:
1. **Số iteration đạt tối đa** (default: 10)
2. **Stagnation**: 10 vòng lặp liên tiếp không cải thiện
3. **Convergence threshold**: Cải thiện < 1s so với vòng trước

---

## 💡 Cách sử dụng

```python
from algorithm.optimization_appa import iterative_appa_with_gradient

result = iterative_appa_with_gradient(
    uavs=uavs_list,
    regions=regions_list,
    V_matrix=V_matrix,
    max_iterations=10,
    convergence_threshold=1.0,
    acs_params={
        'n_ants': 10,
        'n_generations': 50,
        'alpha': 1.0,
        'beta': 2.0,
        'rho': 0.1,
        'epsilon': 0.1,
        'q0': 0.9
    },
    verbose=True
)

best_assignment, history, all_assignments = result
print(f"Best completion time: {history[-1]:.2f}")
```

---

## 📌 Lưu ý

- **UAV indices** trong APPA: 0-based (0 đến n_uavs-1)
- **UAV IDs** trong assignment dict: 1-based (1 đến n_uavs)
- **Region indices** trong APPA: 0-based
- **Region objects** trong assignment: Region instances

Cần chuyển đổi giữa các format khi gọi APPA và lưu trữ results!

