# Optimization APPA - UML Diagrams Collection

## 📊 Danh sách các UML Diagrams

### 1. **Activity Diagram** 🔄
**File**: `OPTIMIZATION_APPA_UML.puml`

Mô tả **luồng hoạt động chi tiết** của toàn bộ thuật toán:

```
┌─ Input & Initialization
│  ├─ Khởi tạo tham số
│  └─ Tạo APPA instance
│
├─ Phase 1 & 2: APPA (Allocation + Ordering)
│  ├─ Tính toán matrices (D, TS, TF)
│  ├─ Phân bổ vùng (ETR-based)
│  ├─ Tối ưu thứ tự (ACS per UAV)
│  └─ Tính thời gian ban đầu T₀
│
├─ Iterative Refinement Loop
│  └─ repeat max_iterations times:
│     ├─ Gradient-Based Refinement
│     │  └─ Evaluate all possible region moves
│     │  └─ Apply best move if improvement > 0
│     │
│     ├─ ACS Reordering
│     │  └─ Reorder each UAV's route
│     │
│     └─ Convergence Check
│        ├─ Calculate new completion time
│        ├─ Check best solution update
│        └─ Check convergence conditions
│
└─ Output & Results
   ├─ Tính toán metrics
   ├─ In kết quả
   └─ Return values
```

**Các điểm quyết định (Decision Points)**:
- ✓ Improvement found? → Reset stagnation
- ✓ Stagnation ≥ 10? → Exit loop
- ✓ T_new < best_time? → Update best
- ✓ |improvement| < threshold? → Exit loop
- ✓ iteration ≤ max_iterations? → Continue

---

### 2. **State Diagram** 🔀
**File**: `OPTIMIZATION_APPA_STATECHART.puml`

Mô tả **các trạng thái và chuyển tiếp trạng thái**:

```
[Start]
   ↓
[Initialization] 
   - Initialize parameters
   - Create APPA instance
   - Precompute matrices
   ↓
[APPA Phase 1: Allocation]
   - Loop: Find UAV with min time
   - Loop: Find region with max ETR
   - Assign & update finish_time
   ↓
[APPA Phase 2: Order Optimization]
   - Loop: Construct tours (ACS)
   - Loop: Update pheromone
   - Find best tour
   ↓
[Initial Evaluation]
   - Convert results to assignment dict
   - Calculate T₀
   - Initialize history
   ↓
[Iterative Loop] ←┐
   │              │
   ├─ Gradient Refinement
   │  ├─ Evaluate moves
   │  ├─ Find best move
   │  └─ Apply if improvement > 0
   │
   ├─ ACS Reordering
   │  └─ Reorder routes
   │
   ├─ Convergence Check
   │  ├─ Check threshold → [Exit: Converged]
   │  ├─ Check stagnation → [Exit: Stagnation]
   │  ├─ Check max_iterations → [Exit: Max reached]
   │  └─ Loop → ┘
   │
   ↓
[Final Output]
   - Calculate metrics
   - Print results
   - Return values
   ↓
[End]
```

**Trạng thái Exit**:
1. `Exit_Converged`: improvement < threshold (1.0s)
2. `Exit_Stagnation`: No improvement for 10 iterations
3. `Max_Iterations_Exit`: iteration > max_iterations

---

### 3. **Class Diagram** 📦
**File**: `OPTIMIZATION_APPA_CLASSES.puml`

Mô tả **cấu trúc classes và mối quan hệ**:

```
┌─────────────────────────────────────┐
│         Utils Package               │
├─────────────────────────────────────┤
│ ◆ Region                            │
│   - id: int                         │
│   - coords: Tuple[float, float]     │
│   - area: float                     │
│                                     │
│ ◆ UAV                               │
│   - id: int                         │
│   - max_velocity: float             │
│   - scan_width: float               │
│                                     │
│ ◆ APPAAlgorithm                     │
│   - solve()                         │
│   - region_allocation_phase()       │
│   - order_optimization_phase()      │
└─────────────────────────────────────┘
         △           △
         │           │
         └─────┬─────┘
              │ uses
              │
    ┌─────────────────────────────────────┐
    │  Optimization_APPA Package          │
    ├─────────────────────────────────────┤
    │ ◆ OptimizationAPPA                  │
    │   + gradient_based_refinement()     │
    │   + estimate_time_change()          │
    │   + iterative_appa_with_gradient()  │
    │   + calculate_path_completion_time()│
    │                                     │
    │ ◆ GradientRefinement                │
    │   - refine()                        │
    │   - find_best_move()                │
    │                                     │
    │ ◆ OrderOptimization                 │
    │   - nearest_neighbor()              │
    │   - insertion_based()               │
    │                                     │
    │ ◆ OptimizationResult                │
    │   - best_assignment: Dict           │
    │   - history: List[float]            │
    │   - all_assignments: List[Dict]     │
    └─────────────────────────────────────┘
```

---

### 4. **Sequence Diagram** 📐
**File**: `OPTIMIZATION_APPA_SEQUENCE.puml`

Mô tả **thứ tự gọi hàm và tương tác giữa các module**:

```
Main
  ↓
OptimizationAPPA.iterative_appa_with_gradient()
  ├─ APPA.solve()
  │  ├─ region_allocation_phase()
  │  └─ order_optimization_phase()
  │
  ├─ Utils.calculate_system_completion_time() → T₀
  │
  └─ Loop [iterations]:
     ├─ GradientRefinement.refine()
     │  ├─ For each move:
     │  │  └─ Utils.calculate_system_completion_time()
     │  │
     │  └─ Return: new_assignment, improved_flag
     │
     ├─ APPA.order_optimization_phase() per UAV
     │  └─ Return: optimized_regions
     │
     ├─ Utils.calculate_system_completion_time() → T_new
     │
     └─ Check convergence
```

---

### 5. **Component Overview Diagram** 🏗️
**File**: `OPTIMIZATION_APPA_OVERVIEW.puml`

Mô tả **tổng quan các components và dependencies**:

```
┌───────────────────────────┐
│    Input Data             │
├───────────────────────────┤
│ • UAVs                    │
│ • Regions                 │
│ • Velocity Matrix         │
└───────┬────────┬──────────┘
        │        │
        ↓        ↓
    ┌────────────────────────────┐
    │ Phase 1: APPA Init         │
    ├────────────────────────────┤
    │ → PrecomputeMatrices       │
    │ → RegionAllocation (ETR)   │
    │ → OrderOptimization (ACS)  │
    └────────────────────────────┘
               ↓
    ┌────────────────────────────┐
    │ Phase 2: Iterative Refine  │
    ├────────────────────────────┤
    │ ⊕ Gradient Refinement      │
    │   - Evaluate all moves     │
    │   - Find & apply best      │
    │                            │
    │ ⊕ ACS Reordering           │
    │   - Reorder routes         │
    │                            │
    │ ⊕ Convergence Analysis     │
    │   - Check thresholds       │
    │   - Check stagnation       │
    └────────────────────────────┘
               ↓
    ┌────────────────────────────┐
    │ Phase 3: Results           │
    ├────────────────────────────┤
    │ ◇ best_assignment: Dict    │
    │ ◇ history: List[float]     │
    │ ◇ all_assignments: List    │
    └────────────────────────────┘
```

---

## 🔍 Hướng dẫn sử dụng các UML Diagrams

### Mở bằng PlantUML:
1. **Online**: https://www.plantuml.com/plantuml/uml/
   - Copy nội dung file `.puml` vào editor
   - Click "Submit" để xem diagram

2. **VS Code Extension**:
   - Cài: PlantUML (`jebbs.plantuml`)
   - Open file `.puml`
   - Alt+D để preview

3. **Command Line**:
   ```bash
   # Cần cài plantuml trước
   plantuml OPTIMIZATION_APPA_UML.puml
   # Output: OPTIMIZATION_APPA_UML.png
   ```

### Export:
- **PNG**: Right-click preview → Export PNG
- **PDF**: https://www.plantuml.com/plantuml/pdf/
- **SVG**: https://www.plantuml.com/plantuml/svg/

---

## 📋 Mapping giữa Diagrams

| Diagram | Tập trung vào | Dùng để |
|---------|-------------|--------|
| Activity | Luồng hoạt động | Hiểu flow tổng thể |
| State | Các trạng thái & transitions | Debug state issues |
| Class | Cấu trúc dữ liệu & relationships | Hiểu architecture |
| Sequence | Thứ tự gọi hàm & timing | Trace execution |
| Overview | Tổng quan components | Quick reference |

---

## 🎓 Learning Path

**Bắt đầu từ**:
1. **Overview** - Hiểu big picture
2. **Activity** - Học chi tiết flow
3. **State** - Hiểu trạng thái & convergence
4. **Sequence** - Trace cách hàm gọi nhau
5. **Class** - Hiểu cấu trúc code

---

## 📝 Ghi chú

**Độ phức tạp**:
- APPA Phase 1: O(n_uavs × n_regions²)
- APPA Phase 2: O(n_uavs × n_ants × n_generations × n_regions²)
- Gradient: O(n_uavs² × n_regions × T_calc)
- **Tổng**: O(iterations × gradient_cost)

**Convergence**:
1. Threshold: improvement < 1.0 second
2. Stagnation: 10 no-improve iterations
3. Max iterations: default = 10

**Khuyến nghị**:
- Sử dụng max_iterations = 5-10
- convergence_threshold = 1.0
- n_generations (ACS) = 30-50

