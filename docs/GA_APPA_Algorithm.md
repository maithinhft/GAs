# GA-APPA: Genetic Algorithm + Ant Colony System cho Coverage Path Planning

## Tổng quan

**GA-APPA** (Genetic Algorithm - Adaptive Path Planning Algorithm) là thuật toán kết hợp hai phương pháp metaheuristic để giải quyết bài toán lập kế hoạch đường bay cho nhiều UAV không đồng nhất:

- **Pha 1 (Region Allocation)**: Sử dụng **Genetic Algorithm (GA)** để phân bổ vùng cho các UAV
- **Pha 2 (Order Optimization)**: Sử dụng **Ant Colony System (ACS)** để tối ưu thứ tự bay qua các vùng

---

## 1. Định nghĩa bài toán

### 1.1 Đầu vào
- $n$: Số lượng UAV
- $m$: Số lượng vùng cần quét
- $V_{i,j}$: Vận tốc quét của UAV $i$ tại vùng $j$ (nếu $V_{i,j} = 0$ thì UAV $i$ không thể quét vùng $j$)
- $v_i^{max}$: Vận tốc di chuyển tối đa của UAV $i$
- $w_i$: Độ rộng quét của UAV $i$
- $A_j$: Diện tích vùng $j$
- $(x_j, y_j)$: Tọa độ tâm vùng $j$

### 1.2 Đầu ra
- Phân bổ vùng cho từng UAV: $\{R_i\}_{i=1}^{n}$ với $R_i$ là tập vùng được gán cho UAV $i$
- Thứ tự bay tối ưu cho mỗi UAV

### 1.3 Mục tiêu
Tối thiểu hóa thời gian hoàn thành tối đa:
$$\min \max_{i \in \{1,...,n\}} T_i$$

với $T_i$ là tổng thời gian hoàn thành của UAV $i$.

---

## 2. Pha 1: Genetic Algorithm cho Region Allocation

### 2.1 Biểu diễn Chromosome

**Cấu trúc**: Chromosome là một mảng có độ dài $m$ (số vùng)
- Gene tại vị trí $j$: chỉ số UAV được gán cho vùng $j$
- Giá trị: $0$ đến $n-1$

**Ví dụ**: Với 4 UAV và 6 vùng:
```
Chromosome: [0, 1, 0, 2, 1, 3]
Ý nghĩa: Region 0→UAV0, Region 1→UAV1, Region 2→UAV0, 
         Region 3→UAV2, Region 4→UAV1, Region 5→UAV3
```

### 2.2 Khởi tạo quần thể thông minh

Quần thể ban đầu được khởi tạo bằng 4 phương pháp:

#### a) Pure Greedy (ETR Heuristic) - 1 cá thể
Sử dụng **Effective Time Ratio** như APPA gốc:
$$ETR_{i,j,k} = \frac{TS_{i,k}}{TF_{i,j,k} + TS_{i,k}}$$

#### b) Stochastic Greedy (ε-greedy) - ~33% quần thể
- Với xác suất $\epsilon$: chọn ngẫu nhiên theo trọng số ETR (roulette wheel)
- Với xác suất $1-\epsilon$: chọn vùng có ETR cao nhất
- Các giá trị $\epsilon$: {0.1, 0.2, 0.3, 0.4, 0.5}

#### c) Workload-Balanced - ~20% quần thể
- Duyệt vùng theo thứ tự ngẫu nhiên
- Gán mỗi vùng cho UAV có workload thấp nhất

#### d) Random Feasible - phần còn lại
- Gán ngẫu nhiên mỗi vùng cho một UAV khả thi

### 2.3 Hàm Fitness

$$fitness = \frac{1}{\max_{i \in UAVs}(T_i) + \epsilon}$$

với:
- $T_i$: thời gian hoàn thành của UAV $i$ (tính bằng Nearest Neighbor heuristic)
- $\epsilon = 10^{-6}$: hằng số nhỏ tránh chia cho 0

**Mục tiêu**: Tối đa hóa fitness ⟺ Tối thiểu hóa max completion time

### 2.4 Toán tử di truyền

#### a) Tournament Selection ($k=3$)
```
1. Chọn ngẫu nhiên k cá thể từ quần thể
2. Trả về cá thể có fitness cao nhất
```
**Độ phức tạp**: $O(k)$

#### b) Uniform Crossover ($p_c = 0.8$)
```
Với mỗi gene j:
    if random() < 0.5:
        child1[j] = parent1[j]
        child2[j] = parent2[j]
    else:
        child1[j] = parent2[j]
        child2[j] = parent1[j]
```
**Độ phức tạp**: $O(m)$

#### c) Random Mutation ($p_m = 0.1$)
```
Với mỗi gene j:
    if random() < p_m:
        gene[j] = UAV khả thi khác ngẫu nhiên
```
**Độ phức tạp**: $O(m)$

#### d) Repair
```
Với mỗi gene j:
    if V[gene[j], j] ≤ 0:
        gene[j] = UAV khả thi ngẫu nhiên
```
**Độ phức tạp**: $O(m \cdot n)$ worst case

### 2.5 Early Stopping

Thuật toán dừng sớm khi:
- Đạt số generation tối đa ($G_{max} = 100$)
- HOẶC không có cải thiện sau $E_{stop} = 15$ generations liên tiếp

### 2.6 Pseudocode Pha 1

```
Algorithm: GA Phase 1 - Region Allocation
Input: n UAVs, m regions, matrices TS, TF, V
Output: Region assignment {R_i} for each UAV i

1.  P ← SmartInitializePopulation(N_pop)  // N_pop = 50
2.  fitness[] ← EvaluatePopulation(P)
3.  best ← argmax(fitness)
4.  no_improvement ← 0
5.  
6.  for g = 1 to G_max do:                // G_max = 100
7.      P' ← Elitism(P, fitness, e)       // e = 2 best individuals
8.      
9.      while |P'| < N_pop do:
10.         p1 ← TournamentSelection(P, fitness, k=3)
11.         p2 ← TournamentSelection(P, fitness, k=3)
12.         c1, c2 ← UniformCrossover(p1, p2, p_c=0.8)
13.         c1 ← Mutate(c1, p_m=0.1)
14.         c2 ← Mutate(c2, p_m=0.1)
15.         c1 ← Repair(c1)
16.         c2 ← Repair(c2)
17.         P' ← P' ∪ {c1, c2}
18.     end while
19.     
20.     P ← P'
21.     fitness[] ← EvaluatePopulation(P)
22.     
23.     if max(fitness) > best_fitness then:
24.         best ← argmax(fitness)
25.         no_improvement ← 0
26.     else:
27.         no_improvement ← no_improvement + 1
28.     
29.     if no_improvement ≥ E_stop then:  // E_stop = 15
30.         break
31. end for
32. 
33. return ChromosomeToAssignment(P[best])
```

---

## 3. Pha 2: Ant Colony System cho Order Optimization

### 3.1 Mô tả

Với mỗi UAV $i$ và tập vùng được gán $R_i$, tìm thứ tự bay tối ưu sử dụng ACS.

### 3.2 Các tham số ACS

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| $N_{ants}$ | 20 | Số kiến |
| $I_{max}$ | 100 | Số iteration tối đa |
| $\alpha$ | 1.0 | Trọng số pheromone |
| $\beta$ | 2.0 | Trọng số heuristic |
| $\rho$ | 0.1 | Tỷ lệ bay hơi local |
| $\epsilon$ | 0.1 | Tỷ lệ bay hơi global |
| $q_0$ | 0.9 | Ngưỡng exploitation |

### 3.3 Quy tắc chọn vùng tiếp theo

$$j = \begin{cases}
\arg\max_{l \in J_k} \{\tau_{il}^\alpha \cdot \eta_{il}^\beta\} & \text{if } q \leq q_0 \text{ (exploitation)}\\
\text{roulette wheel selection} & \text{otherwise (exploration)}
\end{cases}$$

với:
- $\tau_{il}$: pheromone giữa vùng $i$ và $l$
- $\eta_{il} = 1/d_{il}$: heuristic information
- $J_k$: tập vùng chưa thăm của kiến $k$

### 3.4 Cập nhật Pheromone

**Local update** (sau mỗi bước di chuyển):
$$\tau_{ij} \leftarrow (1 - \rho) \cdot \tau_{ij} + \rho \cdot \tau_0$$

**Global update** (sau mỗi iteration):
$$\tau_{ij} \leftarrow (1 - \epsilon) \cdot \tau_{ij} + \epsilon \cdot \frac{1}{L_{best}}$$

---

## 4. Phân tích độ phức tạp

### 4.1 Ký hiệu
- $n$: số UAV
- $m$: số vùng (regions)
- $N_{pop}$: kích thước quần thể GA
- $G$: số generation thực tế (≤ $G_{max}$)
- $N_{ants}$: số kiến trong ACS
- $I$: số iteration ACS thực tế (≤ $I_{max}$)
- $k_i$: số vùng được gán cho UAV $i$

### 4.2 Độ phức tạp thời gian

#### Pha 1: Genetic Algorithm

| Thành phần | Độ phức tạp | Giải thích |
|------------|-------------|------------|
| **Khởi tạo ma trận** | | |
| - Distance matrix $D$ | $O(m^2)$ | Tính khoảng cách giữa $m$ vùng |
| - Scan time matrix $TS$ | $O(n \cdot m)$ | $n$ UAV × $m$ vùng |
| - Flight time matrix $TF$ | $O(n \cdot m^2)$ | $n$ UAV × $m^2$ cặp vùng |
| - Base flight time | $O(n \cdot m)$ | $n$ UAV × $m$ vùng |
| **Khởi tạo quần thể** | | |
| - Greedy chromosome | $O(m^2)$ | Duyệt $m$ vùng, mỗi lần tìm best trong $O(m)$ |
| - Stochastic greedy | $O(N_{pop}/3 \cdot m^2)$ | ~33% quần thể |
| - Balanced | $O(N_{pop}/5 \cdot m \cdot n)$ | ~20% quần thể |
| - Random | $O(N_{pop} \cdot m)$ | Phần còn lại |
| **Một generation** | | |
| - Evaluate 1 chromosome | $O(m + \sum_i k_i^2)$ | Nearest neighbor cho mỗi UAV |
| - Evaluate population | $O(N_{pop} \cdot m \cdot \bar{k})$ | $\bar{k}$: trung bình vùng/UAV |
| - Selection | $O(N_{pop} \cdot k)$ | Tournament selection |
| - Crossover + Mutation | $O(N_{pop} \cdot m)$ | Duyệt từng gene |
| - Repair | $O(N_{pop} \cdot m)$ | Kiểm tra feasibility |

**Tổng hợp Pha 1:**

$$T_{GA} = O(n \cdot m^2) + O(G \cdot N_{pop} \cdot m \cdot \bar{k})$$

Với $\bar{k} = m/n$ (trung bình vùng/UAV):

$$\boxed{T_{GA} = O(n \cdot m^2 + G \cdot N_{pop} \cdot m^2 / n)}$$

#### Pha 2: Ant Colony System (cho mỗi UAV)

| Thành phần | Độ phức tạp | Giải thích |
|------------|-------------|------------|
| Khởi tạo pheromone | $O(k_i^2)$ | Ma trận $k_i \times k_i$ |
| Heuristic matrix | $O(k_i^2)$ | Ma trận $k_i \times k_i$ |
| Một kiến xây tour | $O(k_i^2)$ | $k_i$ bước, mỗi bước tìm trong $O(k_i)$ |
| Một iteration | $O(N_{ants} \cdot k_i^2)$ | $N_{ants}$ kiến |
| Toàn bộ ACS | $O(I \cdot N_{ants} \cdot k_i^2)$ | $I$ iterations |

**Tổng hợp Pha 2 (tất cả UAV):**

$$T_{ACS} = \sum_{i=1}^{n} O(I \cdot N_{ants} \cdot k_i^2)$$

Worst case khi một UAV nhận tất cả vùng:

$$\boxed{T_{ACS} = O(n \cdot I \cdot N_{ants} \cdot (m/n)^2) = O(I \cdot N_{ants} \cdot m^2 / n)}$$

#### Tổng độ phức tạp thời gian

$$\boxed{T_{total} = O\left(n \cdot m^2 + G \cdot N_{pop} \cdot \frac{m^2}{n} + I \cdot N_{ants} \cdot \frac{m^2}{n}\right)}$$

**Với các tham số mặc định** ($N_{pop}=50$, $G_{max}=100$, $N_{ants}=20$, $I_{max}=100$):

$$T_{total} = O\left(m^2 \cdot \left(n + \frac{G \cdot N_{pop} + I \cdot N_{ants}}{n}\right)\right)$$

**Đơn giản hóa:**
- Nếu $n$ nhỏ và $m$ lớn: $T_{total} \approx O(G \cdot N_{pop} \cdot m^2 / n)$
- Nếu $n \approx m$: $T_{total} \approx O(n \cdot m^2)$

### 4.3 Độ phức tạp bộ nhớ

| Thành phần | Kích thước | Giải thích |
|------------|------------|------------|
| **Ma trận tiền xử lý** | | |
| Distance matrix $D$ | $O(m^2)$ | $m \times m$ |
| Scan time matrix $TS$ | $O(n \cdot m)$ | $n \times m$ |
| Flight time matrix $TF$ | $O(n \cdot m^2)$ | $n \times m \times m$ |
| Base flight time | $O(n \cdot m)$ | $n \times m$ |
| Feasibility matrix | $O(n \cdot m)$ | $n \times m$ |
| **GA Phase 1** | | |
| Population | $O(N_{pop} \cdot m)$ | $N_{pop}$ chromosomes |
| Fitness cache | $O(C \cdot m)$ | $C$: cache size (≤10000) |
| Fitness values | $O(N_{pop})$ | Mảng 1D |
| **ACS Phase 2** | | |
| Pheromone matrix | $O(k_{max}^2)$ | $k_{max}$: max vùng/UAV |
| Heuristic matrix | $O(k_{max}^2)$ | $k_{max} \times k_{max}$ |
| Ant tours | $O(N_{ants} \cdot k_{max})$ | Tạm thời |

**Tổng độ phức tạp bộ nhớ:**

$$\boxed{S_{total} = O(n \cdot m^2 + N_{pop} \cdot m + C \cdot m)}$$

**Với giá trị mặc định:**
- Thành phần chiếm chủ đạo: $TF$ matrix với $O(n \cdot m^2)$
- Với $n=8$, $m=50$: ~20,000 phần tử cho $TF$

---

## 5. So sánh với APPA gốc

| Tiêu chí | APPA | GA-APPA |
|----------|------|---------|
| **Pha 1** | Greedy ETR $O(m^2)$ | GA $O(G \cdot N_{pop} \cdot m^2/n)$ |
| **Pha 2** | ACS | ACS (giống nhau) |
| **Thời gian** | Nhanh hơn | Chậm hơn ~3-5x |
| **Chất lượng** | Tốt | Tốt hơn ~5-28% |
| **Khám phá** | Hạn chế (greedy) | Tốt (population-based) |

---

## 6. Tham số được đề xuất

### 6.1 GA Parameters

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `population_size` | 50 | Kích thước quần thể |
| `max_generations` | 100 | Số generation tối đa |
| `crossover_rate` | 0.8 | Xác suất lai ghép |
| `mutation_rate` | 0.1 | Xác suất đột biến |
| `tournament_size` | 3 | Kích thước tournament |
| `elitism_count` | 2 | Số cá thể ưu tú giữ lại |
| `early_stop_generations` | 15 | Số generation không cải thiện để dừng |

### 6.2 ACS Parameters

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `num_ants` | 20 | Số kiến |
| `max_iterations` | 100 | Số iteration tối đa |
| `alpha` | 1.0 | Trọng số pheromone |
| `beta` | 2.0 | Trọng số heuristic |
| `rho` | 0.1 | Tỷ lệ bay hơi local |
| `epsilon` | 0.1 | Tỷ lệ bay hơi global |
| `q0` | 0.9 | Ngưỡng exploitation |

---

## 7. Kết quả thực nghiệm

### 7.1 Điều kiện test
- Python 3.12
- NumPy 1.26
- Chạy 3 lần, lấy trung bình

### 7.2 Kết quả

| Test Case | APPA Time | GA-APPA Time | Generations | Quality Improvement |
|-----------|-----------|--------------|-------------|---------------------|
| 4 UAVs, 20 Regions | 0.042s | 0.143s | 50 | **+12.9%** |
| 6 UAVs, 30 Regions | 0.049s | 0.181s | 39 | **+27.8%** |
| 8 UAVs, 50 Regions | 0.082s | 0.219s | 22 | **+1.7%** |

**Nhận xét:**
- GA-APPA chậm hơn ~3-4x nhưng cho kết quả tốt hơn đáng kể
- Early stopping giúp giảm 50-78% số generation cần thiết
- Cải thiện chất lượng từ 1.7% đến 27.8% tùy bài toán

---

## 8. Ưu điểm và hạn chế

### 8.1 Ưu điểm
1. **Chất lượng giải pháp tốt hơn** do khám phá không gian tìm kiếm rộng hơn
2. **Khởi tạo thông minh** kết hợp greedy và ngẫu nhiên
3. **Early stopping** tránh lãng phí tính toán
4. **Fitness caching** tăng tốc đánh giá
5. **Đa dạng quần thể** với nhiều chiến lược khởi tạo

### 8.2 Hạn chế
1. **Thời gian chạy cao hơn** APPA gốc
2. **Nhiều tham số** cần điều chỉnh
3. **Bộ nhớ lớn hơn** do lưu quần thể và cache

---

## 9. Tài liệu tham khảo

1. Coverage path planning of heterogeneous UAVs based on ACS
2. Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning
3. Dorigo, M., & Gambardella, L. M. (1997). Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem
