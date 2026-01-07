"""
So sánh toàn diện các thuật toán tối ưu hóa cho bài toán UAV Coverage Path Planning

Các thuật toán so sánh:
1. APPA (Original) - Sử dụng ETR heuristic cho Phase 1
2. GA-APPA - Sử dụng Genetic Algorithm cho Phase 1
3. PSO-APPA - Sử dụng Particle Swarm Optimization cho Phase 1
4. GWO-APPA - Sử dụng Grey Wolf Optimizer cho Phase 1
5. Pure GA - Sử dụng GA cho cả allocation và ordering

Các metrics so sánh:
- Max Completion Time (thời gian hoàn thành tối đa)
- Workload Balance (cân bằng tải)
- Efficiency Ratio (tỷ lệ hiệu quả)
- Execution Time (thời gian thực thi)
- Deviation Ratio (tỷ lệ cải thiện)
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import algorithms
from algorithm.appa import APPAAlgorithm
from algorithm.ga_appa import GAPPAAlgorithm
from algorithm.pso_appa import PSOAPPAAlgorithm
from algorithm.gwo_appa import GWOAPPAAlgorithm
from algorithm.ga_pure import PureGAAlgorithm
from utils.create_sample import create_sample
from utils.config import UAV, Region, RANDOM_SEED
from utils.metrics import MetricsCalculator

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Create output directory
os.makedirs('./fig', exist_ok=True)

print("Đã import thành công các thư viện!")
print("="*60)
print("THUẬT TOÁN SO SÁNH:")
print("1. APPA (Original) - ETR heuristic")
print("2. GA-APPA - Genetic Algorithm + ACS")
print("3. PSO-APPA - Particle Swarm Optimization + ACS")
print("4. GWO-APPA - Grey Wolf Optimizer + ACS")
print("5. Pure GA - Genetic Algorithm only")
print("="*60)


# ============================================================================
# 1. Tạo dữ liệu thử nghiệm
# ============================================================================
def generate_test_data(num_uavs=4, num_regions=30, seed=RANDOM_SEED):
    """Tạo dữ liệu thử nghiệm"""
    random.seed(seed)
    np.random.seed(seed)
    
    data = create_sample(
        NUM_UAVS=num_uavs, 
        NUM_REGIONS=num_regions,
        SYSTEM_AREA_RATIO=0.05,
        SYSTEM_DRAG_FACTOR=0.9
    )
    
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = np.array(data['V_matrix'])
    
    return uavs_list, regions_list, V_matrix


# ============================================================================
# 2. Định nghĩa các thuật toán và hàm chạy
# ============================================================================
ALGORITHM_CONFIGS = {
    'APPA': {
        'name': 'APPA (Original)',
        'color': '#3498db',
        'marker': 'o'
    },
    'GA-APPA': {
        'name': 'GA-APPA',
        'color': '#e74c3c',
        'marker': 's'
    },
    'PSO-APPA': {
        'name': 'PSO',
        'color': '#2ecc71',
        'marker': '^'
    },
    'GWO-APPA': {
        'name': 'GWO',
        'color': '#9b59b6',
        'marker': 'D'
    },
    'Pure-GA': {
        'name': 'Pure GA',
        'color': '#f39c12',
        'marker': 'v'
    }
}


def run_algorithm(algorithm_name: str, uavs_list, regions_list, V_matrix) -> Tuple[Dict, float]:
    """Chạy một thuật toán và trả về kết quả + thời gian thực thi"""
    start_time = time.time()
    
    if algorithm_name == 'APPA':
        algo = APPAAlgorithm(uavs_list, regions_list, V_matrix.tolist())
        result = algo.solve()
        
    elif algorithm_name == 'GA-APPA':
        algo = GAPPAAlgorithm(
            uavs_list, regions_list, V_matrix.tolist(),
            ga_population_size=50,
            ga_max_generations=100,
            ga_crossover_rate=0.8,
            ga_mutation_rate=0.1
        )
        result = algo.solve()
        
    elif algorithm_name == 'PSO-APPA':
        algo = PSOAPPAAlgorithm(
            uavs_list, regions_list, V_matrix.tolist(),
            pso_swarm_size=50,
            pso_max_iterations=100
        )
        result = algo.solve()
        
    elif algorithm_name == 'GWO-APPA':
        algo = GWOAPPAAlgorithm(
            uavs_list, regions_list, V_matrix.tolist(),
            gwo_pack_size=50,
            gwo_max_iterations=100
        )
        result = algo.solve()
        
    elif algorithm_name == 'Pure-GA':
        algo = PureGAAlgorithm(
            uavs_list, regions_list, V_matrix.tolist(),
            population_size=100,
            max_generations=150,
            crossover_rate=0.85,
            mutation_rate=0.15
        )
        result = algo.solve()
    
    exec_time = time.time() - start_time
    return result, exec_time


def run_all_algorithms(uavs_list, regions_list, V_matrix, num_runs=5) -> Dict[str, List[Dict]]:
    """Chạy tất cả các thuật toán và thu thập kết quả"""
    all_results = {name: [] for name in ALGORITHM_CONFIGS.keys()}
    metrics_calc = MetricsCalculator(uavs_list, regions_list, V_matrix)
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{num_runs}")
        print('='*60)
        
        for algo_name in ALGORITHM_CONFIGS.keys():
            print(f"  Running {algo_name}...", end=' ')
            result, exec_time = run_algorithm(algo_name, uavs_list, regions_list, V_matrix)
            
            # Calculate metrics
            metrics = metrics_calc.calculate_all_metrics(result, exec_time)
            
            # Store fitness history if available
            for key in ['ga_fitness_history', 'pso_fitness_history', 'gwo_fitness_history']:
                if key in result:
                    metrics[key] = result[key]
            
            all_results[algo_name].append(metrics)
            print(f"Max Time: {metrics['max_completion_time']:.2f}")
    
    return all_results


# ============================================================================
# 3. Tính toán thống kê
# ============================================================================
def calculate_statistics(results_dict: Dict[str, List[Dict]], metric_keys: List[str]) -> Dict[str, Dict]:
    """Tính mean, std, min, max cho các metrics của mỗi thuật toán"""
    stats = {}
    
    for algo_name, metrics_list in results_dict.items():
        stats[algo_name] = {}
        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                stats[algo_name][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    return stats


def print_comparison_table(stats: Dict, metric_keys: List[str]):
    """In bảng so sánh thống kê"""
    print("\n" + "="*120)
    print("BẢNG SO SÁNH THỐNG KÊ")
    print("="*120)
    
    # Header
    header = f"{'Metric':<25}"
    for algo_name in stats.keys():
        header += f"{ALGORITHM_CONFIGS[algo_name]['name']:<20}"
    print(header)
    print("-"*120)
    
    # Data rows
    for metric in metric_keys:
        row = f"{metric:<25}"
        for algo_name in stats.keys():
            if metric in stats[algo_name]:
                mean = stats[algo_name][metric]['mean']
                std = stats[algo_name][metric]['std']
                row += f"{mean:.4f}±{std:.4f}".ljust(20)
            else:
                row += "N/A".ljust(20)
        print(row)
    
    print("="*120)


# ============================================================================
# 4. Biểu đồ so sánh
# ============================================================================
def plot_max_completion_time(stats: Dict):
    """Biểu đồ so sánh Max Completion Time"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    algorithms = list(stats.keys())
    means = [stats[a]['max_completion_time']['mean'] for a in algorithms]
    stds = [stats[a]['max_completion_time']['std'] for a in algorithms]
    colors = [ALGORITHM_CONFIGS[a]['color'] for a in algorithms]
    labels = [ALGORITHM_CONFIGS[a]['name'] for a in algorithms]
    
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}\n(±{std:.2f})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Max Completion Time (s)', fontsize=12)
    ax.set_title('So sánh Max Completion Time giữa các thuật toán', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.35)
    
    # Add best algorithm indicator
    best_idx = np.argmin(means)
    ax.annotate('Best', xy=(best_idx, means[best_idx]), xytext=(best_idx, means[best_idx] * 0.5),
                fontsize=12, color='green', fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_max_completion_time.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_workload_balance(stats: Dict):
    """Biểu đồ so sánh Workload Balance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    algorithms = list(stats.keys())
    colors = [ALGORITHM_CONFIGS[a]['color'] for a in algorithms]
    labels = [ALGORITHM_CONFIGS[a]['name'] for a in algorithms]
    
    # Workload Variance
    variances = [stats[a]['workload_variance']['mean'] for a in algorithms]
    var_stds = [stats[a]['workload_variance']['std'] for a in algorithms]
    
    bars1 = axes[0].bar(labels, variances, yerr=var_stds, capsize=6, color=colors, 
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Workload Variance', fontsize=12)
    axes[0].set_title('Workload Variance (thấp hơn = tốt hơn)', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Workload Balance Index
    balance_idx = [stats[a]['workload_balance_index']['mean'] for a in algorithms]
    balance_stds = [stats[a]['workload_balance_index']['std'] for a in algorithms]
    
    bars2 = axes[1].bar(labels, balance_idx, yerr=balance_stds, capsize=6, color=colors,
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Workload Balance Index', fontsize=12)
    axes[1].set_title('Workload Balance Index (thấp hơn = cân bằng hơn)', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_workload_balance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_metrics(stats: Dict):
    """Biểu đồ so sánh nhiều metrics (normalized)"""
    fig, ax = plt.subplots(figsize=(16, 7))
    
    metrics_to_plot = [
        ('max_completion_time', 'Max Time', 'lower'),
        ('avg_completion_time', 'Avg Time', 'lower'),
        ('workload_balance_index', 'Workload\nBalance', 'lower'),
        ('total_distance', 'Total\nDistance', 'lower'),
        ('efficiency_ratio', 'Efficiency', 'higher'),
        ('avg_uav_utilization', 'Utilization', 'higher'),
    ]
    
    algorithms = list(stats.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.15
    
    # Normalize values
    all_values = {}
    for metric, _, _ in metrics_to_plot:
        values = [stats[a][metric]['mean'] for a in algorithms]
        max_val = max(values)
        all_values[metric] = [v / max_val if max_val > 0 else 0 for v in values]
    
    # Plot bars for each algorithm
    for i, algo in enumerate(algorithms):
        values = [all_values[m[0]][i] for m in metrics_to_plot]
        offset = (i - len(algorithms)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                      label=ALGORITHM_CONFIGS[algo]['name'],
                      color=ALGORITHM_CONFIGS[algo]['color'],
                      alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('So sánh các Metrics giữa các thuật toán (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics_to_plot], fontsize=10)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.set_ylim(0, 1.2)
    
    # Add direction indicators
    for i, (_, _, direction) in enumerate(metrics_to_plot):
        symbol = '↓ better' if direction == 'lower' else '↑ better'
        ax.annotate(symbol, xy=(i, 0.02), ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_execution_time(stats: Dict):
    """Biểu đồ so sánh thời gian thực thi"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    algorithms = list(stats.keys())
    exec_times = [stats[a]['execution_time']['mean'] for a in algorithms]
    exec_stds = [stats[a]['execution_time']['std'] for a in algorithms]
    colors = [ALGORITHM_CONFIGS[a]['color'] for a in algorithms]
    labels = [ALGORITHM_CONFIGS[a]['name'] for a in algorithms]
    
    bars = ax.bar(labels, exec_times, yerr=exec_stds, capsize=8, color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, t, std in zip(bars, exec_times, exec_stds):
        height = bar.get_height()
        ax.annotate(f'{t:.3f}s\n(±{std:.3f})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Execution Time (s)', fontsize=12)
    ax.set_title('So sánh thời gian thực thi', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(exec_times) * 1.4)
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_execution_time.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_deviation_ratio(stats: Dict):
    """Biểu đồ Deviation Ratio so với APPA baseline"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    deviation_metrics = [
        ('max_completion_time', 'Max Time', 'lower'),
        ('avg_completion_time', 'Avg Time', 'lower'),
        ('workload_variance', 'Workload Var', 'lower'),
        ('workload_balance_index', 'Balance Index', 'lower'),
        ('total_distance', 'Total Distance', 'lower'),
        ('efficiency_ratio', 'Efficiency', 'higher'),
    ]
    
    # Use APPA as baseline
    baseline = 'APPA'
    algorithms = [a for a in stats.keys() if a != baseline]
    
    x = np.arange(len(deviation_metrics))
    width = 0.2
    
    for i, algo in enumerate(algorithms):
        deviations = []
        colors = []
        
        for metric, label, direction in deviation_metrics:
            baseline_val = stats[baseline][metric]['mean']
            algo_val = stats[algo][metric]['mean']
            
            if direction == 'lower':
                deviation = ((algo_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else 0
                is_better = deviation < 0
            else:
                deviation = ((algo_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else 0
                is_better = deviation > 0
            
            deviations.append(deviation)
        
        offset = (i - len(algorithms)/2 + 0.5) * width
        bars = ax.bar(x + offset, deviations, width, 
                      label=ALGORITHM_CONFIGS[algo]['name'],
                      color=ALGORITHM_CONFIGS[algo]['color'],
                      alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, dev in zip(bars, deviations):
            height = bar.get_height()
            offset_y = 0.5 if height >= 0 else -2
            ax.annotate(f'{dev:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset_y),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=8, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Deviation Ratio vs APPA (%)', fontsize=12)
    ax.set_title('Deviation Ratio: So với APPA baseline\n(Âm = cải thiện cho metrics "lower is better")', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in deviation_metrics], fontsize=10)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_deviation_ratio.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence_curves(all_results: Dict):
    """Biểu đồ đường cong hội tụ"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get fitness histories from last run
    for algo_name, results in all_results.items():
        if algo_name == 'APPA':
            continue
            
        last_run = results[-1]
        
        fitness_history = None
        for key in ['ga_fitness_history', 'pso_fitness_history', 'gwo_fitness_history']:
            if key in last_run and last_run[key]:
                fitness_history = last_run[key]
                break
        
        if fitness_history:
            # Convert fitness to completion time
            completion_times = [1.0 / (f + 1e-6) if f > 0 else 0 for f in fitness_history]
            generations = range(len(completion_times))
            
            ax.plot(generations, completion_times, 
                    color=ALGORITHM_CONFIGS[algo_name]['color'],
                    linewidth=2, 
                    marker=ALGORITHM_CONFIGS[algo_name]['marker'],
                    markevery=max(1, len(generations)//10),
                    label=ALGORITHM_CONFIGS[algo_name]['name'])
    
    # Add APPA baseline
    appa_baseline = np.mean([r['max_completion_time'] for r in all_results['APPA']])
    ax.axhline(y=appa_baseline, color='#3498db', linestyle='--', linewidth=2, 
               label=f'APPA Baseline ({appa_baseline:.2f})')
    
    ax.set_xlabel('Generation / Iteration', fontsize=12)
    ax.set_ylabel('Max Completion Time', fontsize=12)
    ax.set_title('Đường cong hội tụ của các thuật toán', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_boxplot_comparison(all_results: Dict):
    """Boxplot so sánh phân bố kết quả"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    algorithms = list(all_results.keys())
    colors = [ALGORITHM_CONFIGS[a]['color'] for a in algorithms]
    labels = [ALGORITHM_CONFIGS[a]['name'] for a in algorithms]
    
    # Max Completion Time
    max_times = [[r['max_completion_time'] for r in all_results[a]] for a in algorithms]
    bp1 = axes[0].boxplot(max_times, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Max Completion Time (s)', fontsize=12)
    axes[0].set_title('Phân bố Max Completion Time', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Workload Variance
    workload_vars = [[r['workload_variance'] for r in all_results[a]] for a in algorithms]
    bp2 = axes[1].boxplot(workload_vars, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Workload Variance', fontsize=12)
    axes[1].set_title('Phân bố Workload Variance', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 5. Scalability Test
# ============================================================================
def run_scalability_test(num_uavs=4, regions_range=[10, 20, 30, 40, 50]):
    """Test scalability với số lượng regions khác nhau"""
    results = {
        'regions': [],
    }
    
    for algo_name in ALGORITHM_CONFIGS.keys():
        results[f'{algo_name}_time'] = []
        results[f'{algo_name}_max'] = []
    
    for num_regions in regions_range:
        print(f"\nTesting with {num_regions} regions...")
        
        uavs, regions, v_matrix = generate_test_data(num_uavs=num_uavs, num_regions=num_regions)
        results['regions'].append(num_regions)
        
        for algo_name in ALGORITHM_CONFIGS.keys():
            print(f"  Running {algo_name}...", end=' ')
            result, exec_time = run_algorithm(algo_name, uavs, regions, v_matrix)
            
            results[f'{algo_name}_time'].append(exec_time)
            results[f'{algo_name}_max'].append(result['max_completion_time'])
            print(f"Max: {result['max_completion_time']:.2f}, Time: {exec_time:.3f}s")
    
    return results


def plot_scalability(scalability_results: Dict):
    """Biểu đồ scalability"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    regions = scalability_results['regions']
    
    # Max Completion Time vs Regions
    for algo_name in ALGORITHM_CONFIGS.keys():
        axes[0].plot(regions, scalability_results[f'{algo_name}_max'],
                     color=ALGORITHM_CONFIGS[algo_name]['color'],
                     marker=ALGORITHM_CONFIGS[algo_name]['marker'],
                     linewidth=2, markersize=8,
                     label=ALGORITHM_CONFIGS[algo_name]['name'])
    
    axes[0].set_xlabel('Số lượng Regions', fontsize=12)
    axes[0].set_ylabel('Max Completion Time (s)', fontsize=12)
    axes[0].set_title('Max Completion Time theo số lượng Regions', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Execution Time vs Regions
    for algo_name in ALGORITHM_CONFIGS.keys():
        axes[1].plot(regions, scalability_results[f'{algo_name}_time'],
                     color=ALGORITHM_CONFIGS[algo_name]['color'],
                     marker=ALGORITHM_CONFIGS[algo_name]['marker'],
                     linewidth=2, markersize=8,
                     label=ALGORITHM_CONFIGS[algo_name]['name'])
    
    axes[1].set_xlabel('Số lượng Regions', fontsize=12)
    axes[1].set_ylabel('Execution Time (s)', fontsize=12)
    axes[1].set_title('Thời gian thực thi theo số lượng Regions', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_scalability.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 6. Radar Chart
# ============================================================================
def plot_radar_chart(stats: Dict):
    """Radar chart so sánh các thuật toán"""
    from math import pi
    
    metrics = [
        ('max_completion_time', 'Max Time', 'lower'),
        ('workload_variance', 'Workload Var', 'lower'),
        ('efficiency_ratio', 'Efficiency', 'higher'),
        ('execution_time', 'Exec Time', 'lower'),
        ('total_distance', 'Distance', 'lower'),
    ]
    
    # Normalize and invert where necessary
    algorithms = list(stats.keys())
    
    normalized_data = {}
    for algo in algorithms:
        normalized_data[algo] = []
        
    for metric, label, direction in metrics:
        values = [stats[a][metric]['mean'] for a in algorithms]
        max_val = max(values)
        min_val = min(values)
        
        for i, algo in enumerate(algorithms):
            if direction == 'lower':
                # Invert: lower is better, so we want higher normalized value
                norm_val = 1 - (values[i] - min_val) / (max_val - min_val) if max_val != min_val else 1
            else:
                norm_val = (values[i] - min_val) / (max_val - min_val) if max_val != min_val else 1
            normalized_data[algo].append(norm_val)
    
    # Radar chart
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for algo in algorithms:
        values = normalized_data[algo]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 
                color=ALGORITHM_CONFIGS[algo]['color'],
                linewidth=2, linestyle='solid',
                label=ALGORITHM_CONFIGS[algo]['name'])
        ax.fill(angles, values, 
                color=ALGORITHM_CONFIGS[algo]['color'],
                alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics], fontsize=11)
    
    ax.set_ylim(0, 1)
    ax.set_title('So sánh đa chiều các thuật toán\n(Cao hơn = Tốt hơn)', 
                 fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./fig/all_algorithms_radar.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 7. Print Summary
# ============================================================================
def print_summary(stats: Dict):
    """In tổng kết kết quả"""
    print("\n" + "="*80)
    print("TỔNG KẾT KẾT QUẢ")
    print("="*80)
    
    # Find best algorithm for each metric
    key_metrics = [
        ('max_completion_time', 'lower'),
        ('workload_variance', 'lower'),
        ('efficiency_ratio', 'higher'),
        ('execution_time', 'lower'),
    ]
    
    print("\n📊 Best Algorithm for Each Metric:")
    print("-"*60)
    
    for metric, direction in key_metrics:
        algorithms = list(stats.keys())
        values = [(a, stats[a][metric]['mean']) for a in algorithms]
        
        if direction == 'lower':
            best = min(values, key=lambda x: x[1])
        else:
            best = max(values, key=lambda x: x[1])
        
        print(f"  {metric:<25}: {ALGORITHM_CONFIGS[best[0]]['name']:<15} ({best[1]:.4f})")
    
    # Overall ranking based on max_completion_time
    print("\n🏆 Overall Ranking (by Max Completion Time):")
    print("-"*60)
    
    algorithms = list(stats.keys())
    ranking = sorted(algorithms, key=lambda a: stats[a]['max_completion_time']['mean'])
    
    for i, algo in enumerate(ranking, 1):
        val = stats[algo]['max_completion_time']['mean']
        std = stats[algo]['max_completion_time']['std']
        print(f"  {i}. {ALGORITHM_CONFIGS[algo]['name']:<20}: {val:.4f} ± {std:.4f}")
    
    # Improvement over APPA
    print("\n📈 Improvement over APPA baseline:")
    print("-"*60)
    
    appa_val = stats['APPA']['max_completion_time']['mean']
    for algo in algorithms:
        if algo != 'APPA':
            algo_val = stats[algo]['max_completion_time']['mean']
            improvement = (appa_val - algo_val) / appa_val * 100
            symbol = '✅' if improvement > 0 else '❌'
            print(f"  {symbol} {ALGORITHM_CONFIGS[algo]['name']:<20}: {improvement:+.2f}%")
    
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Tạo dữ liệu mặc định
    print("\n📦 Generating test data...")
    uavs_list, regions_list, V_matrix = generate_test_data(num_uavs=4, num_regions=30)
    print(f"  Số lượng UAV: {len(uavs_list)}")
    print(f"  Số lượng Region: {len(regions_list)}")
    print(f"  V_matrix shape: {V_matrix.shape}")
    
    # Chạy so sánh
    print("\n🚀 Running all algorithms...")
    all_results = run_all_algorithms(uavs_list, regions_list, V_matrix, num_runs=5)
    
    # Tính thống kê
    key_metrics = [
        'max_completion_time',
        'avg_completion_time',
        'workload_variance',
        'workload_std',
        'workload_balance_index',
        'total_distance',
        'efficiency_ratio',
        'avg_uav_utilization',
        'execution_time'
    ]
    
    stats = calculate_statistics(all_results, key_metrics)
    
    # In bảng so sánh
    print_comparison_table(stats, key_metrics)
    
    # Vẽ các biểu đồ
    print("\n📊 Generating plots...")
    
    plot_max_completion_time(stats)
    plot_multiple_metrics(stats)
    plot_execution_time(stats)
    
    # Scalability test
    print("\n📈 Running scalability test...")
    scalability_results = run_scalability_test(num_uavs=4, regions_range=[10, 20, 30, 40, 50])
    plot_scalability(scalability_results)
    
    # In tổng kết
    print_summary(stats)
    
    print("\n✅ Done! All results saved to ./fig/")
