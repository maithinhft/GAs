"""
Test Iterated Local Search (ILS) Algorithm
"""

import time
from utils.create_sample import create_sample
from main import (
    appa_run_sample, 
    ga_run_sample, 
    sdf_run_sample, 
    stca_run_sample, 
    mcaco_run_sample,
    ils_run_sample
)


def test_ils_vs_ga():
    """So sánh ILS vs GA trên nhiều bài toán khác nhau"""
    
    test_cases = [
        {"NUM_UAVS": 3, "NUM_REGIONS": 15, "test_name": "Small (3U, 15R)"},
        {"NUM_UAVS": 4, "NUM_REGIONS": 25, "test_name": "Medium (4U, 25R)"},
        {"NUM_UAVS": 5, "NUM_REGIONS": 35, "test_name": "Large (5U, 35R)"},
    ]
    
    print("=" * 80)
    print("ILS vs GA COMPARISON - Iterated Local Search Algorithm")
    print("=" * 80)
    
    for test_case in test_cases:
        print(f"\n🔍 Test Case: {test_case['test_name']}")
        print(f"   UAVs: {test_case['NUM_UAVS']}, Regions: {test_case['NUM_REGIONS']}")
        print("-" * 80)
        
        # Tạo sample data
        data = create_sample(
            NUM_UAVS=test_case['NUM_UAVS'],
            NUM_REGIONS=test_case['NUM_REGIONS']
        )
        
        algorithms = [
            ("GA (Genetic Algorithm)", ga_run_sample),
            ("ILS (Iterated Local Search)", ils_run_sample),
            ("APPA (Ant + Priority)", appa_run_sample),
            ("SDF (Surplus Demand First)", sdf_run_sample),
        ]
        
        results = {}
        
        for algo_name, algo_func in algorithms:
            try:
                start_time = time.time()
                fitness = algo_func(data)
                elapsed_time = time.time() - start_time
                
                results[algo_name] = {
                    'fitness': fitness,
                    'time': elapsed_time
                }
                
                print(f"   {algo_name:40s} | Fitness: {fitness:10.4f} | Time: {elapsed_time:8.4f}s")
            except Exception as e:
                print(f"   {algo_name:40s} | ERROR: {str(e)[:40]}")
                results[algo_name] = None
        
        # So sánh ILS vs GA
        if "ILS (Iterated Local Search)" in results and "GA (Genetic Algorithm)" in results:
            if results["ILS (Iterated Local Search)"] and results["GA (Genetic Algorithm)"]:
                ils_fitness = results["ILS (Iterated Local Search)"]['fitness']
                ga_fitness = results["GA (Genetic Algorithm)"]['fitness']
                
                diff_percent = ((ils_fitness - ga_fitness) / ga_fitness) * 100
                speedup = results["GA (Genetic Algorithm)"]['time'] / results["ILS (Iterated Local Search)"]['time']
                
                print("-" * 80)
                if ils_fitness < ga_fitness:
                    print(f"   ✅ ILS is BETTER than GA by {abs(diff_percent):.2f}%")
                    print(f"   🚀 ILS is {speedup:.2f}x faster/slower")
                else:
                    print(f"   ⚠️  ILS is worse than GA by {diff_percent:.2f}%")
                    print(f"   🚀 ILS is {speedup:.2f}x faster/slower")


def test_ils_detailed():
    """Chi tiết test ILS với cấu hình khác nhau"""
    
    print("\n" + "=" * 80)
    print("ILS DETAILED ANALYSIS")
    print("=" * 80)
    
        # Test 1: Nhỏ
    print("\n📊 Test 1: Small Problem (3 UAVs, 10 Regions)")
    data = create_sample(NUM_UAVS=3, NUM_REGIONS=10)
    start = time.time()
    fitness = ils_run_sample(data)
    elapsed = time.time() - start
    print(f"   Fitness: {fitness:.4f}")
    print(f"   Time: {elapsed:.4f}s")
    
    # Test 2: Vừa
    print("\n📊 Test 2: Medium Problem (4 UAVs, 30 Regions)")
    data = create_sample(NUM_UAVS=4, NUM_REGIONS=30)
    start = time.time()
    fitness = ils_run_sample(data)
    elapsed = time.time() - start
    print(f"   Fitness: {fitness:.4f}")
    print(f"   Time: {elapsed:.4f}s")
    
    # Test 3: Lớn
    print("\n📊 Test 3: Large Problem (5 UAVs, 50 Regions)")
    data = create_sample(NUM_UAVS=5, NUM_REGIONS=50)
    start = time.time()
    fitness = ils_run_sample(data)
    elapsed = time.time() - start
    print(f"   Fitness: {fitness:.4f}")
    print(f"   Time: {elapsed:.4f}s")


def test_all_algorithms():
    """Tất cả các thuật toán"""
    
    print("\n" + "=" * 80)
    print("ALL ALGORITHMS COMPARISON (ILS included)")
    print("=" * 80)
    
    data = create_sample(NUM_UAVS=4, NUM_REGIONS=25)
    
    algorithms = [
        ("GA (Genetic Algorithm)", ga_run_sample),
        ("ILS (Iterated Local Search)", ils_run_sample),
        ("APPA (Ant + Priority)", appa_run_sample),
        ("SDF (Surplus Demand First)", sdf_run_sample),
        ("STCA", stca_run_sample),
        ("MCACO", mcaco_run_sample),
    ]
    
    print("\n🔍 Test Case: 4 UAVs, 25 Regions")
    print("-" * 80)
    
    results_list = []
    
    for algo_name, algo_func in algorithms:
        try:
            start_time = time.time()
            fitness = algo_func(data)
            elapsed_time = time.time() - start_time
            
            results_list.append((algo_name, fitness, elapsed_time))
            print(f"{algo_name:40s} | Fitness: {fitness:10.4f} | Time: {elapsed_time:8.4f}s")
        except Exception as e:
            print(f"{algo_name:40s} | ERROR: {str(e)[:40]}")
    
    # Sắp xếp theo chất lượng
    print("\n" + "=" * 80)
    print("📊 RANKING BY QUALITY (Fitness)")
    print("=" * 80)
    results_list.sort(key=lambda x: x[1])
    for idx, (name, fitness, time_val) in enumerate(results_list, 1):
        print(f"{idx}. {name:40s} | Fitness: {fitness:10.4f} ⭐" + "⭐" * (5 - idx))


if __name__ == "__main__":
    test_ils_vs_ga()
    print("\n")
    test_ils_detailed()
    print("\n")
    test_all_algorithms()
