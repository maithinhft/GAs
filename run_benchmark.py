import time
import json
import os
import numpy as np
from tabulate import tabulate
from dotenv import load_dotenv

from utils.create_sample import generate_problem_instance
from utils.config import UAV, Region
from algorithm.appa import APPAAlgorithm
from algorithm.llm_appa import LLMAPPAAlgorithm
from algorithm.benchmark import Benchmark

# Load env for API key
load_dotenv()

def run_benchmarks():
    print("Generating sample data...")
    # Parameters
    NUM_UAVS = 4
    NUM_REGIONS = 50  # Reduced for faster LLM processing and cost
    SYSTEM_AREA_RATIO = 0.05
    SYSTEM_DRAG_FACTOR = 0.9
    
    # Generate data
    uavs_list, regions_list, V_matrix = generate_problem_instance(
        NUM_UAVS, 
        NUM_REGIONS,
        SYSTEM_AREA_RATIO,
        SYSTEM_DRAG_FACTOR
    )
    
    print(f"Data generated: {NUM_UAVS} UAVs, {NUM_REGIONS} Regions")
    
    # Initialize Benchmark
    benchmark = Benchmark(uavs_list, regions_list, V_matrix)
    
    # --- Run Original APPA ---
    print("\nRunning Original APPA...")
    start_time = time.time()
    appa = APPAAlgorithm(uavs_list, regions_list, V_matrix)
    appa_result = appa.solve()
    appa_time = time.time() - start_time
    print(f"APPA finished in {appa_time:.4f}s")
    
    # Calculate APPA Metrics
    appa_metrics = benchmark.calculate_metrics(appa_result, appa_time)
    
    # --- Run LLM-APPA ---
    print("\nRunning LLM-APPA...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found. LLM-APPA will fail or fallback.")
    
    start_time = time.time()
    llm_appa = LLMAPPAAlgorithm(uavs_list, regions_list, V_matrix, api_key=api_key)
    llm_result = llm_appa.solve()
    llm_time = time.time() - start_time
    print(f"LLM-APPA finished in {llm_time:.4f}s")
    
    # Calculate LLM-APPA Metrics (with baseline for deviation)
    llm_metrics = benchmark.calculate_metrics(llm_result, llm_time, baseline_metrics=appa_metrics)
    
    # --- Display Results ---
    print("\n=== Benchmark Results ===")
    
    headers = ["Metric", "APPA (Baseline)", "LLM-APPA (Solution)", "Deviation (%)"]
    table_data = []
    
    all_keys = list(appa_metrics.keys())
    # Add Deviation keys that are only in llm_metrics
    for k in llm_metrics.keys():
        if k not in all_keys and "Deviation" in k:
            all_keys.append(k)
            
    for key in all_keys:
        val_appa = appa_metrics.get(key, "N/A")
        val_llm = llm_metrics.get(key, "N/A")
        
        # Format numbers
        if isinstance(val_appa, float):
            val_appa_str = f"{val_appa:.4f}"
        else:
            val_appa_str = str(val_appa)
            
        if isinstance(val_llm, float):
            val_llm_str = f"{val_llm:.4f}"
        else:
            val_llm_str = str(val_llm)
            
        # Calculate deviation for display if not already calculated
        dev_str = ""
        if "Deviation" in key:
            dev_str = val_llm_str # It's already the deviation value
            val_appa_str = "-"
            val_llm_str = "-"
        elif isinstance(val_appa, (int, float)) and isinstance(val_llm, (int, float)) and val_appa != 0:
            dev = ((val_llm - val_appa) / val_appa) * 100
            dev_str = f"{dev:+.2f}%"
        elif val_appa == 0 and val_llm == 0:
             dev_str = "0.00%"
        else:
            dev_str = "N/A"
            
        table_data.append([key, val_appa_str, val_llm_str, dev_str])
        
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save results to file
    with open("benchmark_results.txt", "w") as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\nResults saved to benchmark_results.txt")

if __name__ == "__main__":
    run_benchmarks()
