# GAs - Genetic Algorithms for UAV Coverage Path Planning

## Overview
This project implements and compares APPA (Ant Colony System-based) algorithms for coverage path planning of heterogeneous UAVs. The implementation includes:
- **Original APPA**: Uses Effective Time Ratio (ETR) for Phase 1 (region allocation)
- **LLM-based APPA**: Uses Large Language Models (LLMs) for Phase 1 (region allocation)

Both methods use the same Phase 2 (order optimization) based on Ant Colony System.

## Comprehensive Metrics Evaluation

The project includes comprehensive metrics evaluation similar to the paper, including:

### Time Metrics
- **Max Completion Time**: Maximum time for any UAV to complete its assigned regions
- **Avg Completion Time**: Average completion time across all UAVs
- **Min Completion Time**: Minimum completion time
- **Execution Time**: Algorithm runtime (computational cost)

### Workload Balance Metrics
- **Workload Variance**: Variance of completion times across UAVs (lower = more balanced)
- **Workload Std Dev**: Standard deviation of completion times
- **Workload Balance Index**: Coefficient of variation (lower = better balance)

### Distance Metrics
- **Total Distance**: Total distance traveled by all UAVs
- **Avg Distance per UAV**: Average distance per UAV

### Efficiency Metrics
- **Total Scan Time**: Total time spent scanning regions
- **Total Flight Time**: Total time spent flying between regions
- **Efficiency Ratio**: Scan time / (Scan time + Flight time) - higher is better

### Allocation Metrics
- **Allocation Balance**: Balance of region allocation across UAVs (lower = more balanced)
- **Avg UAV Utilization**: Average utilization of UAVs (completion_time / max_completion_time)

### Comparison Metrics
- **Deviation Ratio**: (Solution - Baseline) / Baseline * 100
  - Used to compare solution quality relative to baseline
  - Negative values indicate improvement

## Usage

See `code.ipynb` for comprehensive comparison experiments with multiple metrics.

## Files
- `algorithm/appa.py`: Original APPA algorithm implementation
- `utils/metrics.py`: Comprehensive metrics calculator
- `code.ipynb`: Main comparison notebook with multiple evaluation metrics
- `main.py`: Utility functions for creating academic tables
