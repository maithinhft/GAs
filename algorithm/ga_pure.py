"""
Pure Genetic Algorithm for UAV Coverage Path Planning
This implementation uses only GA for both region allocation and order optimization
(without APPA Phase 2 ACS)
"""
import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from utils.config import UAV, Region


@dataclass
class PureGAConfig:
    """Configuration for Pure GA parameters"""
    population_size: int = 100
    max_generations: int = 150
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    tournament_size: int = 5
    elitism_count: int = 3
    random_seed: Optional[int] = None


class PureGAAlgorithm:
    """
    Pure Genetic Algorithm for UAV Coverage Path Planning
    
    Chromosome representation: Two-part encoding
    - Part 1: Region allocation (which UAV covers which region)
    - Part 2: Order permutation for each UAV's regions
    
    Fitness: Minimize max completion time across all UAVs
    """
    
    def __init__(
        self,
        uavs_list: List[UAV],
        regions_list: List[Region],
        V_matrix: List[List[float]],
        population_size: int = 100,
        max_generations: int = 150,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        tournament_size: int = 5,
        elitism_count: int = 3
    ):
        self.uavs_list = uavs_list
        self.regions_list = regions_list
        self.V_matrix = np.array(V_matrix)
        
        self.num_uavs = len(uavs_list)
        self.num_regions = len(regions_list)
        
        self.config = PureGAConfig(
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            elitism_count=elitism_count
        )
        
        # Build feasibility matrix
        self.feasibility_matrix = (self.V_matrix > 0).astype(int)
        
        # Precompute matrices
        self.D_matrix = self._compute_distance_matrix()
        self.TS_matrix = self._compute_scan_time_matrix()
        self.TF_matrix = self._compute_flight_time_matrix()
        
        # Track best solution
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute distance matrix between all regions"""
        D = np.zeros((self.num_regions, self.num_regions))
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                if i != j:
                    x1, y1 = self.regions_list[i].coords
                    x2, y2 = self.regions_list[j].coords
                    D[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return D
    
    def _compute_scan_time_matrix(self) -> np.ndarray:
        """Compute scan time matrix"""
        TS = np.full((self.num_uavs, self.num_regions), np.inf)
        for i in range(self.num_uavs):
            for j in range(self.num_regions):
                if self.V_matrix[i][j] > 0:
                    area = self.regions_list[j].area
                    scan_width = self.uavs_list[i].scan_width
                    TS[i][j] = area / (self.V_matrix[i][j] * scan_width)
        return TS
    
    def _compute_flight_time_matrix(self) -> np.ndarray:
        """Compute flight time matrix"""
        TF = np.zeros((self.num_uavs, self.num_regions, self.num_regions))
        for i in range(self.num_uavs):
            for j in range(self.num_regions):
                for k in range(self.num_regions):
                    if j != k:
                        TF[i][j][k] = self.D_matrix[j][k] / self.uavs_list[i].max_velocity
        return TF
    
    def _get_feasible_uavs(self, region_idx: int) -> List[int]:
        """Get list of UAVs that can scan a given region"""
        return [i for i in range(self.num_uavs) if self.feasibility_matrix[i, region_idx] > 0]
    
    def _initialize_population(self) -> List[Dict]:
        """
        Initialize population with random feasible solutions
        Each chromosome contains:
        - allocation: List of UAV indices for each region
        - orders: Dict mapping UAV index to order of regions
        """
        population = []
        
        for _ in range(self.config.population_size):
            # Create random allocation
            allocation = []
            for region_idx in range(self.num_regions):
                feasible_uavs = self._get_feasible_uavs(region_idx)
                if feasible_uavs:
                    allocation.append(random.choice(feasible_uavs))
                else:
                    allocation.append(0)
            
            # Create orders for each UAV's regions
            orders = self._create_orders(allocation)
            
            chromosome = {
                'allocation': allocation,
                'orders': orders
            }
            population.append(chromosome)
        
        return population
    
    def _create_orders(self, allocation: List[int]) -> Dict[int, List[int]]:
        """Create ordering for each UAV's assigned regions"""
        orders = {i: [] for i in range(self.num_uavs)}
        
        for region_idx, uav_idx in enumerate(allocation):
            orders[uav_idx].append(region_idx)
        
        # Shuffle orders to create random permutations
        for uav_idx in orders:
            if len(orders[uav_idx]) > 1:
                random.shuffle(orders[uav_idx])
        
        return orders
    
    def _calculate_fitness(self, chromosome: Dict) -> float:
        """Calculate fitness for a chromosome"""
        completion_times = self._calculate_completion_times(chromosome)
        max_time = max(completion_times.values()) if completion_times else float('inf')
        
        # Fitness is inverse of max completion time
        return 1.0 / (max_time + 1e-6)
    
    def _calculate_completion_times(self, chromosome: Dict) -> Dict[int, float]:
        """Calculate completion time for each UAV"""
        orders = chromosome['orders']
        completion_times = {}
        
        for uav_idx in range(self.num_uavs):
            tour = orders[uav_idx]
            
            if not tour:
                completion_times[uav_idx] = 0.0
                continue
            
            total_time = self._calculate_tour_time(uav_idx, tour)
            completion_times[uav_idx] = total_time
        
        return completion_times
    
    def _calculate_tour_time(self, uav_idx: int, tour: List[int]) -> float:
        """Calculate total time for UAV to complete a tour"""
        if not tour:
            return 0.0
        
        total_time = 0.0
        
        # From base to first region
        first_coords = self.regions_list[tour[0]].coords
        distance = math.sqrt(first_coords[0]**2 + first_coords[1]**2)
        total_time += distance / self.uavs_list[uav_idx].max_velocity
        total_time += self.TS_matrix[uav_idx, tour[0]]
        
        # Between regions
        for i in range(len(tour) - 1):
            total_time += self.TF_matrix[uav_idx, tour[i], tour[i+1]]
            total_time += self.TS_matrix[uav_idx, tour[i+1]]
        
        # Return to base
        last_coords = self.regions_list[tour[-1]].coords
        distance_back = math.sqrt(last_coords[0]**2 + last_coords[1]**2)
        total_time += distance_back / self.uavs_list[uav_idx].max_velocity
        
        return total_time
    
    def _tournament_selection(self, population: List[Dict], fitness_values: List[float]) -> Dict:
        """Select a chromosome using tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.config.tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_values[i])
        return self._copy_chromosome(population[best_idx])
    
    def _copy_chromosome(self, chromosome: Dict) -> Dict:
        """Create a deep copy of a chromosome"""
        return {
            'allocation': chromosome['allocation'].copy(),
            'orders': {k: v.copy() for k, v in chromosome['orders'].items()}
        }
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Perform crossover between two parents"""
        if random.random() > self.config.crossover_rate:
            return self._copy_chromosome(parent1), self._copy_chromosome(parent2)
        
        # Two-point crossover for allocation
        child1_alloc = parent1['allocation'].copy()
        child2_alloc = parent2['allocation'].copy()
        
        if self.num_regions > 2:
            point1 = random.randint(1, self.num_regions - 2)
            point2 = random.randint(point1, self.num_regions - 1)
            
            child1_alloc[point1:point2] = parent2['allocation'][point1:point2]
            child2_alloc[point1:point2] = parent1['allocation'][point1:point2]
        
        # Repair if necessary (ensure feasibility)
        child1_alloc = self._repair_allocation(child1_alloc)
        child2_alloc = self._repair_allocation(child2_alloc)
        
        # Create new orders based on new allocations
        child1 = {
            'allocation': child1_alloc,
            'orders': self._create_orders_from_parents(child1_alloc, parent1['orders'], parent2['orders'])
        }
        child2 = {
            'allocation': child2_alloc,
            'orders': self._create_orders_from_parents(child2_alloc, parent2['orders'], parent1['orders'])
        }
        
        return child1, child2
    
    def _repair_allocation(self, allocation: List[int]) -> List[int]:
        """Repair allocation to ensure feasibility"""
        for region_idx in range(len(allocation)):
            feasible_uavs = self._get_feasible_uavs(region_idx)
            if feasible_uavs and allocation[region_idx] not in feasible_uavs:
                allocation[region_idx] = random.choice(feasible_uavs)
        return allocation
    
    def _create_orders_from_parents(self, allocation: List[int], 
                                     parent1_orders: Dict[int, List[int]], 
                                     parent2_orders: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Create orders for new allocation based on parent orders"""
        orders = {i: [] for i in range(self.num_uavs)}
        
        for region_idx, uav_idx in enumerate(allocation):
            orders[uav_idx].append(region_idx)
        
        # Try to preserve order from parents
        for uav_idx in orders:
            if len(orders[uav_idx]) > 1:
                # Get order preferences from parents
                parent_order = parent1_orders.get(uav_idx, []) + parent2_orders.get(uav_idx, [])
                
                # Sort current regions based on their position in parent order
                def order_key(r):
                    if r in parent_order:
                        return parent_order.index(r)
                    return float('inf')
                
                orders[uav_idx].sort(key=order_key)
        
        return orders
    
    def _mutate(self, chromosome: Dict) -> Dict:
        """Perform mutation on a chromosome"""
        # Allocation mutation
        if random.random() < self.config.mutation_rate:
            # Random region reassignment
            region_idx = random.randint(0, self.num_regions - 1)
            feasible_uavs = self._get_feasible_uavs(region_idx)
            if feasible_uavs:
                old_uav = chromosome['allocation'][region_idx]
                new_uav = random.choice(feasible_uavs)
                
                if old_uav != new_uav:
                    chromosome['allocation'][region_idx] = new_uav
                    # Update orders
                    if region_idx in chromosome['orders'][old_uav]:
                        chromosome['orders'][old_uav].remove(region_idx)
                    chromosome['orders'][new_uav].append(region_idx)
        
        # Order mutation (swap mutation within UAV's regions)
        for uav_idx in range(self.num_uavs):
            if len(chromosome['orders'][uav_idx]) > 1 and random.random() < self.config.mutation_rate:
                # Swap two random positions
                regions = chromosome['orders'][uav_idx]
                idx1, idx2 = random.sample(range(len(regions)), 2)
                regions[idx1], regions[idx2] = regions[idx2], regions[idx1]
        
        # 2-opt improvement for order
        for uav_idx in range(self.num_uavs):
            if len(chromosome['orders'][uav_idx]) > 3 and random.random() < self.config.mutation_rate:
                chromosome['orders'][uav_idx] = self._two_opt(uav_idx, chromosome['orders'][uav_idx])
        
        return chromosome
    
    def _two_opt(self, uav_idx: int, tour: List[int]) -> List[int]:
        """Apply 2-opt improvement to a tour"""
        if len(tour) < 4:
            return tour
        
        improved = True
        best_tour = tour.copy()
        best_time = self._calculate_tour_time(uav_idx, best_tour)
        
        while improved:
            improved = False
            for i in range(1, len(best_tour) - 1):
                for j in range(i + 1, len(best_tour)):
                    new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                    new_time = self._calculate_tour_time(uav_idx, new_tour)
                    
                    if new_time < best_time:
                        best_tour = new_tour
                        best_time = new_time
                        improved = True
                        break
                if improved:
                    break
        
        return best_tour
    
    def solve(self) -> Dict:
        """Run Pure GA algorithm"""
        # Initialize population
        population = self._initialize_population()
        fitness_values = [self._calculate_fitness(c) for c in population]
        
        # Track best solution
        best_idx = np.argmax(fitness_values)
        self.best_chromosome = self._copy_chromosome(population[best_idx])
        self.best_fitness = fitness_values[best_idx]
        
        self.fitness_history = [self.best_fitness]
        
        # Main GA loop
        for generation in range(self.config.max_generations):
            # Create new population
            new_population = []
            
            # Elitism
            sorted_indices = np.argsort(fitness_values)[::-1]
            for i in range(self.config.elitism_count):
                new_population.append(self._copy_chromosome(population[sorted_indices[i]]))
            
            # Fill rest of population
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)
            
            # Update population
            population = new_population
            fitness_values = [self._calculate_fitness(c) for c in population]
            
            # Update best
            current_best_idx = np.argmax(fitness_values)
            if fitness_values[current_best_idx] > self.best_fitness:
                self.best_chromosome = self._copy_chromosome(population[current_best_idx])
                self.best_fitness = fitness_values[current_best_idx]
            
            self.fitness_history.append(self.best_fitness)
        
        # Build result
        completion_times = self._calculate_completion_times(self.best_chromosome)
        max_completion_time = max(completion_times.values()) if completion_times else 0.0
        
        return {
            'paths': self.best_chromosome['orders'],
            'completion_times': completion_times,
            'max_completion_time': max_completion_time,
            'ga_fitness_history': self.fitness_history
        }


if __name__ == "__main__":
    from utils.create_sample import create_sample
    
    data = create_sample(NUM_UAVS=4, NUM_REGIONS=20)
    
    uavs_list = [UAV(**uav_dict) for uav_dict in data['uavs_list']]
    regions_list = [Region(**region) for region in data['regions_list']]
    V_matrix = data['V_matrix']
    
    pure_ga = PureGAAlgorithm(uavs_list, regions_list, V_matrix)
    result = pure_ga.solve()
    
    print("\n=== Pure GA Results ===")
    print(f"Max Completion Time: {result['max_completion_time']:.2f}")
    for uav_idx, path in result['paths'].items():
        print(f"UAV {uav_idx}: {path} (Time: {result['completion_times'][uav_idx]:.2f})")
