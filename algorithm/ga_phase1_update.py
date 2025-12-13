"""
Genetic Algorithm for Phase 1 Region Allocation
Based on the paper: Coverage path planning of heterogeneous UAVs based on ACS
"""
import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class GAConfig:
    """Configuration for GA parameters"""
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elitism_count: int = 2
    early_stop_generations: int = 15  # Stop if no improvement for this many generations
    random_seed: Optional[int] = None


class GAPhase1:
    """
    Genetic Algorithm for Phase 1: Region Allocation
    
    Chromosome representation:
    - Length: number of regions (m)
    - Gene at position j: UAV index (0 to n-1) assigned to region j
    - Example: [0, 1, 0, 2, 1] means region 0→UAV0, region 1→UAV1, etc.
    
    Fitness: Minimize max completion time across all UAVs
    """
    
    def __init__(
        self,
        num_uavs: int,
        num_regions: int,
        TS_matrix: np.ndarray,
        TF_matrix: np.ndarray,
        D_matrix: np.ndarray,
        V_matrix: np.ndarray,
        uavs_max_velocity: List[float],
        regions_coords: List[Tuple[float, float]],
        config: GAConfig = None
    ):
        """
        Initialize GA Phase 1
        
        Args:
            num_uavs: Number of UAVs
            num_regions: Number of regions
            TS_matrix: Scan time matrix [num_uavs x num_regions]
            TF_matrix: Flight time matrix [num_uavs x num_regions x num_regions]
            D_matrix: Distance matrix [num_regions x num_regions]
            V_matrix: Scan velocity matrix [num_uavs x num_regions]
            uavs_max_velocity: List of max velocities for each UAV
            regions_coords: List of (x, y) coordinates for each region
            config: GA configuration parameters
        """
        self.num_uavs = num_uavs
        self.num_regions = num_regions
        self.TS_matrix = TS_matrix
        self.TF_matrix = TF_matrix
        self.D_matrix = D_matrix
        self.V_matrix = V_matrix
        self.uavs_max_velocity = uavs_max_velocity
        self.regions_coords = regions_coords
        
        self.config = config if config else GAConfig()
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Build feasibility matrix: which UAV can scan which region
        self.feasibility_matrix = self._build_feasibility_matrix()
        
        # Track best solution across generations
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        
        # Fitness cache to avoid recalculation
        self._fitness_cache: Dict[tuple, float] = {}
        
        # Precompute base flight times for efficiency
        self._base_flight_time = []
        for uav_idx in range(self.num_uavs):
            base_times = []
            for region_idx in range(self.num_regions):
                coords = self.regions_coords[region_idx]
                dist = math.sqrt(coords[0]**2 + coords[1]**2)
                base_times.append(dist / self.uavs_max_velocity[uav_idx])
            self._base_flight_time.append(base_times)
        
    def _build_feasibility_matrix(self) -> np.ndarray:
        """Build matrix showing which UAV-region assignments are feasible"""
        return (self.V_matrix > 0).astype(int)
    
    def _get_feasible_uavs(self, region_idx: int) -> List[int]:
        """Get list of UAVs that can scan a given region"""
        return [i for i in range(self.num_uavs) if self.feasibility_matrix[i, region_idx] > 0]
    
    def _initialize_population(self) -> List[List[int]]:
        """
        Initialize population with smart initialization:
        1. Pure greedy solution (APPA-style ETR)
        2. Stochastic greedy solutions (epsilon-greedy exploration)
        3. Workload-balanced solutions
        4. Random feasible solutions
        
        Returns:
            List of chromosomes (each chromosome is a list of UAV indices)
        """
        population = []
        
        # 1. Pure greedy solution using ETR heuristic (like APPA) - 1 individual
        greedy = self._get_greedy_chromosome(epsilon=0.0)
        population.append(greedy)
        
        # 2. Stochastic greedy solutions with different epsilon values
        # This allows exploration while still being guided by ETR heuristic
        epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Different exploration rates
        num_stochastic = min(len(epsilon_values) * 3, self.config.population_size // 3)
        for i in range(num_stochastic):
            eps = epsilon_values[i % len(epsilon_values)]
            stochastic_greedy = self._get_greedy_chromosome(epsilon=eps)
            population.append(stochastic_greedy)
        
        # 3. Workload-balanced solutions with randomization
        num_balanced = min(5, self.config.population_size // 5)
        for _ in range(num_balanced):
            balanced = self._get_balanced_chromosome()
            population.append(balanced)
        
        # 4. Fill rest with random feasible solutions
        while len(population) < self.config.population_size:
            chromosome = []
            for region_idx in range(self.num_regions):
                feasible_uavs = self._get_feasible_uavs(region_idx)
                if feasible_uavs:
                    chromosome.append(random.choice(feasible_uavs))
                else:
                    chromosome.append(0)
            population.append(chromosome)
        
        return population
    
    def _get_greedy_chromosome(self, epsilon: float = 0.0) -> List[int]:
        """
        Generate greedy allocation using ETR heuristic with epsilon-greedy exploration.
        
        Args:
            epsilon: Probability of choosing a random feasible region instead of best ETR.
                     0.0 = pure greedy, 1.0 = pure random
        """
        chromosome = [0] * self.num_regions
        assigned = [False] * self.num_regions
        finish_time = [0.0] * self.num_uavs
        last_region = [-1] * self.num_uavs  # -1 means at base
        
        for _ in range(self.num_regions):
            # Find UAV with minimum finish time (with small randomization)
            if epsilon > 0 and random.random() < epsilon * 0.5:
                # Sometimes pick a random UAV (not always the one with min time)
                available_uavs = [i for i in range(self.num_uavs) if finish_time[i] < np.inf]
                if available_uavs:
                    uav_idx = random.choice(available_uavs)
                else:
                    uav_idx = min(range(self.num_uavs), key=lambda i: finish_time[i])
            else:
                uav_idx = min(range(self.num_uavs), key=lambda i: finish_time[i])
            
            # Collect all feasible regions with their ETR values
            candidates = []
            for r in range(self.num_regions):
                if assigned[r]:
                    continue
                if self.V_matrix[uav_idx, r] <= 0:
                    continue
                
                ts = self.TS_matrix[uav_idx, r]
                if ts == np.inf:
                    continue
                
                # Flight time
                if last_region[uav_idx] == -1:
                    tf = self._base_flight_time[uav_idx][r]
                else:
                    tf = self.TF_matrix[uav_idx, last_region[uav_idx], r]
                
                etr = ts / (tf + ts) if (tf + ts) > 0 else 0
                candidates.append((r, etr))
            
            if not candidates:
                # No feasible region for this UAV, mark as done
                finish_time[uav_idx] = np.inf
                # Assign remaining to any feasible UAV
                for r in range(self.num_regions):
                    if not assigned[r]:
                        feasible = self._get_feasible_uavs(r)
                        if feasible:
                            chromosome[r] = feasible[0]
                            assigned[r] = True
                continue
            
            # Epsilon-greedy selection
            if random.random() < epsilon:
                # Exploration: choose randomly from top candidates (roulette wheel)
                # Weight by ETR to still prefer better options
                etrs = [c[1] for c in candidates]
                min_etr = min(etrs)
                weights = [e - min_etr + 0.1 for e in etrs]  # Shift to positive
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                    chosen_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
                    best_region = candidates[chosen_idx][0]
                else:
                    best_region = random.choice(candidates)[0]
            else:
                # Exploitation: choose best ETR
                best_region = max(candidates, key=lambda x: x[1])[0]
            
            # Assign region
            chromosome[best_region] = uav_idx
            assigned[best_region] = True
            
            # Update finish time
            ts = self.TS_matrix[uav_idx, best_region]
            if last_region[uav_idx] == -1:
                tf = self._base_flight_time[uav_idx][best_region]
            else:
                tf = self.TF_matrix[uav_idx, last_region[uav_idx], best_region]
            
            finish_time[uav_idx] += tf + ts
            last_region[uav_idx] = best_region
        
        return chromosome
    
    def _get_balanced_chromosome(self) -> List[int]:
        """
        Generate chromosome with workload balancing
        """
        chromosome = [0] * self.num_regions
        workload = [0.0] * self.num_uavs
        
        # Random order of regions
        order = list(range(self.num_regions))
        random.shuffle(order)
        
        for r in order:
            feasible = self._get_feasible_uavs(r)
            if not feasible:
                continue
            
            # Choose UAV with minimum workload among feasible
            best_uav = min(feasible, key=lambda u: workload[u])
            chromosome[r] = best_uav
            workload[best_uav] += self.TS_matrix[best_uav, r]
        
        return chromosome
    
    def _calculate_completion_time(self, chromosome: List[int]) -> Dict[int, float]:
        """
        Calculate completion time for each UAV given a chromosome
        
        This uses a greedy ordering: for each UAV, visit regions in the order
        that minimizes additional travel time (nearest neighbor heuristic)
        
        Returns:
            Dictionary mapping UAV index to completion time
        """
        # Group regions by assigned UAV
        uav_regions = {i: [] for i in range(self.num_uavs)}
        for region_idx, uav_idx in enumerate(chromosome):
            uav_regions[uav_idx].append(region_idx)
        
        completion_times = {}
        
        for uav_idx in range(self.num_uavs):
            regions = uav_regions[uav_idx]
            
            if not regions:
                completion_times[uav_idx] = 0.0
                continue
            
            # Use nearest neighbor heuristic to order regions
            ordered_regions = self._nearest_neighbor_order(uav_idx, regions)
            
            # Calculate total time
            total_time = self._calculate_tour_time(uav_idx, ordered_regions)
            completion_times[uav_idx] = total_time
        
        return completion_times
    
    def _nearest_neighbor_order(self, uav_idx: int, regions: List[int]) -> List[int]:
        """
        Order regions using nearest neighbor heuristic starting from base
        """
        if len(regions) <= 1:
            return regions
        
        ordered = []
        remaining = set(regions)
        
        # Start from base (find region closest to origin)
        current_pos = (0, 0)
        
        while remaining:
            # Find nearest remaining region
            min_dist = np.inf
            nearest = None
            
            for region_idx in remaining:
                region_coords = self.regions_coords[region_idx]
                dist = math.sqrt(
                    (region_coords[0] - current_pos[0])**2 + 
                    (region_coords[1] - current_pos[1])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest = region_idx
            
            ordered.append(nearest)
            remaining.remove(nearest)
            current_pos = self.regions_coords[nearest]
        
        return ordered
    
    def _calculate_tour_time(self, uav_idx: int, tour: List[int]) -> float:
        """Calculate total time for UAV to complete a tour of regions"""
        if not tour:
            return 0.0
        
        total_time = 0.0
        base_coords = (0, 0)
        
        # From base to first region
        first_coords = self.regions_coords[tour[0]]
        distance = math.sqrt(first_coords[0]**2 + first_coords[1]**2)
        total_time += distance / self.uavs_max_velocity[uav_idx]
        total_time += self.TS_matrix[uav_idx, tour[0]]
        
        # Between regions
        for i in range(len(tour) - 1):
            total_time += self.TF_matrix[uav_idx, tour[i], tour[i+1]]
            total_time += self.TS_matrix[uav_idx, tour[i+1]]
        
        # Return to base
        last_coords = self.regions_coords[tour[-1]]
        distance_back = math.sqrt(last_coords[0]**2 + last_coords[1]**2)
        total_time += distance_back / self.uavs_max_velocity[uav_idx]
        
        return total_time
    
    def _fitness(self, chromosome: List[int]) -> float:
        """
        Calculate fitness of a chromosome with caching
        
        Fitness = 1 / (max_completion_time + epsilon)
        Higher fitness = better solution (lower max completion time)
        """
        # Check cache first
        chrom_tuple = tuple(chromosome)
        if chrom_tuple in self._fitness_cache:
            return self._fitness_cache[chrom_tuple]
        
        completion_times = self._calculate_completion_time(chromosome)
        max_time = max(completion_times.values()) if completion_times else np.inf
        
        if max_time == 0 or max_time == np.inf:
            fitness = 0.0
        else:
            fitness = 1.0 / (max_time + 1e-6)
        
        # Cache result (limit cache size)
        if len(self._fitness_cache) < 10000:
            self._fitness_cache[chrom_tuple] = fitness
        
        return fitness
    
    def _tournament_selection(self, population: List[List[int]], fitness_values: List[float]) -> List[int]:
        """
        Tournament selection: randomly select k individuals and return the best
        """
        tournament_indices = random.sample(range(len(population)), self.config.tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_values[i])
        return population[best_idx].copy()
    
    def _uniform_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Uniform crossover: each gene from parent1 or parent2 with 50% probability
        """
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = []
        child2 = []
        
        for i in range(self.num_regions):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        return child1, child2
    
    def _mutate(self, chromosome: List[int]) -> List[int]:
        """
        Mutation: randomly change gene to a different feasible UAV
        """
        mutated = chromosome.copy()
        
        for i in range(self.num_regions):
            if random.random() < self.config.mutation_rate:
                feasible_uavs = self._get_feasible_uavs(i)
                if len(feasible_uavs) > 1:
                    # Choose a different UAV
                    other_uavs = [u for u in feasible_uavs if u != mutated[i]]
                    if other_uavs:
                        mutated[i] = random.choice(other_uavs)
        
        return mutated
    
    def _repair(self, chromosome: List[int]) -> List[int]:
        """
        Repair infeasible solutions by ensuring all assignments are valid
        """
        repaired = chromosome.copy()
        
        for i in range(self.num_regions):
            if self.V_matrix[repaired[i], i] <= 0:
                # Current assignment is infeasible, find a feasible UAV
                feasible_uavs = self._get_feasible_uavs(i)
                if feasible_uavs:
                    repaired[i] = random.choice(feasible_uavs)
        
        return repaired
    
    def solve(self) -> Dict[int, List[int]]:
        """
        Run GA to find optimal region allocation
        
        Returns:
            Dictionary mapping UAV index to list of assigned region indices
        """
        # Clear fitness cache
        self._fitness_cache = {}
        
        # Initialize population
        population = self._initialize_population()
        
        # Evaluate initial population
        fitness_values = [self._fitness(chrom) for chrom in population]
        
        # Track best
        best_idx = np.argmax(fitness_values)
        self.best_chromosome = population[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]
        self.fitness_history = [self.best_fitness]
        
        # Early stopping tracking
        no_improvement_count = 0
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            new_population = []
            
            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitness_values)[::-1]
            for i in range(self.config.elitism_count):
                new_population.append(population[sorted_indices[i]].copy())
            
            # Generate rest of population through selection, crossover, mutation
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                # Crossover
                child1, child2 = self._uniform_crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Repair
                child1 = self._repair(child1)
                child2 = self._repair(child2)
                
                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)
            
            population = new_population
            
            # Evaluate new population
            fitness_values = [self._fitness(chrom) for chrom in population]
            
            # Update best
            gen_best_idx = np.argmax(fitness_values)
            if fitness_values[gen_best_idx] > self.best_fitness:
                self.best_chromosome = population[gen_best_idx].copy()
                self.best_fitness = fitness_values[gen_best_idx]
                no_improvement_count = 0  # Reset counter
            else:
                no_improvement_count += 1
            
            self.fitness_history.append(self.best_fitness)
            
            # Early stopping check
            if no_improvement_count >= self.config.early_stop_generations:
                break
        
        # Convert best chromosome to region assignment dictionary
        return self._chromosome_to_assignment(self.best_chromosome)
    
    def _chromosome_to_assignment(self, chromosome: List[int]) -> Dict[int, List[int]]:
        """
        Convert chromosome to region assignment dictionary
        
        Returns:
            Dictionary mapping UAV index to list of assigned region indices
        """
        assignment = {i: [] for i in range(self.num_uavs)}
        
        for region_idx, uav_idx in enumerate(chromosome):
            assignment[uav_idx].append(region_idx)
        
        return assignment
    
    def get_fitness_history(self) -> List[float]:
        """Get fitness values over generations for convergence analysis"""
        return self.fitness_history
    
    def get_completion_time_history(self) -> List[float]:
        """Convert fitness history to max completion time history"""
        return [1.0 / (f + 1e-6) - 1e-6 if f > 0 else np.inf for f in self.fitness_history]
