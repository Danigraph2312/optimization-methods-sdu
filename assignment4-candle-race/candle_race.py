#!/usr/bin/env python3
import os
import sys
import time
import math
from random import random, shuffle, randint, sample
from abc import ABC, abstractmethod

# ROAR-NET API Base Classes
class Problem(ABC):
    @abstractmethod
    def empty_solution(self):
        pass
    
    @abstractmethod
    def random_solution(self):
        pass
    
    @abstractmethod
    def heuristic_solution(self):
        pass

class Solution(ABC):
    def __init__(self, problem):
        self.problem = problem
    
    @abstractmethod
    def copy(self):
        pass
    
    @abstractmethod
    def objective_value(self):
        pass
    
    @abstractmethod
    def lower_bound(self):
        pass

class Neighborhood(ABC):
    def __init__(self, problem):
        self.problem = problem
    
    @abstractmethod
    def moves(self, solution):
        pass
    
    @abstractmethod
    def random_move(self, solution):
        pass
    
    @abstractmethod
    def random_moves_without_replacement(self, solution):
        pass

class Move(ABC):
    def __init__(self, neighborhood):
        self.neighborhood = neighborhood
    
    @abstractmethod
    def apply(self, solution):
        pass
    
    @abstractmethod
    def invert(self):
        pass
    
    @abstractmethod
    def objective_value_increment(self, solution):
        pass
    
    @abstractmethod
    def lower_bound_increment(self, solution):
        pass

# Candle Race Specific Implementations
class Village:
    def __init__(self, idx, x, y, h, b):
        self.idx = idx
        self.x = x
        self.y = y
        self.h = h  # initial height
        self.b = b  # burning rate

class CandleRaceProblem(Problem):
    def __init__(self, villages, start):
        self.villages = villages
        self.start = start
        self.n = len(villages)  # Use 'n' consistently
        self._create_distance_matrix()
    
    def _create_distance_matrix(self):
        """Distance matrix where index 0 is the starting village"""
        self.dist = [[0] * (self.n + 1) for _ in range(self.n + 1)]  

        # Distance from start to each village            
        for v in self.villages:
            self.dist[0][v.idx] = abs(self.start[0] - v.x) + abs(self.start[1] - v.y)
         # Distance between candle villages
        for v1 in self.villages:
            for v2 in self.villages:
                if v1.idx != v2.idx:
                    self.dist[v1.idx][v2.idx] = abs(v1.x - v2.x) + abs(v1.y - v2.y)

    def empty_solution(self):
        return CandleRaceSolution(self, [])
    
    def random_solution(self):
        route = list(range(1, self.n+1))
        shuffle(route)
        # Visit between 1 and all villages (now handles partial visits naturally)
        k = randint(1, self.n)
        return CandleRaceSolution(self, route[:k])
    
    def heuristic_solution(self):
        unvisited = list(range(1, self.n+1))
        route = []
        current_time = 0
        
        while unvisited:
            best_score = -1
            best_village = None
            
            for v_idx in unvisited:
                v = self.villages[v_idx-1]
                travel_time = self.dist[0][v_idx] if not route else self.dist[route[-1]][v_idx]
                arrival_time = current_time + travel_time
                score = max(0, v.h - v.b * arrival_time)
                
                if score > best_score:
                    best_score = score
                    best_village = v_idx
                    best_time = arrival_time
            
            if best_score > 0:  # Only add if positive score
                route.append(best_village)
                unvisited.remove(best_village)
                current_time = best_time
            else:
                break  # Stop if no positive scores left
        
        return CandleRaceSolution(self, route)

class CandleRaceSolution(Solution):
    def __init__(self, problem, route):
        super().__init__(problem)
        self.route = route
        self._score = None
        self._times = None
    
    def copy(self):
        return CandleRaceSolution(self.problem, self.route.copy())
    
    def _calculate_times_and_score(self):
        """Calculate arrival times and score for the current route"""
        current_time = 0
        prev_pos = self.problem.start
        total = 0
        times = []
        
        for v_idx in self.route:
            v = self.problem.villages[v_idx-1]
            travel_time = self.problem.dist[0][v_idx] if not times else self.problem.dist[self.route[len(times)-1]][v_idx]
            current_time += travel_time
            times.append(current_time)
            total += max(0, v.h - v.b * current_time)
            prev_pos = (v.x, v.y)
        
        self._times = times
        self._score = total
        return total
    
    def objective_value(self):
        if self._score is None:
            self._calculate_times_and_score()
        return self._score
    
    def lower_bound(self):
        # For this problem, we don't have a lower bound calculation
        return self.objective_value()

class InsertNeighborhood(Neighborhood):
    def moves(self, solution):
        for i in range(len(solution.route)):
            for j in range(len(solution.route)+1):
                if i != j and j != i+1:
                    yield InsertMove(self, i, j)
    
    def random_move(self, solution):
        if len(solution.route) < 2:
            return None
        i = randint(0, len(solution.route)-1)
        j = randint(0, len(solution.route))
        while j == i or j == i+1:
            j = randint(0, len(solution.route))
        return InsertMove(self, i, j)
    
    def random_moves_without_replacement(self, solution):
        moves = []
        for i in range(len(solution.route)):
            for j in range(len(solution.route)+1):
                if i != j and j != i+1:
                    moves.append((i, j))
        shuffle(moves)
        for i, j in moves:
            yield InsertMove(self, i, j)

class InsertMove(Move):
    def __init__(self, neighborhood, i, j):
        super().__init__(neighborhood)
        self.i = i
        self.j = j
    
    def __str__(self):
        return f"InsertMove({self.i}->{self.j})"
    
    def apply(self, solution):
        new_route = solution.route.copy()
        v = new_route.pop(self.i)
        new_route.insert(self.j, v)
        return CandleRaceSolution(solution.problem, new_route)
    
    def invert(self):
        if self.j > self.i:
            return InsertMove(self.neighborhood, self.j-1, self.i)
        else:
            return InsertMove(self.neighborhood, self.j, self.i+1)
    
    def objective_value_increment(self, solution):
        # For efficiency, we calculate the exact increment
        new_solution = self.apply(solution)
        return new_solution.objective_value() - solution.objective_value()
    
    def lower_bound_increment(self, solution):
        # For this problem, we use the exact increment as lower bound
        return self.objective_value_increment(solution)

class SwapNeighborhood(Neighborhood):
    def moves(self, solution):
        for i in range(len(solution.route)):
            for j in range(i+1, len(solution.route)):
                yield SwapMove(self, i, j)
    
    def random_move(self, solution):
        if len(solution.route) < 2:
            return None
        i, j = sample(range(len(solution.route)), 2)
        return SwapMove(self, min(i,j), max(i,j))
    
    def random_moves_without_replacement(self, solution):
        moves = []
        for i in range(len(solution.route)):
            for j in range(i+1, len(solution.route)):
                moves.append((i, j))
        shuffle(moves)
        for i, j in moves:
            yield SwapMove(self, i, j)

class SwapMove(Move):
    def __init__(self, neighborhood, i, j):
        super().__init__(neighborhood)
        self.i = i
        self.j = j
    
    def __str__(self):
        return f"SwapMove({self.i}<->{self.j})"
    
    def apply(self, solution):
        new_route = solution.route.copy()
        new_route[self.i], new_route[self.j] = new_route[self.j], new_route[self.i]
        return CandleRaceSolution(solution.problem, new_route)
    
    def invert(self):
        return self  # Swap is its own inverse
    
    def objective_value_increment(self, solution):
        # For efficiency, we calculate the exact increment
        new_solution = self.apply(solution)
        return new_solution.objective_value() - solution.objective_value()
    
    def lower_bound_increment(self, solution):
        # For this problem, we use the exact increment as lower bound
        return self.objective_value_increment(solution)

# Metaheuristic Implementations
def variable_neighborhood_search(problem, max_time=60):
    current = problem.heuristic_solution()
    best = current.copy()
    
    start_time = time.time()
    while time.time() - start_time < max_time:
        # First neighborhood - Insert
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            for move in InsertNeighborhood(problem).random_moves_without_replacement(current):
                delta = move.objective_value_increment(current)
                if delta > 0:
                    current = move.apply(current)
                    improved = True
                    if current.objective_value() > best.objective_value():
                        best = current.copy()
                    break
        
        # Second neighborhood - Swap
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            for move in SwapNeighborhood(problem).random_moves_without_replacement(current):
                delta = move.objective_value_increment(current)
                if delta > 0:
                    current = move.apply(current)
                    improved = True
                    if current.objective_value() > best.objective_value():
                        best = current.copy()
                    break
        
        # Perturbation
        current = perturb(current)
    
    return best

def perturb(solution):
    """Double-bridge perturbation (4-opt)"""
    if len(solution.route) < 4:
        return solution.copy()
    
    # Select 4 distinct indices
    indices = sorted(sample(range(len(solution.route)), 4))
    i, j, k, l = indices
    
    # Perform double-bridge: A-B-C-D becomes A-C-B-D
    new_route = solution.route[:i] + solution.route[k:l] + solution.route[j:k] + solution.route[i:j] + solution.route[l:]
    
    return CandleRaceSolution(solution.problem, new_route)

# Main Program
def read_input(file=sys.stdin):
    try:
        n = int(file.readline())
        x0, y0 = map(int, file.readline().split())
        villages = []
        
        # Read until EOF or n villages
        for idx in range(1, n+1):
            line = file.readline().strip()
            if not line:  # EOF reached
                print(f"Warning: Expected {n} villages but got {len(villages)}", file=sys.stderr)
                break
                
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid data for village {idx}")
            x, y, h, b = map(int, parts)
            villages.append(Village(idx, x, y, h, b))
            
        return villages, (x0, y0)
    except Exception as e:
        raise ValueError(f"Input parsing failed: {str(e)}")


def main():
    import sys
    import os
    import time
    from datetime import datetime
    
    # Enhanced logging that always shows up
    def log(message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)
    
    # 1. Verify input file exists
    if len(sys.argv) != 2:
        log("ERROR: Missing input file")
        log("Usage: ./candle_race.py inputfile")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        log(f"ERROR: Input file '{input_file}' not found")
        sys.exit(1)
    
    # 2. Prepare output filename
    output_file = os.path.splitext(input_file)[0] + ".out"
    log(f"Processing input file: {input_file}")
    log(f"Will write output to: {output_file}")
    
    try:
        # 3. Read input with explicit file handling
        log("Reading input data...")
        with open(input_file, 'r') as f:
            villages, start = read_input(f)
        
        if not villages:
            log("ERROR: No villages loaded from input file")
            sys.exit(1)
            
        log(f"Successfully loaded {len(villages)} villages")
        
        # 4. Solve the problem
        problem = CandleRaceProblem(villages, start)
        log("Starting optimization...")
        start_time = time.time()
        best_solution = variable_neighborhood_search(problem)
        elapsed = time.time() - start_time
        log(f"Optimization completed in {elapsed:.2f} seconds")
        
        # 5. Write output with verification
        log(f"Best solution score: {best_solution.objective_value()}")
        log(f"Writing solution to {output_file}...")
        
        with open(output_file, 'w') as f:
            for village_idx in best_solution.route:
                f.write(f"{village_idx}\n")
        
        # Verify output was written
        if os.path.exists(output_file):
            log("Successfully created output file")
            log(f"Output contains {len(best_solution.route)} villages")
        else:
            log("ERROR: Failed to create output file!")
            sys.exit(1)
            
    except Exception as e:
        log(f"FATAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
     main()
