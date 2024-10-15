import ga
import time

def test_empty(pcd):
    print("------------------------------------------------------------------")
    print("Empty interview:")
    sol = ga.Interview(-1, {})
    print(f"Fitness: {pcd.fitness(sol, pcd.df)}")
    if not pcd.check_correctness(sol):
        print("Error")
    print("------------------------------------------------------------------")

def test_random(pcd, nquestions):
    print("------------------------------------------------------------------")
    print("Random interview:")
    sol = pcd.initialization_one(pcd.df, pcd.k, list(range(0, int(nquestions))))
    print(pcd.interview_to_string(sol))
    print(f"Fitness: {pcd.fitness(sol, pcd.df)}")
    if not pcd.check_correctness(sol):
        print(" Error")
    print("------------------------------------------------------------------")

def test_greedy(pcd):
    print("------------------------------------------------------------------")
    print("Greedy algorithm:")
    t0 = time.perf_counter()
    sol = pcd.greedy()
    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    print(pcd.interview_to_string(sol))
    print(f"Fitness: {pcd.fitness(sol, pcd.df)}")
    print(f"Execution time: {elapsed_time} seconds")
    if not pcd.check_correctness(sol):
        print(" Error")
    print("------------------------------------------------------------------")

def test_ga(pcd, mutation, rounds, population):
    print("------------------------------------------------------------------")
    print("Basic genetic algorithm:")
    t0 = time.perf_counter()
    sol = pcd.genetic_algorithm(mutation, rounds, population)
    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    print(pcd.interview_to_string(sol))
    print(f"Fitness: {pcd.fitness(sol, pcd.df)}")
    print(f"Rounds: {rounds}")
    print(f"Population: {population}")
    print(f"Execution time: {elapsed_time} seconds")
    if not pcd.check_correctness(sol):
        print(" Error")
    print("------------------------------------------------------------------")

def test_ga_greedy(pcd, mutation, greedyMutation, rounds, population):
    print("------------------------------------------------------------------")
    print("Reinforced genetic algorithm:")
    t0 = time.perf_counter()
    sol = pcd.ga_with_greedy(mutation, greedyMutation, rounds, population)
    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    print(pcd.interview_to_string(sol))
    print(f"Fitness: {pcd.fitness(sol, pcd.df)}")
    print(f"Rounds: {rounds}")
    print(f"Population: {population}")
    print(f"Execution time: {elapsed_time} seconds")
    if not pcd.check_correctness(sol):
        print(" Error")
    print("------------------------------------------------------------------")

def mult_test(pcd, df, n, mutation, greedymutation, rounds, population):
    total_elapsed_time_ga = 0
    total_fitness_ga = 0
    
    total_elapsed_time_ga_greedy = 0
    total_fitness_ga_greedy = 0

    with open("ga.txt", "w") as ga_file, open("ga_greedy.txt", "w") as ga_greedy_file:
        for _ in range(n):
            t0 = time.perf_counter()
            sol_ga = pcd.genetic_algorithm(mutation, rounds, population)
            t1 = time.perf_counter()
            
            elapsed_time_ga = t1 - t0
            fitness_ga = pcd.fitness(sol_ga, df)
            
            total_elapsed_time_ga += elapsed_time_ga
            total_fitness_ga += fitness_ga
            
            ga_file.write(f"{fitness_ga}\n")
        
        for _ in range(n):
            t0 = time.perf_counter()
            sol_ga_greedy = pcd.ga_with_greedy(mutation, greedymutation, rounds, population)
            t1 = time.perf_counter()
            
            elapsed_time_ga_greedy = t1 - t0
            fitness_ga_greedy = pcd.fitness(sol_ga_greedy, df)
            
            total_elapsed_time_ga_greedy += elapsed_time_ga_greedy
            total_fitness_ga_greedy += fitness_ga_greedy
            
            ga_greedy_file.write(f"{fitness_ga_greedy}\n")
    
    t0 = time.perf_counter()
    sol_greedy = pcd.greedy()
    t1 = time.perf_counter()
    
    elapsed_time_greedy = t1 - t0
    fitness_greedy = pcd.fitness(sol_greedy, df)

    average_elapsed_time_ga = total_elapsed_time_ga / n
    average_fitness_ga = total_fitness_ga / n
    
    average_elapsed_time_ga_greedy = total_elapsed_time_ga_greedy / n
    average_fitness_ga_greedy = total_fitness_ga_greedy / n

    results = {
        'Greedy algorithm': {
            'elapsed_time': elapsed_time_greedy,
            'fitness': fitness_greedy
        },
        'Genetic algorithm': {
            'average_elapsed_time': average_elapsed_time_ga,
            'average_fitness': average_fitness_ga
        },
        'GA with greedy': {
            'average_elapsed_time': average_elapsed_time_ga_greedy,
            'average_fitness': average_fitness_ga_greedy
        }
    }

    return results

# Características privadas (como referencia):
"""
('SEXO', 1, 0.25, 0.75),
('P0c', 3, 0.05, 0.5),
('P16', 1, 0.02, 0.2),
('P16A', 0, 0.98, 0.8),
('P19', 1, 0.2, 0.8),
('P19A', 0, 0.2, 0.8),
('P20', 0, 0.2, 0.8),
('P21', 0, 0.2, 0.8),
('P21A_1', 0, 0.2, 0.8),
('P21A_2', 0, 0.2, 0.8),
('P21A_3', 0, 0.2, 0.8),
('P21A_4', 0, 0.2, 0.8),
('P21A_5', 0, 0.2, 0.8),
('P21A_6', 0, 0.2, 0.8),
('P21A_96', 0, 0.2, 0.8),
('P21A_98', 0, 0.2, 0.8),
('P21A_99', 0, 0.2, 0.8),
('P21B_1', 0, 0.2, 0.8),
('P21B_2', 0, 0.2, 0.8),
('P21B_3', 0, 0.2, 0.8),
('P21B_4', 0, 0.2, 0.8),
('P21B_5', 0, 0.2, 0.8),
('P21B_6', 0, 0.2, 0.8),
('P21B_7', 0, 0.2, 0.8),
('P21B_96', 0, 0.2, 0.8),
('P21B_98', 0, 0.2, 0.8),
('P21B_99', 0, 0.2, 0.8),
('P22', 0, 0.2, 0.8),
('P23', 0, 0.2, 0.8)
"""
