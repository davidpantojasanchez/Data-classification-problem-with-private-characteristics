import heuristic_algorithms
import pandas as pd
import numpy as np
import random
import time
from multiprocessing import Process

def private(df, n):
    tuples_list = []
    q_columns = [col for col in df.columns if col.startswith('q')]

    for _ in range(n):
        q_col = np.random.choice(q_columns)
        
        x = np.random.choice(df[q_col])
        
        count_x = (df[q_col] == x).sum()
        ratio_x = count_x / len(df[q_col])
        
        lower_bound = np.random.uniform(0, ratio_x/2)
        upper_bound = np.random.uniform(max(ratio_x*1.5, 0.99), 1)
        
        tuple_entry = (q_col, x, lower_bound, upper_bound)
        tuples_list.append(tuple_entry)
    
    return tuples_list

def random_instance(num_candidate_types, num_questions):
    data = {
        f"q{i}": np.random.randint(1, 6, num_candidate_types) for i in range(num_questions)
    }
    data['QUANTITY'] = np.random.randint(1, 101, num_candidate_types)
    data['SINCERIDAD'] = np.random.randint(0, 2, num_candidate_types)
    df = pd.DataFrame(data)
    return df

def test_parameters(rounds_ga=100, population_ga=[20], values_k_ga=[10], mutation_ga=[round(i * 0.1, 1) for i in range(11)],
                    population_ins_min=2000, population_ins_max=4000, characteristics_ins_min=150, characteristics_ins_max=300,
                    private_ins_min=5, private_ins_max=5, num_instances=1, num_executions=1, show_progress = False,
                    test_greedy = True, test_bga = True, test_rga = True, suffix=''):
    
    open(f'instances_{suffix}.txt', 'w')
    open(f'results_{suffix}.txt', 'w')
    for i in range(0, num_instances):
        population_ins = random.randint(population_ins_min, population_ins_max)
        characteristics_ins = random.randint(characteristics_ins_min, characteristics_ins_max+1)
        private_ins = random.randint(private_ins_min, private_ins_max)

        print(f'\nProcessing instance {i}')
        with open(f'results_{suffix}.txt', 'a') as file_results:
            file_results.write(f'INSTANCE {i}\n')

        df = random_instance(population_ins, characteristics_ins)
        df.to_csv(f'instance_{suffix}_{i}.csv', index=False)

        k = random.choice(values_k_ga)
        p = private(df, private_ins)
        nquestions = len(df.columns) - 1
        num_rows, num_columns = df.shape
        with open(f'instances_{suffix}.txt', 'a') as file_instances:
            file_instances.write(f'{i} -> Candidate types: {num_rows}, Characteristics: {num_columns}, Private characteristics: {len(p)}, Question limit: {k} \n')
        prog_dir = ''
        if show_progress:
            prog_dir = f'progress_{suffix}.txt'
        heuristic = heuristic_algorithms.Heuristic(df=df, nquestions=int(nquestions), P=p, k=k, progress_dir=prog_dir)
        
        if test_greedy:
            t = time.perf_counter()
            sol = heuristic.greedy()
            t = time.perf_counter() - t
            with open(f'results_{suffix}.txt', 'a') as file_results:
                file_results.write(f'GREEDY         -  {round(heuristic.fitness(sol, df), 4)} fitness, {t/60} minutes\n')

        for population in population_ga:
            print(f"* {population} population size")
            for i in mutation_ga:
                print(f"* {i} mutation ratio")

                bga_best_fit = 0
                bga_avg_t = 0
                rga_best_fit = 0
                rga_avg_t = 0

                if test_bga:
                    for exec_n in range(num_executions):
                        t = time.perf_counter()
                        sol = heuristic.genetic_algorithm(i, rounds_ga, population)
                        t = time.perf_counter() - t

                        with open(f'results_{suffix}.txt', 'a') as file_results:
                            file_results.write(f'BASIC GA       -  {population} population size, {i} mutation ratio, {exec_n} execution, {round(heuristic.fitness(sol, df), 4)} fitness, {t/60} minutes\n')

                        bga_best_fit = max(heuristic.fitness(sol, df), bga_best_fit)
                        bga_avg_t += t/60
                
                if test_rga:
                    for exec_n in range(num_executions):
                        t = time.perf_counter()
                        sol = heuristic.ga_with_greedy(i, 0.5, rounds_ga, population,)
                        t = time.perf_counter() - t
                        with open(f'results_{suffix}.txt', 'a') as file_results:
                            file_results.write(f'REINFORCED GA  -  {population} population size, {i} mutation ratio, {exec_n} execution, {round(heuristic.fitness(sol, df), 4)} fitness, {t/60} minutes\n')

                        rga_best_fit = max(heuristic.fitness(sol, df), rga_best_fit)
                        rga_avg_t += t/60

                with open(f'results_{suffix}.txt', 'a') as file_results:
                    file_results.write(f'BASIC GA best fitness: {round(bga_best_fit, 4)}\n')
                    file_results.write(f'BASIC GA average time: {round(bga_avg_t/3.0, 4)}\n')
                    file_results.write(f'REINFORCED GA best fitness: {round(rga_best_fit, 4)}\n')
                    file_results.write(f'REINFORCED GA average time: {round(rga_avg_t/3.0, 4)}\n')


def test_backtracking_algorithm(num_candidate_types, num_questions, k, p_number, num_executions):
    total_time = 0
    print(f'\n{num_questions} questions:\n')
    for i in range(num_executions):
        df = random_instance(num_candidate_types, num_questions)
        p = private(df, p_number)
        heuristic = heuristic_algorithms.Heuristic(df=df, nquestions=int(num_questions), P=p, k=k, progress_dir="")
        t = time.perf_counter()
        __, fitness = heuristic.backtracking(df, 10, list(range(num_questions)))
        t = time.perf_counter() - t
        total_time += t
        print(fitness)
        print(f'Time: {t}')
    
    with open(f'exact_algorithm.txt', 'a') as file_results:
        file_results.write(f'Average time ({num_questions} questions): {total_time / num_executions}\n')



def test_backtracking_algorithm_time():
    for i in range(11, 12):
        test_backtracking_algorithm(500, i, 10, 2, 5)


if __name__ == '__main__':

    process1 = Process(target=test_parameters, args=(400, [20], [10,11,12,13,14,15], [0.2],
                                                     2000, 4000, 150, 300,
                                                     4, 9, 2, 3, False,
                                                     True, True, True, 'AA'))
    process2 = Process(target=test_parameters, args=(400, [20], [10,11,12,13,14,15], [0.2],
                                                     2000, 4000, 150, 300,
                                                     4, 9, 2, 3, False,
                                                     True, True, True, 'AB'))
    process3 = Process(target=test_parameters, args=(400, [20], [10,11,12,13,14,15], [0.2],
                                                     2000, 4000, 150, 300,
                                                     4, 9, 2, 3, False,
                                                     True, True, True, 'AC'))
    process4 = Process(target=test_parameters, args=(400, [20], [10,11,12,13,14,15], [0.2],
                                                     2000, 4000, 150, 300,
                                                     4, 9, 2, 3, False,
                                                     True, True, True, 'AD'))
    process5 = Process(target=test_parameters, args=(400, [20], [10,11,12,13,14,15], [0.2],
                                                     2000, 4000, 150, 300,
                                                     4, 9, 2, 3, False,
                                                     True, True, True, 'AE'))
    process6 = Process(target=test_parameters, args=(400, [20], [10,11,12,13,14,15], [0.2],
                                                     2000, 4000, 150, 300,
                                                     4, 9, 2, 3, False,
                                                     True, True, True, 'AF'))
    
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process6.start()

    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()
    process6.join()
