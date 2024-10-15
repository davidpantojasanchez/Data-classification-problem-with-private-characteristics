import ga
import test_methods as tm
import pandas
import numpy as np
import random
import csv

FILE = "3443_num.csv"

df_aux = pandas.read_csv(FILE, sep=";")
df = df_aux.iloc[:, 4:188]
df.drop('EDAD', axis=1, inplace=True)
df.drop('PROV', axis=1, inplace=True)
df.drop('MUN', axis=1, inplace=True)
df.drop('CCAA', axis=1, inplace=True)
df.drop('INTENCIONG', axis=1, inplace=True)
df.drop('INTENCIONGALTER', axis=1, inplace=True)
df.drop('SIMPATIA', axis=1, inplace=True)
df.drop('CERCANIA', axis=1, inplace=True)
df.drop('NACEXTR', axis=1, inplace=True)
df.drop('PAISNAC2', axis=1, inplace=True)

dfs = df_aux['SINCERIDAD'].replace({4 : 0})
dfs.replace({3 : 0}, inplace=True)
dfs.replace({8 : 0}, inplace=True)
dfs.replace({2 : 0}, inplace=True)
dfs.replace({1 : 1}, inplace=True)
df = df.assign(SINCERIDAD=dfs)

exclude_columns = ['SEXO', 'P0c', 'P16', 'P16A', 'P19', 'P19A', 'P20', 'P21', 'P22', 'P23', 'SINCERIDAD']

def remove_random_columns_and_rows(df, c, r, exclude_columns):
    exclude_columns_set = set(exclude_columns)
    removable_columns = [col for col in df.columns if col not in exclude_columns_set]

    columns_to_remove = random.randint(0, min(c, len(removable_columns)))
    if columns_to_remove > 0:
        columns_to_drop = np.random.choice(removable_columns, columns_to_remove, replace=False)
        df = df.drop(columns=columns_to_drop)

    rows_to_remove = random.randint(0, min(r, len(df)))
    if rows_to_remove > 0:
        rows_to_drop = np.random.choice(df.index, rows_to_remove, replace=False)
        df = df.drop(rows_to_drop)

    return df

def remove_random_elements(lst, a, b):
    elements_to_remove = random.randint(a, min(b, len(lst)))
    if elements_to_remove > 0:
        indices_to_remove = random.sample(range(len(lst)), elements_to_remove)
        indices_to_remove.sort(reverse=True)
        for index in indices_to_remove:
            del lst[index]
    return lst

def calc_private(lowerBound, upperBound, df1):
    pSexo = len(df1[df1['SEXO'] == 1]) / len(df1)
    pP0c = len(df1[df1['P0c'] == 3]) / len(df1)
    pP16 = len(df1[df1['P16'] == 1]) / len(df1)
    pP16A = len(df1[df1['P16A'] == 0]) / len(df1)
    pP19 = len(df1[df1['P19'] == 1]) / len(df1)
    pP19A = len(df1[df1['P19A'] == 0]) / len(df1)
    pP20 = len(df1[df1['P20'] == 0]) / len(df1)
    pP21 = len(df1[df1['P21'] == 0]) / len(df1)
    pP22 = len(df1[df1['P22'] == 0]) / len(df1)
    pP23 = len(df1[df1['P23'] == 0]) / len(df1)
    return [
        ('SEXO', 1, pSexo*lowerBound, pSexo*upperBound),
        ('P0c', 3, pP0c*lowerBound, pP0c*upperBound),
        ('P16', 1, pP16*lowerBound, pP16*upperBound),
        ('P16A', 0, pP16A*lowerBound, pP16A*upperBound),
        ('P19', 1, pP19*lowerBound, pP19*upperBound),
        ('P19A', 0, pP19A*lowerBound, pP19A*upperBound),
        ('P20', 0, pP20*lowerBound, pP20*upperBound),
        ('P21', 0, pP21*lowerBound, pP21*upperBound),
        ('P22', 0, pP22*lowerBound, pP22*upperBound),
        ('P23', 0, pP23*lowerBound, pP23*upperBound)
    ]

def add_dicts(dict1, dict2):
    result = {}
    for key in dict1:
        result[key] = {}
        for sub_key in dict1[key]:
            result[key][sub_key] = dict1[key][sub_key] + dict2[key][sub_key]
    return result

mutation = 0.05
greedyMutation = 0.4
rounds_ga = 2
population = 10
iterations_mult_test = 1
num_instances = 10
values_k = [4,5,6,7]

r = {
        'Greedy algorithm': {
            'elapsed_time': 0,
            'fitness': 0
        },
        'Genetic algorithm': {
            'average_elapsed_time': 0,
            'average_fitness': 0
        },
        'GA with greedy': {
            'average_elapsed_time': 0,
            'average_fitness': 0
        }
    }

results_list = []
for i in range(0, num_instances):
    print(i+1)
    k = random.choice(values_k)
    new_df = remove_random_columns_and_rows(df[:], 60, 4000, exclude_columns)
    p = calc_private(random.uniform(0.25, 0.8), random.uniform(1.25, 4), new_df)
    new_p = remove_random_elements(p, 0, 9)
    nquestions = len(new_df.columns) - 1

    pcd = ga.pcd(new_df, int(nquestions), new_p, k)

    results = tm.mult_test(pcd, new_df, iterations_mult_test, mutation, greedyMutation, rounds_ga, population)
    print(results)
    r = add_dicts(r, results)

    greedy_fitness = results['Greedy algorithm']['fitness']
    ga_fitness = results['Genetic algorithm']['average_fitness']
    ga_greedy_fitness = results['GA with greedy']['average_fitness']
    results_list.append([f'I{i+1}', -greedy_fitness, -ga_fitness, -ga_greedy_fitness])

csv_file = 'results.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['dataset', 'greedy', 'ga', 'ga_greedy'])
    writer.writerows(results_list)
