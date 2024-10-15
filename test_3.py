import ga
import test_methods as tm
import pandas

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

exclude_columns = ['SEXO', 'P0c', 'P16', 'P16A', 'P19', 'P20', 'P21', 'P22', 'P23', 'SINCERIDAD']
clms = [col for col in df.columns[94:] if col not in exclude_columns]
df = df.drop(columns=clms)
df = df.iloc[:3000]
print(df)

def calc_private():
    lowerBound = 0.75
    upperBound = 1.25
    pSexo = len(df[df['SEXO'] == 1]) / len(df)
    pP20 = len(df[df['P20'] == 0]) / len(df)
    pP21 = len(df[df['P21'] == 0]) / len(df)
    pP22 = len(df[df['P22'] == 0]) / len(df)
    pP23 = len(df[df['P23'] == 0]) / len(df)
    return [
        ('SEXO', 1, pSexo*lowerBound, pSexo*upperBound),
        ('P20', 0, pP20*lowerBound, pP20*upperBound),
        ('P21', 0, pP21*lowerBound, pP21*upperBound),
        ('P22', 0, pP22*lowerBound, pP22*upperBound),
        ('P23', 0, pP23*lowerBound, pP23*upperBound)
    ]

mutation = 0.05
greedyMutation = 0.4 # indica la proporción de mutaciones que son voraces (en el algorítmo genético con voraz)
rounds = 10
population = 10  # tiene que ser un número par (para el cruzamiento)
k = 6
iterations = 10
nquestions = len(df.columns) - 1
p = calc_private()
pcd = ga.pcd(df, int(nquestions), p, k)

tm.test_empty(pcd)
results = tm.mult_test(pcd, df, iterations, mutation, greedyMutation, rounds, population)
print(results)
