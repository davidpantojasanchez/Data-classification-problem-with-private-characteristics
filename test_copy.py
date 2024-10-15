import ga
import test_methods as tm
import pandas

FILE = "3443_num.csv"

df_aux = pandas.read_csv(FILE, sep=";")
print(df_aux)
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

def calc_private():
    lowerBound = 0.5
    upperBound = 1.5
    pSexo = len(df[df['SEXO'] == 1]) / len(df)
    pP0c = len(df[df['P0c'] == 3]) / len(df)
    pP16 = len(df[df['P16'] == 1]) / len(df)
    pP16A = len(df[df['P16A'] == 0]) / len(df)
    pP19 = len(df[df['P19'] == 1]) / len(df)
    pP19A = len(df[df['P19A'] == 0]) / len(df)
    pP20 = len(df[df['P20'] == 0]) / len(df)
    pP21 = len(df[df['P21'] == 0]) / len(df)
    return [
        ('SEXO', 1, pSexo*lowerBound, pSexo*upperBound),
        ('P0c', 3, pP0c*lowerBound, pP0c*upperBound),
        ('P16', 1, pP16*lowerBound, pP16*upperBound),
        ('P16A', 0, pP16A*lowerBound, pP16A*upperBound),
        ('P19', 1, pP19*lowerBound, pP19*upperBound),
        ('P19A', 0, pP19A*lowerBound, pP19A*upperBound),
        ('P20', 0, pP20*lowerBound, pP20*upperBound),
        ('P21', 0, pP21*lowerBound, pP21*upperBound)
    ]

mutation = 0.05
greedyMutation = 0.4
rounds = 100
population = 10  # has to be even (for crossover)
k = 4
iterations = 10
nquestions = len(df.columns) - 1
p = calc_private()
pcd = ga.pcd(df, int(nquestions), p, k)

tm.test_ga(pcd, mutation, rounds, 4)
tm.test_ga(pcd, mutation, rounds, 30)
tm.test_ga(pcd, mutation, rounds, 20)
tm.test_ga(pcd, mutation, rounds, 30)
