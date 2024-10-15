import random
import copy
import math
import pandas
from functools import partial
import time

# Interview

class Interview:
     def __init__(self, value, children):  # "children" is a map from answers to subinterviews
        self.value = value
        self.children = children

# GDCOP

class pcd:

    mutationChance:float
    greedyMutationChance:float
    rounds:int
    population:int

    nquestions:int

    P:list
    k:int

    tFitness:int

    df:pandas.DataFrame

    triesGreedy = 1000
    triesGreedyGA = 200

    def __init__(self, df, nquestions, P, k):
        self.df = df
        self.nquestions = int(nquestions)
        self.P = P
        self.k = k
    
    # Greedy algorithm
    def greedy(self):
        partial_fitness = partial(self.fitness_for_order, df=self.df)
        Q = list(range(0, self.nquestions))
        Q.sort(key=partial_fitness, reverse=True)
        return self.greedy_aux(self.df, self.k, Q, 0, self.triesGreedy)

    def greedy_aux(self, df, depth, Q, i, tries):
        # Leaf
        if (len(Q) == 0 or depth == 0 or i == len(Q) or tries == 0):
            return Interview(-1, {})
        
        value = Q[i]
        column_name = df.columns[value]
        unique_values = df[column_name].unique()
        children = {}

        # Checks if an answer infers private information
        for v in unique_values:
            new_df = self.restrict_df(df, column_name, v)
            if not self.satisfiesPrivate(new_df):
                return self.greedy_aux(df, depth, copy.copy(Q), i + 1, tries - 1)
        
        # If not, it generates the children using the greedyt algorithm
        Q.remove(value)
        for v in unique_values:
            new_df = self.restrict_df(df, column_name, v)
            partial_fitness = partial(self.fitness_for_order, df=new_df)
            newQ = copy.copy(Q)
            newQ.sort(key=partial_fitness, reverse=True)
            children[str(v)] = self.greedy_aux(new_df, depth - 1, newQ, 0, tries)
        
        return Interview(value, children)
    
    def fitness_for_order(self, element, df):
        column_name = df.columns[element]
        unique_values = df[column_name].unique()
        children = {}
        for v in unique_values:
            children[str(v)] = Interview(-1, {})
        return self.fitness(Interview(element, children), df)
    
    # Genetic algorithm
    def genetic_algorithm(self, mutation, rounds, population):
        self.mutationChance = mutation
        self.greedyMutationChance = 0.0
        self.rounds = rounds
        self.population = population

        tInitialization = 0
        tCombinations = 0
        tSelection = 0

        df = self.df[:]
        t0 = time.perf_counter()
        interviews = self.initialization(self.population)
        tInitialization += time.perf_counter() - t0

        for i in range(0, self.rounds):
            t0 = time.perf_counter()
            interviews = interviews + self.combinations(interviews)
            tCombinations += time.perf_counter() - t0
            t0 = time.perf_counter()
            interviews = self.selections(interviews, df[:])
            tSelection += time.perf_counter() - t0
        
        bestI = interviews[0]
        bestF = self.fitness(interviews[0], df[:])
        for i in range(1, len(interviews)):
            if self.fitness(interviews[i], df[:]) > bestF:
                bestI = interviews[i]
                bestF = self.fitness(interviews[i], df[:])
        return bestI
    
    # Reinforced genetic algorithm
    # Uses the greedy algorithm to initialize the population and to complete the solution
    def ga_with_greedy(self, mutation, greedyMutation, rounds, population):
        self.mutationChance = mutation
        self.greedyMutationChance = greedyMutation
        self.rounds = rounds
        self.population = population

        df = self.df[:]
        interviews = self.initialization(int(self.population - 2))
        interviews.append(self.greedy())
        prev_mutation = self.mutationChance
        prev_gm = self.greedyMutationChance
        prev_k = self.k
        self.mutationChance = 0.25
        self.greedyMutationChance = 0
        self.k = min(self.k, 3)
        interviews.append(self.mutation(self.greedy()))
        self.mutationChance = prev_mutation
        self.greedyMutationChance = prev_gm
        self.k = prev_k

        for i in range(0, self.rounds):
            interviews = interviews + self.combinations(interviews)
            interviews = self.selections(interviews, df[:])
        
        bestI = interviews[0]
        bestF = self.fitness(interviews[0], df[:])
        for i in range(1, len(interviews)):
            if self.fitness(interviews[i], df[:]) > bestF:
                bestI = interviews[i]
                bestF = self.fitness(interviews[i], df[:])
        
        self.reinforcement(df, bestI, self.k, list(range(0, self.nquestions)))
        return bestI
    
    # Used by ga_with_greedy. Reemplaces the leaves of a solution with correct subinterviews constructed greedily
    def reinforcement(self, df, interview, k, Q):
        if interview.value != -1:
            Q.remove(interview.value)
            column_name = df.columns[interview.value]
            unique_values = df[column_name].unique()

            for v in unique_values:
                new_df = self.restrict_df(df, column_name, v)
                self.reinforcement(new_df, interview.children[str(v)], k - 1, copy.copy(Q))
        else:
            partial_fitness = partial(self.fitness_for_order, df=df)
            Q.sort(key=partial_fitness, reverse=True)
            self.greedy_aux(df, k, Q, 0, self.triesGreedyGA)

    # Random initialization of the population. Ensures that every interview is correct
    def initialization(self, ninterviews):
        r = []
        for _ in range(0, ninterviews):
            r.append(self.initialization_one(self.df, self.k, list(range(0, self.nquestions))))
        return r
    
    # Construction of a random interview. It ensures it is correct: it has an adequate depth and does not infer private information
    def initialization_one(self, df, depth, Q):
        if (len(Q) == 0 or depth == 0):
            return Interview(-1, {})
        
        value = random.choice(Q)
        children = {}
        Q.remove(value)
        column_name = df.columns[value]
        unique_values = df[column_name].unique()

        for v in unique_values:
            new_df = self.restrict_df(df, column_name, v)
            if not self.satisfiesPrivate(new_df):
                return self.initialization_one(df, depth - 1, copy.copy(Q)) # the depth decreases here to accelerate the process when it does not find a valid question easily
                
            children[str(v)] = self.initialization_one(new_df, depth - 1, copy.copy(Q))
        
        return Interview(value, children)
    
    # Calculates how good is an interview for a certain population
    def fitness(self, interview, df):
        t0 = time.perf_counter()
        value = interview.value
        if value == -1:
            return self.fitness_leaf(df)
        column_name = df.columns[value]
        unique_values = df[column_name].unique()

        for v in unique_values:
            new_df = self.restrict_df(df, column_name, v)
            fit = min(1, self.fitness(interview.children[str(v)], new_df))
        return fit
    
    # Calculates how good is a branch of the interview, consulting the average fitness of the remaining population
    def fitness_leaf(self, df):
        fitness_column = df.columns[-1]
        fit = df[fitness_column].mean()
        return max(fit, 1 - fit)
    
    # From a population, a question "column_name" and an answer "value", generates the remaining population that would answer "value" to the question "column_name"
    def restrict_df(self, df, column_name, value):
        filtered_df_aux = df[df[column_name] == value]
        filtered_df = filtered_df_aux[:]
        return filtered_df

    # Checks if the remaining population (in one of the leaves of the interview) has not inferred private information
    def satisfiesPrivate(self, df):
        ok = True
        count_total = len(df)
        for (question, answer, a, b) in self.P:
            count_qa = df[question].value_counts().get(answer, 0)
            if not (a <= (count_qa / count_total) <= b):
                ok = False
        return ok


    # Crossover algorithm
    # Creates a new interview, where the root question is a question that has not been asked in the interviews
    # The subinterviews resulting from each possible answer to the root question are one of the parents (at random)
    def crossover(self, interview1, interview2):
        valid_questions = set(range(0, self.nquestions)) - set(self.nodes(interview1)) - set(self.nodes(interview2))

        privateQuestions = set()
        for (question, _, _, _) in self.P:
            privateQuestions.add(self.df.columns.get_loc(question))
        valid_questions -= privateQuestions

        column_name = "error"
        unique_values = set()
        value = -1
        ok = False
        for _ in range(0, self.k):
            if len(valid_questions) == 0:
                res = self.initialization_one(self.df, self.k, list(range(0, self.nquestions)))
                return res
            
            value = random.choice(list(valid_questions))
            valid_questions -= {value}
            column_name = self.df.columns[value]
            unique_values = set(self.df[column_name].unique())

            childrenRoot = {}
            for v in unique_values:
                childrenRoot[str(v)] = Interview(-1, {})
            if (self.priv_violations(Interview(value, childrenRoot), self.df) == 0):
                ok = True
                break

        if not ok:
            res = self.initialization_one(self.df, self.k, list(range(0, self.nquestions)))
            return res

        children = {}
        for v in unique_values:
            new_df = self.restrict_df(self.df, column_name, v)
            if random.random() < 0.5:
                children[str(v)] = self.prune(new_df, interview1, self.k - 1)
            else:
                children[str(v)] = self.prune(new_df, interview2, self.k - 1)
        
        res = Interview(value, children)
        return res
    
    # Prunes an interview without modifying the original interview, replacing the subinterviews that infer private information with empty subinterviews
    def prune(self, df, interview, k):
        children = {}
        column_name = df.columns[interview.value]
        unique_values = df[column_name].unique()
        for v in unique_values:
            new_df = self.restrict_df(df, column_name, v)
            if not self.satisfiesPrivate(new_df):
                return Interview(-1, {})
            if str(v) in interview.children:
                children[str(v)] = self.prune(new_df, interview.children[str(v)], k - 1)
            else:
                children[str(v)] = Interview(-1, {})
        return Interview(interview.value, children)
    
    # Traverses a random path, and on each node it has a small chance of replacing the rest of the interview with a random and correct interview, or one constructed greedily
    def mutation(self, interview):
        return self.mutation_aux(self.df, interview, self.k, list(range(0, self.nquestions)))
    
    def mutation_aux(self, df, interview, depth, Q):
        value = interview.value
        if value == -1:
            return interview
        Q.remove(value)
        if random.random() < self.mutationChance:
            if random.random() < self.greedyMutationChance:
                partial_fitness = partial(self.fitness_for_order, df=df)
                Q.sort(key=partial_fitness, reverse=True)
                return self.greedy_aux(df, depth, Q, 0, self.triesGreedyGA)
            else:
                return self.initialization_one(df, depth, Q)
        column_name = df.columns[value]
        unique_values = df[column_name].unique()
        v = random.choice(unique_values)
        new_df = self.restrict_df(df, column_name, v)
        interview.children[str(v)] = self.mutation_aux(new_df, interview.children[str(v)], depth - 1, Q)
        return interview

    # Pairs the interviews of the population, crosses them and mutates the resulting interviews
    def combinations(self, interviews):
        children = [("error", {})] * (len(interviews))
        random.shuffle(interviews)
        for i in range(1, len(interviews), 2):
            children[i-1] = self.mutation(self.crossover(interviews[i-1], interviews[i]))
            children[i] = self.mutation(self.crossover(interviews[i], interviews[i-1]))
        return children

    # Creates the next population with tournament selection. Pairs the interviews and selects the best one of each pair
    def selections(self, interviews, df):
        random.shuffle(interviews)
        half = math.floor(len(interviews)/2)
        for i in range(0, half):
            if self.fitness(interviews[i + half], df[:]) > self.fitness(interviews[i], df[:]):
                interviews[i] = interviews[i + half]
        return interviews[:half]
    
    # Returns the questions of an interview
    def nodes(self, interview):
        ns = []
        ns.append(interview.value)
        for child in interview.children.values():
            ns = ns + self.nodes(child)
        return ns
    
    # Returns a string that represents an interview
    def interview_to_string(self, interview):
        return self.interview_to_string_aux(interview, " ")

    def interview_to_string_aux(self, interview, space):
        if interview.value == -1:
            return space + "hoja\n"
        r = space + self.df.columns[interview.value] + '\n'
        for key, node in interview.children.items():
            r += space + "[" + key + "]" + '\n' + self.interview_to_string_aux(node, space + " ")
        return r
    
    # Given an interview and a population, returns the number of times that interview infers private information
    def priv_violations(self, interview, df):
        column_number = interview.value
        priv_violations = 0
        if column_number == -1:
            return self.priv_leaf(df)
        column_name = df.columns[column_number]
        for value in set(df[column_name]):
            filtered_df_aux = df[df[column_name] == value]
            filtered_df = filtered_df_aux[:]
            pv = self.priv_violations(interview.children[str(value)], filtered_df)
            priv_violations += pv
        return priv_violations
    
    def priv_leaf(self, df):
        priv_violations = 0
        count_total = len(df)
        for (q, a, var_min, var_max) in self.P:
            count_qa = df[q].value_counts().get(a, 0)
            if not (var_min <= (float(count_qa) / float(count_total)) <= var_max):
                priv_violations += 1
        return priv_violations
    
    # Returns the depth of an interview
    def depth_interview(self, interview:Interview):
        if interview.value == -1:
            return 0
        depth = 0
        for c in interview.children.values():
            depth = max(depth, 1 + self.depth_interview(c))
        return depth
    
    # Checks if an interview is correct
    def check_correctness(self, interview):
        return self.depth_interview(interview) <= self.k and self.priv_violations(interview, self.df) == 0
