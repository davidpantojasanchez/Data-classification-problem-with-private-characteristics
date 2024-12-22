import random
import copy
import math
import pandas
import time
from functools import partial

FITNESS_COLUMN = 'SINCERIDAD'
QUANTITY_COLUMN = 'QUANTITY'

# Interview

class Interview:
     def __init__(self, value, children):  # "children" is a map from answers to subinterviews
        self.value = value
        self.children = children

class Counter:
    def __init__(self, value):
        self.val = value
    
    def zero(self):
        return self.val <= 0
    
    def decrease(self):
        self.val -= 1

# Heuristic algorithms

class Heuristic:

    # Variables of the heuristic algorithms

    mutationChance:float
    greedyMutationChance:float
    rounds:int
    population:int

    nquestions:int          # Number of questions of the instance of the problem. We can obtain it from the DataFrame, but it is useful to have it here
    TRIES = 100             # Number of times the genetic algorithms can try again with a new question
    prog_dir:str            # If we want to measure the performance of the genetic algorithms after every few rounds, we can indicate here where to store that information

    # Variables of the instance of the problem

    P:list
    k:int
    df:pandas.DataFrame

    def __init__(self, df, nquestions, P, k, progress_dir=''):
        self.df = df
        self.nquestions = int(nquestions)
        self.P = P
        self.k = k
        self.prog_dir = progress_dir
    
    # Greedy algorithm
    def greedy(self):
        return self.greedy_aux(self.df, self.k, list(range(self.nquestions)))

    def greedy_aux(self, df, depth, Q):
        # Leaf
        if (len(Q) == 0 or depth == 0):
            return Interview(-1, {})
        
        partial_fitness = partial(self.fitness_for_order, df=self.df)
        Q.sort(key=partial_fitness, reverse=True)
        
        # Traverses the questions in descending order of fitness
        preserves_privacy = True
        for i in range(len(Q)):
            value = Q[i]
            column_name = df.columns[value]
            unique_values = df[column_name].unique()
            # Checks if an answer infers private information. If not, it keeps the value and breaks the loop.
            for v in unique_values:
                new_df = self.restrict_df(df, column_name, v)
                if not self.preserves_privacy_leaf(new_df):
                    preserves_privacy = False
                    break
            if preserves_privacy:
                break
        # If every question infers private information, it ends the path of the tree
        if not preserves_privacy:
            return Interview(-1, {})

        # Generates the children using the greedy algorithm
        Q.remove(value)
        children = {}
        for v in unique_values:
            new_df = self.restrict_df(df, column_name, v)
            newQ = copy.copy(Q)
            children[str(v)] = self.greedy_aux(new_df, depth - 1, newQ)
        
        return Interview(value, children)
    
    # Auxiliary variation of the fitness function that is used to order the questions
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
            interviews = interviews + self.combinations(interviews, False)
            tCombinations += time.perf_counter() - t0
            t0 = time.perf_counter()
            interviews = self.selections(interviews, df[:])
            tSelection += time.perf_counter() - t0
            
            # Show progress
            if (i % 50 == 0 or (i % 10 == 0) and i < 50) and self.prog_dir != '':
                bestI = interviews[0]
                bestF = self.fitness(interviews[0], df[:])
                for j in range(1, len(interviews)):
                    if self.fitness(interviews[j], df[:]) > bestF:
                        bestI = interviews[j]
                        bestF = self.fitness(interviews[j], df[:])
                print(f'Round {i}: {bestF}')
                with open(self.prog_dir, 'a') as file_results:
                    file_results.write(f'({i}, {bestF})\n')
        
        bestI = interviews[0]
        bestF = self.fitness(interviews[0], df[:])
        for i in range(1, len(interviews)):
            if self.fitness(interviews[i], df[:]) > bestF:
                bestI = interviews[i]
                bestF = self.fitness(interviews[i], df[:])
        return bestI
    
    # Reinforced genetic algorithm
    # Uses the greedy algorithm to initialize the population and to complete the solution
    def ga_with_greedy(self, mutation, greedy_mutation, rounds, population):
        self.mutationChance = mutation
        self.greedyMutationChance = greedy_mutation
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
        interviews.append(self.mutation(interview=self.greedy(), is_greedy=True))
        self.mutationChance = prev_mutation
        self.greedyMutationChance = prev_gm
        self.k = prev_k

        for i in range(0, self.rounds):
            interviews = interviews + self.combinations(interviews, True)
            interviews = self.selections(interviews, df[:])
        
        bestI = interviews[0]
        bestF = self.fitness(interviews[0], df[:])
        for i in range(1, len(interviews)):
            if self.fitness(interviews[i], df[:]) > bestF:
                bestI = interviews[i]
                bestF = self.fitness(interviews[i], df[:])
        
        self.reinforcement(df, bestI, self.k, list(range(0, self.nquestions)))
        return bestI
    
    # Used by ga_with_greedy. Replaces the leaves of a solution with correct subinterviews constructed greedily
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
            #self.greedy_aux(df, k, Q, 0, Counter(self.TRIES))
            self.greedy()

    # Random initialization of the population. Ensures that every interview is correct
    def initialization(self, ninterviews):
        r = []
        for _ in range(0, ninterviews):
            r.append(self.initialization_one(self.df, self.k, list(range(0, self.nquestions)), Counter(self.TRIES)))
        return r
    
    # Construction of a random interview. It ensures it is correct: it has an adequate depth and does not infer private information
    def initialization_one(self, df, depth, Q, tries):
        if (len(Q) == 0 or depth == 0 or tries.zero()):
            return Interview(-1, {})
        
        value = random.choice(Q)
        children = {}
        Q.remove(value)
        column_name = df.columns[value]
        unique_values = df[column_name].unique()

        for v in unique_values:
            new_df = self.restrict_df(df, column_name, v)
            if not self.preserves_privacy_leaf(new_df):
                tries.decrease()
                return self.initialization_one(df, depth, copy.copy(Q), tries)
                
            children[str(v)] = self.initialization_one(new_df, depth - 1, copy.copy(Q), tries)
        
        return Interview(value, children)
    
    # Calculates how good an interview is for a certain population
    def fitness(self, interview, df):
        value = interview.value
        if value == -1:
            return self.fitness_leaf(df)
        column_name = df.columns[value]
        unique_values = df[column_name].unique()

        for v in unique_values:
            new_df = self.restrict_df(df, column_name, v)
            fit = min(1, self.fitness(interview.children[str(v)], new_df))
        return fit
    
    # Calculates how good a path of the interview is, consulting the average fitness of the remaining population
    def fitness_leaf(self, df):
        fit = (df['SINCERIDAD'] * df['QUANTITY']).sum() / df['QUANTITY'].sum()
        return max(fit, 1 - fit)
    
    # From a population, a question "column_name" and an answer "value", generates the remaining population that would answer "value" to the question "column_name"
    def restrict_df(self, df, column_name, value):
        filtered_df_aux = df[df[column_name] == value]
        filtered_df = filtered_df_aux[:]
        return filtered_df

    # Crossover algorithm
    # Creates a new interview, where the root question is a question that has not been asked in the interviews
    # The subinterviews resulting from each possible answer to the root question are one of the parents (at random)
    def crossover(self, interview1, interview2):
        valid_questions = set(range(0, self.nquestions)) - set(self.nodes(interview1)) - set(self.nodes(interview2))

        privateQuestions = set()
        for (question, _, _, _) in self.P:
            privateQuestions.add(self.df.columns.get_loc(question))
        valid_questions -= privateQuestions
        valid_questions -= {'Quantity'}
        
        # Searches for a question that does not infer private information
        column_name = "error"
        unique_values = set()
        value = -1
        ok = False
        for _ in range(0, self.k):
            if len(valid_questions) == 0:
                res = self.initialization_one(self.df, self.k, list(range(0, self.nquestions)), Counter(self.TRIES))
                return res
            
            value = random.choice(list(valid_questions))
            valid_questions -= {value}
            column_name = self.df.columns[value]
            unique_values = set(self.df[column_name].unique())

            childrenRoot = {}
            for v in unique_values:
                childrenRoot[str(v)] = Interview(-1, {})
            if self.preserves_privacy(Interview(value, childrenRoot), self.df):
                ok = True
                break
        # If every question infers private information, returns a random interview
        if not ok:
            res = self.initialization_one(self.df, self.k, list(range(0, self.nquestions)), Counter(self.TRIES))
            return res

        # Generates the children of the root node, using the parent interviews and pruning them, if needed, to ensure the correctness of the resulting interview
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
            if not self.preserves_privacy_leaf(new_df):
                return Interview(-1, {})
            if str(v) in interview.children:
                children[str(v)] = self.prune(new_df, interview.children[str(v)], k - 1)
            else:
                children[str(v)] = Interview(-1, {})
        return Interview(interview.value, children)
    
    # Has a small probability, depending on mutationChance, of replacing a random subinterview of the tree with a random one, or one constructed greedily
    def mutation(self, interview, is_greedy):
        if random.random() < self.mutationChance:
            return self.mutation_aux(self.df, interview, self.k, random.randint(0, self.k), list(range(0, self.nquestions)), is_greedy)
        else:
            return interview
    
    def mutation_aux(self, df, interview, depth, where_to_mutate, Q, is_greedy):
        value = interview.value
        if value == -1:
            return interview
        Q.remove(value)
        if where_to_mutate == depth:
            if random.random() < self.greedyMutationChance and is_greedy:
                partial_fitness = partial(self.fitness_for_order, df=df)
                Q.sort(key=partial_fitness, reverse=True)
                #return self.greedy_aux(df, depth, Q, 0, Counter(self.TRIES))
                return self.greedy()
            else:
                return self.initialization_one(df, depth, Q, Counter(self.TRIES))
        column_name = df.columns[value]
        unique_values = df[column_name].unique()
        v = random.choice(unique_values)
        new_df = self.restrict_df(df, column_name, v)
        interview.children[str(v)] = self.mutation_aux(new_df, interview.children[str(v)], depth - 1, where_to_mutate, Q, is_greedy)
        return interview

    # Pairs the interviews of the population, crosses them and mutates the resulting interviews
    def combinations(self, interviews, is_greedy):
        children:list[Interview] = [Interview("Error", []) for _ in range (len(interviews))]
        random.shuffle(interviews)
        for i in range(1, len(interviews), 2):
            children[i-1] = self.mutation(self.crossover(interviews[i-1], interviews[i]), is_greedy)
            children[i] = self.mutation(self.crossover(interviews[i], interviews[i-1]), is_greedy)
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
            return space + "leaf\n"
        r = space + self.df.columns[interview.value] + '\n'
        for key, node in interview.children.items():
            r += space + "[" + key + "]" + '\n' + self.interview_to_string_aux(node, space + " ")
        return r
    
    # Given an interview and a population, returns the number of times that interview infers private information
    def preserves_privacy(self, interview, df):
        column_number = interview.value
        if column_number == -1:
            return self.preserves_privacy_leaf(df)
        column_name = df.columns[column_number]
        for value in set(df[column_name]):
            filtered_df_aux = df[df[column_name] == value]
            filtered_df = filtered_df_aux[:]
            if not self.preserves_privacy(interview.children[str(value)], filtered_df):
                return False
        return True
    
    # Checks if the remaining population (in one of the leaves of the interview) has not inferred private information
    def preserves_privacy_leaf(self, df):
        count_total = df['QUANTITY'].sum()
        for (q, a, var_min, var_max) in self.P:
            count_qa = df[df[q] == a]['QUANTITY'].sum()
            if not (var_min <= (float(count_qa) / float(count_total)) <= var_max):
                return False
        return True
    
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
        return self.depth_interview(interview) <= self.k and self.preserves_privacy(interview, self.df)
