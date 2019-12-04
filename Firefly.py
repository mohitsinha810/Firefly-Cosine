import numpy as np
import random
from copy import deepcopy
import math

from Text import generate_weight_matrix, compute_longest_path_weight
from greedy import greedy_optimizer

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

class FireflyOptimizer(object):
    def __init__(self, fitness_fun, docs, docs_representation, max_length, population_size, survival_rate, mutation_rate, reproduction_rate, maximization=False, sentences_rep=None):
        # np.random.seed(123)

        self._fitness_fun = fitness_fun
        self._population_size = population_size
        self._survival_rate = survival_rate
        self._mutation_rate = mutation_rate
        self._reproduction_rate = reproduction_rate
        self._maximization = maximization

        self._fireflies  = []
        self._light_intensity = []

        self._docs = docs
        self._docs_representation = docs_representation
        self._sentences_rep = sentences_rep
        self._max_length = max_length

        self._docs_weights = generate_weight_matrix(self._docs_representation)
        self._N = len(self._docs_representation)
        self._M = max(self._docs_weights)
        self._A = compute_longest_path_weight(self._docs_representation)

        self._sentences = []
        self._sentence_tokens = []
        for title, doc in docs:
            # self._sentences.append(title)
            self._sentence_tokens.append(tokenizer.tokenize(title))
            self._sentences.extend(doc)
            for s in doc:
                self._sentence_tokens.append(tokenizer.tokenize(s))



    def _inilialize_population(self):
        # print len(self._sentences)
        print 'Initializing fireflies';
        self._fireflies = np.random.randint(low=0, high=2, size=(self._population_size, len(self._sentences)))
        self._fireflies = self._fireflies.astype('float64')


    def _inilialize_light_intensities(self):
        print 'Initializing light intensities'

        index = 0
        for firefly in self._fireflies:
            sys_summary = self._create_summary(firefly)
            # print sys_summary
            if self._sentences_rep != None:
                score = self._fitness_fun(sys_summary, self._A, self._sentences_rep, self._docs, self._N, self._M)
            else:
                score = self._fitness_fun(sys_summary, self._A, self._docs, self._N, self._M)
            self._light_intensity.append(score)
        print self._light_intensity


    def run_firefly(self, iteration):
        
        self._inilialize_population()

        # print len(self._fireflies)
        # print len(self._sentences)

        self._inilialize_light_intensities()

        firefly_best = []
        firefly_best_score = 0
        noChange = 0
        alpha = 1
        

        for iter in range(iteration):
            if noChange >= 500:
                alpha += 1
            for i in range(self._population_size):
                for j in range(self._population_size):
                    if self._light_intensity[j] > self._light_intensity[i]:
                        self._move_firefly(i, j, alpha)

            # sorted_fireflies = [x for _,x in sorted(zip(self._light_intensity,self._fireflies))]
            # self._fireflies = np.array(sorted_fireflies)
            new_firefly_best = self._fireflies[-1]
            new_firefly_best = new_firefly_best + alpha * ( np.random.rand(len(self._sentences)) - 0.5 )
            new_firefly_best = self._normalize(new_firefly_best)
            new_sys_summary = self._create_summary(new_firefly_best)
            sys_summary = self._create_summary(firefly_best)
            new_score = self._fitness_fun(new_sys_summary, self._A, self._docs, self._N, self._M)
            if new_score > firefly_best_score:
                firefly_best = new_firefly_best
                firefly_best_score = new_score
                noChange = 0
            else:
                self._fireflies[-1] = firefly_best_score
                noChange += 1
            # if score > self._light_intensity[-1]:
            self._light_intensity[-1] = firefly_best_score
            self._fireflies[-1] = firefly_best

            print "Iteration--", iter, " : ", self._fireflies[-1]


        self._light_intensity.sort()
        print self._light_intensity
        print self._create_summary(self._fireflies[-1])


    def _move_firefly(self, i, j, alpha):

        firefly_i = self._fireflies[i]
        firefly_j = self._fireflies[j]
        # print firefly_i
        # print firefly_j

        new_firefly = firefly_i + 1 * math.e**(-1 * self._distance(firefly_i, firefly_j)**2) \
                      * (firefly_j - firefly_i) + 2*alpha * ( np.random.rand(len(self._sentences)) - 0.5 )
        normalized_new_firefly = self._normalize(new_firefly)
        sys_summary = []
        index = 0
        # print new_firefly
        sys_summary = self._create_summary(normalized_new_firefly)
        new_light_intensity = self._fitness_fun(sys_summary, self._A, self._docs, self._N, self._M)

        # print 'New: ', new_light_intensity
        # print self._light_intensity[i]
        # print self._light_intensity[j]

        # if new_light_intensity > self._light_intensity[i]:
        self._fireflies[i] = new_firefly
        self._light_intensity[i] = new_light_intensity



    def _distance(self, vector_1, vector_2):
        return np.linalg.norm(vector_1 - vector_2)


    def _normalize(self, x):
        return (x - min(x)) / (max(x) - min(x))

    def _create_summary(self, firefly):
        #print firefly
        sys_summary = []

        updatedFirefly = {}

        for i in range(len(firefly)):
            updatedFirefly[i] = firefly[i]

        # sort the firefly on the basis of values
        sortedFirefly = sorted(updatedFirefly.items(), key=lambda kv: kv[1], reverse=True)

        # Limit the firefly to the summary length
        noOfSentences = 0
        for key, value in sortedFirefly:
            sentence = self._sentences[key]
            sys_summary.append(sentence)
            noOfSentences += 1
            if noOfSentences == 5:
                break

        return sys_summary


    def _create_final_summary(self, firefly):
        # print firefly

        indices = []
        for i in range(len(self._sentences)):
            indices.append(i)
        sorted_indices = [x for _,x in sorted(zip(firefly,indices))]

        top_indices = sorted_indices[:self._max_length]

        sys_summary = []
        for index in top_indices:
            sys_summary.append(self._sentences[index])
        return sys_summary



    # def _normalize(self,dat, out_range=(-1, 1)):
    #     domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

    #     def interp(x):
    #         return out_range[0] * (1.0 - x) + out_range[1] * x

    #     def uninterp(x):
    #         b = 0
    #         if (domain[1] - domain[0]) != 0:
    #             b = domain[1] - domain[0]
    #         else:
    #             b =  1.0 / domain[1]
    #         return (x - domain[0]) / b

    #     return interp(uninterp(dat)) 


    # def _generate_random_population(self, n):
    #     population = []
    #     for i in xrange(n):
    #         population.append(self._create_random_individual())
    #     return population

    # def _score_population(self, population):
    #     scored_population = []
    #     for individual in population:
    #         # score = self._fitness_fun(individual, self._docs)
    #         if self._sentences_rep != None:
    #             score = self._fitness_fun(individual, self._docs_representation, self._sentences_rep)
    #         else:
    #             score = self._fitness_fun(individual, self._docs_representation)
    #         scored_population.append((individual, score))

    #     return scored_population

    # def _select_survivors(self, scored_population):
    #     sorted_population = sorted(scored_population, key=lambda tup: tup[1], reverse=self._maximization)

    #     percentage_winner = 0.5

    #     to_keep = int(self._survival_rate * self._population_size)
    #     number_winners = int(percentage_winner * to_keep)
    #     winners = [tup[0] for tup in sorted_population[:number_winners]]

    #     losers = sorted_population[number_winners:]

    #     number_losers = int((1 - percentage_winner) * to_keep)

    #     survivors = deepcopy(winners)
    #     random_scores = np.random.rand(len(losers))

    #     sorted_losers = sorted(zip(losers, random_scores), key=lambda tup: tup[1])
    #     loser_survivors = [tup[0][0] for tup in sorted_losers[:number_losers]]

    #     survivors.extend(loser_survivors)
    #     return survivors, winners

    # def _new_generation(self, scored_population):
    #     new_generation, winners = self._select_survivors(scored_population)
    #     new_generation = self._mutate(new_generation)
    #     new_generation.extend(self._reproduction(winners, len(new_generation)))
    #     individuals_to_create = self._population_size - len(new_generation)
    #     new_generation.extend(self._generate_random_population(individuals_to_create))

    #     return new_generation

    # def _len_individual(self, individual):
    #     len_ = 0
    #     for sentence in individual:
    #         len_ += len(tokenizer.tokenize(sentence))
    #     return len_

    # def _mutate(self, population, mutation_rate="auto"):
    #     if mutation_rate == "auto":
    #         mutation_rate = self._mutation_rate

    #     nb_mutant = int(mutation_rate * len(population))

    #     random_scores = np.random.rand(len(population))
    #     sorted_population = sorted(zip(population, random_scores), key=lambda tup: tup[1])
    #     mutants = [tup[0] for tup in sorted_population[:nb_mutant]]

    #     mutated = []
    #     i = 0
    #     for mutant in mutants:
    #         to_mutate = deepcopy(mutant)

    #         sentence_to_remove = random.choice(to_mutate)
    #         idx = to_mutate.index(sentence_to_remove)
    #         del to_mutate[idx]

    #         available_size = self._max_length - self._len_individual(to_mutate)

    #         available_sentences = [s[0] for s in zip(self._sentences, self._sentence_tokens) if len(s[1]) <= available_size]
    #         if available_sentences != []:
    #             i += 1
    #             sentence_to_add = random.choice(available_sentences)
    #             to_mutate.append(sentence_to_add)

    #             mutated.append(to_mutate)

    #     population.extend(mutated)
    #     return population

    # def _reproduction(self, population_winners, population_size, reproduction_rate="auto"):
    #     if reproduction_rate == "auto":
    #         reproduction_rate = self._reproduction_rate

    #     parents = []
    #     number_families = int(reproduction_rate * population_size)

    #     for i in xrange(number_families):
    #         parents.append(random.sample(population_winners, 2))

    #     children = []
    #     for father, mother in parents:
    #         genetic_pool = [s for s in self._sentences if s in father]
    #         genetic_pool.extend([s for s in self._sentences if s in mother])

    #         random_scores = np.random.rand(len(genetic_pool))

    #         scored_sentences = zip(self._sentences, random_scores)
    #         sorted_sentences = sorted(scored_sentences, key=lambda tup: tup[1], reverse=True)
    #         child = greedy_optimizer(sorted_sentences, self._max_length)

    #         children.append(child)

    #     return children

    # def initial_population(self):
    #     initial_population = self._generate_random_population(self._population_size)
    #     print "initial population len:", len(initial_population)
    #     return initial_population

    # def _is_better(self, scored_individual, best_scored_individual):
    #     if self._maximization:
    #         return scored_individual[1] > best_scored_individual[1]
    #     return scored_individual[1] < best_scored_individual[1]

    # def evolve(self, epoch):
    #     population = self.initial_population()
    #     if self._maximization:
    #         best_individual = (None, -10000)
    #     else:
    #         best_individual = (None, 10000)
    #     for i in xrange(epoch):
    #         print "Iteration: ", i, " -- best individual: ", best_individual[0]
    #         print
    #         scored_population = self._score_population(population)
    #         sorted_population = sorted(scored_population, key=lambda tup: tup[1], reverse=self._maximization)
    #         best_individual_in_generation = sorted_population[0]

    #         if self._is_better(best_individual_in_generation, best_individual):
    #             best_individual = best_individual_in_generation

    #         population = self._new_generation(scored_population)

    #     return best_individual