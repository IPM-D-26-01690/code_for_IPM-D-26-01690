# -*- coding: utf-8 -*-
"""DHHO single-objective feature-selection algorithm.

This self-contained implementation includes CVT initialization,
Hamming-distance-guided global search, and multi-point mutation local search.
"""

import copy
import os
import random
import math
import numpy
import pandas as pd
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import bernoulli
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import xlsxwriter

TRAIN_INDEX = None
TEST_INDEX = None
FITNESS_CACHE = None


def hamming_distance_to_generators(solution, generators):
    """Return all generator indices having maximum bit agreement."""
    agreement = numpy.sum(generators == solution, axis=1)
    return numpy.flatnonzero(agreement == numpy.max(agreement)).tolist()


def cvt_initialization(population_size, dimension):
    """Create a binary population using the CVT initialization strategy."""
    generators = numpy.random.randint(0, 2, (population_size, dimension))
    assigned = [[] for _ in range(population_size)]
    extra_solutions = numpy.random.randint(0, 2, (2 * population_size, dimension))

    for extra_index, solution in enumerate(extra_solutions):
        candidates = hamming_distance_to_generators(solution, generators)
        smallest_group = min(len(assigned[index]) for index in candidates)
        candidates = [
            index for index in candidates if len(assigned[index]) == smallest_group
        ]
        selected = candidates[numpy.random.randint(0, len(candidates))]
        assigned[selected].append(extra_index)

    centroids = generators.astype(float)
    for generator_index, members in enumerate(assigned):
        if members:
            centroids[generator_index] += numpy.sum(extra_solutions[members], axis=0)
        centroids[generator_index] /= 1 + len(members)

    return numpy.array(
        [
            [int(bernoulli.rvs(probability)) for probability in centroid]
            for centroid in centroids
        ]
    )


def single_get_ObjectFunction(solution, dim, x, y, z, costTotal):
    while sum(solution) < 2:
        solution = numpy.random.randint(0, 2, dim)
    cache_key = tuple(solution.astype(int))
    if cache_key in FITNESS_CACHE:
        return FITNESS_CACHE[cache_key]
    column_use = (solution == 1)
    x_test = x.columns[column_use]
    z_test = z.columns[column_use]
    clf = OneVsOneClassifier(SVC(C=1.0, kernel='rbf', cache_size=1500))
    X_train, X_test = x.loc[TRAIN_INDEX, x_test], x.loc[TEST_INDEX, x_test]
    y_train, y_test = y.loc[TRAIN_INDEX], y.loc[TEST_INDEX]
    clf.fit(X_train, y_train)
    objective_1 = 1 - accuracy_score(y_test, clf.predict(X_test))
    costSum = 0
    for i in z_test:
        costSum += z[i]
    objective_2 = 1.0 * sum(column_use) / dim
    objective_3 = 1.0 * costSum[0] / costTotal
    fitness = objective_1 * 0.98 + objective_2 * 0.01 + objective_3 * 0.01
    result = fitness, objective_1, objective_2, objective_3, costSum[0]
    FITNESS_CACHE[cache_key] = result
    return result


def transform_S(solution, dim):
    solution_fit = solution * 1
    for j in range(solution.shape[0]):
        a = 1 / (1 + numpy.exp(-solution[j]))
        solution_fit[j] = 1 if a >= 0.5 else 0
    while sum(solution_fit) < 1:
        b = 1 - 2 * numpy.random.rand(dim)
        solution_fit = b * 1
        for j in range(solution_fit.shape[0]):
            a = 1 / (1 + numpy.exp(-solution_fit[j]))
            solution_fit[j] = 1 if a >= 0.5 else 0
    return solution_fit


def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
        math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    step = u / (numpy.abs(v) ** (1 / beta))
    return step


def delete_dup(list_data, dim, ignored_index):
    for i in range(list_data.shape[0]):
        for j in range(list_data.shape[0]):
            if j in ignored_index:
                continue
            if all(list_data[i] == list_data[j]) and i != j:
                list_data[j] = numpy.random.randint(0, 2, (1, dim))
    return list_data


def compute_diversity(population):
    n = population.shape[0]
    if n <= 1: return 1.0
    dim = population.shape[1]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += numpy.sum(population[i] != population[j])
    return total / (n * (n - 1) / 2 * dim)


def hamming_guided_indices(population):
    """Precompute Hamming-guided companion indices once per iteration."""
    population_size = population.shape[0]
    dis = numpy.zeros((population_size, population_size))
    for i in range(population_size):
        for j in range(i + 1, population_size):
            dis[i, j] = numpy.sum(population[i, :] != population[j, :])
            dis[j, i] = dis[i, j]

    indicator = numpy.sum(dis, axis=0)
    selected = numpy.zeros(population_size, dtype=int)
    for idx in range(population_size):
        idx_indicator = indicator.copy()
        idx_indicator[idx] = 0
        max_value = numpy.max(idx_indicator)
        candidates = numpy.where(idx_indicator == max_value)[0]
        if candidates.shape[0] == 1:
            selected[idx] = candidates[0]
        else:
            farthest = candidates[dis[idx, candidates] == numpy.max(dis[idx, candidates])]
            selected[idx] = farthest[numpy.random.randint(0, farthest.shape[0])]
    return selected


def multi_point_mutation(solution, dim, mutation_rate):
    mutated = solution.copy()
    num_bits = max(1, int(mutation_rate * dim))
    indices = numpy.random.choice(dim, size=num_bits, replace=False)
    mutated[indices] = 1 - mutated[indices]
    return mutated


def compressed_population_size(original_size, dim):
    """Return the late-stage active population size."""
    if dim >= 1000:
        return max(5, math.ceil(original_size * 0.25))
    return max(4, math.ceil(original_size * 0.50))


def shrink_population_by_fitness(population, fitnessrecord, f1record, f2record,
                                 f3record, costrecord, target_size,
                                 rabbit_location=None, rabbit_values=None):
    """Keep elite solutions when local search starts to reduce fitness calls."""
    current_size = population.shape[0]
    target_size = max(2, min(target_size, current_size))
    if target_size >= current_size:
        return (population, fitnessrecord, f1record, f2record, f3record,
                costrecord, current_size)

    combined_population = population
    combined_fitness = fitnessrecord
    combined_f1 = f1record
    combined_f2 = f2record
    combined_f3 = f3record
    combined_cost = costrecord

    if (rabbit_location is not None and rabbit_values is not None
            and numpy.isfinite(rabbit_values[0])):
        combined_population = numpy.vstack((combined_population, rabbit_location.reshape(1, -1)))
        combined_fitness = numpy.append(combined_fitness, rabbit_values[0])
        combined_f1 = numpy.append(combined_f1, rabbit_values[1])
        combined_f2 = numpy.append(combined_f2, rabbit_values[2])
        combined_f3 = numpy.append(combined_f3, rabbit_values[3])
        combined_cost = numpy.append(combined_cost, rabbit_values[4])

    order = numpy.argsort(combined_fitness)
    keep = []
    for idx in order:
        if not any(numpy.array_equal(combined_population[idx], combined_population[j]) for j in keep):
            keep.append(idx)
        if len(keep) == target_size:
            break
    if len(keep) < target_size:
        for idx in order:
            if idx not in keep:
                keep.append(idx)
            if len(keep) == target_size:
                break

    keep = numpy.array(keep)
    return (combined_population[keep, :].copy(),
            combined_fitness[keep].copy(),
            combined_f1[keep].copy(),
            combined_f2[keep].copy(),
            combined_f3[keep].copy(),
            combined_cost[keep].copy(),
            target_size)

def dhho(datasetName,
         enable_improve1=True,
         enable_improve2=True,
         enable_improve3=True,    # Multi-point Mutation LS
         algo_name='DHHO'):
    global TRAIN_INDEX, TEST_INDEX, FITNESS_CACHE
    FITNESS_CACHE = {}
    start1 = time.time()

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    inputdata = os.path.join(project_dir, 'Dataset', datasetName + '.csv')
    inputdata1 = os.path.join(project_dir, 'Dataset', datasetName + '-cost.csv')
    dataset = pd.read_csv(inputdata, header=None)
    dataset1 = pd.read_csv(inputdata1, header=None)
    all_indices = numpy.arange(len(dataset))
    TRAIN_INDEX, TEST_INDEX = train_test_split(
        all_indices, test_size=0.30, random_state=42, stratify=dataset.iloc[:, -1]
    )
    path = os.path.join(project_dir, 'Results', algo_name,
                        datasetName + '_' + algo_name + '_' + str(time.time()) + '.xlsx')
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    workbook = xlsxwriter.Workbook(path)
    worksheet1 = workbook.add_worksheet()
    ColName = ["fitness", 'f1', 'f2', 'f3', 'cost', 'time']
    for i in range(len(ColName)):
        worksheet1.write(0, i, ColName[i])
    worksheet2 = workbook.add_worksheet()

    x = dataset.iloc[:, 0:-1]
    x = pd.DataFrame(StandardScaler().fit_transform(x))
    y = dataset.iloc[:, -1]
    z = dataset1.iloc[:]
    dim = x.shape[1]
    costTotal = 0.0
    for i in range(dim):
        costTotal += z[i]
    costTotal = costTotal[0]

    low_dim = ['CKD']
    if datasetName in low_dim:
        population_size = 8
        T = 100
    else:
        population_size = 20
        T = 200
    original_population_size = population_size
    population_compressed = False

    lb, ub = -6, 6

    Rabbit_Location = numpy.zeros(dim)
    Rabbit_Energy = float("inf")
    Rabbit_f1 = float("inf")
    Rabbit_f2 = float("inf")
    Rabbit_f3 = float("inf")
    Rabbit_cost = float("inf")

    best_in_history = []
    fitnessrecord = numpy.zeros(population_size)
    f1record = numpy.zeros(population_size)
    f2record = numpy.zeros(population_size)
    f3record = numpy.zeros(population_size)
    costrecord = numpy.zeros(population_size)

    if enable_improve1:
        population = cvt_initialization(population_size, dim)
        population = numpy.array(population)
        population = delete_dup(population, dim, [])
    else:
        population = numpy.random.uniform(lb, ub, (population_size, dim))
        for i in range(population_size):
            population[i, :] = transform_S(population[i, :], dim)

    for i in range(population_size):
        fv = single_get_ObjectFunction(population[i, :], dim, x, y, z, costTotal)
        fitnessrecord[i], f1record[i], f2record[i], f3record[i], costrecord[i] = fv

    best_idx = numpy.argmin(fitnessrecord)
    best_individual = copy.deepcopy(population[best_idx, :])
    best_fitness = fitnessrecord[best_idx]
    best_f1 = f1record[best_idx]
    best_f2 = f2record[best_idx]
    best_f3 = f3record[best_idx]
    best_cost = costrecord[best_idx]

    stagnation_counter = 0

    t = 0
    while t < T:
        ignored_index = []
        start = time.time()

        ls_start = 0.3 * T if (enable_improve3 and dim <= 10) else 0.5 * T
        ls_active = enable_improve3 and t > ls_start
        if ls_active and not population_compressed:
            target_size = compressed_population_size(original_population_size, dim)
            old_population_size = population_size
            (population, fitnessrecord, f1record, f2record, f3record,
             costrecord, population_size) = shrink_population_by_fitness(
                population, fitnessrecord, f1record, f2record, f3record, costrecord,
                target_size, Rabbit_Location,
                (Rabbit_Energy, Rabbit_f1, Rabbit_f2, Rabbit_f3, Rabbit_cost))
            population_compressed = True
            best_idx = numpy.argmin(fitnessrecord)
            best_individual = copy.deepcopy(population[best_idx, :])
            best_fitness = fitnessrecord[best_idx]
            best_f1 = f1record[best_idx]
            best_f2 = f2record[best_idx]
            best_f3 = f3record[best_idx]
            best_cost = costrecord[best_idx]
            print(f"[{datasetName}] late-stage active population: "
                  f"{old_population_size}->{population_size}")

        if ls_active:
            n_samples = x.shape[0]
            if n_samples > 2000:
                L = 15; mutation_rate = 0.25; reduce_interval = 4
            elif n_samples > 500:
                L = 12; mutation_rate = 0.28; reduce_interval = 3
            elif dim <= 10:
                L = 15; mutation_rate = 0.50; reduce_interval = 2
            else:
                L = 10; mutation_rate = 0.30; reduce_interval = 3
            for k in range(L):
                if k % reduce_interval == 0 and sum(best_individual) > 2:
                    mutated = best_individual.copy()
                    selected = numpy.where(mutated == 1)[0]
                    n_remove = 2 if dim <= 10 and len(selected) >= 4 else 1
                    mutated[numpy.random.choice(selected, size=n_remove, replace=False)] = 0
                else:
                    mutated = multi_point_mutation(best_individual, dim, mutation_rate)
                fit, f1_v, f2_v, f3_v, cost_v = single_get_ObjectFunction(
                    mutated, dim, x, y, z, costTotal)
                if fit < best_fitness:
                    best_individual = mutated
                    best_fitness = fit
                    best_f1 = f1_v
                    best_f2 = f2_v
                    best_f3 = f3_v
                    best_cost = cost_v

        if best_fitness < Rabbit_Energy - 1e-12:
            Rabbit_Energy = best_fitness
            Rabbit_Location = best_individual.copy()
            Rabbit_f1 = best_f1
            Rabbit_f2 = best_f2
            Rabbit_f3 = best_f3
            Rabbit_cost = best_cost
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        best_in_history.append(Rabbit_Energy)

        E1 = 2 * (1 - (t / T))
        population_diversity = compute_diversity(population)
        hamming_indices = hamming_guided_indices(population) if population_diversity > 0.25 else None
        for i in range(population_size):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0

            # Exploration
            if abs(Escaping_Energy) >= 1:
                q = random.random()
                if q < 0.5:
                    if enable_improve2 and hamming_indices is not None:
                        maxId = hamming_indices[i]
                        population[i, :] = (population[maxId, :]
                                            - random.random() * abs(
                            population[maxId, :] - 2 * random.random() * population[i, :]))
                    else:
                        rand_idx = random.randint(0, population_size - 1)
                        population[i, :] = (population[rand_idx, :]
                                            - random.random() * abs(
                            population[rand_idx, :] - 2 * random.random() * population[i, :]))
                else:
                    population[i, :] = (
                        (Rabbit_Location - population.mean(0))
                        - random.random() * ((ub - lb) * random.random() + lb))
                population[i, :] = transform_S(population[i, :], dim)

            # Exploitation
            else:
                r = random.random()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege
                    population[i, :] = Rabbit_Location - Escaping_Energy * abs(
                        Rabbit_Location - population[i, :])
                    population[i, :] = transform_S(population[i, :], dim)

                if r >= 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege
                    Js = 2 * (1 - random.random())
                    population[i, :] = ((Rabbit_Location - population[i, :])
                                        - Escaping_Energy * abs(
                        Js * Rabbit_Location - population[i, :]))
                    population[i, :] = transform_S(population[i, :], dim)

                if r < 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft + dives
                    ignored_index.append(i)
                    Js = 2 * (1 - random.random())
                    X1 = (Rabbit_Location - Escaping_Energy * abs(
                        Js * Rabbit_Location - population[i, :]))
                    X1 = transform_S(X1, dim)
                    fv1 = single_get_ObjectFunction(X1, dim, x, y, z, costTotal)
                    if fv1[0] < fitnessrecord[i]:
                        population[i, :] = X1.copy()
                        fitnessrecord[i], f1record[i], f2record[i], f3record[i], costrecord[i] = fv1
                    else:
                        X2 = (Rabbit_Location - Escaping_Energy * abs(
                            Js * Rabbit_Location - population[i, :])
                              + numpy.multiply(numpy.random.randn(dim), Levy(dim)))
                        X2 = transform_S(X2, dim)
                        fv2 = single_get_ObjectFunction(X2, dim, x, y, z, costTotal)
                        if fv2[0] < fitnessrecord[i]:
                            population[i, :] = X2.copy()
                            fitnessrecord[i], f1record[i], f2record[i], f3record[i], costrecord[i] = fv2

                if r < 0.5 and abs(Escaping_Energy) < 0.5:  # Hard + dives
                    ignored_index.append(i)
                    Js = 2 * (1 - random.random())
                    X1 = (Rabbit_Location - Escaping_Energy * abs(
                        Js * Rabbit_Location - population.mean(0)))
                    X1 = transform_S(X1, dim)
                    fv1 = single_get_ObjectFunction(X1, dim, x, y, z, costTotal)
                    if fv1[0] < fitnessrecord[i]:
                        population[i, :] = X1.copy()
                        fitnessrecord[i], f1record[i], f2record[i], f3record[i], costrecord[i] = fv1
                    else:
                        X2 = (Rabbit_Location - Escaping_Energy * abs(
                            Js * Rabbit_Location - population.mean(0))
                              + numpy.multiply(numpy.random.randn(dim), Levy(dim)))
                        X2 = transform_S(X2, dim)
                        fv2 = single_get_ObjectFunction(X2, dim, x, y, z, costTotal)
                        if fv2[0] < fitnessrecord[i]:
                            population[i, :] = X2.copy()
                            fitnessrecord[i], f1record[i], f2record[i], f3record[i], costrecord[i] = fv2

        population = delete_dup(population, dim, ignored_index)

        for i in range(population_size):
            if i in ignored_index:
                continue
            fv = single_get_ObjectFunction(population[i, :], dim, x, y, z, costTotal)
            fitnessrecord[i], f1record[i], f2record[i], f3record[i], costrecord[i] = fv

        best_idx = numpy.argmin(fitnessrecord)
        best_individual = copy.deepcopy(population[best_idx, :])
        best_fitness = fitnessrecord[best_idx]
        best_f1 = f1record[best_idx]
        best_f2 = f2record[best_idx]
        best_f3 = f3record[best_idx]
        best_cost = costrecord[best_idx]

        if dim <= 7 and t % 3 == 0:
            worst_idx = numpy.argmax(fitnessrecord)
            best_inject, best_inject_fit = None, float('inf')
            best_inject_fv = None
            for _ in range(5):
                cand = numpy.random.randint(0, 2, dim)
                while sum(cand) < 2:
                    cand = numpy.random.randint(0, 2, dim)
                fv = single_get_ObjectFunction(cand, dim, x, y, z, costTotal)
                if fv[0] < best_inject_fit:
                    best_inject_fit = fv[0]
                    best_inject = cand
                    best_inject_fv = fv
            population[worst_idx] = best_inject
            fitnessrecord[worst_idx], f1record[worst_idx], f2record[worst_idx], f3record[worst_idx], costrecord[worst_idx] = best_inject_fv

        t += 1
        end = time.time()
        if t % 10 == 0 or t == 1:
            print(f"[{datasetName}] iter {t}/{T}, best_f1={best_f1:.6f}, "
                  f"stagn={stagnation_counter}, time={end - start:.2f}s")

    if best_fitness < Rabbit_Energy:
        Rabbit_Energy = best_fitness
        Rabbit_Location = best_individual.copy()
        Rabbit_f1 = best_f1
        Rabbit_f2 = best_f2
        Rabbit_f3 = best_f3
        Rabbit_cost = best_cost
    best_in_history.append(Rabbit_Energy)

    end1 = time.time()
    worksheet1.write(1, 0, Rabbit_Energy)
    worksheet1.write(1, 1, Rabbit_f1)
    worksheet1.write(1, 2, Rabbit_f2)
    worksheet1.write(1, 3, Rabbit_f3)
    worksheet1.write(1, 4, Rabbit_cost)
    worksheet1.write(1, 5, end1 - start1)
    for i in range(dim):
        worksheet1.write(1, 6 + i, Rabbit_Location[i])
    for k in range(len(best_in_history)):
        worksheet2.write(1, k, best_in_history[k])
    workbook.close()

    print(f"[{datasetName}] DONE. fit={Rabbit_Energy:.6f}, f1={Rabbit_f1:.6f}, "
          f"f2={Rabbit_f2:.4f}, cost={Rabbit_cost:.2f}, time={end1 - start1:.1f}s")

    return Rabbit_Energy, Rabbit_f1, Rabbit_f2, Rabbit_f3, Rabbit_cost



if __name__ == "__main__":
    dhho("CKD")
