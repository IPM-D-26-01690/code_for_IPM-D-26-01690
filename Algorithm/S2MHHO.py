# -*- coding: utf-8 -*-
"""S2MHHO multi-objective feature-selection algorithm.

This standalone release implementation runs the complete S2MHHO, containing
all three strategies described in the manuscript:

  S1: Circle Map, a=0.5, b=0.2, Initial_value=0.35
  S2: mut_rate = alpha(t)/sqrt(dim), alpha: 0.50 to 0.08
  S3: GA operator(proM=2)
"""
import os
import sys
import random
import math
import numpy
import pandas as pd
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import xlsxwriter


TRAIN_INDEX = None
TEST_INDEX = None


def circle_map(initial_value, iterations):
    """Generate a Circle-map chaotic sequence."""
    value = initial_value
    sequence = []
    a = 0.5
    b = 0.2
    for _ in range(iterations):
        value = (value + b - (a / (2 * math.pi)) * math.sin(2 * math.pi * value)) % 1
        sequence.append(value)
    return sequence


def sigmoid_binary_mapping(solution):
    """Map a continuous position to a binary solution with at least two features."""
    mapped = numpy.zeros_like(solution)
    for index, value in enumerate(solution):
        mapped[index] = 1 if 1 / (1 + numpy.exp(-value)) > 0.5 else 0
    while numpy.sum(mapped) < 2:
        candidate = 1 - 2 * numpy.random.rand(solution.shape[0])
        mapped = (1 / (1 + numpy.exp(-candidate)) > 0.5).astype(float)
    return mapped


def get_ObjectFunction(population, dim, x, y, z, ignored_index, particles_F):
    f = []
    for i in range(population.shape[0]):
        if i in ignored_index:
            f.append(particles_F[i])
            continue
        while (sum(population[i]) < 2):
            population[i] = numpy.random.randint(0, 2, dim)
        column_use = (population[i] == 1)
        x_test = x.columns[column_use]
        z_test = z.columns[column_use]
        clf = OneVsOneClassifier(SVC(C=1.0, kernel='rbf', cache_size=1500))
        X_train, X_test = x.loc[TRAIN_INDEX, x_test], x.loc[TEST_INDEX, x_test]
        y_train, y_test = y.loc[TRAIN_INDEX], y.loc[TEST_INDEX]
        clf.fit(X_train, y_train)
        fitness_1 = 1 - accuracy_score(y_test, clf.predict(X_test))
        costSum = 0
        for i in z_test:
            costSum += z[i]
        number = [fitness_1, 1.0 * (sum(column_use)) / dim, costSum[0]]
        f.append(number)
    return f


def single_get_ObjectFunction(population, dim, x, y, z):
    while (sum(population) < 2):
        population = numpy.random.randint(0, 2, dim)
    column_use = (population == 1)
    x_test = x.columns[column_use]
    z_test = z.columns[column_use]
    clf = OneVsOneClassifier(SVC(C=1.0, kernel='rbf', cache_size=1500))
    X_train, X_test = x.loc[TRAIN_INDEX, x_test], x.loc[TEST_INDEX, x_test]
    y_train, y_test = y.loc[TRAIN_INDEX], y.loc[TEST_INDEX]
    clf.fit(X_train, y_train)
    fitness_1 = 1 - accuracy_score(y_test, clf.predict(X_test))
    costSum = 0
    for i in z_test:
        costSum += z[i]
    number = [fitness_1, 1.0 * (sum(column_use)) / dim, costSum[0]]
    return number


def dominates(x, y):
    return (all(x <= y) and any(x < y))


def updateArchive(Archive_X, Archive_F, population, particles_F):
    Archive_temp_X = numpy.vstack((Archive_X, population))
    Archive_temp_F = numpy.vstack((Archive_F, particles_F))
    o = numpy.zeros(Archive_temp_F.shape[0])
    for i in range(0, Archive_temp_F.shape[0]):
        for j in range(0, Archive_temp_F.shape[0]):
            if i != j:
                if dominates(Archive_temp_F[j], Archive_temp_F[i]):
                    o[i] = 1
                    break
    Archive_member_no = 0
    Archive_X_updated, Archive_F_updated = [], []
    for i in range(Archive_temp_F.shape[0]):
        if o[i] == 0:
            Archive_member_no += 1
            Archive_X_updated.append(Archive_temp_X[i])
            Archive_F_updated.append(Archive_temp_F[i])
    return Archive_X_updated, Archive_F_updated, Archive_member_no


def RankingProcess(Archive_F, obj_no):
    if len(Archive_F) == 1:
        my_min = [Archive_F[0][0], Archive_F[0][1], Archive_F[0][2]]
        my_max = [Archive_F[0][0], Archive_F[0][1], Archive_F[0][2]]
    else:
        my_min = [min(Archive_F[:, 0]), min(Archive_F[:, 1]), min(Archive_F[:, 2])]
        my_max = [max(Archive_F[:, 0]), max(Archive_F[:, 1]), max(Archive_F[:, 2])]
    r = [(my_max[0] - my_min[0]) / 10, (my_max[1] - my_min[1]) / 10, (my_max[2] - my_min[2]) / 10]
    ranks = numpy.zeros(len(Archive_F))
    for i in range(len(Archive_F)):
        ranks[i] = 0
        for j in range(len(Archive_F)):
            flag = 0
            for k in range(obj_no):
                if math.fabs(Archive_F[j][k] - Archive_F[i][k]) <= r[k]:
                    flag += 1
            if flag == obj_no:
                ranks[i] += 1
    return ranks


def RouletteWheelSelection(weights):
    accumulation = numpy.cumsum(weights)
    p = random.random()
    end = accumulation[-1]
    lst = [0]
    for i in range(weights.shape[0]):
        lst.append(1.0 * accumulation[i] / end)
    chosen_index = -1
    for i in range(accumulation.shape[0]):
        if lst[i] < p <= lst[i + 1]:
            chosen_index = i
            break
    return chosen_index if chosen_index >= 0 else 0


def handleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize):
    for i in range(len(Archive_F) - ArchiveMaxSize):
        index = RouletteWheelSelection(Archive_mem_ranks)
        Archive_X = numpy.vstack((Archive_X[0:index], Archive_X[index + 1:Archive_member_no]))
        Archive_F = numpy.vstack((Archive_F[0:index], Archive_F[index + 1:Archive_member_no]))
        Archive_mem_ranks = numpy.hstack((Archive_mem_ranks[0:index], Archive_mem_ranks[index + 1:Archive_member_no]))
        Archive_member_no -= 1
    return 1.0 * Archive_X, 1.0 * Archive_F, 1.0 * Archive_mem_ranks, Archive_member_no


def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    return numpy.divide(u, zz)


def transform(population1, chromosome_length):
    population_fit = population1 * 1
    for j in range(population1.shape[0]):
        a = 1 / (1 + numpy.exp(-population1[j]))
        population_fit[j] = 1 if a >= 0.5 else 0
    while (sum(population_fit) < 2):
        b = 1 - 2 * numpy.random.rand(chromosome_length)
        population_fit = b * 1
        for j in range(population_fit.shape[0]):
            a = 1 / (1 + numpy.exp(-population_fit[j]))
            population_fit[j] = 1 if a >= 0.5 else 0
    return population_fit


def delete_duplicates(lst, dim, ignored_index):
    for i in range(lst.shape[0]):
        for j in range(lst.shape[0]):
            if j in ignored_index: continue
            if all(lst[i] == lst[j]) and i != j:
                lst[j] = numpy.random.randint(0, 2, (1, dim))
    return lst


def OperatorGA(parent):
    Parent1 = parent[0:math.floor(parent.shape[0] / 2), :]
    Parent2 = parent[math.floor(parent.shape[0] / 2):math.floor(parent.shape[0] / 2) * 2, :]
    N, D = Parent1.shape[0], Parent1.shape[1]
    k = numpy.array([[random.random() for _ in range(D)] for _ in range(N)])
    for i in range(N):
        for j in range(D):
            k[i, j] = 1 if k[i, j] < 0.5 else 0
    Offspring1, Offspring2 = Parent1.copy(), Parent2.copy()
    for i in range(N):
        for j in range(D):
            if k[i, j] == 1:
                Offspring1[i, j] = Parent2[i, j]
                Offspring2[i, j] = Parent1[i, j]
    Offspring = numpy.vstack((Offspring1, Offspring2))
    proM = 3
    p = proM / D
    Site = numpy.array([[random.random() for _ in range(D)] for _ in range(2 * N)])
    for i in range(2 * N):
        for j in range(D):
            if Site[i, j] < p:
                Offspring[i, j] = 1 - Offspring[i, j]
    return Offspring

def run(datasetName):
    """Run the complete S2MHHO on one of the six released datasets."""
    use_s1 = True
    use_s2 = True
    use_s3 = True
    global TRAIN_INDEX, TEST_INDEX
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

    output_dir = os.path.join(project_dir, 'Results', 'S2MHHO')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, datasetName + '_' + str(time.time()) + '.xlsx')

    workbook = xlsxwriter.Workbook(path)
    worksheet1 = workbook.add_worksheet("generation_30")
    worksheet2 = workbook.add_worksheet("generation_60")
    worksheet3 = workbook.add_worksheet("generation_100")
    ColName = ["classification_error", "feature_ratio", "feature_cost"]
    for i in range(len(ColName)):
        worksheet1.write(0, i, ColName[i])
        worksheet2.write(0, i, ColName[i])
        worksheet3.write(0, i, ColName[i])

    x = dataset.iloc[:, 0:-1]
    x = pd.DataFrame(StandardScaler().fit_transform(x))
    y = dataset.iloc[:, -1]
    z = dataset1.iloc[:]

    population_size = 8
    T_val = 100
    dim = x.shape[1]
    obj_no = 3
    lb, ub = -6, 6

    Archive_F1, Archive_F2 = [], []

    if use_s3:
        ESNum_base = math.ceil(population_size * 0.75)
    else:
        ESNum_base = 0
    ES_X = numpy.zeros((max(ESNum_base, 1), dim))
    ES_Y = numpy.ones((max(ESNum_base, 1), obj_no)) * float("inf")

    ArchiveMaxSize = 8
    Archive_member_no = 0
    Archive_X = numpy.zeros((ArchiveMaxSize, dim))
    Archive_F = numpy.ones((ArchiveMaxSize, obj_no)) * float("inf")

    Rabbit_Location = numpy.zeros(dim)
    Rabbit_Energy = float("inf") * numpy.ones(3)

    if use_s1:
        pops = circle_map(0.35, population_size * dim)
        k = 0
        population = numpy.zeros((population_size, dim))
        for i in range(population_size):
            for j in range(dim):
                population[i, j] = pops[k]
                k += 1
        for i in range(population_size):
            population[i] = sigmoid_binary_mapping(population[i])
    else:
        population = numpy.random.randint(0, 2, (population_size, dim))

    t = 0
    ignored_index = []
    particles_F = []

    while t < T_val:
        start_iter = time.time()
        population = delete_duplicates(population, dim, ignored_index)

        particles_F = get_ObjectFunction(population, dim, x, y, z, ignored_index, particles_F)
        particles_F = numpy.array(particles_F)
        ignored_index = []

        Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, population, particles_F)
        if Archive_member_no > ArchiveMaxSize:
            Archive_mem_ranks = RankingProcess(numpy.array(Archive_F), obj_no)
            Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no = handleFullArchive(
                numpy.array(Archive_X), numpy.array(Archive_F), Archive_member_no, Archive_mem_ranks, ArchiveMaxSize)
        else:
            Archive_mem_ranks = RankingProcess(numpy.array(Archive_F), obj_no)

        OffSize = 0
        popindex = numpy.lexsort((particles_F[:, 1], particles_F[:, 2], particles_F[:, 0]))

        if use_s3:
            Temp_X = numpy.array(Archive_X)
            Temp_F = numpy.array(Archive_F)
            idx_arch = numpy.lexsort((Temp_F[:, 1], Temp_F[:, 2], Temp_F[:, 0]))
            ESNum = min(ESNum_base, Archive_member_no)
            if ESNum >= 2:
                for i in range(ESNum):
                    ES_X[i, :] = Temp_X[idx_arch[i], :]
                    ES_Y[i, :] = Temp_F[idx_arch[i], :]
                Offspring = OperatorGA(ES_X)
                OffSize = Offspring.shape[0]
                kk = population_size
                for i in range(0, OffSize):
                    population[popindex[kk - 1]] = Offspring[i]
                    kk -= 1

        index = RouletteWheelSelection(1.0 / (numpy.array(Archive_mem_ranks) + 1))
        Rabbit_Energy = Archive_F[index]
        Rabbit_Location = Archive_X[index]

        if t == 30:
            for q in range(len(Archive_F)):
                Archive_F1.append(Archive_F[q])
        if t == 60:
            for q in range(len(Archive_F)):
                Archive_F2.append(Archive_F[q])

        E1 = 2 * (1 - (t / T_val))
        jj = 0
        while jj < (population_size - OffSize):
            i = popindex[jj]
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0

            # -------- Exploration phase --------
            if abs(Escaping_Energy) >= 1:
                q = random.random()
                rand_Hawk_index = math.floor(population_size * random.random())
                X_rand = population[rand_Hawk_index, :]
                if q < 0.5:
                    population[i, :] = X_rand - random.random() * abs(X_rand - 2 * random.random() * population[i, :])
                else:
                    population[i, :] = (Rabbit_Location - population.mean(0)) - random.random() * ((ub - lb) * random.random() + lb)
                population[i] = transform(population[i], dim)

                if use_s2:
                    alpha = 0.72 - 0.62 * (t / T_val)
                    mut_rate = alpha / math.sqrt(dim)
                    for j in range(dim):
                        if random.random() < mut_rate:
                            population[i, j] = 1 - population[i, j]

            # -------- Exploitation phase --------
            else:
                r = random.random()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege
                    population[i, :] = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - population[i, :])
                    population[i] = transform(population[i], dim)

                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege
                    J = 2 * (1 - random.random())
                    population[i, :] = (Rabbit_Location - population[i, :]) - Escaping_Energy * abs(J * Rabbit_Location - population[i, :])
                    population[i] = transform(population[i], dim)

                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege + dives
                    ignored_index.append(i)
                    J = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(J * Rabbit_Location - population[i, :])
                    X1 = transform(X1, dim)
                    pX1 = single_get_ObjectFunction(X1, dim, x, y, z)
                    if dominates(pX1, particles_F[i]):
                        population[i, :] = X1.copy()
                        particles_F[i, :] = pX1.copy()
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(J * Rabbit_Location - population[i, :]) + numpy.multiply(numpy.random.randn(dim), Levy(dim))
                        X2 = transform(X2, dim)
                        pX2 = single_get_ObjectFunction(X2, dim, x, y, z)
                        if dominates(pX2, particles_F[i]):
                            population[i, :] = X2.copy()
                            particles_F[i, :] = pX2.copy()
                    jj += 1

                elif r < 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege + dives
                    ignored_index.append(i)
                    J = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(J * Rabbit_Location - population.mean(0))
                    X1 = transform(X1, dim)
                    pX1 = single_get_ObjectFunction(X1, dim, x, y, z)
                    if dominates(pX1, particles_F[i]):
                        population[i, :] = X1.copy()
                        particles_F[i, :] = pX1.copy()
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(J * Rabbit_Location - population.mean(0)) + numpy.multiply(numpy.random.randn(dim), Levy(dim))
                        X2 = transform(X2, dim)
                        pX2 = single_get_ObjectFunction(X2, dim, x, y, z)
                        if dominates(pX2, particles_F[i]):
                            population[i, :] = X2.copy()
                            particles_F[i, :] = pX2.copy()
                    jj += 1

                if use_s2:
                    alpha_light = (0.72 - 0.62 * (t / T_val)) * 0.12
                    mut_rate_light = alpha_light / math.sqrt(dim)
                    for j in range(dim):
                        if random.random() < mut_rate_light:
                            population[i, j] = 1 - population[i, j]

            jj += 1

        t += 1
        end_iter = time.time()
        if t % 20 == 0:
            remaining = end_iter + (T_val - t) * (end_iter - start_iter) if t < T_val else 0
            print(f"  iteration {t}, elapsed {end_iter - start_iter:.2f}s", end="")
            if t < T_val:
                print(f", estimated remaining {remaining - end_iter:.0f}s")
            else:
                print()

    Archive_F1 = list(set([tuple(t) for t in Archive_F1]))
    Archive_F1.sort(key=lambda x: x[0])
    Archive_F1 = numpy.array(Archive_F1)

    Archive_F2 = list(set([tuple(t) for t in Archive_F2]))
    Archive_F2.sort(key=lambda x: x[0])
    Archive_F2 = numpy.array(Archive_F2)

    Archive_X, Archive_F, Archive_member_no = updateArchive(Archive_X, Archive_F, population, particles_F)
    Archive_F = list(set([tuple(t) for t in Archive_F]))
    Archive_F.sort(key=lambda x: x[0])
    Archive_F = numpy.array(Archive_F)

    for i in range(len(Archive_F1)):
        worksheet1.write(i + 1, 0, Archive_F1[i][0])
        worksheet1.write(i + 1, 1, Archive_F1[i][1])
        worksheet1.write(i + 1, 2, Archive_F1[i][2])
    for i in range(len(Archive_F2)):
        worksheet2.write(i + 1, 0, Archive_F2[i][0])
        worksheet2.write(i + 1, 1, Archive_F2[i][1])
        worksheet2.write(i + 1, 2, Archive_F2[i][2])
    for i in range(len(Archive_F)):
        worksheet3.write(i + 1, 0, Archive_F[i][0])
        worksheet3.write(i + 1, 1, Archive_F[i][1])
        worksheet3.write(i + 1, 2, Archive_F[i][2])

    end1 = time.time()
    runtime = end1 - start1
    worksheet3.write(len(Archive_F) + 2, 0, runtime)
    workbook.close()
    print(f"  completed in {runtime:.1f}s, Pareto solutions={len(Archive_F)}")
    return Archive_F


if __name__ == "__main__":
    supported_datasets = ['CKD']
    datasets = sys.argv[1:] or supported_datasets
    unknown = [name for name in datasets if name not in supported_datasets]
    if unknown:
        raise SystemExit(
            'Unknown dataset(s): ' + ', '.join(unknown)
            + '. Choose from: ' + ', '.join(supported_datasets)
        )
    for dataset in datasets:
        print(f"Running S2MHHO on {dataset} ...")
        run(dataset)
