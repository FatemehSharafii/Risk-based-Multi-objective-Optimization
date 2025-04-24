import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GAfunctions import GeneticAlgorithmFuncs



# Problem Definition
nVar = 6
df= pd.read_csv('steel2.csv')
allowed_values={
    'fy': [235,355],
    'L': [6, 12, 18],
    'D/L': [1,1/2,1/3,1/4],
    'aq': list(np.arange(0.1, 0.4 + 0.05, 0.05)),
    'ag': list(np.arange(0.6, 1 + 0.2, 0.2)), 
    'p': df['Plastic section modulus'].tolist()
}

# Call Functions for GA
GAmethods=GeneticAlgorithmFuncs()

# Number of Objective Functions
nObj = len(GAmethods.TotalCost({'fy': np.random.choice(allowed_values['fy']), 'L': np.random.choice(allowed_values['L']), 
                      'D/L': np.random.choice(allowed_values['D/L']), 'aq': np.random.choice(allowed_values['aq']),
                       'ag': np.random.choice(allowed_values['ag']), 'p': np.random.choice(allowed_values['p'])}))

# NSGA-II Parameters
MaxIt = 20
nPop = 8
pCrossover = 0.7
nCrossover = 2 * round(pCrossover * nPop / 2)
pMutation = 0.4
nMutation = round(pMutation * nPop)
mu = 0.02
# sigma = 0.1 * (VarMax - VarMin)
sigma_initial= 0.5


# Initialization
empty_individual = {
    'Position': [],
    'Cost': [],
    'Rank': [],
    'DominationSet': [],
    'DominatedCount': [],
    'CrowdingDistance': []
}

pop = [empty_individual.copy() for _ in range(nPop)]

for i in range(nPop):
    pop[i]['Position'] = {'fy': np.random.choice(allowed_values['fy']), 'L': np.random.choice(allowed_values['L']), 
                      'D/L': np.random.choice(allowed_values['D/L']), 'aq': np.random.choice(allowed_values['aq']),
                       'ag': np.random.choice(allowed_values['ag']), 'p': np.random.choice(allowed_values['p'])}
    pop[i]['Cost'] = GAmethods.TotalCost(pop[i]['Position'])


# Non-Dominated Sorting
pop, F = GAmethods.NonDominatedSorting(pop)


# Calculate Crowding Distance
pop = GAmethods.CalcCrowdingDistance(pop, F)

# Sort Population
pop, F = GAmethods.SortPopulation(pop)

################################## NSGA-II Main Loop ######################################
for it in range(MaxIt):
    # Crossover
    popc = []
    for k in range(nCrossover // 2):
        i1 = np.random.randint(0, nPop)
        p1 = pop[i1]
        i2 = np.random.randint(0, nPop)
        p2 = pop[i2]
        # child1_pos, child2_pos = GAmethods.Crossover(p1['Position'], p2['Position'])
        child1_pos, child2_pos = GAmethods.Crossover(p1['Position'], p2['Position'])
        child1 = empty_individual.copy()
        child2 = empty_individual.copy()
        child1['Position'] = child1_pos
        child2['Position'] = child2_pos
        child1['Cost'] = GAmethods.TotalCost(child1['Position'])
        child2['Cost'] = GAmethods.TotalCost(child2['Position'])
        popc.append(child1)
        popc.append(child2)

    # Mutation
    # sigma= sigma_initial * (1 - it/MaxIt)
    popm = []
    for k in range(nMutation):
        i = np.random.randint(0, nPop)
        p = pop[i]
        mutant = empty_individual.copy()
        mutant['Position'] = GAmethods.Mutate(p['Position'], mu, allowed_values)
        mutant['Cost'] = GAmethods.TotalCost(mutant['Position'])
        popm.append(mutant)

    # Merge
    pop = pop + popc + popm

    # for ind in pop:
    #     ind['DominationSet'] = []
    #     ind['DominatedCount'] = 0
    #     ind['Rank'] = 0
    #     ind['CrowdingDistance'] = 0.0  # or float('inf') depending on your logic
    # Non-Dominated Sorting
    pop, F = GAmethods.NonDominatedSorting(pop)

    # Calculate Crowding Distance
    pop = GAmethods.CalcCrowdingDistance(pop, F)

    # Sort Population
    pop = GAmethods.SortPopulation(pop)

    # Truncate
    pop = pop[:nPop]

    # Non-Dominated Sorting
    pop, F = GAmethods.NonDominatedSorting(pop)

    # Calculate Crowding Distance
    pop = GAmethods.CalcCrowdingDistance(pop, F)

    # Sort Population
    pop = GAmethods.SortPopulation(pop)

    # Store F1
    F1 = [pop[i] for i in F[0]]

    # Show Iteration Information
    print(f'Iteration {it + 1}: Number of F1 Members = {len(F1)}')

    # Plot F1 Costs
    plt.figure(1)
    GAmethods.PlotCosts(F1)
