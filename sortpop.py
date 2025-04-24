## sort population
def SortPopulation(pop):
    # Sort Based on Crowding Distance
    CDSO = np.argsort([ind['CrowdingDistance'] for ind in pop])[::-1]
    pop = [pop[i] for i in CDSO]

    # Sort Based on Rank
    RSO = np.argsort([ind['Rank'] for ind in pop])
    pop = [pop[i] for i in RSO]

    # Update Fronts
    Ranks = [ind['Rank'] for ind in pop]
    MaxRank = max(Ranks)
    F = [[] for _ in range(MaxRank)]
    for r in range(MaxRank):
        F[r] = [i for i, rank in enumerate(Ranks) if rank == r + 1]  # Python index starts at 0, MATLAB at 1

    return pop, F