## Non Dominated sorting
def NonDominatedSorting(pop):
    nPop = len(pop)
    
    # Initialize domination-related attributes
    for i in range(nPop):
        pop[i]['DominationSet'] = []
        pop[i]['DominatedCount'] = 0

    F = [[]]

    # Compare each pair of individuals
    for i in range(nPop):
        for j in range(i + 1, nPop):
            p = pop[i]
            q = pop[j]

            if Dominates(p, q):
                p['DominationSet'].append(j)
                q['DominatedCount'] += 1
            elif Dominates(q, p):
                q['DominationSet'].append(i)
                p['DominatedCount'] += 1

            pop[i] = p
            pop[j] = q

        if pop[i]['DominatedCount'] == 0:
            pop[i]['Rank'] = 1
            F[0].append(i)

    k = 0

    while True:
        Q = []

        for i in F[k]:
            p = pop[i]

            for j in p['DominationSet']:
                q = pop[j]
                q['DominatedCount'] -= 1

                if q['DominatedCount'] == 0:
                    q['Rank'] = k + 2
                    Q.append(j)

                pop[j] = q

        if not Q:
            break

        F.append(Q)
        k += 1

    return pop, F