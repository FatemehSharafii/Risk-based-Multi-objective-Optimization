## Calc crowding distance
import numpy as np

def CalcCrowdingDistance(pop, F):
    nF = len(F)
    
    for k in range(nF):
        front = F[k]
        costs = np.array([pop[i]['Cost'] for i in front]).T  # shape: (nObj, n)
        
        nObj, n = costs.shape
        d = np.zeros((n, nObj))
        
        for j in range(nObj):
            cj = costs[j]
            so = np.argsort(cj)
            
            d[so[0], j] = np.inf
            d[so[-1], j] = np.inf
            
            if np.abs(cj[so[0]] - cj[so[-1]]) < 1e-10:
                continue  # avoid division by zero
            
            for i in range(1, n - 1):
                d[so[i], j] = abs(cj[so[i + 1]] - cj[so[i - 1]]) / abs(cj[so[0]] - cj[so[-1]])

        for i in range(n):
            pop[front[i]]['CrowdingDistance'] = np.sum(d[i])
    
    return pop