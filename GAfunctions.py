import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FORM1 import ReliabilityAnalysis

class GeneticAlgorithmFuncs:
    def __init__(self):
        pass


    def Crossover(self, x1, x2):
        # Arithmetic crossover 
        # x1_vals = np.array(list(x1.values()))
        # x2_vals = np.array(list(x2.values()))
        
        # alpha = np.random.rand(*x1_vals.shape)
        
        # y1_vals = alpha * x1_vals + (1 - alpha) * x2_vals
        # y2_vals = alpha * x2_vals + (1 - alpha) * x1_vals

        #     # Convert back to dicts (preserving the same keys)
        # keys = list(x1.keys())
        # y1 = dict(zip(keys, y1_vals))
        # y2 = dict(zip(keys, y2_vals))
        
        # return y1, y2

        # Random picking crossover for discrete variables
        y1 = {}
        y2 = {}
        for key in x1:
            if np.random.rand() < 0.5:
                y1[key] = x1[key]
                y2[key] = x2[key]
            else:
                y1[key] = x2[key]
                y2[key] = x1[key]
        return y1, y2



    def Mutate(self, x, mu, allowed_values):
        # x = np.array(x, dtype=float)
        # nVar = x.size

        # nMu = int(np.ceil(mu * nVar))
        
        # # Randomly select nMu unique indices
        # j = np.random.choice(nVar, nMu, replace=False)

        # y = x.copy()
        # y[j] = x[j] + sigma * np.random.randn(nMu)

        # return y

        # Discrete Mutation
        import copy
        y = copy.deepcopy(x)
        
        keys = list(x.keys())
        nVar = len(keys)
        nMu = int(np.ceil(mu * nVar))
        
        # Choose random variable keys to mutate
        mutate_keys = np.random.choice(keys, nMu, replace=False)
        
        for key in mutate_keys:
            current_val = x[key]
            options = [v for v in allowed_values[key] if v != current_val]
            if options:
                y[key] = np.random.choice(options)  # mutate to a different valid value
        
        return y



    def SortPopulation(self, pop):
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




    def CalcCrowdingDistance(self, pop, F):
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




    def Dominates(self, x, y):
        # If x or y are dicts (as a replacement for MATLAB structs), extract 'Cost'
        if isinstance(x, dict) and 'Cost' in x:
            x = x['Cost']
        if isinstance(y, dict) and 'Cost' in y:
            y = y['Cost']
        
        x = np.array(x)
        y = np.array(y)
        
        return np.all(x <= y) and np.any(x < y)



    def NonDominatedSorting(self, pop):
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

                if self.Dominates(p, q):
                    p['DominationSet'].append(j)
                    q['DominatedCount'] += 1
                elif self.Dominates(q, p):
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
    


    # Cost Function
    def TotalCost(self, model):

        def CreateModel(model):

            # Calculate D
            D= model['L']* model['D/L']

            # density of RC slab
            p_rcs=2500        #kg/m3

            #Calculate h_rcs (slab depth)
            # h_min = max(120, D*1000 / 30)   #mm unit
            h_min = D*1000 / 30   #mm unit
            h_max = D*1000 / 20             #mm unit
            h_min_rounded = int(np.ceil(h_min / 10) * 10)
            h_max_rounded = int(np.floor(h_max / 10) * 10)
            h_choices = np.arange(h_min_rounded, h_max_rounded + 10, 10)
            h = np.random.choice(h_choices)

            # Calculate g (permanent load)
            qk=2.5
            g_k=qk*((1/model['aq'])-1)
            g_sw= g_k* model['ag']
            g_nsl=g_k-g_sw


            # Cross_section Area
            df= pd.read_csv('steel2.csv')
            print(model['p'])
            A= float(df.loc[df['Plastic section modulus'] == model['p'], 'Area'].values[0])
            mass= 7850 * model['L'] * A
            model_des={'fy': model['fy'], 'L': model['L'], 'D': D, 'q': qk, 'gnsl': g_nsl, 'gsw': g_sw, 'p': model['p'], 'mass':mass}

            object_reliability= ReliabilityAnalysis()
            pf=object_reliability.HLRF_Algorithm(20, model_des)
            model_des['pf']=pf

            return model_des




        model_des=CreateModel(model)
        w=1/50
        i=0.03
        Demolition_cost= 0

        SVSL= 2.775E06
        CC=30
        Acol= 4 * model_des['D'] * model_des['L']
        Ncol= model_des['pf'] * (0.26 * Acol**0.89)/ (CC**0.7)

        Cc= model_des['mass'] * (1.06 + 0.1 * 1.13)
        Co= (Cc + Demolition_cost) * w * i
        Cf= SVSL * Ncol * Acol * CC



        # Penalty: pf constraint
        pf_max = 1e-05
        if model_des['pf'] >= pf_max:
            penalty = 1e8 * (model_des['pf'] - pf_max)  
            Cf += penalty


        Cs= Cc + Co
        z=[Cs, Cf]
        return z





    def PlotCosts(self, pop):
        Costs = np.array([ind['Cost'] for ind in pop]).T  # Convert the list of Costs into a numpy array and transpose
        
        plt.scatter(Costs[:, 0], Costs[:, 1], color='r', marker='*', s=80)  # 's' is the marker size
        plt.xlabel('1st Objective')
        plt.ylabel('2nd Objective')
        plt.grid(True)
        plt.show()