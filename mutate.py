import numpy as np

def Mutate(x, mu, sigma):
    x = np.array(x, dtype=float)
    nVar = x.size

    nMu = int(np.ceil(mu * nVar))
    
    # Randomly select nMu unique indices
    j = np.random.choice(nVar, nMu, replace=False)

    y = x.copy()
    y[j] = x[j] + sigma * np.random.randn(nMu)

    return y