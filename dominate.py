## Dominates
import numpy as np

def Dominates(x, y):
    # If x or y are dicts (as a replacement for MATLAB structs), extract 'Cost'
    if isinstance(x, dict) and 'Cost' in x:
        x = x['Cost']
    if isinstance(y, dict) and 'Cost' in y:
        y = y['Cost']
    
    x = np.array(x)
    y = np.array(y)
    
    return np.all(x <= y) and np.any(x < y)
