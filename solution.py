import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 396317433 

def solution(x: np.array, y: np.array) -> bool:
    pval = MMD(compute_kernel="rbf", gamma=1).test(x, y)[1]
    return pval < 0.01
