import pandas as pd
import numpy as np
from scipy.stats import anderson_ksamp

chat_id = 396317433 

def solution(x: np.array, y: np.array) -> bool:
    pval = anderson_ksamp([x, y]).pvalue
    static, critical, _ = anderson_ksamp([x, y])
    return static > critical[4]
