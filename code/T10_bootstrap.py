import numpy as np
from scipy.stats import norm

def ecdf_strict(x):
    x_sorted = np.sort(x)
    n = len(x)
    
    def F(t):
        return np.searchsorted(x_sorted, t, side='left') / n
    
    return F

sample = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
           4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
             5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
             6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 
             8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9]

F_theor = norm.cdf(sample, loc=4.77, scale=np.sqrt(6.28))
F = ecdf_strict(sample)
F_ecsp = np.concatenate((np.array([F(sample[j]) for j in range(100)]), [1]))
delta_1 = -1
for j in range(100):
    delta_b = max(abs(F_theor[j] - F_ecsp[j]), abs(F_theor[j] - F_ecsp[j+1]))
    delta_1 = max(delta_1, delta_b)

delta_1 = delta_1 * 10
print(delta_1)


def bootstrap(a: float, D: float, delta_abs: float, count: int) -> float:
    deltas = np.zeros(50000)
    for i in range(count):
        x_i = np.random.normal(loc=a, scale=np.sqrt(D), size=100)
        a_new = np.mean(x_i)
        D_new = np.mean((x_i - a_new)**2)
        x_i.sort()
        F_theor = norm.cdf(x_i, loc=a_new, scale=np.sqrt(D_new))
        F = ecdf_strict(x_i)
        F_ecsp = np.concatenate((np.array([F(x_i[j]) for j in range(100)]), [1]))
        delta = -1
        for j in range(100):
            delta_b = max(abs(F_theor[j] - F_ecsp[j]), abs(F_theor[j] - F_ecsp[j+1]))
            delta = max(delta, delta_b)
        deltas[i] = delta
    ans = 0
    deltas *= 10
    for j in range(count):
        ans = ans + 1 if deltas[j] >= delta_abs else ans
    return ans/count

print(bootstrap(4.77, 6.28, delta_1, 50000))