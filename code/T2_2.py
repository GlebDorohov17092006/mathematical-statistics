import numpy as np
from scipy.stats import t, f

data = {
    1: np.array([83, 85]),
    2: np.array([84, 85, 85, 86, 86, 87]),
    3: np.array([86, 87, 87, 87, 88, 88, 88, 88, 88, 89, 90]),
    4: np.array([89, 90, 90, 91]),
    5: np.array([90, 92])
}

X = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],

    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],

    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],

    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],

    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],

])

Y = np.array(
    [83, 85, 84, 85, 85, 86, 86, 87, 86, 87, 87, 87, 88, 88, 88, 88, 88, 89, 90, 89, 90, 90, 91, 90, 92]
).T

def influence_age(X: np.ndarray, Y: np.ndarray, a: float = 0.05, b: float = 0.05) -> tuple[np.ndarray, list[bool], float, bool]:
    n, p = X.shape
    
    F = X.T @ X
    F1 = np.linalg.inv(F)
    beta = F1 @ X.T @ Y
    
    Y_pred = X @ beta
    residuals = Y - Y_pred
    RSS = residuals @ residuals
    
    p_values = []
    for i in range(p):
        delta = beta[i] * np.sqrt(n - p) / np.sqrt(RSS * F1[i][i])
        p_value = 2 * (1 - t.cdf(abs(delta), n - p))
        p_values.append((p_value, i))
    
    p_values.sort()
    significance_beta = [False] * p
    for i in range(p):
        significance_beta[p_values[i][1]] = (p_values[i][0] < a)
    
    Y_mean = np.mean(Y)
    TSS = np.sum((Y - Y_mean) ** 2)
    R2 = 1 - RSS / TSS
    
    F_stat = ((TSS - RSS) / (p - 1)) / (RSS / (n - p))
    p_value_R2 = 1 - f.cdf(F_stat, p - 1, n - p)
    significance_R2 = p_value_R2 < a
    
    return beta, significance_beta, R2, significance_R2, p_values, p_value_R2

def pairwise_comparison_averages(X: np.ndarray, Y: np.ndarray, beta: np.ndarray, a: float = 0.05) -> tuple[list[list[bool]], list[list[float]]]:
    n, p = X.shape
    residuals = Y - X @ beta
    RSS = residuals @ residuals
    F1 = np.linalg.inv(X.T @ X)
    
    result = [[False] * p for _ in range(p)]
    p_value_matrix = [[0.0] * p for _ in range(p)]
    
    pairs = []
    for i in range(p):
        for j in range(i + 1, p):
            delta = (beta[i] - beta[j]) / np.sqrt(RSS * (F1[i][i] + F1[j][j] - 2 * F1[i][j]) / (n - p))
            p_val = 2 * (1 - t.cdf(abs(delta), n - p))
            pairs.append((p_val, i, j))
    
    pairs.sort(key=lambda x: x[0])
    
    m = len(pairs)
    for idx, (p_val, i, j) in enumerate(pairs):
        if p_val < a / (m - idx):
            result[i][j] = True
            result[j][i] = True
            p_value_matrix[i][j] = p_val
            p_value_matrix[j][i] = p_val
        else:
            p_value_matrix[i][j] = p_val
            p_value_matrix[j][i] = p_val
    
    return result, p_value_matrix


beta, significance_beta, R2, significance_R2, p_values, p_value_R2 = influence_age(X, Y)

p_values_dict = {idx: p_val for p_val, idx in p_values}

print("а)Результаты регрессионного анализа влияния возраста на содержание Ig A:")
for i, coef in enumerate(beta):
    p_val = p_values_dict[i]
    if i == 0:
        print(f"   β{i} (группа 1): {coef}, значим: {significance_beta[i]}, p_value = {p_val}")
    else:
        print(f"   β{i} (группа {i+1}): {coef}, значим: {significance_beta[i]}, p_value = {p_val}")
print(f"   R² = {R2}, значимость R²: {significance_R2}, p_value = {p_value_R2}")
print("\nb) Попарное сравнение средних:")
comparison, p_value_matrix = pairwise_comparison_averages(X, Y, beta)

groups = ["1", "2", "3", "4", "5"]
print("      ", end="")
for g in groups:
    print(f"{g:>6}", end="")
print()
for i in range(len(groups)):
    print(f"  {groups[i]}   ", end="")
    for j in range(len(groups)):
        if i == j:
            print(f"{' - ':>6}", end="")
        else:
            print(f"{str(comparison[i][j]):>6}", end="")
    print()

print("\nМатрица p-value:")
print("      ", end="")
for g in groups:
    print(f"{g:>10}", end="")
print()
for i in range(len(groups)):
    print(f"  {groups[i]}   ", end="")
    for j in range(len(groups)):
        if i == j:
            print(f"{' - ':>10}", end="")
        else:
            print(f"{p_value_matrix[i][j]:>10.6f}", end="")
    print()