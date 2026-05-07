import numpy as np
from scipy.stats import t, f, norm
import matplotlib.pyplot as plt

def generation_vector() -> tuple[np.ndarray, float]:
    rng = np.random.default_rng()
    x = rng.uniform(-1, 1, 5)
    M_y = 2 + 3*x[0] - 2*x[1] + x[2] + x[3] - x[4]
    y1 = rng.normal(M_y, 1.5)
    return (np.array([1, x[0], x[1], x[2], x[3], x[4]]), y1)

def generation_sample(n: int) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.empty((n,6)), np.empty((n,))
    for i in range(n):
        X[i], Y[i] = generation_vector()
    return X, Y 
        
def is_multicollinearity(X: np.ndarray) -> list[bool]:
    X_bufer = X[:, 1:] 
    
    multicollinearity = [True] * X_bufer.shape[1]
    
    for i in range(X_bufer.shape[1]):
        if not multicollinearity[i]:
            continue
        
        Y = X_bufer[:, i]
        
        predictors = []
        for j in range(X_bufer.shape[1]):
            if j != i and multicollinearity[j]:
                predictors.append(X_bufer[:, j])
        
        X_reg = np.column_stack([np.ones(X_bufer.shape[0])] + predictors)
        
        beta, significance_beta, R2, significance_R2, p_values = coefficients_regression_and_determination(X_reg, Y)
        
        if R2 > 0.7:
            multicollinearity[i] = False
    
    return multicollinearity
 
def coefficients_regression_and_determination(X: np.ndarray, Y: np.ndarray, a: float = 0.05, b: float = 0.05) -> tuple[np.ndarray, list[bool], float, bool, list[tuple[float, int]]]: 
    F = X.T @ X
    F1 = np.linalg.inv(F)
    beta = F1 @ X.T @ Y
    Y_pred = X @ beta      
    residuals = Y - Y_pred          
    RSS = residuals @ residuals
    p_values = []
    for i in range(X.shape[1]):
        delta = beta[i] * np.sqrt(X.shape[0] - X.shape[1])/np.sqrt(RSS * F1[i][i])
        p_value = 2 * (1 - t.cdf(abs(delta), X.shape[0] - X.shape[1]))
        p_values.append((p_value,i))
    p_values.sort()
    significance_beta = [1 for _ in range(X.shape[1])]
    for i in range(X.shape[1]):
        significance_beta[p_values[i][1]] = (p_values[i][0] < a/(X.shape[1] - i))

    Y_mean = np.mean(Y)
    TSS = np.sum((Y - Y_mean) ** 2)
    R2 = 1 - RSS/TSS
    delta = ((TSS - RSS) / (X.shape[1] - 1)) / (RSS / (X.shape[0] - X.shape[1]))
    p_value = 1 - f.cdf(delta, X.shape[1] - 1, X.shape[0] - X.shape[1])
    significance_R2 = p_value < b
    return beta, significance_beta, R2, significance_R2, p_values

def value_and_confidence_interval(X: np.ndarray, Y: np.ndarray, beta: np.ndarray, b: float = 0.95) -> tuple[float, tuple[float, float]]:
    x0 = np.array([1, 0, 0, 0, 0, 0])
    
    y0_pred = x0 @ beta
    
    n, p = X.shape
    residuals = Y - X @ beta
    RSS = residuals @ residuals
    
    F = X.T @ X
    F1 = np.linalg.inv(F)
    
    delta = t.ppf((1 + b) / 2, n - p) * np.sqrt(RSS * (1 + x0 @ F1 @ x0) / (n - p))
    
    ci_lower = y0_pred - delta
    ci_upper = y0_pred + delta
    
    return y0_pred, (ci_lower, ci_upper)

def independence_errors(X: np.ndarray, Y: np.ndarray, beta: np.ndarray, a: float = 0.05) -> bool:
    residuals = Y - X @ beta
    n = len(residuals)
    
    I = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if residuals[i] < residuals[j]:
                I += 1
    
    M = n * (n - 1) / 4
    sigma = np.sqrt(n**3 / 36)
    
    delta = (I - M) / sigma
    
    p_value = 2 * (1 - norm.cdf(abs(delta)))
    
    return p_value > a

def normal_errors(X: np.ndarray, Y: np.ndarray, beta: np.ndarray, N: int = 50000, a: float = 0.05) -> bool:
    residuals = Y - X @ beta
    n = len(residuals)
    
    residuals_sorted = np.sort(residuals)
    sigma_hat = np.std(residuals, ddof=1)
    
    F_emp = np.arange(1, n + 1) / n
    
    F_theor = norm.cdf(residuals_sorted, loc=0, scale=sigma_hat)
    
    delta_abs = np.sqrt(n) * np.max(np.abs(F_emp - F_theor))
    
    deltas = np.zeros(N)
    
    for i in range(N):
        e_star = np.random.normal(0, sigma_hat, n)
        e_star_sorted = np.sort(e_star)
        sigma_star = np.std(e_star, ddof=1)
        
        F_emp_star = np.arange(1, n + 1) / n
        F_theor_star = norm.cdf(e_star_sorted, loc=0, scale=sigma_star)
        
        delta_star = np.sqrt(n) * np.max(np.abs(F_emp_star - F_theor_star))
        deltas[i] = delta_star
    
    m = np.sum(deltas >= delta_abs)
    p_value = m / N
    
    return p_value > a

def emissions(X: np.ndarray, Y: np.ndarray, beta: np.ndarray):
    residuals = Y - X @ beta
    
    n = len(residuals)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.boxplot(residuals)
    plt.title('Boxplot остатков')
    plt.ylabel('Значения остатков')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(1, n + 1), residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.axhline(y=2 * np.std(residuals), color='orange', linestyle='--', linewidth=1, label='2σ')
    plt.axhline(y=-2 * np.std(residuals), color='orange', linestyle='--', linewidth=1)
    plt.title('График остатков e(i) от номера наблюдения')
    plt.xlabel('Номер наблюдения i')
    plt.ylabel('Остатки e(i)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def cross_validation(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]
    CVSS = 0

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        Y_train = np.delete(Y, i, axis=0)
        X_test = X[i].reshape(1, -1)
        Y_test = Y[i]
        
        beta_train, _, _, _, _ = coefficients_regression_and_determination(X_train, Y_train)
        
        Y_pred = (X_test @ beta_train).item()
        
        CVSS += (Y_test - Y_pred) ** 2

    Y_mean = np.mean(Y)
    TSS = np.sum((Y - Y_mean) ** 2)

    return (TSS - CVSS) / TSS

def adequacy_of_regression(X: np.ndarray, Y: np.ndarray, beta: np.ndarray, x_fix: np.ndarray, N: int = 5, a: float = 0.05) -> bool:
    
    true_mean = 2 + 3*x_fix[0] - 2*x_fix[1] + x_fix[2] + x_fix[3] - x_fix[4]
    
    repeated_y = []
    for _ in range(N):
        rng = np.random.default_rng()
        repeated_y.append(rng.normal(true_mean, 1.5))
    
    y_mean = np.mean(repeated_y)
    l = len(repeated_y)
    sigma1_2 = np.sum((repeated_y - y_mean) ** 2) / (l - 1)
    
    n, p = X.shape
    residuals = Y - X @ beta
    RSS = residuals @ residuals
    sigma2_2 = RSS / (n - p)
    
    F_stat = sigma2_2 / sigma1_2
    
    p_value = 1 - f.cdf(F_stat, l - 1, n - p)
    
    return p_value > a

def delete_min_and_replay_bc(X: np.ndarray, Y: np.ndarray, p_values: list[tuple[float, int]], a: float = 0.05):
    min_p_idx = p_values[-1][1]
    min_p_val = p_values[-1][0]
    
    X_new = np.delete(X, min_p_idx, axis=1)
    
    beta_new, significance_beta_new, R2_new, significance_R2_new, p_values_new = coefficients_regression_and_determination(X_new, Y)
    
    residuals_old = Y - X @ beta
    residuals_new = Y - X_new @ beta_new
    RSS_old = residuals_old @ residuals_old
    RSS_new = residuals_new @ residuals_new
    
    n, p_old = X.shape
    p_new = X_new.shape[1]
    
    F_stat = ((RSS_new - RSS_old) / (p_old - p_new)) / (RSS_old / (n - p_old))
    p_value_F = 1 - f.cdf(F_stat, p_old - p_new, n - p_old)
    
    return beta_new, significance_beta_new, R2_new, significance_R2_new, p_values_new, min_p_idx, min_p_val, p_value_F > a

def comparison_bootstrap(X: np.ndarray, Y: np.ndarray, p_values: list[tuple[float, int]], N: int = 10000, a: float = 0.05) -> bool:
    min_p_idx = p_values[-1][1]
    X_new = np.delete(X, min_p_idx, axis=1)
    
    beta_old = beta
    beta_new, _, _, _, _ = coefficients_regression_and_determination(X_new, Y)
    
    residuals_old = Y - X @ beta_old
    residuals_new = Y - X_new @ beta_new
    RSS_old = residuals_old @ residuals_old
    RSS_new = residuals_new @ residuals_new
    
    h_abs = RSS_new - RSS_old
    
    n = X.shape[0]
    deltas = []
    
    for _ in range(N):
        idx = np.random.choice(n, n, replace=True)
        
        X_boot = X[idx]
        Y_boot = Y[idx]
        X_new_boot = X_new[idx]
        
        beta_old_boot, _, _, _, _ = coefficients_regression_and_determination(X_boot, Y_boot)
        beta_new_boot, _, _, _, _ = coefficients_regression_and_determination(X_new_boot, Y_boot)
        
        residuals_old_boot = Y_boot - X_boot @ beta_old_boot
        residuals_new_boot = Y_boot - X_new_boot @ beta_new_boot
        RSS_old_boot = residuals_old_boot @ residuals_old_boot
        RSS_new_boot = residuals_new_boot @ residuals_new_boot
        
        h_boot = RSS_new_boot - RSS_old_boot
        deltas.append(h_boot - h_abs)
    
    deltas.sort()
    
    alpha = 1 - a
    lower_idx = int(a * N) - 1
    upper_idx = N - 1
    
    ci_lower = h_abs - deltas[upper_idx]
    ci_upper = h_abs - deltas[lower_idx]
    
    return ci_lower <= 0 <= ci_upper

X, Y = generation_sample(50)

multicollinearity_result = is_multicollinearity(X)
print("a) Мультиколлинеарность (True - оставляем, False - отбрасываем):\n", multicollinearity_result)

beta, significance_beta, R2, significance_R2, p_values = coefficients_regression_and_determination(X, Y)
print("\nb) Коэффициенты регрессии:")
for i, coef in enumerate(beta):
    print(f"   β{i}: {coef:.4f}, значим: {significance_beta[i]}, p_value = {p_values[i][0]}")
print(f"   R² = {R2:.4f}, значимость R²: {significance_R2}")

y0_pred, ci = value_and_confidence_interval(X, Y, beta)
print(f"\nd) Предсказание при x_k=0: {y0_pred:.4f}")
print(f"   95% доверительный интервал: ({ci[0]:.4f}, {ci[1]:.4f})")

indep = independence_errors(X, Y, beta)
print(f"\ne) Ошибки независимы: {indep}")

normality = normal_errors(X, Y, beta)
print(f"\nf) Ошибки нормально распределены: {normality}")

print("\ng) Графики для анализа выбросов:")
emissions(X, Y, beta)


Rcv = cross_validation(X, Y)
print("\nh) Кросс-валидация регрессии:")
print(f"Rcv = {Rcv:.4f}")

x_fix = np.array([1, 2, 1, 0, 3])
result = adequacy_of_regression(X, Y, beta, x_fix)
print("\ni) Адекватность регрессии:")
print(f"Модель адекватна: {result}")

beta_new, significance_beta_new, R2_new, significance_R2_new, p_values_new, min_p_idx, min_p_val, non_significant = delete_min_and_replay_bc(X, Y, p_values)

print(f"\nj) Удаляем переменную ξ{min_p_idx} (p-value = {min_p_val:.6f})")
print(f"   H0: удаленная переменная не значима = {non_significant}")
print("\nРегрессия после удаления:")
for i, coef in enumerate(beta_new):
    print(f"   β{i}: {coef:.4f}, значим: {significance_beta_new[i]}, p-value = {p_values_new[i][0]:.6f}")
print(f"   R² = {R2_new:.4f}, значимость R²: {significance_R2_new}")

result_bootstrap = comparison_bootstrap(X, Y, p_values)
print("\nk) Бутстреп сравнение регрессий:")
print(f"   Модели не различаются значимо (ноль внутри ДИ): {result_bootstrap}")
