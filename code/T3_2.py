import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 0], [0, 1], [1, 1]])
Y = np.array([1, 5, 2])

class Regression:
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def regression(self) -> np.ndarray:
        F = self.X.T @ self.X
        F1 = np.linalg.inv(F)
        beta = F1 @ self.X.T @ self.Y
        return beta

    def ridge(self, lambdas: np.ndarray = np.linspace(0, 10, 100)) -> tuple[float, np.ndarray, np.ndarray]:
        n = self.X.shape[0]
        cvss_values = []
        
        for lam in lambdas:
            cvss = 0
            for i in range(n):
                X_train = np.delete(self.X, i, axis=0)
                Y_train = np.delete(self.Y, i, axis=0)
                X_test = self.X[i].reshape(1, -1)
                Y_test = self.Y[i]
                
                beta_ridge = np.linalg.inv(X_train.T @ X_train + lam * np.eye(X_train.shape[1])) @ X_train.T @ Y_train
                
                Y_pred = (X_test @ beta_ridge).item()
                cvss += (Y_test - Y_pred) ** 2
            cvss_values.append(cvss)
        
        best_idx = np.argmin(cvss_values)
        best_lambda = lambdas[best_idx]
        
        beta_best = np.linalg.inv(self.X.T @ self.X + best_lambda * np.eye(self.X.shape[1])) @ self.X.T @ self.Y
        
        plt.figure(figsize=(10, 6))
        plt.plot(lambdas, cvss_values, 'b-', linewidth=2)
        plt.scatter(best_lambda, cvss_values[best_idx], color='red', s=100, zorder=5)
        plt.xlabel('Lambda')
        plt.ylabel('CVSS')
        plt.title('График CVSS(lambda) для Ridge регрессии')
        plt.grid(True)
        plt.show()
        
        return best_lambda, beta_best, cvss_values

    def lasso(self, lambdas: np.ndarray = np.linspace(0, 10, 100)) -> tuple[float, np.ndarray, np.ndarray]:
        n = self.X.shape[0]
        cvss_values = []
    
        for lam in lambdas:
            cvss = 0
            for i in range(n):
                X_train = np.delete(self.X, i, axis=0)
                Y_train = np.delete(self.Y, i, axis=0)
                X_test = self.X[i].reshape(1, -1)
                Y_test = self.Y[i]
            
                beta_lasso = self._coordinate_descent(X_train, Y_train, lam)
            
                Y_pred = (X_test @ beta_lasso).item()
                cvss += (Y_test - Y_pred) ** 2
            cvss_values.append(cvss)
    
        best_idx = np.argmin(cvss_values)
        best_lambda = lambdas[best_idx]
    
        beta_best = self._coordinate_descent(self.X, self.Y, best_lambda)
    
        plt.figure(figsize=(10, 6))
        plt.plot(lambdas, cvss_values, 'b-', linewidth=2)
        plt.scatter(best_lambda, cvss_values[best_idx], color='red', s=100, zorder=5)
        plt.xlabel('Lambda')
        plt.ylabel('CVSS')
        plt.title('График CVSS(lambda) для Lasso регрессии')
        plt.grid(True)
        plt.show()
    
        return best_lambda, beta_best, cvss_values

    def _coordinate_descent(self, X: np.ndarray, Y: np.ndarray, lam: float, max_iter: int = 1000, tol: float = 1e-4) -> np.ndarray:
        n, p = X.shape
        beta = np.zeros(p)

        for _ in range(max_iter):
            beta_old = beta.copy()

            for j in range(p):
                r = Y - X @ beta + beta[j] * X[:, j]
                rho = X[:, j] @ r
                if rho > lam:
                    beta[j] = (rho - lam) / (X[:, j] @ X[:, j])
                elif rho < -lam:
                    beta[j] = (rho + lam) / (X[:, j] @ X[:, j])
                else:
                    beta[j] = 0

            if np.linalg.norm(beta - beta_old) < tol:
                break
                
        return beta

regression = Regression(X, Y)
beta = regression.regression()
print(f"a)Коэффиценты регрессии: {beta}")
best_lambda, beta_ridge, cvss_values = regression.ridge()
print(f"b)Оптимальная lambda: {best_lambda:.4f}")
print(f"Коэффициенты Ridge регрессии: {beta_ridge}")
best_lambda, beta_lasso, cvss_values = regression.lasso()
print(f"Оптимальная lambda: {best_lambda:.4f}")
print(f"Коэффициенты Lasso регрессии: {np.round(beta_lasso, 8)}")