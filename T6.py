import numpy as np
from scipy.stats import norm, pareto


def generate_sample(theta: float, size: int) -> np.ndarray:
    if theta <= 1:
        raise ValueError("Параметр тета не больше 1")
    if size <= 0:
        raise ValueError("Размер выборки не положителен")
    
    return pareto.rvs(theta - 1, size=size)


class ConfidenceInterval:
    def __init__(self, sample: np.ndarray, beta: float):
        self.__sample = sample
        self.__beta = beta if 0 < beta < 1 else 0.95
        self.__size = len(sample)

    def __normal_quantile(self, p, mu=0, sigma=1):
        if np.any((p <= 0) | (p >= 1)):
            raise ValueError("p должно быть в интервале (0, 1)")
        if sigma <= 0:
            raise ValueError("sigma должна быть положительной")
        return norm.ppf(p, loc=mu, scale=sigma)

    def get_tetha(self) -> float:
        return 1 + self.__size/np.sum(np.log(self.__sample))
    
    def get_median(self) -> float:
        return np.median(self.__sample)
    
    def interval_median_accurate(self) -> tuple[float, float]:
        t1, t2 = -self.__size * np.log(np.min(self.__sample))/np.log((1 - self.__beta)/2), -self.__size * np.log(np.min(self.__sample))/np.log((1 + self.__beta)/2)
        return (np.exp2(t1), np.exp2(t2))
    
    def interval_tetha_asymptotic(self) -> tuple[float, float]:
        tetha = self.get_tetha()
        left = tetha - (tetha - 1)*self.__normal_quantile((1 + self.__beta)/2)/np.sqrt(self.__size)
        right = tetha - (tetha - 1)*self.__normal_quantile((1 - self.__beta)/2)/np.sqrt(self.__size)
        return (left, right)
    
    def noparametric_bootstrap_median(self, count: int) -> tuple[float, float]:
        if count<=0:
            raise ValueError("количество генераций должно быть > 0")
        
        delta = np.zeros(count)
        median = self.get_median()
        for i in range(count):
            x_i = np.random.choice(self.__sample, size=self.__size, replace=True)
            delta[i] = np.median(x_i) - median
        delta.sort()
        k1, k2 = int((1 - self.__beta) * count/2 ), int((1 + self.__beta) * count/2 )
        return (median - delta[k2], median - delta[k1])
 
    def parametric_bootstrap_median(self, count: int) -> tuple[float, float]:
        if count<=0:
            raise ValueError("количество генераций должно быть > 0")
        
        delta = np.zeros(count)
        median = self.get_median()
        for i in range(count):
            x_i = generate_sample(self.get_tetha(), self.__size)
            delta[i] = np.median(x_i) - median
        delta.sort()
        k1, k2 = int((1 - self.__beta) * count/2 ), int((1 + self.__beta) * count/2)
        return (median - delta[k2], median - delta[k1])

    def noparametric_bootstrap_tetha(self, count: int) -> tuple[float, float]:
        if count<=0:
            raise ValueError("количество генераций должно быть > 0")
        
        delta = np.zeros(count)
        tetha = self.get_tetha()
        for i in range(count):
            x_i = np.random.choice(self.__sample, size=self.__size, replace=True)
            delta[i] = 1 + self.__size/np.sum(np.log(x_i)) - tetha
        delta.sort()
        k1, k2 = int((1 - self.__beta) * count/2 ), int((1 + self.__beta) * count/2 )
        return (tetha - delta[k2], tetha - delta[k1])

    def parametric_bootstrap_tetha(self, count: int) -> tuple[float, float]:
        if count<=0:
            raise ValueError("количество генераций должно быть > 0")
        
        delta = np.zeros(count)
        tetha = self.get_tetha()
        for i in range(count):
            x_i = generate_sample(tetha, self.__size)
            delta[i] = 1 + self.__size/np.sum(np.log(x_i)) - tetha
        delta.sort()
        k1, k2 = int((1 - self.__beta) * count/2 ), int((1 + self.__beta) * count/2)
        return (tetha - delta[k2], tetha - delta[k1])



def main():
    sample = generate_sample(4, 100)
    intervals = ConfidenceInterval(sample, 0.95)
    print(f"Θ = {intervals.get_tetha()}")
    print(f"Медиана = {intervals.get_median()}")
    print(f"Точный доверительный интервал медианы = {intervals.interval_median_accurate()}")
    print(f"Асимптотический доверительный интервал Θ = {intervals.interval_tetha_asymptotic()}")
    print(f"Непараметрический bootstrap Θ = {intervals.noparametric_bootstrap_tetha(1000)}")
    print(f"Параметрический bootstrap Θ = {intervals.parametric_bootstrap_tetha(1000)}")
    print(f"Непараметрический bootstrap медианы = {intervals.noparametric_bootstrap_median(50000)}")
    print(f"Параметрический bootstrap медианы = {intervals.parametric_bootstrap_median(50000)}")

if __name__ == "__main__":
    main()