import numpy as np
from scipy.stats import norm


def generate_sample(theta: float, size: int) -> np.ndarray:
    if theta <= 0:
        raise ValueError("Параметр тета не положительный")
    if size <= 0:
        raise ValueError("Размер выборки не положителен")
    
    rng = np.random.default_rng()
    return rng.uniform(theta, 2 * theta, size=size)


class ConfidenceInterval:
    def __init__(self, sample: np.ndarray, beta: float):
        self.__sample = sample
        self.__beta = beta if 0 < beta < 1 else 0.95
        self.__size = len(sample)

    def __max(self) -> float:
        return np.max(self.__sample)

    def __normal_quantile(self, p, mu=0, sigma=1):
        if np.any((p <= 0) | (p >= 1)):
            raise ValueError("p должно быть в интервале (0, 1)")
        if sigma <= 0:
            raise ValueError("sigma должна быть положительной")
        return norm.ppf(p, loc=mu, scale=sigma)
    
    def get_tetha_omm(self) -> float:
        return 2/3 * np.mean(self.__sample)
    
    def get_tetha_omp1(self) -> float:
        return self.__max()/2
    
    def get_tetha_omp2(self) -> float:
        return (self.__size + 1)/(2 * self.__size + 1) * self.__max()
    
    def accurate_interval(self) -> tuple[float, float]:
        t1, t2 = 1 + ((1 - self.__beta)/2)**(1/self.__size), 1 + ((1 + self.__beta)/2)**(1/self.__size)
        M = self.__max()
        return (M/t2, M/t1)
    
    def asymptotic_interval(self) -> tuple[float, float]:
        t1, t2 = self.__normal_quantile((1 + self.__beta)/2), self.__normal_quantile((1 - self.__beta)/2)
        sample2 = self.__sample**2
        left =  2/3 * np.mean(self.__sample) - t1 * 2/3 * np.sqrt((np.mean(sample2) - np.mean(self.__sample)**2)/self.__size)
        rigth = 2/3 * np.mean(self.__sample) - t2 * 2/3 * np.sqrt((np.mean(sample2) - np.mean(self.__sample)**2)/self.__size)
        return (left, rigth)

    def parametric_bootstrap_omm(self, count) -> tuple[float, float]:
        if count<=0:
            raise ValueError("количество генераций должно быть > 0")
        
        delta = np.zeros(count)
        tetha = self.get_tetha_omm()
        for i in range(count):
            x_i = generate_sample(tetha, self.__size)
            delta[i] = 2/3 * np.mean(x_i) - tetha
        delta.sort()
        k1, k2 = int((1 - self.__beta) * count/2 ), int((1 + self.__beta) * count/2)
        return (tetha - delta[k2], tetha - delta[k1])

    def parametric_bootstrap_omp(self, count) -> tuple[float, float]:
        if count<=0:
            raise ValueError("количество генераций должно быть > 0")
        
        delta = np.zeros(count)
        tetha = self.get_tetha_omp2()
        for i in range(count):
            x_i = generate_sample(tetha, self.__size)
            delta[i] = np.max(x_i)*(self.__size + 1)/(2 * self.__size + 1) - tetha
        delta.sort()
        k1, k2 = int((1 - self.__beta) * count/2 ), int((1 + self.__beta) * count/2)
        return (tetha - delta[k2], tetha - delta[k1])

    def noparametric_bootstrap_omm(self, count: int) -> tuple[float, float]:
        if count<=0:
            raise ValueError("количество генераций должно быть > 0")
        
        delta = np.zeros(count)
        tetha = self.get_tetha_omm()
        for i in range(count):
            x_i = np.random.choice(self.__sample, size=self.__size, replace=True)
            delta[i] = 2/3 * np.mean(x_i) - tetha
        delta.sort()
        k1, k2 = int((1 - self.__beta) * count/2 ), int((1 + self.__beta) * count/2 )
        return (tetha - delta[k2], tetha - delta[k1])
    
    def noparametric_bootstrap_omp(self, count: int) -> tuple[float, float]:
        if count<=0:
            raise ValueError("количество генераций должно быть > 0")
        
        delta = np.zeros(count)
        tetha = self.get_tetha_omp2()
        for i in range(count):
            x_i = np.random.choice(self.__sample, size=self.__size, replace=True)
            delta[i] = np.max(x_i)*(self.__size + 1)/(2 * self.__size + 1) - tetha
        delta.sort()
        k1, k2 = int((1 - self.__beta) * count/2 ), int((1 + self.__beta) * count/2)
        return (tetha - delta[k2], tetha - delta[k1])


def main():
    sample = generate_sample(1, 100)
    intervals = ConfidenceInterval(sample, 0.95)
    print(f"Θ1 = {intervals.get_tetha_omm()}")
    print(f"Θ2(смещенная) = {intervals.get_tetha_omp1()}")
    print(f"Θ2(несмещенная) = {intervals.get_tetha_omp2()}")
    print(f"Точный доверительный интервал = {intervals.accurate_interval()}")
    print(f"Асимптотический доверительный интервал(ОММ) = {intervals.asymptotic_interval()}")
    print(f"Непараметрический bootstrap(omm) = {intervals.noparametric_bootstrap_omm(1000)}")
    print(f"Непараметрический bootstrap(omp) = {intervals.noparametric_bootstrap_omp(1000)}")
    print(f"Параметрический bootstrap(omm) = {intervals.parametric_bootstrap_omm(50000)}")
    print(f"Параметрический bootstrap(omp) = {intervals.parametric_bootstrap_omp(50000)}")

if __name__ == "__main__":
    main()
