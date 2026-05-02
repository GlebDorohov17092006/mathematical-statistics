import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

def sample_generation(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    u = rng.random(n)  
    sample = -np.log(1 - u)
    return sample

def a(sample: np.ndarray) -> tuple:
    mode = stats.mode(sample).mode
    median = np.median(sample)
    range_value = np.max(sample) - np.min(sample)

    M_x = np.sum(sample)/len(sample)
    m_3 = np.sum((sample - M_x)**3)/len(sample)
    m_2 = (np.sum((sample - M_x)**2)/len(sample))**1.5

    coefficient_asymmetry = m_3/m_2

    return (mode, median, range_value, coefficient_asymmetry)


def F(sample: np.ndarray) -> None:
    x = np.sort(sample)
    y = np.arange(1, len(x) + 1)/len(x)
    plt.figure(figsize=(10, 6))
    plt.step(x, y, where='post', linewidth=2, label='Эмпирическая F(x)')
    plt.scatter(x, y, color='red', s=50, zorder=5, label='Наблюдения')
    x_theor = np.linspace(0, max(x) * 1.1, 1000)
    y_theor = 1 - np.exp(-x_theor)
    plt.plot(x_theor, y_theor, 'g-', linewidth=2, label=r'Теоретическая $F(x) = 1 - e^{-x}$')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title('Эмпирическая и теоретическая функции распределения')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.xlim(0, max(x) * 1.1)
    plt.show()


def p(sample: np.ndarray) -> None:
    n = len(sample)
    
    K = round(1 + np.log2(n))

    x_max, x_min = np.max(sample), np.min(sample)
    
    data_range = x_max - x_min
    
    delta_x = data_range / K
    
    bins = np.linspace(x_min, x_max, K + 1)
    
    plt.figure(figsize=(12, 6))
    counts, bins_edges, patches = plt.hist(sample, bins=bins, density=True, 
                                           alpha=0.7, color='skyblue', 
                                           edgecolor='black', linewidth=1.2,
                                           label='Эмпирическая плотность p(x)')
    
    x_theor = np.linspace(0, x_max, 10000)
    y_theor = np.exp(-x_theor)

    plt.plot(x_theor, y_theor, 'r-', linewidth=2, 
             label=r'Теоретическая плотность $p(x) = e^{-x}$')
    
    for bin_edge in bins_edges:
        plt.axvline(x=bin_edge, color='gray', linestyle='--', 
                   alpha=0.3, linewidth=0.5)
    
    plt.xlabel('x')
    plt.ylabel('Плотность')
    plt.title(f'Гистограмма (K={K}, Δx={delta_x})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, x_max)
    plt.tight_layout()
    plt.show()


def boxplot(sample: np.ndarray) -> None:

    q1 = np.percentile(sample, 25)
    q3 = np.percentile(sample, 75)
    iqr = q3 - q1
    
    lower_bound = max(q1 - 1.5 * iqr, np.min(sample))
    upper_bound = min(q3 + 1.5 * iqr, np.max(sample))
    
    outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
    
    print(f"  Q1: {q1:.4f}")
    print(f"  Q3: {q3:.4f}")
    print(f"  IQR: {iqr:.4f}")
    print(f"  Нижняя граница: {lower_bound:.4f}")
    print(f"  Верхняя граница: {upper_bound:.4f}")
    print(f"  Количество выбросов: {len(outliers)}")

    if len(outliers) > 0:
        print(f"  Выбросы: {np.sort(outliers)}")
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(sample)
    
    plt.ylabel('Значения')
    plt.title('Boxplot')
    plt.grid(True, alpha=0.5, axis='y')
    plt.show()

def b(sample: np.ndarray) -> None:
    F(sample)
    p(sample)
    boxplot(sample)

def c(sample: np.ndarray, N: int) -> None:
    arr = np.zeros(N)
    for i in range(N):
        x_i = np.sum(np.random.choice(sample, size=len(sample), replace=True))/len(sample)
        arr[i] = x_i
    
    K = round(1 + np.log2(N))
    
    x_min = np.min(arr)
    x_max = np.max(arr)
    bins = np.linspace(x_min, x_max, K + 1)
    
    plt.figure(figsize=(16, 9))
    plt.hist(arr, bins=bins, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black')
    
    x_theor = np.linspace(x_min, x_max, 10000)
    y_theor = stats.norm.pdf(x_theor, loc=1, scale=1/5)
    plt.plot(x_theor, y_theor, 'r-', linewidth=2, 
             label=r'$N(1, \frac{1}{25})$')
    
    plt.xlabel('среднее арифметическое')
    plt.ylabel('Плотность')
    plt.title(f'Бутстреп распределение среднего арифметического (N={N})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def d(sample: np.ndarray, N:int) -> None:
    arr = np.zeros(N)
    for i in range(N):
        x_i = np.random.choice(sample, size=len(sample), replace=True)

        M_x = np.sum(x_i)/len(x_i)
        m_3 = np.sum((x_i - M_x)**3)/len(x_i)
        m_2 = (np.sum((x_i - M_x)**2)/len(x_i))**1.5

        arr[i] = m_3/m_2

    arr_sort = np.sort(arr)
    k = 0
    for i in range(N):
        if arr_sort[i] >= 1:
            break
        k+=1
    print(f"P(coeff_assimetry < 1) = {k/N}")

    K = round(1 + np.log2(N))
    
    x_min = np.min(arr)
    x_max = np.max(arr)
    bins = np.linspace(x_min, x_max, K + 1)
    
    plt.figure(figsize=(16, 9))
    plt.hist(arr, bins=bins, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black')

    plt.xlabel('Медиана')
    plt.ylabel('Плотность')
    plt.title(f'Бутстреп распределение коэффицента ассиметрии (N={N})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def p_med(x: float) -> float:
    p = np.exp(-x) if x >= 0 else 0
    F = 1 - np.exp(-x) if x>=0 else 0
    arr = np.array([(i - 25*F) * math.comb(25, i) * (F**(i - 1)) * ((1 - F)**(24 - i)) for i in range(13, 26)])
    return p * sum(arr)

def e(sample: np.ndarray, N: int) -> None:
    arr = np.zeros(N)
    for i in range(N):
        x_i = np.median(np.random.choice(sample, size=len(sample), replace=True))
        arr[i] = x_i
    
    K = round(1 + np.log2(N))
    
    x_min = np.min(arr)
    x_max = np.max(arr)
    bins = np.linspace(x_min, x_max, K + 1)
    
    plt.figure(figsize=(16, 9))
    plt.hist(arr, bins=bins, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black')
    
    x_theor = np.linspace(x_min, x_max, 10000)
    y_theor = np.zeros(10000)
    for i in range(len(x_theor)):
        y_theor[i] = p_med(x_theor[i])

    plt.plot(x_theor, y_theor, 'r-', linewidth=2, 
             label=r'p_median(x)')
    
    plt.xlabel('Медиана')
    plt.ylabel('Плотность')
    plt.title(f'Бутстреп распределение медианы (N={N})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

sample = sample_generation(25)

print(a(sample))
b(sample)
c(sample,1000)
d(sample, 1000)
e(sample, 1000)