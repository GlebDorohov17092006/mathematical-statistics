import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

alpha = 0.05
q = norm.ppf(1 - alpha)
sigma = np.sqrt(7/6)

def power(theta, q, sigma):
    return 1 - norm.cdf(q - theta / sigma)

theta = np.linspace(-2, 6, 500)
y = power(theta, q, sigma)

plt.figure(figsize=(8, 5))
plt.plot(theta, y, 'b-', linewidth=2)

plt.axhline(y=alpha, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

plt.xlabel('θ')
plt.ylabel('W(θ)')
plt.title('Функция мощности')
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.xlim(-2, 6)

plt.tight_layout()
plt.show()