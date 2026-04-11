import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

n1 = 139
n2 = 1000
df1 = n1 - 1
df2 = n2 - 1
alpha = 0.05

F_lower = f.ppf(alpha / 2, df1, df2)
F_upper = f.ppf(1 - alpha / 2, df1, df2)

def power(theta):
    return 1 - f.cdf(F_upper / theta, df1, df2) + f.cdf(F_lower / theta, df1, df2)

theta_range = np.linspace(0.3, 2.5, 500)
power_values = [power(t) for t in theta_range]

plt.figure(figsize=(10, 6))
plt.plot(theta_range, power_values, 'b-', linewidth=2)
plt.xlabel('θ', fontsize=12)
plt.ylabel('W', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0.3, 2.5)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()