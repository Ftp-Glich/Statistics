import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_lab4', exist_ok=True)

n = 100
np.random.seed(42)

distributions = {
    'Normal': lambda: np.random.normal(0, 1, n),
    'Cauchy': lambda: np.random.standard_cauchy(n),
    'Laplace': lambda: np.random.laplace(0, 1/np.sqrt(2), n),
    'Poisson': lambda: np.random.poisson(10, n),
    'Uniform': lambda: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
}

x_ranges = {
    'Normal': (-4, 4),
    'Cauchy': (-4, 4),
    'Laplace': (-4, 4),
    'Poisson': (6, 14),
    'Uniform': (-4, 4)
}

h_values = [0.1, 0.3, 0.5, 1.0, 'scott']
h_labels = ['h=0.1', 'h=0.3', 'h=0.5', 'h=1.0', 'Scott (auto)']
colors = ['red', 'orange', 'green', 'blue', 'purple']

print("=" * 60)
print("LAB 4: KDE Bandwidth Comparison")
print("=" * 60)

for dist_name, dist_func in distributions.items():
    print(f"\nProcessing {dist_name}...")
    
    sample = dist_func()
    
    if dist_name == 'Cauchy':
        sample = sample[(sample > -10) & (sample < 10)]
        sample = sample[:n]
    
    x_min, x_max = x_ranges[dist_name]
    sample = sample[(sample >= x_min) & (sample <= x_max)]
    
    x_plot = np.linspace(x_min, x_max, 1000)
    
    if dist_name == 'Normal':
        y_theoretical = stats.norm.pdf(x_plot, loc=0, scale=1)
    elif dist_name == 'Cauchy':
        y_theoretical = stats.cauchy.pdf(x_plot, loc=0, scale=1)
    elif dist_name == 'Laplace':
        y_theoretical = stats.laplace.pdf(x_plot, loc=0, scale=1/np.sqrt(2))
    elif dist_name == 'Poisson':
        x_plot = np.arange(x_min, x_max+1)
        y_theoretical = stats.poisson.pmf(x_plot, mu=10)
    elif dist_name == 'Uniform':
        y_theoretical = stats.uniform.pdf(x_plot, loc=-np.sqrt(3), scale=2*np.sqrt(3))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    # Гистограмма для справки
    axes[0].hist(sample, bins='sqrt', density=True, alpha=0.3, 
                color='gray', edgecolor='black', label='Histogram')
    axes[0].set_title('Histogram + Theoretical')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)
    
    if dist_name != 'Poisson':
        axes[0].plot(x_plot, y_theoretical, 'k-', linewidth=2, label='Theoretical')
        axes[0].legend(fontsize=8)
    else:
        axes[0].stem(x_plot, y_theoretical, linefmt='k-', markerfmt='ko', 
                    basefmt=' ', label='Theoretical')
        axes[0].legend(fontsize=8)
    
    for idx, (h, label, color) in enumerate(zip(h_values, h_labels, colors), start=1):
        if h == 'scott':
            kde = gaussian_kde(sample) 
            y_kde = kde(x_plot)
        else:
            kde = gaussian_kde(sample, bw_method=h / np.std(sample))
            y_kde = kde(x_plot)
        
        axes[idx].hist(sample, bins='sqrt', density=True, alpha=0.2, 
                      color='gray', edgecolor='black')
        axes[idx].plot(x_plot, y_kde, color=color, linewidth=2, label=f'KDE {label}')
        
        if dist_name != 'Poisson':
            axes[idx].plot(x_plot, y_theoretical, 'k--', linewidth=1.5, label='Theoretical')
        
        axes[idx].set_title(f'KDE with {label}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('Density')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    axes[-1].axis('off')
    
    plt.suptitle(f'{dist_name} Distribution - KDE Bandwidth Comparison (n={n})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results_lab4/{dist_name.lower()}_bandwidth_comparison.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved bandwidth comparison for {dist_name}")

print("\nCalculating MSE for different h values...")

sample_normal = np.random.normal(0, 1, n)
x_eval = np.linspace(-4, 4, 200)
y_true = stats.norm.pdf(x_eval, loc=0, scale=1)

h_test = np.linspace(0.1, 2.0, 50)
mse_values = []

for h in h_test:
    kde = gaussian_kde(sample_normal, bw_method=h / np.std(sample_normal))
    y_est = kde(x_eval)
    mse = np.mean((y_est - y_true) ** 2)
    mse_values.append(mse)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(h_test, mse_values, 'b-', linewidth=2)
ax.axvline(x=1.06 * np.std(sample_normal) * n**(-1/5), color='red', 
          linestyle='--', label="Scott's rule")
ax.set_xlabel('Bandwidth h')
ax.set_ylabel('Mean Squared Error (MSE)')
ax.set_title('MSE vs Bandwidth for Normal Distribution (n=100)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results_lab4/mse_vs_bandwidth.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ MSE plot saved")
print("\n🎉 All bandwidth comparison results saved to 'results_lab4' folder")