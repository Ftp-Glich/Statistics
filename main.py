import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_lab2', exist_ok=True)

sample_sizes = [10, 100, 1000]
np.random.seed(42)

distributions = {
    'Normal': lambda n: np.random.normal(0, 1, n),
    'Cauchy': lambda n: np.random.standard_cauchy(n),
    'Laplace': lambda n: np.random.laplace(0, 1/np.sqrt(2), n),
    'Poisson': lambda n: np.random.poisson(10, n),
    'Uniform': lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
}

def calculate_mean(sample):
    return np.mean(sample)

def calculate_median(sample):
    return np.median(sample)

def calculate_midrange(sample):
    return (np.min(sample) + np.max(sample)) / 2

def calculate_midhinge(sample):
    q1 = np.percentile(sample, 25)
    q3 = np.percentile(sample, 75)
    return (q1 + q3) / 2

def calculate_trimmed_mean(sample, proportion=0.1):
    return stats.trim_mean(sample, proportion)

characteristics = {
    'Mean': (calculate_mean, 'blue', 'solid'),
    'Median': (calculate_median, 'red', 'dashed'),
    'Midrange': (calculate_midrange, 'green', 'dashdot'),
    'Midhinge': (calculate_midhinge, 'orange', 'dotted'),
    'Trimmed_Mean': (calculate_trimmed_mean, 'purple', 'solid')
}

for dist_name, dist_func in distributions.items():
    print(f"Processing {dist_name}...")
    
    for n in sample_sizes:
        sample = dist_func(n)
        
        if dist_name == 'Cauchy':
            mask = (sample > -10) & (sample < 10)
            if np.sum(mask) < n * 0.5:
                sample = dist_func(n)
            else:
                sample = sample[mask]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if dist_name == 'Poisson':
            bins = np.arange(min(sample), max(sample) + 2) - 0.5
            ax.hist(sample, bins=bins, density=True, alpha=0.6, 
                    color='skyblue', edgecolor='black', label='Histogram')
            x = np.arange(0, 21)
            pmf = stats.poisson.pmf(x, mu=10)
            ax.plot(x, pmf, 'k-', linewidth=2, label='Theoretical')
        else:
            ax.hist(sample, bins='sqrt', density=True, alpha=0.6, 
                    color='skyblue', edgecolor='black', label='Histogram')
            x = np.linspace(-5, 5, 1000) if dist_name != 'Uniform' else np.linspace(-3, 3, 1000)
            if dist_name == 'Normal':
                pdf = stats.norm.pdf(x, loc=0, scale=1)
            elif dist_name == 'Cauchy':
                pdf = stats.cauchy.pdf(x, loc=0, scale=1)
            elif dist_name == 'Laplace':
                pdf = stats.laplace.pdf(x, loc=0, scale=1/np.sqrt(2))
            elif dist_name == 'Uniform':
                pdf = stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2*np.sqrt(3))
            ax.plot(x, pdf, 'k-', linewidth=2, label='Theoretical')
        
        for char_name, (char_func, color, linestyle) in characteristics.items():
            try:
                value = char_func(sample)
                if np.isfinite(value):
                    ax.axvline(value, color=color, linestyle=linestyle, 
                              linewidth=2, label=char_name)
            except:
                pass
        
        ax.set_title(f'{dist_name} Distribution, n={n}')
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        if dist_name == 'Cauchy':
            ax.set_xlim(-10, 10)
        elif dist_name == 'Uniform':
            ax.set_xlim(-3, 3)
        
        plt.tight_layout()
        plt.savefig(f'results_lab2/{dist_name.lower()}_n{n}_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved 3 histograms for {dist_name}")

print("\nAll histograms with characteristics saved to 'results_lab2' folder")