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
    'Mean': (calculate_mean, 'blue', 'solid', 2.5),
    'Median': (calculate_median, 'red', 'dashed', 2.5),
    'Midrange': (calculate_midrange, 'green', 'dashdot', 2.5),
    'Midhinge': (calculate_midhinge, 'orange', 'dotted', 2.5),
    'Trimmed_Mean': (calculate_trimmed_mean, 'purple', 'solid', 2.5)
}

xlim_params = {
    'Normal': (-4, 4),
    'Cauchy': (-5, 5),
    'Laplace': (-4, 4),
    'Poisson': (0, 20),
    'Uniform': (-2, 2)
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
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        x_min, x_max = xlim_params[dist_name]
        
        if dist_name == 'Poisson':
            bins = np.arange(max(x_min, min(sample)), min(x_max, max(sample) + 1))
            ax.hist(sample, bins=bins, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', label='Histogram')
            x = np.arange(int(x_min), int(x_max) + 1)
            pmf = stats.poisson.pmf(x, mu=10)
            ax.plot(x, pmf, 'k-', linewidth=2.5, label='Theoretical PMF', zorder=5)
        else:
            bins = np.linspace(x_min, x_max, int(np.sqrt(n)) + 1)
            ax.hist(sample, bins=bins, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', label='Histogram', range=(x_min, x_max))
            x = np.linspace(x_min, x_max, 1000)
            if dist_name == 'Normal':
                pdf = stats.norm.pdf(x, loc=0, scale=1)
            elif dist_name == 'Cauchy':
                pdf = stats.cauchy.pdf(x, loc=0, scale=1)
            elif dist_name == 'Laplace':
                pdf = stats.laplace.pdf(x, loc=0, scale=1/np.sqrt(2))
            elif dist_name == 'Uniform':
                pdf = stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2*np.sqrt(3))
            ax.plot(x, pdf, 'k-', linewidth=2.5, label='Theoretical PDF', zorder=5)
        
        char_values = {}
        for char_name, (char_func, color, linestyle, linewidth) in characteristics.items():
            try:
                value = char_func(sample)
                if np.isfinite(value) and x_min <= value <= x_max:
                    ax.axvline(value, color=color, linestyle=linestyle, 
                              linewidth=linewidth, label=char_name, zorder=10)
                    char_values[char_name] = value
            except:
                pass
        
        legend_elements = [plt.Line2D([0], [0], color=color, linestyle=linestyle, 
                                      linewidth=linewidth, label=name) 
                          for name, (func, color, linestyle, linewidth) in characteristics.items()]
        
        ax.set_xlim(x_min, x_max)
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.15)
        
        ax.set_title(f'{dist_name} Distribution, n={n}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        
        legend1 = ax.legend(handles=legend_elements, fontsize=9, loc='upper right', 
                           title='Characteristics', title_fontsize=10)
        ax.add_artist(legend1)
        ax.legend(fontsize=9, loc='upper left')
        
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        for i, (char_name, value) in enumerate(char_values.items()):
            offset = 0.02 * i
            ax.text(value, y_max * (1.05 + offset), f'{value:.3f}', 
                   ha='center', fontsize=9, rotation=90, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'results_lab2/{dist_name.lower()}_n{n}_histogram.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved 3 histograms for {dist_name}")

print("\nAll histograms with characteristics saved to 'results_lab2' folder")