import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_lab2', exist_ok=True)

sample_sizes = [10, 100, 1000]
n_repetitions = 1000
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
    'Mean': calculate_mean,
    'Median': calculate_median,
    'Midrange': calculate_midrange,
    'Midhinge': calculate_midhinge,
    'Trimmed_Mean': calculate_trimmed_mean
}

results = {}

for dist_name, dist_func in distributions.items():
    print(f"Processing {dist_name}...")
    results[dist_name] = {}
    
    for n in sample_sizes:
        results[dist_name][n] = {}
        
        for char_name, char_func in characteristics.items():
            values = []
            
            for _ in range(n_repetitions):
                sample = dist_func(n)
                
                if dist_name == 'Cauchy':
                    sample = sample[(sample > -100) & (sample < 100)]
                    if len(sample) < n * 0.5:
                        sample = dist_func(n)
                
                try:
                    value = char_func(sample)
                    if np.isfinite(value):
                        values.append(value)
                except:
                    pass
            
            if len(values) > 0:
                E_z = np.mean(values)
                D_z = np.var(values)
                
                results[dist_name][n][char_name] = {
                    'E': E_z,
                    'D': D_z,
                    'values': values
                }

with open('results_lab2/numerical_results.txt', 'w') as f:
    f.write("LABORATORY WORK #2 - RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    for dist_name in distributions.keys():
        f.write(f"\n{dist_name.upper()}\n")
        f.write("-" * 80 + "\n")
        
        for n in sample_sizes:
            f.write(f"\nSample size n = {n}\n")
            f.write(f"{'Characteristic':<15} {'E(z)':<20} {'D(z)':<20}\n")
            f.write("-" * 55 + "\n")
            
            for char_name in characteristics.keys():
                if char_name in results[dist_name][n]:
                    E = results[dist_name][n][char_name]['E']
                    D = results[dist_name][n][char_name]['D']
                    f.write(f"{char_name:<15} {E:<20.6f} {D:<20.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")

for dist_name in distributions.keys():
    fig, axes = plt.subplots(len(sample_sizes), len(characteristics), 
                             figsize=(15, 3*len(sample_sizes)))
    
    if len(sample_sizes) == 1:
        axes = axes.reshape(1, -1)
    
    for i, n in enumerate(sample_sizes):
        for j, char_name in enumerate(characteristics.keys()):
            if char_name in results[dist_name][n]:
                values = results[dist_name][n][char_name]['values']
                
                if len(sample_sizes) > 1 and len(characteristics) > 1:
                    ax = axes[i, j]
                elif len(sample_sizes) == 1 and len(characteristics) > 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                
                ax.hist(values, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
                ax.axvline(results[dist_name][n][char_name]['E'], color='red', linestyle='--', linewidth=2)
                ax.set_title(f'{char_name}\nn={n}, E={results[dist_name][n][char_name]["E"]:.3f}')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dist_name} Distribution - Characteristics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results_lab2/{dist_name.lower()}_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()

for dist_name in distributions.keys():
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(15, 4))
    
    if len(sample_sizes) == 1:
        axes = [axes]
    
    for idx, n in enumerate(sample_sizes):
        char_names = list(characteristics.keys())
        E_values = []
        D_values = []
        
        for char_name in char_names:
            if char_name in results[dist_name][n]:
                E_values.append(results[dist_name][n][char_name]['E'])
                D_values.append(results[dist_name][n][char_name]['D'])
        
        x = np.arange(len(E_values))
        width = 0.35
        
        axes[idx].bar(x - width/2, E_values, width, label='E(z)', alpha=0.7)
        axes[idx].bar(x + width/2, D_values, width, label='D(z)', alpha=0.7)
        axes[idx].set_xlabel('Characteristics')
        axes[idx].set_title(f'n = {n}')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([cn.replace('_', '\n') for cn in char_names], rotation=0, ha='center', fontsize=8)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{dist_name} - Comparison of Characteristics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results_lab2/{dist_name.lower()}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nAll results saved to 'results_lab2' folder")
print("Files created:")
print("  - numerical_results.txt")
print("  - [distribution]_characteristics.png (for each distribution)")
print("  - [distribution]_comparison.png (for each distribution)")