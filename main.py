import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_lab3', exist_ok=True)

sample_sizes = [20, 100]
n_repetitions = 1000
np.random.seed(42)

distributions = {
    'Normal': lambda n: np.random.normal(0, 1, n),
    'Cauchy': lambda n: np.random.standard_cauchy(n),
    'Laplace': lambda n: np.random.laplace(0, 1/np.sqrt(2), n),
    'Poisson': lambda n: np.random.poisson(10, n),
    'Uniform': lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
}

def calculate_outliers_tukey(sample):
    Q1 = np.percentile(sample, 25)
    Q3 = np.percentile(sample, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
    return outliers, len(outliers) / len(sample)

print("=" * 60)
print("LABORATORY WORK #3: Tukey Boxplot and Outlier Analysis")
print("=" * 60)

for dist_name, dist_func in distributions.items():
    print(f"\nProcessing {dist_name}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, n in enumerate(sample_sizes):
        sample = dist_func(n)
        
        if dist_name == 'Cauchy':
            sample = sample[(sample > -10) & (sample < 10)]
            sample = sample[:n]
        
        outliers, outlier_ratio = calculate_outliers_tukey(sample)
        
        if dist_name == 'Poisson':
            sns.boxplot(y=sample, ax=axes[idx], palette=['skyblue'])
            axes[idx].set_ylabel('Value')
        else:
            sns.boxplot(y=sample, ax=axes[idx], palette=['skyblue'])
            axes[idx].set_ylabel('Value')
        
        axes[idx].set_title(f'n = {n}\nOutliers: {len(outliers)} ({outlier_ratio*100:.1f}%)')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        for i, outlier in enumerate(outliers):
            axes[idx].axhline(y=outlier, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.suptitle(f'{dist_name} Distribution - Tukey Boxplots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results_lab3/{dist_name.lower()}_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved boxplots for {dist_name}")

print("\n" + "=" * 60)
print("Calculating outlier proportions (1000 repetitions)...")
print("=" * 60)

outlier_results = {}

for dist_name, dist_func in distributions.items():
    print(f"\nProcessing {dist_name}...")
    outlier_results[dist_name] = {}
    
    for n in sample_sizes:
        outlier_ratios = []
        total_outliers = 0
        
        for rep in range(n_repetitions):
            sample = dist_func(n)
            
            if dist_name == 'Cauchy':
                sample = sample[(sample > -100) & (sample < 100)]
                if len(sample) < n * 0.5:
                    sample = dist_func(n)
                else:
                    sample = sample[:n] if len(sample) >= n else sample
            
            _, ratio = calculate_outliers_tukey(sample)
            outlier_ratios.append(ratio)
            total_outliers += ratio
        
        avg_outlier_ratio = np.mean(outlier_ratios)
        std_outlier_ratio = np.std(outlier_ratios)
        min_ratio = np.min(outlier_ratios)
        max_ratio = np.max(outlier_ratios)
        
        outlier_results[dist_name][n] = {
            'avg_ratio': avg_outlier_ratio,
            'std_ratio': std_outlier_ratio,
            'min_ratio': min_ratio,
            'max_ratio': max_ratio,
            'all_ratios': outlier_ratios
        }
        
        print(f"  n={n}: Avg outlier ratio = {avg_outlier_ratio:.4f} ± {std_outlier_ratio:.4f}")
        print(f"       Range: [{min_ratio:.4f}, {max_ratio:.4f}]")

with open('results_lab3/outlier_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("LABORATORY WORK #3 - OUTLIER ANALYSIS RESULTS\n")
    f.write("=" * 70 + "\n\n")
    
    for dist_name in distributions.keys():
        f.write(f"\n{dist_name.upper()}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Sample Size':<15} {'Avg Ratio':<15} {'Std':<15} {'Min':<15} {'Max':<15}\n")
        f.write("-" * 70 + "\n")
        
        for n in sample_sizes:
            avg = outlier_results[dist_name][n]['avg_ratio']
            std = outlier_results[dist_name][n]['std_ratio']
            min_r = outlier_results[dist_name][n]['min_ratio']
            max_r = outlier_results[dist_name][n]['max_ratio']
            f.write(f"{n:<15} {avg:<15.6f} {std:<15.6f} {min_r:<15.6f} {max_r:<15.6f}\n")
        
        f.write("\n")

print("\n✅ Outlier analysis saved to 'results_lab3/outlier_analysis.txt'")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dist_names = list(distributions.keys())
x = np.arange(len(dist_names))
width = 0.35

for idx, n in enumerate(sample_sizes):
    avg_ratios = [outlier_results[dist][n]['avg_ratio'] * 100 for dist in dist_names]
    std_ratios = [outlier_results[dist][n]['std_ratio'] * 100 for dist in dist_names]
    
    axes[idx].bar(x, avg_ratios, width, 
                  alpha=0.7, color='skyblue', edgecolor='black', label='Avg %')
    axes[idx].set_xlabel('Distribution')
    axes[idx].set_ylabel('Outlier Percentage (%)')
    axes[idx].set_title(f'Sample Size n = {n}')
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(dist_names, rotation=15, ha='right')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.suptitle('Average Outlier Proportion by Distribution and Sample Size', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results_lab3/outlier_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Comparison plot saved")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (dist_name, dist_func) in enumerate(distributions.items()):
    n = 100
    sample = dist_func(n)
    
    if dist_name == 'Cauchy':
        sample = sample[(sample > -10) & (sample < 10)]
        sample = sample[:n]
    
    outliers, outlier_ratio = calculate_outliers_tukey(sample)
    
    if dist_name != 'Poisson':
        x = np.linspace(-5, 5, 1000)
        if dist_name == 'Normal':
            pdf = stats.norm.pdf(x, loc=0, scale=1)
        elif dist_name == 'Cauchy':
            pdf = stats.cauchy.pdf(x, loc=0, scale=1)
        elif dist_name == 'Laplace':
            pdf = stats.laplace.pdf(x, loc=0, scale=1/np.sqrt(2))
        elif dist_name == 'Uniform':
            pdf = stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2*np.sqrt(3))
        
        axes[idx].hist(sample, bins='sqrt', density=True, alpha=0.6, 
                      color='skyblue', edgecolor='black', label='Histogram')
        axes[idx].plot(x, pdf, 'r-', linewidth=2, label='Theoretical')
    else:
        axes[idx].hist(sample, bins=range(min(sample), max(sample)+2), 
                      density=True, alpha=0.6, color='skyblue', 
                      edgecolor='black', label='Histogram')
    
    Q1 = np.percentile(sample, 25)
    Q3 = np.percentile(sample, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[idx].axvline(x=lower_bound, color='red', linestyle='--', 
                     linewidth=2, label=f'Outlier bounds')
    axes[idx].axvline(x=upper_bound, color='red', linestyle='--', linewidth=2)
    
    for outlier in outliers:
        axes[idx].axvline(x=outlier, color='orange', linestyle=':', 
                         linewidth=1, alpha=0.7)
    
    axes[idx].set_title(f'{dist_name}\nOutliers: {len(outliers)} ({outlier_ratio*100:.1f}%)')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

axes[-1].axis('off')

plt.suptitle('Distribution Histograms with Tukey Outlier Bounds (n=100)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results_lab3/distributions_with_outliers.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Distribution histograms saved")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, n in enumerate(sample_sizes):
    ratios = []
    dist_labels = []
    
    for dist_name in dist_names:
        all_ratios = outlier_results[dist_name][n]['all_ratios']
        ratios.append(all_ratios)
        dist_labels.append(dist_name)
    
    bp = axes[idx].boxplot(ratios, labels=dist_labels, patch_artist=True,
                          boxprops=dict(facecolor='skyblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))
    
    axes[idx].set_xlabel('Distribution')
    axes[idx].set_ylabel('Outlier Proportion')
    axes[idx].set_title(f'Sample Size n = {n}')
    axes[idx].set_xticklabels(dist_labels, rotation=15, ha='right')
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].axhline(y=0.007, color='green', linestyle='--', 
                     linewidth=1.5, label='Theoretical (0.7%)')

plt.suptitle('Distribution of Outlier Proportions (1000 repetitions)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results_lab3/outlier_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Outlier proportion boxplots saved")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n{'Distribution':<15} {'n=20':<15} {'n=100':<15}")
print("-" * 45)
for dist_name in dist_names:
    ratio_20 = outlier_results[dist_name][20]['avg_ratio'] * 100
    ratio_100 = outlier_results[dist_name][100]['avg_ratio'] * 100
    print(f"{dist_name:<15} {ratio_20:<15.2f}% {ratio_100:<15.2f}%")

print("\n All results saved to 'results_lab3' folder")
print("Files created:")
print("  - [distribution]_boxplots.png (5 files)")
print("  - outlier_analysis.txt")
print("  - outlier_comparison.png")
print("  - distributions_with_outliers.png")
print("  - outlier_boxplots.png")