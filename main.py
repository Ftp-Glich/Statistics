import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_lab4', exist_ok=True)

sample_sizes = [20, 60, 100]
np.random.seed(42)

distributions = {
    'Normal': lambda n: np.random.normal(0, 1, n),
    'Cauchy': lambda n: np.random.standard_cauchy(n),
    'Laplace': lambda n: np.random.laplace(0, 1/np.sqrt(2), n),
    'Poisson': lambda n: np.random.poisson(10, n),
    'Uniform': lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
}

x_ranges = {
    'Normal': (-4, 4),
    'Cauchy': (-4, 4),
    'Laplace': (-4, 4),
    'Poisson': (6, 14),
    'Uniform': (-4, 4)
}

def empirical_cdf(sample, x):
    return np.sum(sample <= x) / len(sample)

print("=" * 60)
print("LABORATORY WORK #4: Empirical CDF and Kernel Density")
print("=" * 60)

for dist_name, dist_func in distributions.items():
    print(f"\nProcessing {dist_name}...")
    
    x_min, x_max = x_ranges[dist_name]
    
    fig_ecdf, axes_ecdf = plt.subplots(1, 3, figsize=(15, 4))
    fig_kde, axes_kde = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, n in enumerate(sample_sizes):
        sample = dist_func(n)
        
        if dist_name == 'Cauchy':
            sample = sample[(sample > -10) & (sample < 10)]
            sample = sample[:n]
        
        if dist_name == 'Poisson':
            sample_plot = np.clip(sample, x_min, x_max)
        else:
            sample_plot = sample
        
        sample_plot = sample_plot[(sample_plot >= x_min) & (sample_plot <= x_max)]
        
        x_ecdf = np.linspace(x_min, x_max, 1000)
        y_ecdf = np.array([np.mean(sample_plot <= xi) for xi in x_ecdf])
        
        if dist_name == 'Normal':
            y_theoretical_cdf = stats.norm.cdf(x_ecdf, loc=0, scale=1)
        elif dist_name == 'Cauchy':
            y_theoretical_cdf = stats.cauchy.cdf(x_ecdf, loc=0, scale=1)
        elif dist_name == 'Laplace':
            y_theoretical_cdf = stats.laplace.cdf(x_ecdf, loc=0, scale=1/np.sqrt(2))
        elif dist_name == 'Poisson':
            y_theoretical_cdf = stats.poisson.cdf(np.arange(x_min, x_max+1), mu=10)
            x_theoretical = np.arange(x_min, x_max+1)
        elif dist_name == 'Uniform':
            y_theoretical_cdf = stats.uniform.cdf(x_ecdf, loc=-np.sqrt(3), scale=2*np.sqrt(3))
        
        if dist_name == 'Poisson':
            axes_ecdf[idx].step(np.sort(sample_plot), np.arange(1, len(sample_plot)+1) / len(sample_plot), 
                               where='post', label='Empirical CDF', color='blue')
            axes_ecdf[idx].stairs(np.cumsum(stats.poisson.pmf(np.arange(x_min, x_max+1), 10)), 
                                 np.arange(x_min, x_max+2), label='Theoretical CDF', color='red')
        else:
            axes_ecdf[idx].step(np.sort(sample_plot), np.arange(1, len(sample_plot)+1) / len(sample_plot), 
                               where='post', label='Empirical CDF', color='blue')
            axes_ecdf[idx].plot(x_ecdf, y_theoretical_cdf, 'r-', linewidth=2, label='Theoretical CDF')
        
        axes_ecdf[idx].set_title(f'n = {n}')
        axes_ecdf[idx].set_xlabel('x')
        axes_ecdf[idx].set_ylabel('F(x)')
        axes_ecdf[idx].legend(fontsize=8)
        axes_ecdf[idx].grid(True, alpha=0.3)
        
        if dist_name != 'Poisson':
            kde = gaussian_kde(sample_plot)
            y_kde = kde(x_ecdf)
            
            if dist_name == 'Normal':
                y_theoretical_pdf = stats.norm.pdf(x_ecdf, loc=0, scale=1)
            elif dist_name == 'Cauchy':
                y_theoretical_pdf = stats.cauchy.pdf(x_ecdf, loc=0, scale=1)
            elif dist_name == 'Laplace':
                y_theoretical_pdf = stats.laplace.pdf(x_ecdf, loc=0, scale=1/np.sqrt(2))
            elif dist_name == 'Uniform':
                y_theoretical_pdf = stats.uniform.pdf(x_ecdf, loc=-np.sqrt(3), scale=2*np.sqrt(3))
            
            axes_kde[idx].hist(sample_plot, bins='sqrt', density=True, alpha=0.3, 
                              color='gray', edgecolor='black', label='Histogram')
            axes_kde[idx].plot(x_ecdf, y_kde, 'b-', linewidth=2, label='Kernel Density')
            axes_kde[idx].plot(x_ecdf, y_theoretical_pdf, 'r--', linewidth=2, label='Theoretical PDF')
        else:
            x_poisson = np.arange(x_min, x_max+1)
            y_theoretical_pmf = stats.poisson.pmf(x_poisson, mu=10)
            
            axes_kde[idx].hist(sample_plot, bins=np.arange(x_min-0.5, x_max+1.5), 
                              density=True, alpha=0.3, color='gray', 
                              edgecolor='black', label='Histogram')
            axes_kde[idx].stem(x_poisson, y_theoretical_pmf, linefmt='r--', markerfmt='ro', 
                              basefmt=' ', label='Theoretical PMF')
        
        axes_kde[idx].set_title(f'n = {n}')
        axes_kde[idx].set_xlabel('x')
        axes_kde[idx].set_ylabel('Density')
        axes_kde[idx].legend(fontsize=8)
        axes_kde[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'{dist_name} Distribution - Empirical CDF', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_ecdf.savefig(f'results_lab4/{dist_name.lower()}_ecdf.png', dpi=300, bbox_inches='tight')
    plt.close(fig_ecdf)
    
    plt.suptitle(f'{dist_name} Distribution - Kernel Density Estimate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_kde.savefig(f'results_lab4/{dist_name.lower()}_kde.png', dpi=300, bbox_inches='tight')
    plt.close(fig_kde)
    
    print(f"  Saved ECDF and KDE plots for {dist_name}")

print("\n" + "=" * 60)
print("Creating comparison plots...")
print("=" * 60)

n = 100
fig_comparison, axes = plt.subplots(5, 2, figsize=(12, 15))

for idx, (dist_name, dist_func) in enumerate(distributions.items()):
    sample = dist_func(n)
    
    if dist_name == 'Cauchy':
        sample = sample[(sample > -10) & (sample < 10)]
        sample = sample[:n]
    
    x_min, x_max = x_ranges[dist_name]
    sample = sample[(sample >= x_min) & (sample <= x_max)]
    
    x_ecdf = np.linspace(x_min, x_max, 1000)
    y_ecdf = np.array([np.mean(sample <= xi) for xi in x_ecdf])
    
    if dist_name == 'Normal':
        y_theoretical_cdf = stats.norm.cdf(x_ecdf, loc=0, scale=1)
        y_theoretical_pdf = stats.norm.pdf(x_ecdf, loc=0, scale=1)
    elif dist_name == 'Cauchy':
        y_theoretical_cdf = stats.cauchy.cdf(x_ecdf, loc=0, scale=1)
        y_theoretical_pdf = stats.cauchy.pdf(x_ecdf, loc=0, scale=1)
    elif dist_name == 'Laplace':
        y_theoretical_cdf = stats.laplace.cdf(x_ecdf, loc=0, scale=1/np.sqrt(2))
        y_theoretical_pdf = stats.laplace.pdf(x_ecdf, loc=0, scale=1/np.sqrt(2))
    elif dist_name == 'Poisson':
        x_poisson = np.arange(x_min, x_max+1)
        y_theoretical_cdf = stats.poisson.cdf(x_poisson, mu=10)
        y_theoretical_pdf = stats.poisson.pmf(x_poisson, mu=10)
        x_ecdf = x_poisson
    elif dist_name == 'Uniform':
        y_theoretical_cdf = stats.uniform.cdf(x_ecdf, loc=-np.sqrt(3), scale=2*np.sqrt(3))
        y_theoretical_pdf = stats.uniform.pdf(x_ecdf, loc=-np.sqrt(3), scale=2*np.sqrt(3))
    
    axes[idx, 0].step(np.sort(sample), np.arange(1, len(sample)+1) / len(sample), 
                     where='post', label='Empirical CDF', color='blue', linewidth=1.5)
    if dist_name == 'Poisson':
        axes[idx, 0].stairs(np.cumsum(y_theoretical_pdf), np.arange(x_min, x_max+2), 
                           label='Theoretical CDF', color='red', linewidth=1.5)
    else:
        axes[idx, 0].plot(x_ecdf, y_theoretical_cdf, 'r-', linewidth=2, label='Theoretical CDF')
    axes[idx, 0].set_title(f'{dist_name} - ECDF')
    axes[idx, 0].set_xlabel('x')
    axes[idx, 0].set_ylabel('F(x)')
    axes[idx, 0].legend(fontsize=7)
    axes[idx, 0].grid(True, alpha=0.3)
    
    axes[idx, 1].hist(sample, bins='sqrt', density=True, alpha=0.3, 
                     color='gray', edgecolor='black', label='Histogram')
    if dist_name != 'Poisson':
        kde = gaussian_kde(sample)
        y_kde = kde(x_ecdf)
        axes[idx, 1].plot(x_ecdf, y_kde, 'b-', linewidth=2, label='Kernel Density')
        axes[idx, 1].plot(x_ecdf, y_theoretical_pdf, 'r--', linewidth=2, label='Theoretical PDF')
    else:
        axes[idx, 1].stem(x_poisson, y_theoretical_pdf, linefmt='r--', markerfmt='ro', 
                         basefmt=' ', label='Theoretical PMF')
    axes[idx, 1].set_title(f'{dist_name} - KDE')
    axes[idx, 1].set_xlabel('x')
    axes[idx, 1].set_ylabel('Density')
    axes[idx, 1].legend(fontsize=7)
    axes[idx, 1].grid(True, alpha=0.3)

plt.suptitle('Comparison of All Distributions (n=100)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results_lab4/all_distributions_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig_comparison)
