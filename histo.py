import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 10

sample_sizes = [10, 100, 1000]
np.random.seed(42)  


print("=" * 60)
print("1. Normal distribution N(0, 1)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n in enumerate(sample_sizes):
    sample = np.random.normal(loc=0, scale=1, size=n)
    
    axes[idx].hist(sample, bins='sqrt', density=True, alpha=0.6, 
                   color='skyblue', edgecolor='black', label='Historam')
    
    x = np.linspace(-4, 4, 1000)
    pdf = stats.norm.pdf(x, loc=0, scale=1)
    axes[idx].plot(x, pdf, 'r-', linewidth=2, label='Theoretical density')
    
    axes[idx].set_title(f'n = {n}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('density')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Normal distribution N(0, 1)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('normal_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("2. Cauchy distribution C(0, 1)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n in enumerate(sample_sizes):
    sample = np.random.standard_cauchy(size=n)
    sample = sample[(sample > -10) & (sample < 10)]
    
    axes[idx].hist(sample, bins='sqrt', density=True, alpha=0.6, 
                   color='lightcoral', edgecolor='black', label='Histogram')
    
    x = np.linspace(-10, 10, 1000)
    pdf = stats.cauchy.pdf(x, loc=0, scale=1)
    axes[idx].plot(x, pdf, 'r-', linewidth=2, label='Theoretical density')
    
    axes[idx].set_title(f'n = {n}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('density')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(-10, 10)

plt.suptitle('Cauchy distribution C(0, 1)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cauchy_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("3. Laplace distribution  L(0, 1/√2)")
print("=" * 60)

b = 1/np.sqrt(2)  

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n in enumerate(sample_sizes):
    sample = np.random.laplace(loc=0, scale=b, size=n)
    
    axes[idx].hist(sample, bins='sqrt', density=True, alpha=0.6, 
                   color='lightgreen', edgecolor='black', label='Histogram')
    
    x = np.linspace(-5, 5, 1000)
    pdf = stats.laplace.pdf(x, loc=0, scale=b)
    axes[idx].plot(x, pdf, 'r-', linewidth=2, label='Theoretical density')
    
    axes[idx].set_title(f'n = {n}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('density')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle(r'Laplace distribution $L(0, 1/\sqrt{2})$', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('laplace_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("4. Poisson distribution P(10)")
print("=" * 60)

mu = 10

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n in enumerate(sample_sizes):
    sample = np.random.poisson(lam=mu, size=n)
    
    bins = np.arange(min(sample), max(sample) + 2) - 0.5
    axes[idx].hist(sample, bins=bins, density=True, alpha=0.6, 
                   color='orange', edgecolor='black', label='Histogram', rwidth=0.8)
    
    x = np.arange(0, 21)
    pmf = stats.poisson.pmf(x, mu=mu)
    axes[idx].stem(x, pmf, linefmt='r-', markerfmt='ro', basefmt=' ', label='Theoretical')
    
    axes[idx].set_title(f'n = {n}')
    axes[idx].set_xlabel('k')
    axes[idx].set_ylabel('Probability')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Poisson distribution P(10)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('poisson_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("5. Uniform distribution U(-√3, √3)")
print("=" * 60)

a = -np.sqrt(3)
b = np.sqrt(3)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n in enumerate(sample_sizes):
    sample = np.random.uniform(low=a, high=b, size=n)
    
    axes[idx].hist(sample, bins='sqrt', density=True, alpha=0.6, 
                   color='mediumpurple', edgecolor='black', label='Histogram')
    
    x = np.linspace(-3, 3, 1000)
    pdf = stats.uniform.pdf(x, loc=a, scale=b-a)
    axes[idx].plot(x, pdf, 'r-', linewidth=2, label='Theoretical density', zorder=5)
    
    axes[idx].set_title(f'n = {n}')
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('density')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(-3, 3)

plt.suptitle(r'Uniform distribution $U(-\sqrt{3}, \sqrt{3})$', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('uniform_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Statistics alalysis")
print("=" * 60)

distributions = {
    'Normal': lambda n: np.random.normal(0, 1, n),
    'Cauchy': lambda n: np.random.standard_cauchy(n),
    'Laplace': lambda n: np.random.laplace(0, 1/np.sqrt(2), n),
    'Poisson': lambda n: np.random.poisson(10, n),
    'Uniform': lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
}

for dist_name, generator in distributions.items():
    print(f"\n{dist_name}:")
    print("-" * 40)
    for n in sample_sizes:
        sample = generator(n)
        if dist_name == 'Cauchy':
            sample = sample[(sample > -10) & (sample < 10)]
        
        print(f"  n={len(sample):4d}: mean={np.mean(sample):7.3f}, "
              f"std={np.std(sample):7.3f}, "
              f"min={np.min(sample):7.3f}, max={np.max(sample):7.3f}")

print("\n" + "=" * 60)
print("=" * 60)