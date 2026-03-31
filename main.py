import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

def generate_bivariate_normal(n, rho, mean=None, cov=None):
    if mean is None:
        mean = [0, 0]
    if cov is None:
        cov = [[1, rho], [rho, 1]]
    return np.random.multivariate_normal(mean, cov, n)

def generate_mixture(n, rho1=0.9, rho2=-0.9, weight1=0.9):
    n1 = int(n * weight1)
    n2 = n - n1
    
    sample1 = generate_bivariate_normal(n1, rho1, cov=[[1, rho1], [rho1, 1]])
    sample2 = generate_bivariate_normal(n2, rho2, cov=[[10, 10*rho2], [10*rho2, 10]])
    
    return np.vstack([sample1, sample2])

def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

def spearman_correlation(x, y):
    return spearmanr(x, y)[0]

def quadrant_correlation(x, y):
    median_x = np.median(x)
    median_y = np.median(y)
    
    q1 = np.sum((x > median_x) & (y > median_y))
    q2 = np.sum((x < median_x) & (y > median_y))
    q3 = np.sum((x < median_x) & (y < median_y))
    q4 = np.sum((x > median_x) & (y < median_y))
    
    n = len(x)
    return (q1 + q3 - q2 - q4) / n

def calculate_correlations(x, y):
    pearson = pearson_correlation(x, y)
    spearman = spearman_correlation(x, y)
    quadrant = quadrant_correlation(x, y)
    return pearson, spearman, quadrant
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    vals, vecs = np.linalg.eigh(cov)
    
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    theta = np.arctan2(vecs[1, 0], vecs[0, 0])
    
    width, height = 2 * n_std * np.sqrt(vals)
    
    ell = Ellipse((np.mean(x), np.mean(y)), width, height, 
                  angle=np.degrees(theta), facecolor=facecolor, **kwargs)
    
    return ax.add_patch(ell)

def plot_scatter_with_ellipse(x, y, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.5, s=10, edgecolors='none')
    confidence_ellipse(x, y, ax, n_std=2, edgecolor='red', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def run_simulation():
    np.random.seed(42)
    
    sample_sizes = [20, 60, 100]
    rho_values = [0, 0.5, 0.9]
    n_repetitions = 1000
    
    results = {}
    
    print("=" * 80)
    print("BIVARIATE NORMAL DISTRIBUTION")
    print("=" * 80)
    
    for n in sample_sizes:
        for rho in rho_values:
            pearson_results = []
            spearman_results = []
            quadrant_results = []
            
            for _ in range(n_repetitions):
                sample = generate_bivariate_normal(n, rho)
                x, y = sample[:, 0], sample[:, 1]
                
                p, s, q = calculate_correlations(x, y)
                pearson_results.append(p)
                spearman_results.append(s)
                quadrant_results.append(q)
            
            key = f"n{n}_rho{rho}"
            results[key] = {
                'pearson': {
                    'mean': np.mean(pearson_results),
                    'variance': np.var(pearson_results)
                },
                'spearman': {
                    'mean': np.mean(spearman_results),
                    'variance': np.var(spearman_results)
                },
                'quadrant': {
                    'mean': np.mean(quadrant_results),
                    'variance': np.var(quadrant_results)
                }
            }
            
            print(f"\nSample size: {n}, True rho: {rho}")
            print(f"Pearson:   mean = {results[key]['pearson']['mean']:.4f}, variance = {results[key]['pearson']['variance']:.6f}")
            print(f"Spearman:  mean = {results[key]['spearman']['mean']:.4f}, variance = {results[key]['spearman']['variance']:.6f}")
            print(f"Quadrant:  mean = {results[key]['quadrant']['mean']:.4f}, variance = {results[key]['quadrant']['variance']:.6f}")
            
            if n == 100:
                sample = generate_bivariate_normal(n, rho)
                x, y = sample[:, 0], sample[:, 1]
                plot_scatter_with_ellipse(x, y, f'Bivariate Normal (n={n}, rho={rho})', 
                                         f'bivariate_n{n}_rho{rho}.png')
    
    print("\n" + "=" * 80)
    print("MIXTURE DISTRIBUTION")
    print("=" * 80)
    
    for n in sample_sizes:
        pearson_results = []
        spearman_results = []
        quadrant_results = []
        
        for _ in range(n_repetitions):
            sample = generate_mixture(n)
            x, y = sample[:, 0], sample[:, 1]
            
            p, s, q = calculate_correlations(x, y)
            pearson_results.append(p)
            spearman_results.append(s)
            quadrant_results.append(q)
        
        key = f"mixture_n{n}"
        results[key] = {
            'pearson': {
                'mean': np.mean(pearson_results),
                'variance': np.var(pearson_results)
            },
            'spearman': {
                'mean': np.mean(spearman_results),
                'variance': np.var(spearman_results)
            },
            'quadrant': {
                'mean': np.mean(quadrant_results),
                'variance': np.var(quadrant_results)
            }
        }
        
        print(f"\nSample size: {n}")
        print(f"Pearson:   mean = {results[key]['pearson']['mean']:.4f}, variance = {results[key]['pearson']['variance']:.6f}")
        print(f"Spearman:  mean = {results[key]['spearman']['mean']:.4f}, variance = {results[key]['spearman']['variance']:.6f}")
        print(f"Quadrant:  mean = {results[key]['quadrant']['mean']:.4f}, variance = {results[key]['quadrant']['variance']:.6f}")
        
        if n == 100:
            sample = generate_mixture(n)
            x, y = sample[:, 0], sample[:, 1]
            plot_scatter_with_ellipse(x, y, f'Mixture Distribution (n={n})', 
                                     f'mixture_n{n}.png')
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)
    
    print("\nBivariate Normal Distribution Results:")
    print("-" * 80)
    print(f"{'n':<6} {'rho':<6} {'Pearson Mean':<15} {'Pearson Var':<15} {'Spearman Mean':<15}")
    print("-" * 80)
    for n in sample_sizes:
        for rho in rho_values:
            key = f"n{n}_rho{rho}"
            r = results[key]
            print(f"{n:<6} {rho:<6} {r['pearson']['mean']:<15.4f} {r['pearson']['variance']:<15.6f} {r['spearman']['mean']:<15.4f}")
    
    print("\nMixture Distribution Results:")
    print("-" * 80)
    print(f"{'n':<6} {'Pearson Mean':<15} {'Pearson Var':<15} {'Spearman Mean':<15}")
    print("-" * 80)
    for n in sample_sizes:
        key = f"mixture_n{n}"
        r = results[key]
        print(f"{n:<6} {r['pearson']['mean']:<15.4f} {r['pearson']['variance']:<15.6f} {r['spearman']['mean']:<15.4f}")
    
    return results

if __name__ == "__main__":
    results = run_simulation()