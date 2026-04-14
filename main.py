import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

def generate_x_values(start=-1.8, end=2.0, step=0.2):
    return np.arange(start, end + step, step)

def generate_y_values(x, a_true=2.0, b_true=2.0, sigma=1.0):
    epsilon = np.random.normal(0, sigma, len(x))
    y = a_true + b_true * x + epsilon
    return y

def add_outliers(y, indices, values):
    y_outlier = y.copy()
    for idx, val in zip(indices, values):
        y_outlier[idx] += val
    return y_outlier

def least_squares_method(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    b_hat = numerator / denominator
    a_hat = y_mean - b_hat * x_mean
    
    return a_hat, b_hat

def least_absolute_deviations_method(x, y):
    def objective(params):
        a, b = params
        residuals = y - (a + b * x)
        return np.sum(np.abs(residuals))
    
    initial_guess = [0, 0]
    result = minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-6)
    
    a_hat, b_hat = result.x
    return a_hat, b_hat

def calculate_errors(a_hat, b_hat, a_true=2.0, b_true=2.0):
    delta_a = abs(a_hat - a_true)
    delta_b = abs(b_hat - b_true)
    
    relative_error_a = (delta_a / a_true) * 100
    relative_error_b = (delta_b / b_true) * 100
    
    return delta_a, relative_error_a, delta_b, relative_error_b

def plot_regression(x, y, a, b, title, filename, outliers=False):
    plt.figure(figsize=(10, 6))
    
    if outliers:
        plt.scatter(x, y, color='red', alpha=0.6, label='Data with outliers', zorder=2)
    else:
        plt.scatter(x, y, color='blue', alpha=0.6, label='Data', zorder=2)
    
    y_pred = a + b * x
    plt.plot(x, y_pred, 'g-', linewidth=2, label=f'Regression: y = {a:.2f} + {b:.2f}x')
    
    true_line = 2 + 2 * x
    plt.plot(x, true_line, 'r--', linewidth=1.5, label='True: y = 2 + 2x', alpha=0.7)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_simulation():
    np.random.seed(42)
    
    a_true = 2.0
    b_true = 2.0
    
    print("=" * 100)
    print("LABORATORY WORK #6: LINEAR REGRESSION")
    print("=" * 100)
    
    x = generate_x_values()
    y = generate_y_values(x, a_true, b_true)
    
    print("\nCASE 1: WITHOUT OUTLIERS")
    print("-" * 100)
    
    a_mls, b_mls = least_squares_method(x, y)
    a_mlad, b_mlad = least_absolute_deviations_method(x, y)
    
    delta_a_mls, rel_a_mls, delta_b_mls, rel_b_mls = calculate_errors(a_mls, b_mls, a_true, b_true)
    delta_a_mlad, rel_a_mlad, delta_b_mlad, rel_b_mlad = calculate_errors(a_mlad, b_mlad, a_true, b_true)
    
    print(f"{'Method':<10} {'a':<12} {'Δa':<12} {'δa, %':<12} {'b':<12} {'Δb':<12} {'δb, %':<12}")
    print("-" * 100)
    print(f"{'MLS':<10} {a_mls:<12.4f} {delta_a_mls:<12.4f} {rel_a_mls:<12.2f} {b_mls:<12.4f} {delta_b_mls:<12.4f} {rel_b_mls:<12.2f}")
    print(f"{'MLAD':<10} {a_mlad:<12.4f} {delta_a_mlad:<12.4f} {rel_a_mlad:<12.2f} {b_mlad:<12.4f} {delta_b_mlad:<12.4f} {rel_b_mlad:<12.2f}")
    
    plot_regression(x, y, a_mls, b_mls, 'Least Squares Method (No Outliers)', 'regression_mls_no_outliers.png')
    plot_regression(x, y, a_mlad, b_mlad, 'Least Absolute Deviations (No Outliers)', 'regression_mlad_no_outliers.png')
    
    print("\nCASE 2: WITH OUTLIERS (y[0] += 10, y[19] -= 10)")
    print("-" * 100)
    
    y_outliers = add_outliers(y, [0, 19], [10, -10])
    
    a_mls_out, b_mls_out = least_squares_method(x, y_outliers)
    a_mlad_out, b_mlad_out = least_absolute_deviations_method(x, y_outliers)
    
    delta_a_mls_out, rel_a_mls_out, delta_b_mls_out, rel_b_mls_out = calculate_errors(a_mls_out, b_mls_out, a_true, b_true)
    delta_a_mlad_out, rel_a_mlad_out, delta_b_mlad_out, rel_b_mlad_out = calculate_errors(a_mlad_out, b_mlad_out, a_true, b_true)
    
    print(f"{'Method':<10} {'a':<12} {'Δa':<12} {'δa, %':<12} {'b':<12} {'Δb':<12} {'δb, %':<12}")
    print("-" * 100)
    print(f"{'MLS':<10} {a_mls_out:<12.4f} {delta_a_mls_out:<12.4f} {rel_a_mls_out:<12.2f} {b_mls_out:<12.4f} {delta_b_mls_out:<12.4f} {rel_b_mls_out:<12.2f}")
    print(f"{'MLAD':<10} {a_mlad_out:<12.4f} {delta_a_mlad_out:<12.4f} {rel_a_mlad_out:<12.2f} {b_mlad_out:<12.4f} {delta_b_mlad_out:<12.4f} {rel_b_mlad_out:<12.2f}")
    
    plot_regression(x, y_outliers, a_mls_out, b_mls_out, 'Least Squares Method (With Outliers)', 
                   'regression_mls_with_outliers.png', outliers=True)
    plot_regression(x, y_outliers, a_mlad_out, b_mlad_out, 'Least Absolute Deviations (With Outliers)', 
                   'regression_mlad_with_outliers.png', outliers=True)
    
    print("\n" + "=" * 100)
    print("COMPARISON: ROBUSTNESS TO OUTLIERS")
    print("=" * 100)
    
    print("\nLeast Squares Method:")
    print(f"  Change in a: {abs(a_mls_out - a_mls):.4f}")
    print(f"  Change in b: {abs(b_mls_out - b_mls):.4f}")
    
    print("\nLeast Absolute Deviations:")
    print(f"  Change in a: {abs(a_mlad_out - a_mlad):.4f}")
    print(f"  Change in b: {abs(b_mlad_out - b_mlad):.4f}")
    
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    
    print("\nWITHOUT OUTLIERS:")
    print(f"{'Method':<10} {'a':<12} {'Δa':<12} {'δa, %':<12} {'b':<12} {'Δb':<12} {'δb, %':<12}")
    print("-" * 100)
    print(f"{'MLS':<10} {a_mls:<12.4f} {delta_a_mls:<12.4f} {rel_a_mls:<12.2f} {b_mls:<12.4f} {delta_b_mls:<12.4f} {rel_b_mls:<12.2f}")
    print(f"{'MLAD':<10} {a_mlad:<12.4f} {delta_a_mlad:<12.4f} {rel_a_mlad:<12.2f} {b_mlad:<12.4f} {delta_b_mlad:<12.4f} {rel_b_mlad:<12.2f}")
    
    print("\nWITH OUTLIERS:")
    print(f"{'Method':<10} {'a':<12} {'Δa':<12} {'δa, %':<12} {'b':<12} {'Δb':<12} {'δb, %':<12}")
    print("-" * 100)
    print(f"{'MLS':<10} {a_mls_out:<12.4f} {delta_a_mls_out:<12.4f} {rel_a_mls_out:<12.2f} {b_mls_out:<12.4f} {delta_b_mls_out:<12.4f} {rel_b_mls_out:<12.2f}")
    print(f"{'MLAD':<10} {a_mlad_out:<12.4f} {delta_a_mlad_out:<12.4f} {rel_a_mlad_out:<12.2f} {b_mlad_out:<12.4f} {delta_b_mlad_out:<12.4f} {rel_b_mlad_out:<12.2f}")
    
    return {
        'x': x,
        'y_clean': y,
        'y_outliers': y_outliers,
        'results': {
            'mls_no_outliers': (a_mls, b_mls, delta_a_mls, rel_a_mls, delta_b_mls, rel_b_mls),
            'mlad_no_outliers': (a_mlad, b_mlad, delta_a_mlad, rel_a_mlad, delta_b_mlad, rel_b_mlad),
            'mls_with_outliers': (a_mls_out, b_mls_out, delta_a_mls_out, rel_a_mls_out, delta_b_mls_out, rel_b_mls_out),
            'mlad_with_outliers': (a_mlad_out, b_mlad_out, delta_a_mlad_out, rel_a_mlad_out, delta_b_mlad_out, rel_b_mlad_out)
        }
    }

if __name__ == "__main__":
    results = run_simulation()