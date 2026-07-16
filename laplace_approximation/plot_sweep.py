"""
plot_sweep.py
=============
Loads sweep_results.pkl and generates all plots.
Run this locally after scp-ing the results back from Gilbreth.

Usage:
    python plot_sweep.py
    python plot_sweep.py --results sweep_results.pkl
"""

import argparse
import pickle
import numpy as np
from matplotlib import pyplot as pp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='sweep_results.pkl')
    return parser.parse_args()

def main():
    args    = parse_args()
    with open(args.results, 'rb') as f:
        data = pickle.load(f)

    results     = data['results']
    delta_true  = data['delta_true']
    omega0_true = data['omega0_true']

    M2_vals  = [r['M2']          for r in results]
    d_biases = [abs(r['d_bias']) for r in results]
    w_biases = [abs(r['w_bias']) for r in results]
    d_widths = [r['d_width']     for r in results]
    w_widths = [r['w_width']     for r in results]
    runtimes = [r['runtime']     for r in results]

    # --- Plot 1: convergence metrics ---
    fig, axes = pp.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(M2_vals, d_biases, 'o-', label='|bias| δ')
    axes[0].plot(M2_vals, w_biases, 's-', label='|bias| ω₀')
    axes[0].set_xlabel('M2'); axes[0].set_title('Bias vs M2')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(M2_vals, d_widths, 'o-', label='95% CI width δ')
    axes[1].plot(M2_vals, w_widths, 's-', label='95% CI width ω₀')
    axes[1].set_xlabel('M2'); axes[1].set_title('95% CI width vs M2')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(M2_vals, runtimes, 'o-', color='gray')
    axes[2].set_xlabel('M2'); axes[2].set_title('Runtime [s] vs M2')
    axes[2].grid(True)

    pp.suptitle(f'M2 sweep  (true δ={delta_true:.4f}, ω₀={omega0_true:.4f})')
    pp.tight_layout()
    pp.savefig('sweep_convergence.png', dpi=150)
    pp.show()

    # --- Plot 2: marginal posteriors for each M2 ---
    n_M2 = len(results)
    fig, axes = pp.subplots(n_M2, 2, figsize=(10, 3 * n_M2))
    axes = np.array(axes).reshape(n_M2, 2)

    for row, r in enumerate(results):
        axes[row,0].hist(r['theta_samples'][:,0], bins=30, density=True,
                         color='steelblue', alpha=0.8)
        axes[row,0].axvline(delta_true,        color='r', linestyle='--', label='true')
        axes[row,0].axvline(r['d_mean'],        color='k', linestyle=':',  label='est')
        axes[row,0].set_title(f'M2={r["M2"]} — δ marginal')
        axes[row,0].legend()

        axes[row,1].hist(r['theta_samples'][:,1], bins=30, density=True,
                         color='steelblue', alpha=0.8)
        axes[row,1].axvline(omega0_true,       color='r', linestyle='--', label='true')
        axes[row,1].axvline(r['w_mean'],        color='k', linestyle=':',  label='est')
        axes[row,1].set_title(f'M2={r["M2"]} — ω₀ marginal')
        axes[row,1].legend()

    pp.tight_layout()
    pp.savefig('sweep_marginals.png', dpi=150)
    pp.show()

    # --- Print summary table ---
    print(f"\n{'M2':>4}  {'grid':>6}  {'runtime':>8}  "
          f"{'d_bias':>8}  {'w_bias':>8}  {'d_CI_w':>8}  {'w_CI_w':>8}")
    print('-' * 65)
    for r in results:
        print(f"{r['M2']:>4}  {r['n_grid']:>6}  {r['runtime']:>7.1f}s  "
              f"{r['d_bias']:>+8.4f}  {r['w_bias']:>+8.4f}  "
              f"{r['d_width']:>8.4f}  {r['w_width']:>8.4f}")

if __name__ == '__main__':
    main()