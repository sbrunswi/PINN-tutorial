"""
LAP_M2_sweep.py — follows the same structure as the notebook.
Usage:
    python LAP_M2_sweep.py
    python LAP_M2_sweep.py --m2_values 3 5 10 20
    python LAP_M2_sweep.py --data_path /path/to/massdamper_data.pkl
"""

import argparse
import numpy as np
import functools
import pickle
import time
import torch

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument('--m2_values', type=int, nargs='+', default=[3, 5, 10, 20])
parser.add_argument('--data_path', type=str,
                    default='/Users/anshgrover/AIDA3/PINN-tutorial/data/massdamper_data.pkl')
parser.add_argument('--output',    type=str, default='sweep_results.pkl')
parser.add_argument('--n_samples', type=int, default=1000)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# --- Data ---
with open(args.data_path, 'rb') as f:
    data = pickle.load(f)

data_sampled_random_with_noise = data['data_sampled_random_with_noise']
model_name = 'No external force'
sparsity   = 'sparcity level 100'

t_obs = torch.tensor(data_sampled_random_with_noise[model_name][sparsity]['t'], dtype=torch.float64).to(device)
y_obs = torch.tensor(data_sampled_random_with_noise[model_name][sparsity]['x'], dtype=torch.float64).to(device)
n = len(t_obs)
p = 1

# --- Physical parameters ---
m_true      = 2.0
c_true      = 0.3
k_true      = 0.2
delta_true  = c_true / (2 * m_true)
omega0_true = np.sqrt(k_true / m_true)
x0          = [1.0, 0.0]
print(f"delta_true={delta_true:.4f}  omega0_true={omega0_true:.4f}")

# --- Theorem 2 step size ---
K              = 4
h              = (t_obs[1] - t_obs[0]).item()
theorem2_bound = n ** (-5 / (2 * K))
m_sub          = int(np.ceil(h / theorem2_bound))
print(f"m_sub={m_sub}  step_size={h/m_sub:.4f}s")

# --- Prior hyperparameters ---
mu_x1 = y_obs[0].item()
c     = 100.0
a     = 0.1
b     = 0.01

# --- ODE ---
def msd(t, y, delta, omega0):
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float64)
    x, v = y[0], y[1]
    return torch.stack([v, -2*delta*v - omega0**2*x])

# --- RK4 ---
def rk4_integrate(f, t_eval, y0, m_sub):
    if not torch.is_tensor(y0):
        y0 = torch.tensor(y0, dtype=torch.float64, device=device)
    else:
        y0 = y0.to(device)
    if not torch.is_tensor(t_eval):
        t_eval = torch.tensor(t_eval, dtype=torch.float64, device=device)
    else:
        t_eval = t_eval.to(device)
    Y = [y0]
    y = y0.to(device)
    for i in range(1, len(t_eval)):
        dt = (t_eval[i] - t_eval[i-1]).item()
        h  = dt / m_sub
        yi = y
        ti = t_eval[i-1].item()
        for _ in range(m_sub):
            k1 = f(ti,       yi            )
            k2 = f(ti + h/2, yi + h/2 * k1)
            k3 = f(ti + h/2, yi + h/2 * k2)
            k4 = f(ti + h,   yi + h   * k3)
            yi = yi + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            ti += h
        y = yi
        Y.append(y)
    return torch.stack(Y)

# --- g_n ---
def compute_gn(x_pred, y_obs):
    if not torch.is_tensor(x_pred):
        x_pred = torch.tensor(x_pred, dtype=torch.float64)
    if not torch.is_tensor(y_obs):
        y_obs  = torch.tensor(y_obs,  dtype=torch.float64)
    return torch.mean((y_obs - x_pred)**2)

# --- find x1hat ---
def find_x1hat_torch(theta):
    delta, omega0 = theta[0], theta[1]
    x1 = torch.tensor([mu_x1], requires_grad=True, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([x1], lr=1.0, max_iter=100)
    def closure():
        optimizer.zero_grad()
        f    = functools.partial(msd, delta=delta, omega0=omega0)
        Y    = rk4_integrate(f, t_obs,
                             torch.stack([x1[0], torch.tensor(0.0, dtype=torch.float64)]),
                             m_sub)
        gn   = compute_gn(Y[:, 0], y_obs)
        loss = n * gn + (x1[0] - mu_x1)**2 / c
        loss.backward()
        return loss
    optimizer.step(closure)
    return x1[0]

# --- compute_u ---
def compute_u(x1_hat, theta):
    delta, omega0 = theta
    f      = functools.partial(msd, delta=delta, omega0=omega0)
    Y      = rk4_integrate(f, t_obs, [x1_hat, 0.0], m_sub)
    gn     = compute_gn(Y[:, 0], y_obs)
    u      = n * gn + (x1_hat - mu_x1)**2 / c
    return u, Y[:, 0]

# --- g_n_dd via autograd ---
def compute_gn_dd_autograd(x1_val, theta):
    delta, omega0 = theta
    x1     = torch.tensor(x1_val, requires_grad=True, dtype=torch.float64)
    y0     = torch.stack([x1, torch.tensor(0.0, dtype=torch.float64)])
    f      = lambda t, y: msd(t, y, delta, omega0)
    Y      = rk4_integrate(f, t_obs, y0, m_sub)
    gn     = compute_gn(Y[:, 0], y_obs)
    dgn    = torch.autograd.grad(gn, x1, create_graph=True)[0]
    d2gn   = torch.autograd.grad(dgn, x1)[0]
    return d2gn.item()

# --- lap_posterior ---
def lap_posterior(theta):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.float64, device=device)
    else:
        theta = theta.to(device)
    delta, omega0   = theta[0], theta[1]
    theta_t         = (delta, omega0)
    x1_hat          = find_x1hat_torch(theta).detach()
    u, _            = compute_u(x1_hat, theta_t)
    if u <= 0:
        return torch.tensor(-torch.inf, dtype=torch.float64)
    gn_dd = compute_gn_dd_autograd(x1_hat.item(), theta_t)
    H     = n * gn_dd + 2.0 / c
    if H <= 0:
        return torch.tensor(-torch.inf, dtype=torch.float64)
    v      = torch.log(torch.tensor(H, dtype=torch.float64))
    log_pi = -(n * p / 2 + a) * torch.log(u / 2 + b) - 0.5 * v
    return log_pi

def lap_posterior_u(theta):
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.float64, device=device)
    else:
        theta = theta.to(device)
    delta, omega0   = theta[0], theta[1]
    theta_t         = (delta, omega0)
    x1_hat          = find_x1hat_torch(theta).detach()
    u, _            = compute_u(x1_hat, theta_t)
    if u <= 0:
        return torch.tensor(-torch.inf, dtype=torch.float64), torch.tensor(float('nan'))
    gn_dd = compute_gn_dd_autograd(x1_hat.item(), theta_t)
    H     = n * gn_dd + 2.0 / c
    if H <= 0:
        return torch.tensor(-torch.inf, dtype=torch.float64), torch.tensor(float('nan'))
    v      = torch.log(torch.tensor(H, dtype=torch.float64))
    log_pi = -(n * p / 2 + a) * torch.log(u / 2 + b) - 0.5 * v
    return log_pi, u

# --- coarse-to-fine theta0 search ---
def coarse_to_fine_sweep(delta_range, omega0_range, n_points=5):
    best_lp, best_theta = -np.inf, None
    for d in np.linspace(*delta_range,  n_points):
        for w in np.linspace(*omega0_range, n_points):
            lp = lap_posterior(torch.tensor([d, w], dtype=torch.float64)).item()
            if lp > best_lp:
                best_lp, best_theta = lp, np.array([d, w])
    print(f"  best: delta={best_theta[0]:.4f} omega0={best_theta[1]:.4f} log_pi={best_lp:.2f}")
    return best_theta

print("\nFinding theta0...")
best1  = coarse_to_fine_sweep((0.01, 0.2),                    (0.1, 0.6),                    n_points=5)
best2  = coarse_to_fine_sweep((best1[0]-0.05, best1[0]+0.05), (best1[1]-0.10, best1[1]+0.10), n_points=5)
best3  = coarse_to_fine_sweep((best2[0]-0.01, best2[0]+0.01), (best2[1]-0.02, best2[1]+0.02), n_points=5)
theta0 = torch.tensor(best3, dtype=torch.float64)
print(f"theta0 = {theta0.numpy()}")

# --- LAP sweep function ---
def LAP_M2_sweep(theta0, M2):
    theta0_np = np.array([float(theta0[0]), float(theta0[1])])

    H     = torch.autograd.functional.hessian(lap_posterior,
                torch.tensor(theta0_np, dtype=torch.float64))
    neg_H = -H.numpy()
    eigenvalues, U = np.linalg.eigh(neg_H)
    min_pos        = eigenvalues[eigenvalues > 0].min()
    eigenvalues    = np.where(eigenvalues <= 0, min_pos, eigenvalues)
    A              = U @ np.diag(eigenvalues ** -0.5)

    M1, eta = 3, 1e-5
    z_1d    = np.linspace(-4, 4, 2*M1 + 1)
    A_bounds, B_bounds = [], []
    for dim in range(len(theta0_np)):
        active = []
        for z_val in z_1d:
            z          = np.zeros(len(theta0_np)); z[dim] = z_val
            theta_test = theta0_np + A @ z
            lp         = lap_posterior(theta_test)
            if float(lp) > np.log(eta):
                active.append(z_val)
        A_bounds.append(min(active) if active else -4)
        B_bounds.append(max(active) if active else  4)
    print(f"  coarse bounds: z[0]=[{A_bounds[0]:.2f},{B_bounds[0]:.2f}]  z[1]=[{A_bounds[1]:.2f},{B_bounds[1]:.2f}]")

    z0_1d = np.linspace(A_bounds[0], B_bounds[0], 2*M2 + 1)
    z1_1d = np.linspace(A_bounds[1], B_bounds[1], 2*M2 + 1)
    Z0, Z1  = np.meshgrid(z0_1d, z1_1d)
    log_pi  = np.zeros_like(Z0)
    u_grid  = np.zeros_like(Z0)

    for i in range(Z0.shape[0]):
        for j in range(Z0.shape[1]):
            z           = np.array([Z0[i,j], Z1[i,j]])
            theta_ij    = theta0_np + A @ z
            lp, u       = lap_posterior_u(theta_ij)
            log_pi[i,j] = float(lp)
            u_grid[i,j] = float(u)
        print(f"  row {i+1}/{Z0.shape[0]} done", flush=True)

    log_pi_shifted = log_pi - np.max(log_pi)
    pi             = np.exp(log_pi_shifted)
    pi            /= pi.sum()

    flat_pi       = pi.flatten()
    flat_idx      = np.random.choice(len(flat_pi), size=args.n_samples, p=flat_pi)
    i_idx, j_idx  = np.unravel_index(flat_idx, pi.shape)
    theta_samples = np.array([theta0_np + A @ np.array([Z0[i,j], Z1[i,j]])
                               for i, j in zip(i_idx, j_idx)])
    u_flat       = u_grid.flatten()[flat_idx]
    tau2_samples = np.array([
        np.random.gamma(shape=n*p/2 + a, scale=1.0 / (u_flat[i] / 2 + b))
        for i in range(args.n_samples)
    ])

    return tau2_samples, theta_samples, Z0, Z1, A, log_pi, u_grid, pi

# --- run sweep ---
all_results = []

for M2 in args.m2_values:
    print(f"\n{'='*50}")
    print(f"M2 = {M2}   grid = {(2*M2+1)**2} points")
    print('='*50)

    t_start = time.time()
    tau2_samples, theta_samples, Z0, Z1, A, log_pi, u_grid, pi = LAP_M2_sweep(theta0, M2)
    runtime = time.time() - t_start

    d_mean  = np.mean(theta_samples[:,0])
    w_mean  = np.mean(theta_samples[:,1])
    d_ci    = np.percentile(theta_samples[:,0], [2.5, 97.5])
    w_ci    = np.percentile(theta_samples[:,1], [2.5, 97.5])

    result = {
        'M2': M2, 'n_grid': (2*M2+1)**2, 'runtime': runtime,
        'd_mean': d_mean,              'w_mean': w_mean,
        'd_bias': d_mean-delta_true,   'w_bias': w_mean-omega0_true,
        'd_width': d_ci[1]-d_ci[0],   'w_width': w_ci[1]-w_ci[0],
        'd_ci': d_ci, 'w_ci': w_ci,
        'theta_samples': theta_samples, 'tau2_samples': tau2_samples,
        'log_pi': log_pi, 'Z0': Z0, 'Z1': Z1, 'A': A, 'pi': pi,
        'theta0': theta0.numpy(),
    }
    all_results.append(result)

    print(f"runtime : {runtime:.1f}s")
    print(f"d  : mean={d_mean:.4f}  bias={d_mean-delta_true:+.4f}  95%CI=[{d_ci[0]:.4f},{d_ci[1]:.4f}]  width={d_ci[1]-d_ci[0]:.4f}")
    print(f"w0 : mean={w_mean:.4f}  bias={w_mean-omega0_true:+.4f}  95%CI=[{w_ci[0]:.4f},{w_ci[1]:.4f}]  width={w_ci[1]-w_ci[0]:.4f}")

# --- save ---
with open(args.output, 'wb') as f:
    pickle.dump({'results': all_results, 'delta_true': delta_true,
                 'omega0_true': omega0_true, 'args': vars(args)}, f)
print(f"\nResults saved to {args.output}")