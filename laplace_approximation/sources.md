# Laplace Transform videos:

### 1. https://www.youtube.com/watch?v=7UvtU75NXTg:
Introduction to Laplace transform along with derivation from Fourier transform. Explained Laplace as a more generalized version of Fourier Transform that can work for any function. 

Transformation equations:
$$
\bar{f}(s) = \int_{0}^{\infin}f(t)e^{-st}dt
$$
$$f(s) = \frac{1}{2\pi i}\int_{\gamma-i\infin}^{\gamma+i\infin}\bar{f}(s)e^{st}ds$$



### 2. https://www.youtube.com/watch?v=5hPD7CF0_54
Laplace transform properties:
$$L\{\frac{df}{dt}\} = s\bar{f}(s) - \bar{f}(0)$$
$$L\{\frac{d^2f}{dt}\} = s^2\bar{f}(s) - s\bar{f}(0) - f'(0)$$
$$L\{f(t) * g(t)\} = \bar{f}(s) * \bar{g}(s)$$

Examples:

1. $$f(t) = 1$$
$$\bar{f}(s) = \int_{0}^{\infin}1 \cdot e^{-st}dt$$
$$=\bigg[\frac{-1}{s}e^{-st}\bigg]_0^{\infin}$$
$$=\frac{1}{s}$$
2. $$f(t) = e^{at}$$
$$\bar{f}(s) = \int_{0}^{\infin}e^{at} \cdot e^{-st}dt$$
$$= \int_{0}^{\infin}e^{(a-s)t}dt$$
$$=\bigg[\frac{1}{a-s}e^{(a-s)t}\bigg]_0^{\infin}$$
$$=\frac{1}{s-a}$$

### 3. https://www.youtube.com/watch?v=iBde8qOW0h0
Converting ordinary differential equations into algebraic equations using Laplace transform. 

Example:

ODE of spring mass damper system. Differential equation of this system:
$$\ddot{x} + \frac{c}{m}\dot{x} + \frac{k}{m}x = 0$$
Assume $m = 1, c = 5, k = 4$.

Initial conditions are $x(0) = 2, \dot{x}(0) = -5$. This means that position at time 0 was 2 and velocity was -5.

Next Laplace transform every term of this equation:
$$L\{\ddot{x}\} + 5L\{\dot{x}\} + 4\bar{x} = 0$$
$$ =  (s^2\bar{x} - sx(0) - \dot{x}(0)) + 5(s\bar{x} - x(0)) + 4\bar{x} = 0$$
$$(s^2 + 5s + 4)\bar{x}(s) = 2s + 5$$

In the above equation, the LHS is the characteristic polynomial of the ODE and the RHS is the initial conditions. 

Now,
$$\bar{x}(s) = \frac{2s+5}{s^2+5s+4}$$
We know that,
$$L\{e^{at}\} = \frac{1}{s-a}$$
Using this,
$$\frac{2s+5}{s^2+5s+4} = \frac{2s+5}{(s+4)(s+1)}$$
$$= \frac{1}{s+4} + \frac{1}{s+1}$$
Now we have the equation as a sum of two easy functions that we can inverse Laplace transform.

So the solution can be simplified to:
$$x(t) = e^{-4t} + e^{-t}$$

In the example above, the denominator factored into real roots $(s+4)(s+1)$, making partial fractions straightforward. This is because the system was overdamped ( friction dominates and the system slides back to rest without oscillating.)

When the system is underdamped, the spring dominates over friction and the system oscillates before settling. This means the roots of the characteristic polynomial are complex, and we cannot simply decompose the denominator as we did above.

For a general second order ODE:
$$\ddot{x} + \frac{c}{m}\dot{x} + \frac{k}{m}x = 0$$

We define:
- $\delta = \frac{c}{2m}$  (damping factor)
- $\omega_0 = \sqrt{\frac{k}{m}}$ (natural frequency)
- $\omega = \sqrt{\omega_0^2 - \delta^2}$ (actual observed oscillation due to damping)

The system is underdamped when $\delta < \omega_0$. In this case the denominator does not factor into real roots, so instead we complete the square:

$$s^2 + 2\delta s + \omega_0^2 = (s+\delta)^2 + \omega^2$$

To invert this, we use Euler's formula $e^{i\omega t} = \cos(\omega t) + i\sin(\omega t)$, which gives two standard Laplace pairs:

$$L\{e^{-\delta t}\cos(\omega t)\} = \frac{s+\delta}{(s+\delta)^2+\omega^2} \qquad L\{e^{-\delta t}\sin(\omega t)\} = \frac{\omega}{(s+\delta)^2+\omega^2}$$

In the notebook, we use $x(0)=1, \dot{x}(0)=0$. This gives the solution:

$$y(t) = e^{-\delta t}\cos(\omega t) + \frac{\delta}{\omega}e^{-\delta t}\sin(\omega t)$$

# Laplace Based Approximate Posterior Inference for Differential Equation Models

This paper proposed an approximation method for obtaining the posterior distribution of parameters in differential equation models using classic numerical solvers like Runge-Kutta and Laplace approximations.

## Paper summary (same as literature review):

Laplace Based Approximate Posterior Inference for Differential Equation Models, Dass, Sarat C. and Lee, Jaeyong and Lee, Kyoungjae and Park, Jonghun (https://arxiv.org/pdf/1607.07203)

This paper uses Laplace approximation to develop a fast, accurate method for approximating posterior distribution of ODEs using Bayesian Inference. This was done in order to bypass computationally heavy techniques like MCMC.

The methodology involved two steps. First classic one-step numerical methods were used to approximate the solution of the differential equation at observed time points. Then, Laplace approximation were used for marginalization of nuisance parameters resulting in simplified posterior paramters.

The results were compared to some existing methods like MCMC, parameter cascading, GP-ODE, and AGM. Measured outcomes included: accuracy of parameter estimates (via RMSE, bias), log-likelihood at estimates, and computational time and stability.

**Results:**

1. The LAP approach produces posterior estimates close to the true posteriors even with small sample sizes.
2. Approximation error reduced with increase in steps, showing convergence
3. Computation times were substantially lower than classic Bayesian methods like MCMC.

**Limitations:**
1. The accuracy depends on choice of numerical solver and number of steps. Runge-Kutta generally produced better results. 
2. For high-dimensional parameter spaces, grid sampling was computationally prohibitive.
3. The method applies primarily to first order differential equations. Higher parameter equations require extra methods. 

# Summary of Math: Laplace Approximated Posterior (LAP) for ODE Models

### 1. The Problem

ODE with unknown parameters:
$$\dot{x}(t) = f(x, t; \theta), \quad x_i = x(t_i)$$

Observations with noise:
$$y_i = x_i + \varepsilon_i, \quad \varepsilon_i \sim N_p(0, \sigma^2 I_p)$$

Goal: estimate $\theta$, initial condition $x_1$, and precision $\tau^2 = \frac{1}{\sigma^2}$

---

### 2. Priors

$$\tau^2 \sim \text{Gamma}(a, b)$$

This is chosen for conjugacy, letting us integrate out $\tau^2$ analytically later.

$$x_1 \mid \tau^2 \sim N_p(\mu_{x_1},\ c\tau^{-2} I_p)$$

$$\pi(\theta) \rightarrow \text{arbitrary}$$

---

### 3. Full Joint Posterior

Define the mean squared residual:
$$g_n(x_1, \theta) = \frac{1}{n}\sum_{i=1}^{n} \|y_i - x_i(\theta, x_1)\|^2$$

Multiplying the likelihood by the priors and simplifying:
$$\pi(\theta, \tau^2, x_1 \mid y_n) \propto (\tau^2)^{\frac{(n+1)p}{2}+a-1} \cdot e^{-\frac{\tau^2}{2}\left(ng_n(x_1,\theta)+\frac{\|x_1-\mu_{x_1}\|^2}{c}+2b\right)} \cdot \pi(\theta)$$

---

### 4. Two Approximations to Marginalize $x_1$

Since we cannot integrate out $x_1$ analytically, two approximations are used.

#### a) Numerical ODE Solver (Runge-Kutta)

$x_i(\theta, x_1)$ has no closed form for most ODEs, so we approximate it numerically:

$$x_i \approx x_{i-1} + h\phi(x_{i-1}, t_{i-1}; \theta)$$

where $\phi$ is the Runge-Kutta update rule and $h$ is the step size. The global error is:

$$\text{Error} = O(h^K), \quad K = \text{order of method}$$

#### b) Laplace Approximation of $x_1$

We need to integrate out $x_1$ from

$$\int\pi(x_1|\tau^2)P(y_n|\theta, \tau^2, x_1) dx_1$$

We approximate the integral by using Laplace approximation and expanding around its minimum. 

Find the optimal $\hat{x}_1(\theta)$:
$$\hat{x}_1(\theta) = \underset{x_1}{\text{argmin}} \left[ng_n(x_1, \theta) + \frac{\|x_1 - \mu_{x_1}\|^2}{c}\right]$$

Define:
$$u(\theta) = ng_n(\hat{x}_1) + \frac{\|\hat{x}_1 - \mu_{x_1}\|^2}{c}$$
$$H = n\ddot{g}_n(\hat{x}_1) + \frac{2}{c}I_p$$

where $\ddot{g}_n$ is the Hessian of $g_n$ with respect to $x_1$. $H$ captures the curvature of the posterior around the optimal $\hat{x}_1$.

After integrating out $x_1$, the joint posterior becomes:

$$\pi(\theta, \tau^2 \mid y_n) \propto \pi(\theta) \times (\tau^2)^{\frac{np}{2}+a-1} \times e^{-\frac{\tau^2}{2}u(\theta)} \times \det(H)^{-\frac{1}{2}}$$

$$\text{Approximation error: } O(n^{-3/2})$$

---

### 5. Marginalize $\tau^2$ Using Conjugacy

Since the prior on $\tau^2$ is Gamma and the posterior of $\tau^2 \mid \theta, y_n$ is also Gamma (by conjugacy), we can integrate it out analytically:

$$\tau^2 \mid \theta, y_n \sim \text{Gamma}\left(\frac{np}{2}+a,\ \frac{u(\theta)}{2}+b\right)$$

Integrating this out gives the marginal posterior of $\theta$ alone:

$$\pi(\theta \mid y_n) \propto \frac{\pi(\theta)}{\left(\frac{1}{2}u(\theta)+b\right)^{\frac{np}{2}+a}} \times \det(H)^{\frac{1}{2}}$$

---

### 6. Sampling Algorithm

Since $\pi(\theta \mid y_n)$ has no closed form, sample $\theta$ using:
- Grid sampling when $\dim(\theta) \leq 4$ — evaluate the posterior on a fine grid and sample directly
- Griddy Gibbs sampling when $\dim(\theta) \geq 5$

---
# Pseudocode: LAP for ODE Models

**Input:** Observations $y_1, \dots, y_n$, ODE $\dot{x}(t) = f(x,t;\theta)$, priors on $\theta, \tau^2, x_1$

**Output:** Posterior samples of $\theta$ and $\tau^2$

---

**Set** step size $h/m = O(n^{-5/(2K)})$

**Construct** grid $G_\theta$ over parameter space

**For** each $\theta$ in $G_\theta$:

$\quad$ 1. Approximate $x_1, \dots, x_n$ via Runge-Kutta: $x_i \leftarrow x_{i-1} + (h/m)\phi(x_{i-1}, t_{i-1}; \theta)$

$\quad$ 2. Compute $g_n(x_1, \theta) = \frac{1}{n}\sum_{i=1}^{n} \|y_i - x_i\|^2$

$\quad$ 3. **Laplace approximation** — marginalize $x_1$:

$\quad\quad$ a. Find $\hat{x}_1(\theta) = \underset{x_1}{\text{argmin}}\left[ng_n + \frac{\|x_1 - \mu_{x_1}\|^2}{c}\right]$ via Newton-Raphson

$\quad\quad$ b. Compute $u(\theta) = ng_n(\hat{x}_1) + \frac{\|\hat{x}_1 - \mu_{x_1}\|^2}{c}$

$\quad\quad$ c. Compute $H = n\ddot{g}_n(\hat{x}_1) + \frac{2}{c}I_p$

$\quad\quad$ d. Approximate $\int \pi(x_1 \mid \tau^2)\, L(\theta, \tau^2, x_1)\, dx_1 \approx (\tau^2)^{np/2}\, e^{-\frac{\tau^2}{2}u(\theta)}\, \det(H)^{-1/2}$

$\quad$ 4. **Marginalize $\tau^2$** — by conjugacy $\tau^2 \mid \theta, y_n \sim \text{Gamma}\!\left(\frac{np}{2}+a,\ \frac{u(\theta)}{2}+b\right)$, so integrate analytically:

$$\pi(\theta \mid y_n) \propto \frac{\pi(\theta)}{\left(\frac{u(\theta)}{2}+b\right)^{\frac{np}{2}+a}} \cdot \det(H)^{1/2}$$

**Normalize** across grid and draw $\theta^{(1)}, \dots, \theta^{(N)} \sim \pi^d(\theta \mid y_n)$

**For** each $\theta^{(i)}$, sample $\tau^{2(i)} \sim \text{Gamma}\!\left(\frac{np}{2}+a,\ \frac{u(\theta^{(i)})}{2}+b\right)$

**Return** $\{\theta^{(i)}, \tau^{2(i)}\}$

## Relevant code sources for LAP method
1. https://github.com/leekjstat/dem-LAPinfer/tree/master (end to end implementation of the LAP method as an R package)
2. https://github.com/LaplacesDemonR/LaplacesDemon (provides a complete and self-contained Bayesian environment within R. For example, this package includes dozens of MCMC algorithms, Laplace Approximation, iterative quadrature, Variational Bayes, parallelization, big data, PMC, over 100 examples in the Examples vignette, dozens of additional probability distributions, numerous MCMC diagnostics, Bayes factors, posterior predictive checks, a variety of plots, elicitation, parameter and variable importance, Bayesian forms of test statistics)