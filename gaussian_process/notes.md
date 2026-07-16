# Kernel Methods & Gaussian Processes

## 1. Kernels:

A kernel function is a symmetric function that gives the inner product of two points after mapping them into some feature space:

k(x, x') = φ(x)ᵗφ(x')

φ(x) is the feature space mapping. The point of a kernel is you can compute this inner product without ever building φ(x) by hand.

Simplest kernel: linear kernel:

k(x, x') = xᵗx'

Types of kernels:

- Stationary kernel: only depends on the difference x − x'
  k(x, x') = k(x − x')
- Homogeneous kernel: only depends on the distance between points
  k(x, x') = k(‖x − x'‖)

## 2. Constructing kernels

Two ways to get a kernel:

**Explicit** — pick basis functions φᵢ(x):

k(x, x') = φ(x)ᵗφ(x') = Σᵢ φᵢ(x)φᵢ(x')

**Implicit** — start from a simple expression and it turns out to secretly be an inner product. Example, 2D input:

k(x,z) = (xᵗz)² = (x₁z₁ + x₂z₂)²

### Gram matrix

Given vectors v₁...vₙ, the Gram matrix is the n×n matrix of inner products:

Gᵢⱼ = ⟨vᵢ, vⱼ⟩

same as AᵗA, where A has the vectors as columns.

### Positive semidefinite

A matrix M is positive semidefinite if:

xᵗMx ≥ 0 for all x in ℝⁿ

A function k(x,x') is a valid kernel only if its Gram matrix is positive semidefinite, for any choice of points.

### Gaussian kernel

k(x,x') = exp( −‖x−x'‖² / 2σ² )

### Rules for building new kernels from old ones

If k₁ and k₂ are valid kernels, so are:

- c·k₁(x,x'), for constant c > 0
- f(x)·k₁(x,x')·f(x'), for any function f
- q(k₁(x,x')), for a polynomial q with positive coefficients
- exp(k₁(x,x'))
- k₁(x,x') + k₂(x,x')
- k₁(x,x') · k₂(x,x')
- k₃(φ(x), φ(x')), where φ maps x into ℝᴹ and k₃ is a valid kernel
- xᵗAx', for symmetric positive semidefinite A
- ka(xa,xa') + kb(xb,xb'), splitting x into (xa, xb)
- ka(xa,xa') · kb(xb,xb')

### Invariance

A kernel can be built to ignore certain transformations:

- Translation: x → x + c
- Rotation: x → Rx
- Scale: x' = cx

### Generative kernels

Build a kernel from a generative model instead of a feature map.

Simple version:

k(x,x') = p(x)·p(x')

General version, with a latent variable i and weights p(i):

k(x,x') = Σᵢ p(x|i) p(x'|i) p(i)

### Fisher kernel

Fisher score:

g(θ,x) = ∇θ ln p(x|θ)

Fisher information matrix:

F = Eₓ[ g(θ,x) g(θ,x)ᵗ ]

Fisher kernel:

k(x,x') = g(θ,x)ᵗ F⁻¹ g(θ,x')

## 3. Radial Basis Function (RBF) networks

Setup: inputs x₁...x_N, targets t₁...t_N. We want a smooth function that hits every target exactly.

f(x) = Σₙ wₙ h(‖x − xₙ‖)

weights wₙ found by least squares.

### Noisy inputs

If the input x is noisy (noise ξ ~ ν(ξ)), minimize the expected sum of squares:

E = (1/2) Σₙ ∫ { y(xₙ+ξ) − tₙ }² ν(ξ) dξ

The solution is:

y(x) = Σₙ tₙ h(x − xₙ)

with

h(x − xₙ) = ν(x − xₙ) / Σₘ ν(x − xₘ)

So the prediction is a weighted average of the targets, where the weights come from the noise distribution centered at each point.

## 4. Nadaraya-Watson model (kernel regression)

Training set: {xₙ, tₙ}

### Step 1: model the joint density

Use a Parzen estimator (non-parametric way to estimate a PDF from a finite set of points) to model p(x,t):

p(x,t) = (1/N) Σₙ f(x − xₙ, t − tₙ)

f(x,t) is a density placed at each data point.

### Step 2: get y(x) as a conditional mean

y(x) = E[t|x] = ∫ t p(t|x) dt = ∫t p(x,t) dt / ∫ p(x,t) dt

Plug in the Parzen estimate:

y(x) = [ Σₙ ∫ t f(x−xₙ, t−tₙ) dt ] / [ Σₘ ∫ f(x−xₘ, t−tₘ) dt ]

### Step 3: assume zero mean

Assume each component density has zero mean in t:

∫ f(x,t) t dt = 0

After a change of variables this simplifies to:

y(x) = [ Σₙ g(x−xₙ) tₙ ] / [ Σₘ g(x−xₘ) ]

where

g(x) = ∫ f(x,t) dt

### Step 4: write as a kernel

y(x) = Σₙ k(x, xₙ) tₙ

where

k(x,xₙ) = g(x−xₙ) / Σₘ g(x−xₘ)

So y(x) is just a weighted average of the training targets, where the weight on each target depends on how close x is to that training point.

## 5. Gaussian Processes

### Definition

A Gaussian process is a probability distribution over functions y(x), such that the values of y(x) at any finite set of points x₁...x_N are jointly Gaussian.

For a 2D input, this is also called a Gaussian random field.

A GP is fully described by a mean function (usually taken as zero) and a covariance function (the kernel).

### GP for regression

Noise model:

tₙ = yₙ + εₙ,  p(tₙ|yₙ) = N(tₙ|yₙ, β⁻¹)

GP prior on noisy targets given y:

p(t|y) = N(t|y, β⁻¹I_N)

GP prior on the function values:

p(y) = N(y|0, K)

Marginalize out y:

p(t) = N(t|0, C)

with

C(xₙ,xₘ) = k(xₙ,xₘ) + β⁻¹δₙₘ

### Predicting a new point

Joint distribution over all N+1 targets:

p(t_{N+1}) = N(t_{N+1} | 0, C_{N+1})

Split the covariance matrix:

C_{N+1} = [ C_N   k  ]
          [ kᵗ    c  ]

- C_N: N×N covariance of training points
- k: N×1 vector, kₙ = k(xₙ, x_{N+1})
- c = k(x_{N+1}, x_{N+1}) + β⁻¹

Conditioning the joint Gaussian gives the predictive distribution, with:

m(x_{N+1}) = kᵗ C_N⁻¹ t

σ²(x_{N+1}) = c − kᵗ C_N⁻¹ k

These two formulas are the core result of GP regression. The mean is a weighted sum of the observed targets. The variance is small near training points and grows where there's no nearby data.

### Learning the hyperparameters

The kernel has parameters θ (e.g. length scale), plus noise precision β. Learn them by maximizing the log marginal likelihood:

ln p(t|θ) = −(1/2) ln|C_N| − (1/2) tᵗC_N⁻¹t − (N/2) ln(2π)

Maximize with gradient methods. Gradient:

∂ln p(t|θ)/∂θᵢ = −(1/2) Tr(C_N⁻¹ ∂C_N/∂θᵢ) + (1/2) tᵗC_N⁻¹ (∂C_N/∂θᵢ) C_N⁻¹ t

This isn't convex, so there can be multiple local maxima.

### GP for classification

Classification needs probabilities in (0,1), but a GP's output is any real number.

Fix: put a GP on a latent function a(x), then squash it with a logistic sigmoid:

y = σ(a)

p(t|a) = σ(a)ᵗ (1−σ(a))^{1−t}

This makes the target Bernoulli. Since p(t|a) is no longer Gaussian, predictions need an approximation (Laplace approximation, variational inference, or expectation propagation).