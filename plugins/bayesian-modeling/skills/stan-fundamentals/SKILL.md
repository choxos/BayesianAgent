---
name: stan-fundamentals
description: Foundational knowledge for writing Stan 2.37 models including program structure, type system, distributions, and best practices. Use when creating or reviewing Stan models.
---

# Stan Fundamentals

## When to Use This Skill

- Writing new Stan models from scratch
- Understanding Stan program structure
- Learning Stan syntax and conventions
- Translating models from other languages to Stan
- Optimizing existing Stan code

## Program Structure

Stan models have up to 7 blocks in this exact order:

```stan
functions { }           // User-defined functions
data { }                // Input data declarations
transformed data { }    // Data preprocessing
parameters { }          // Model parameters
transformed parameters { } // Derived parameters
model { }               // Log probability
generated quantities { }  // Posterior predictions
```

All blocks are optional. Empty string is valid (but useless) Stan program.

## Type System Quick Reference

### Scalars
```stan
int n;                    // Integer
real x;                   // Real number
complex z;                // Complex number
```

### Vectors and Matrices
```stan
vector[N] v;              // Column vector
row_vector[N] r;          // Row vector
matrix[M, N] A;           // Matrix
```

### Arrays (Modern Syntax)
```stan
array[N] real x;          // 1D array of reals
array[M, N] int y;        // 2D array of integers
array[J] vector[K] theta; // Array of vectors
```

### Constrained Types
```stan
real<lower=0> sigma;              // Non-negative
real<lower=0, upper=1> p;         // Probability
simplex[K] theta;                 // Sums to 1
ordered[K] c;                     // Ascending
corr_matrix[K] Omega;             // Correlation
cov_matrix[K] Sigma;              // Covariance
cholesky_factor_corr[K] L_Omega;  // Cholesky correlation
```

## Key Distributions

### Continuous (SD parameterization!)
```stan
y ~ normal(mu, sigma);      // sigma is SD
y ~ student_t(nu, mu, sigma);
y ~ cauchy(mu, sigma);
y ~ exponential(lambda);
y ~ gamma(alpha, beta);
y ~ beta(a, b);
y ~ lognormal(mu, sigma);
```

### Discrete
```stan
y ~ bernoulli(theta);
y ~ binomial(n, theta);
y ~ poisson(lambda);
y ~ neg_binomial_2(mu, phi);
y ~ categorical(theta);
```

### Multivariate
```stan
y ~ multi_normal(mu, Sigma);        // Sigma is COVARIANCE
y ~ multi_normal_cholesky(mu, L);
y ~ lkj_corr(eta);
```

## Essential Patterns

### Vectorization
```stan
// GOOD - Efficient
y ~ normal(mu, sigma);

// BAD - Slow
for (n in 1:N) y[n] ~ normal(mu[n], sigma);
```

### Non-Centered Parameterization
```stan
parameters {
  vector[J] theta_raw;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  theta_raw ~ std_normal();
}
```

### Target Syntax
```stan
// These are equivalent:
y ~ normal(mu, sigma);
target += normal_lpdf(y | mu, sigma);
```

## Common Priors

```stan
// Location parameters
mu ~ normal(0, 10);

// Scale parameters
sigma ~ exponential(1);
sigma ~ cauchy(0, 2.5);  // half-Cauchy when sigma > 0

// Probabilities
theta ~ beta(1, 1);  // Uniform on (0,1)

// Regression coefficients
beta ~ normal(0, 2.5);

// Correlation matrices
Omega ~ lkj_corr(2);  // eta=2 favors identity
```

## R Integration (cmdstanr)

```r
library(cmdstanr)
mod <- cmdstan_model("model.stan")
fit <- mod$sample(data = stan_data, chains = 4)
fit$summary()
fit$cmdstan_diagnose()
```

## Key Differences from BUGS

| Feature | Stan | BUGS/JAGS |
|---------|------|-----------|
| Normal | `normal(mu, sigma)` SD | `dnorm(mu, tau)` precision |
| MVN | `multi_normal(mu, Sigma)` cov | `dmnorm(mu, Omega)` precision |
| Execution | Sequential (order matters) | Declarative (order doesn't matter) |
| Sampling | HMC/NUTS | Gibbs/Metropolis |
