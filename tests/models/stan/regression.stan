// Linear Regression Model for Testing
// BayesianAgent - Stan 2.37

data {
  int<lower=0> N;           // Number of observations
  int<lower=0> K;           // Number of predictors
  matrix[N, K] X;           // Predictor matrix
  vector[N] y;              // Outcome variable
}

parameters {
  real alpha;               // Intercept
  vector[K] beta;           // Coefficients
  real<lower=0> sigma;      // Error SD
}

model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 5);
  sigma ~ exponential(1);

  // Likelihood
  y ~ normal(alpha + X * beta, sigma);
}

generated quantities {
  array[N] real y_rep;
  real log_lik[N];

  for (n in 1:N) {
    real mu_n = alpha + X[n] * beta;
    y_rep[n] = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma);
  }
}
