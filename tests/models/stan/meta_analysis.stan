// Random Effects Meta-Analysis Model for Testing
// BayesianAgent - Stan 2.37
// Non-centered parameterization

data {
  int<lower=0> K;                    // Number of studies
  vector[K] y;                       // Effect estimates
  vector<lower=0>[K] se;             // Standard errors
}

parameters {
  real mu;                           // Overall effect
  real<lower=0> tau;                 // Between-study SD
  vector[K] eta;                     // Standardized study effects
}

transformed parameters {
  vector[K] theta;
  theta = mu + tau * eta;
}

model {
  // Priors
  mu ~ normal(0, 1);
  tau ~ cauchy(0, 0.5);
  eta ~ std_normal();

  // Likelihood
  y ~ normal(theta, se);
}

generated quantities {
  real theta_new;                    // Predictive distribution
  real I2;                           // Heterogeneity statistic

  theta_new = normal_rng(mu, tau);
  I2 = square(tau) / (square(tau) + mean(square(se)));
}
