// Hierarchical Model (Eight Schools) for Testing
// BayesianAgent - Stan 2.37
// Non-centered parameterization for robust sampling

data {
  int<lower=0> J;                    // Number of groups
  array[J] real y;                   // Observed effects
  array[J] real<lower=0> sigma;      // Known standard errors
}

parameters {
  real mu;                           // Population mean
  real<lower=0> tau;                 // Between-group SD
  vector[J] theta_raw;               // Standardized group effects
}

transformed parameters {
  vector[J] theta;
  for (j in 1:J)
    theta[j] = mu + tau * theta_raw[j];
}

model {
  // Hyperpriors
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);

  // Group effects (non-centered)
  theta_raw ~ std_normal();

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  array[J] real y_rep;
  real theta_new;

  for (j in 1:J)
    y_rep[j] = normal_rng(theta[j], sigma[j]);

  theta_new = normal_rng(mu, tau);
}
