// AR(1) Time Series Model for Testing
// BayesianAgent - Stan 2.37

data {
  int<lower=0> T;          // Number of time points
  vector[T] y;             // Time series observations
}

parameters {
  real mu;                            // Mean
  real<lower=-1, upper=1> phi;        // AR coefficient (stationary)
  real<lower=0> sigma;                // Innovation SD
}

model {
  // Priors
  mu ~ normal(0, 10);
  phi ~ uniform(-1, 1);
  sigma ~ exponential(1);

  // Stationary initial distribution
  y[1] ~ normal(mu, sigma / sqrt(1 - square(phi)));

  // AR(1) likelihood (vectorized)
  y[2:T] ~ normal(mu + phi * (y[1:(T-1)] - mu), sigma);
}

generated quantities {
  vector[T] y_rep;
  real y_forecast;

  y_rep[1] = normal_rng(mu, sigma / sqrt(1 - square(phi)));
  for (t in 2:T)
    y_rep[t] = normal_rng(mu + phi * (y_rep[t-1] - mu), sigma);

  // One-step-ahead forecast
  y_forecast = normal_rng(mu + phi * (y[T] - mu), sigma);
}
