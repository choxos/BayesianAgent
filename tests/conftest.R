# Test Configuration for BayesianAgent
# This file sets up the test environment for Stan and JAGS model testing

# ============================================================================
# Dependencies
# ============================================================================

required_packages <- c("testthat", "cmdstanr", "R2jags", "posterior", "bayesplot")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (pkg == "cmdstanr") {
      install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
    } else {
      install.packages(pkg)
    }
  }
}

# Install missing packages
invisible(lapply(required_packages, install_if_missing))

# Load packages
library(testthat)
library(posterior)

# ============================================================================
# Test Configuration
# ============================================================================

# Convergence thresholds
RHAT_THRESHOLD <- 1.1
ESS_THRESHOLD <- 100
MAX_DIVERGENCES <- 0

# Test MCMC settings (short chains for CI)
TEST_CHAINS <- 2
TEST_WARMUP <- 200
TEST_SAMPLING <- 200
TEST_SEED <- 12345

# ============================================================================
# Helper Functions
# ============================================================================

#' Check if cmdstanr is available and CmdStan is installed
stan_available <- function() {
  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    return(FALSE)
  }
  tryCatch({
    cmdstanr::cmdstan_path()
    TRUE
  }, error = function(e) FALSE)
}

#' Check if JAGS is available
jags_available <- function() {
  if (!requireNamespace("R2jags", quietly = TRUE)) {
    return(FALSE)
  }
  tryCatch({
    # Try to find JAGS
    system("which jags", intern = TRUE, ignore.stderr = TRUE)
    TRUE
  }, error = function(e) FALSE)
}

#' Generate regression test data
generate_regression_data <- function(N = 100, K = 3, seed = 42) {
  set.seed(seed)

  # True parameters
  true_alpha <- 2.0
  true_beta <- rnorm(K, 0, 1)
  true_sigma <- 0.5

  # Generate data
  X <- matrix(rnorm(N * K), N, K)
  y <- true_alpha + X %*% true_beta + rnorm(N, 0, true_sigma)

  list(
    data = list(N = N, K = K, X = X, y = as.vector(y)),
    true_values = list(
      alpha = true_alpha,
      beta = true_beta,
      sigma = true_sigma
    )
  )
}

#' Generate hierarchical test data (8 schools style)
generate_hierarchical_data <- function(J = 8, seed = 42) {
  set.seed(seed)

  # True parameters
  true_mu <- 5
  true_tau <- 3
  true_theta <- rnorm(J, true_mu, true_tau)
  true_sigma <- c(15, 10, 16, 11, 9, 11, 10, 18)

  # Generate data
  y <- rnorm(J, true_theta, true_sigma)

  list(
    data = list(J = J, y = y, sigma = true_sigma),
    true_values = list(
      mu = true_mu,
      tau = true_tau,
      theta = true_theta
    )
  )
}

#' Generate time series test data (AR1)
generate_ar1_data <- function(T = 200, seed = 42) {
  set.seed(seed)

  # True parameters
  true_mu <- 5
  true_phi <- 0.7
  true_sigma <- 1

  # Generate AR(1) process
  y <- numeric(T)
  y[1] <- rnorm(1, true_mu, true_sigma / sqrt(1 - true_phi^2))
  for (t in 2:T) {
    y[t] <- true_mu + true_phi * (y[t-1] - true_mu) + rnorm(1, 0, true_sigma)
  }

  list(
    data = list(T = T, y = y),
    true_values = list(
      mu = true_mu,
      phi = true_phi,
      sigma = true_sigma
    )
  )
}

#' Generate survival test data
generate_survival_data <- function(N = 150, seed = 42) {
  set.seed(seed)

  # True parameters
  true_shape <- 1.5
  true_scale_intercept <- 2
  true_beta <- 0.5

  # Generate data
  x <- rbinom(N, 1, 0.5)
  scale <- exp(true_scale_intercept + true_beta * x)
  t_event <- rweibull(N, true_shape, scale)

  # Random censoring
  t_censor <- runif(N, 0, max(t_event) * 0.8)
  event <- as.integer(t_event <= t_censor)
  time <- pmin(t_event, t_censor)

  list(
    data = list(N = N, time = time, event = event, x = x),
    true_values = list(
      shape = true_shape,
      alpha = true_scale_intercept,
      beta = true_beta
    )
  )
}

#' Generate meta-analysis test data
generate_meta_data <- function(K = 10, seed = 42) {
  set.seed(seed)

  # True parameters
  true_mu <- 0.3
  true_tau <- 0.15

  # Generate study-specific effects
  theta <- rnorm(K, true_mu, true_tau)
  se <- runif(K, 0.05, 0.2)
  y <- rnorm(K, theta, se)

  list(
    data = list(K = K, y = y, se = se),
    true_values = list(
      mu = true_mu,
      tau = true_tau
    )
  )
}

#' Check Stan fit diagnostics
check_stan_diagnostics <- function(fit, true_values = NULL) {
  summary_df <- fit$summary()

  diagnostics <- list(
    max_rhat = max(summary_df$rhat, na.rm = TRUE),
    min_ess_bulk = min(summary_df$ess_bulk, na.rm = TRUE),
    min_ess_tail = min(summary_df$ess_tail, na.rm = TRUE),
    num_divergent = sum(fit$diagnostic_summary()$num_divergent),
    num_max_treedepth = sum(fit$diagnostic_summary()$num_max_treedepth)
  )

  diagnostics$converged <-
    diagnostics$max_rhat < RHAT_THRESHOLD &&
    diagnostics$min_ess_bulk > ESS_THRESHOLD &&
    diagnostics$num_divergent == MAX_DIVERGENCES

  # Parameter recovery
  if (!is.null(true_values)) {
    diagnostics$recovery <- list()
    for (param in names(true_values)) {
      if (param %in% summary_df$variable) {
        row <- summary_df[summary_df$variable == param, ]
        true_val <- true_values[[param]]
        if (length(true_val) == 1) {
          in_ci <- true_val >= row$q5 && true_val <= row$q95
          diagnostics$recovery[[param]] <- list(
            true = true_val,
            estimate = row$mean,
            in_90_ci = in_ci
          )
        }
      }
    }
  }

  return(diagnostics)
}

#' Check JAGS fit diagnostics
check_jags_diagnostics <- function(fit, true_values = NULL) {
  summary_stats <- fit$BUGSoutput$summary

  diagnostics <- list(
    max_rhat = max(summary_stats[, "Rhat"], na.rm = TRUE),
    min_neff = min(summary_stats[, "n.eff"], na.rm = TRUE),
    DIC = fit$BUGSoutput$DIC,
    pD = fit$BUGSoutput$pD
  )

  diagnostics$converged <-
    diagnostics$max_rhat < RHAT_THRESHOLD &&
    diagnostics$min_neff > ESS_THRESHOLD

  return(diagnostics)
}

#' Run Stan model test
run_stan_test <- function(model_file, stan_data, true_values = NULL) {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  mod <- cmdstan_model(model_file)
  fit <- mod$sample(
    data = stan_data,
    seed = TEST_SEED,
    chains = TEST_CHAINS,
    parallel_chains = TEST_CHAINS,
    iter_warmup = TEST_WARMUP,
    iter_sampling = TEST_SAMPLING,
    refresh = 0,
    show_messages = FALSE
  )

  diagnostics <- check_stan_diagnostics(fit, true_values)
  list(fit = fit, diagnostics = diagnostics)
}

#' Run JAGS model test
run_jags_test <- function(model_file, jags_data, params, true_values = NULL) {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  output <- capture.output({
    fit <- jags(
      data = jags_data,
      parameters.to.save = params,
      model.file = model_file,
      n.chains = TEST_CHAINS,
      n.iter = TEST_WARMUP + TEST_SAMPLING,
      n.burnin = TEST_WARMUP,
      n.thin = 1,
      DIC = TRUE,
      progress.bar = "none"
    )
  })

  diagnostics <- check_jags_diagnostics(fit, true_values)
  list(fit = fit, diagnostics = diagnostics)
}

# Print test configuration
cat("BayesianAgent Test Configuration\n")
cat("================================\n")
cat("Stan available:", stan_available(), "\n")
cat("JAGS available:", jags_available(), "\n")
cat("Rhat threshold:", RHAT_THRESHOLD, "\n")
cat("ESS threshold:", ESS_THRESHOLD, "\n")
cat("Test chains:", TEST_CHAINS, "\n")
cat("Test warmup:", TEST_WARMUP, "\n")
cat("Test sampling:", TEST_SAMPLING, "\n")
