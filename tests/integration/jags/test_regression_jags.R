# JAGS Regression Model Integration Tests
# BayesianAgent

source(here::here("tests/conftest.R"))

context("JAGS Regression Models")

test_that("Linear regression model runs in JAGS", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  # Generate test data
  test_data <- generate_regression_data(N = 100, K = 3)

  # Model file
  model_file <- here::here("tests/models/jags/regression.txt")
  expect_true(file.exists(model_file))

  # Parameters to monitor
  params <- c("alpha", "beta", "sigma")

  # Run model
  output <- capture.output({
    fit <- jags(
      data = test_data$data,
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

  # Check convergence
  diagnostics <- check_jags_diagnostics(fit)

  expect_lt(diagnostics$max_rhat, RHAT_THRESHOLD,
            info = paste("Max Rhat:", diagnostics$max_rhat))
  expect_gt(diagnostics$min_neff, ESS_THRESHOLD / 2,
            info = paste("Min n.eff:", diagnostics$min_neff))
})

test_that("JAGS regression recovers parameters", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  test_data <- generate_regression_data(N = 200, K = 2)
  model_file <- here::here("tests/models/jags/regression.txt")
  params <- c("alpha", "beta", "sigma")

  output <- capture.output({
    fit <- jags(
      data = test_data$data,
      parameters.to.save = params,
      model.file = model_file,
      n.chains = 2,
      n.iter = 5000,
      n.burnin = 2500,
      n.thin = 1,
      DIC = TRUE,
      progress.bar = "none"
    )
  })

  summ <- fit$BUGSoutput$summary

  # Check alpha recovery
  alpha_mean <- summ["alpha", "mean"]
  alpha_lower <- summ["alpha", "2.5%"]
  alpha_upper <- summ["alpha", "97.5%"]

  expect_true(test_data$true_values$alpha >= alpha_lower &&
              test_data$true_values$alpha <= alpha_upper,
              info = paste("True alpha:", test_data$true_values$alpha,
                          "CI:", alpha_lower, "-", alpha_upper))
})

test_that("JAGS uses precision parameterization correctly", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  # Simple test to verify precision vs SD
  set.seed(42)
  y <- rnorm(100, 5, 2)  # True SD = 2

  jags_data <- list(
    N = 100,
    K = 1,
    y = y,
    X = matrix(rep(0, 100), ncol = 1)
  )

  model_file <- here::here("tests/models/jags/regression.txt")
  params <- c("sigma", "alpha")

  output <- capture.output({
    fit <- jags(
      data = jags_data,
      parameters.to.save = params,
      model.file = model_file,
      n.chains = 2,
      n.iter = 3000,
      n.burnin = 1000,
      progress.bar = "none"
    )
  })

  summ <- fit$BUGSoutput$summary
  sigma_mean <- summ["sigma", "mean"]

  # Should recover approximately sigma = 2
  expect_gt(sigma_mean, 1.5, info = paste("Sigma estimate:", sigma_mean))
  expect_lt(sigma_mean, 2.5, info = paste("Sigma estimate:", sigma_mean))
})
