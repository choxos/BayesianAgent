# JAGS Hierarchical Model Integration Tests
# BayesianAgent

source(here::here("tests/conftest.R"))

context("JAGS Hierarchical Models")

test_that("Hierarchical model runs in JAGS", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  # Generate test data
  test_data <- generate_hierarchical_data(J = 8)

  # Model file
  model_file <- here::here("tests/models/jags/hierarchical.txt")
  expect_true(file.exists(model_file))

  # Parameters to monitor
  params <- c("mu", "tau", "theta")

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
})

test_that("JAGS hierarchical model recovers mu", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  test_data <- generate_hierarchical_data(J = 8)
  model_file <- here::here("tests/models/jags/hierarchical.txt")
  params <- c("mu", "tau")

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

  # Check mu recovery
  mu_mean <- summ["mu", "mean"]
  mu_lower <- summ["mu", "2.5%"]
  mu_upper <- summ["mu", "97.5%"]

  expect_true(test_data$true_values$mu >= mu_lower &&
              test_data$true_values$mu <= mu_upper,
              info = paste("True mu:", test_data$true_values$mu,
                          "CI:", mu_lower, "-", mu_upper))
})

test_that("DIC is computed for hierarchical model", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  test_data <- generate_hierarchical_data(J = 8)
  model_file <- here::here("tests/models/jags/hierarchical.txt")

  output <- capture.output({
    fit <- jags(
      data = test_data$data,
      parameters.to.save = c("mu"),
      model.file = model_file,
      n.chains = 2,
      n.iter = 2000,
      n.burnin = 1000,
      DIC = TRUE,
      progress.bar = "none"
    )
  })

  expect_true(!is.na(fit$BUGSoutput$DIC),
              info = "DIC should be computed")
  expect_true(is.finite(fit$BUGSoutput$DIC),
              info = "DIC should be finite")
})
