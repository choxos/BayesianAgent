# JAGS Meta-Analysis Model Integration Tests
# BayesianAgent

source(here::here("tests/conftest.R"))

context("JAGS Meta-Analysis Models")

test_that("Random effects meta-analysis runs in JAGS", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  # Generate test data
  test_data <- generate_meta_data(K = 10)

  # Model file
  model_file <- here::here("tests/models/jags/meta_analysis.txt")
  expect_true(file.exists(model_file))

  # Parameters to monitor
  params <- c("mu", "tau", "tau2", "theta")

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

test_that("JAGS meta-analysis recovers overall effect", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  test_data <- generate_meta_data(K = 20)  # More studies
  model_file <- here::here("tests/models/jags/meta_analysis.txt")
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
  mu_lower <- summ["mu", "2.5%"]
  mu_upper <- summ["mu", "97.5%"]

  expect_true(test_data$true_values$mu >= mu_lower &&
              test_data$true_values$mu <= mu_upper,
              info = paste("True mu:", test_data$true_values$mu,
                          "CI:", mu_lower, "-", mu_upper))
})

test_that("Heterogeneity variance is estimated", {
  skip_if_not(jags_available(), "JAGS not available")

  library(R2jags)

  test_data <- generate_meta_data(K = 15)
  model_file <- here::here("tests/models/jags/meta_analysis.txt")

  output <- capture.output({
    fit <- jags(
      data = test_data$data,
      parameters.to.save = c("tau", "tau2"),
      model.file = model_file,
      n.chains = 2,
      n.iter = 3000,
      n.burnin = 1500,
      DIC = TRUE,
      progress.bar = "none"
    )
  })

  summ <- fit$BUGSoutput$summary

  # tau2 should be non-negative
  tau2_samples <- fit$BUGSoutput$sims.list$tau2
  expect_true(all(tau2_samples >= 0),
              info = "tau2 (heterogeneity variance) should be non-negative")
})
