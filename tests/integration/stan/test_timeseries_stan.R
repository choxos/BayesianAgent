# Stan Time Series Model Integration Tests
# BayesianAgent

source(here::here("tests/conftest.R"))

context("Stan Time Series Models")

test_that("AR(1) model compiles and samples", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  # Generate test data
  test_data <- generate_ar1_data(T = 200)

  # Compile model
  model_file <- here::here("tests/models/stan/ar1.stan")
  expect_true(file.exists(model_file))

  mod <- cmdstan_model(model_file)

  # Run test
  fit <- mod$sample(
    data = test_data$data,
    seed = TEST_SEED,
    chains = TEST_CHAINS,
    parallel_chains = TEST_CHAINS,
    iter_warmup = TEST_WARMUP,
    iter_sampling = TEST_SAMPLING,
    refresh = 0,
    show_messages = FALSE
  )

  # Check convergence
  diagnostics <- check_stan_diagnostics(fit, test_data$true_values)

  expect_lt(diagnostics$max_rhat, RHAT_THRESHOLD,
            info = paste("Max Rhat:", diagnostics$max_rhat))
  expect_gt(diagnostics$min_ess_bulk, ESS_THRESHOLD,
            info = paste("Min ESS bulk:", diagnostics$min_ess_bulk))
})

test_that("AR(1) recovers true parameters", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  test_data <- generate_ar1_data(T = 500)  # Longer series for better recovery
  model_file <- here::here("tests/models/stan/ar1.stan")
  mod <- cmdstan_model(model_file)

  fit <- mod$sample(
    data = test_data$data,
    seed = TEST_SEED,
    chains = 2,
    iter_warmup = 500,
    iter_sampling = 500,
    refresh = 0,
    show_messages = FALSE
  )

  summ <- fit$summary()

  # Check phi recovery (most important for AR model)
  phi_row <- summ[summ$variable == "phi", ]
  expect_true(test_data$true_values$phi >= phi_row$q5 &&
              test_data$true_values$phi <= phi_row$q95,
              info = paste("True phi:", test_data$true_values$phi,
                          "Estimate:", phi_row$mean))

  # Check mu recovery
  mu_row <- summ[summ$variable == "mu", ]
  expect_true(test_data$true_values$mu >= mu_row$q5 &&
              test_data$true_values$mu <= mu_row$q95,
              info = paste("True mu:", test_data$true_values$mu,
                          "Estimate:", mu_row$mean))
})

test_that("AR(1) stationarity constraint is respected", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  test_data <- generate_ar1_data(T = 200)
  model_file <- here::here("tests/models/stan/ar1.stan")
  mod <- cmdstan_model(model_file)

  fit <- mod$sample(
    data = test_data$data,
    seed = TEST_SEED,
    chains = 2,
    iter_warmup = 200,
    iter_sampling = 200,
    refresh = 0,
    show_messages = FALSE
  )

  # All phi samples should be between -1 and 1
  phi_draws <- as.vector(fit$draws("phi"))
  expect_true(all(phi_draws > -1 & phi_draws < 1),
              info = "All phi samples should satisfy stationarity constraint")
})
