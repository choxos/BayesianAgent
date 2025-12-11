# Stan Regression Model Integration Tests
# BayesianAgent

source(here::here("tests/conftest.R"))

context("Stan Regression Models")

test_that("Linear regression model compiles and samples", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  # Generate test data
  test_data <- generate_regression_data(N = 100, K = 3)

  # Compile model
  model_file <- here::here("tests/models/stan/regression.stan")
  expect_true(file.exists(model_file))

  mod <- cmdstan_model(model_file)

  # Run short test
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
  expect_equal(diagnostics$num_divergent, 0,
               info = paste("Divergences:", diagnostics$num_divergent))

  # Check parameter recovery
  summ <- fit$summary()
  alpha_row <- summ[summ$variable == "alpha", ]
  sigma_row <- summ[summ$variable == "sigma", ]

  expect_true(test_data$true_values$alpha >= alpha_row$q5 &&
              test_data$true_values$alpha <= alpha_row$q95,
              info = "True alpha should be in 90% CI")
  expect_true(test_data$true_values$sigma >= sigma_row$q5 &&
              test_data$true_values$sigma <= sigma_row$q95,
              info = "True sigma should be in 90% CI")
})

test_that("Regression model syntax is valid", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  model_file <- here::here("tests/models/stan/regression.stan")
  mod <- cmdstan_model(model_file, compile = FALSE)

  # This will throw an error if syntax is invalid
  expect_no_error(mod$check_syntax())
})

test_that("Regression model generates predictions", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  test_data <- generate_regression_data(N = 50, K = 2)
  model_file <- here::here("tests/models/stan/regression.stan")
  mod <- cmdstan_model(model_file)

  fit <- mod$sample(
    data = test_data$data,
    seed = TEST_SEED,
    chains = 1,
    iter_warmup = 100,
    iter_sampling = 100,
    refresh = 0,
    show_messages = FALSE
  )

  # Check generated quantities exist
  y_rep <- fit$draws("y_rep")
  expect_equal(dim(y_rep)[3], test_data$data$N)
})
