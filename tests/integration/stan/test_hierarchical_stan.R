# Stan Hierarchical Model Integration Tests
# BayesianAgent

source(here::here("tests/conftest.R"))

context("Stan Hierarchical Models")

test_that("Hierarchical model (Eight Schools) compiles and samples", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  # Generate test data
  test_data <- generate_hierarchical_data(J = 8)

  # Compile model
  model_file <- here::here("tests/models/stan/hierarchical.stan")
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
  expect_gt(diagnostics$min_ess_bulk, ESS_THRESHOLD / 2,  # Hierarchical models may have lower ESS
            info = paste("Min ESS bulk:", diagnostics$min_ess_bulk))

  # Check mu recovery
  summ <- fit$summary()
  mu_row <- summ[summ$variable == "mu", ]
  expect_true(test_data$true_values$mu >= mu_row$q5 &&
              test_data$true_values$mu <= mu_row$q95,
              info = "True mu should be in 90% CI")
})

test_that("Non-centered parameterization reduces divergences", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  # Data that typically causes issues with centered parameterization
  test_data <- generate_hierarchical_data(J = 8)
  test_data$data$sigma <- rep(20, 8)  # Large SEs = weak data

  model_file <- here::here("tests/models/stan/hierarchical.stan")
  mod <- cmdstan_model(model_file)

  fit <- mod$sample(
    data = test_data$data,
    seed = TEST_SEED,
    chains = 2,
    iter_warmup = 300,
    iter_sampling = 300,
    refresh = 0,
    show_messages = FALSE
  )

  # Non-centered should have few/no divergences even with weak data
  diagnostics <- check_stan_diagnostics(fit)

  # Allow some divergences but should be minimal

  expect_lte(diagnostics$num_divergent, 5,
             info = paste("Divergences:", diagnostics$num_divergent))
})

test_that("Hierarchical model generates predictions", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  test_data <- generate_hierarchical_data(J = 8)
  model_file <- here::here("tests/models/stan/hierarchical.stan")
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

  # Check generated quantities
  y_rep <- fit$draws("y_rep")
  theta_new <- fit$draws("theta_new")

  expect_equal(dim(y_rep)[3], test_data$data$J)
  expect_equal(dim(theta_new)[3], 1)
})
