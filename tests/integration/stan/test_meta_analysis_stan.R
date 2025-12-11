# Stan Meta-Analysis Model Integration Tests
# BayesianAgent

source(here::here("tests/conftest.R"))

context("Stan Meta-Analysis Models")

test_that("Random effects meta-analysis compiles and samples", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  # Generate test data
  test_data <- generate_meta_data(K = 10)

  # Compile model
  model_file <- here::here("tests/models/stan/meta_analysis.stan")
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

test_that("Meta-analysis recovers overall effect", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  test_data <- generate_meta_data(K = 20)  # More studies
  model_file <- here::here("tests/models/stan/meta_analysis.stan")
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

  # Check mu recovery
  mu_row <- summ[summ$variable == "mu", ]
  expect_true(test_data$true_values$mu >= mu_row$q5 &&
              test_data$true_values$mu <= mu_row$q95,
              info = paste("True mu:", test_data$true_values$mu,
                          "CI:", mu_row$q5, "-", mu_row$q95))
})

test_that("I2 heterogeneity statistic is computed", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  test_data <- generate_meta_data(K = 10)
  model_file <- here::here("tests/models/stan/meta_analysis.stan")
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

  # Check I2 is computed and bounded
  I2_draws <- as.vector(fit$draws("I2"))
  expect_true(all(I2_draws >= 0 & I2_draws <= 1),
              info = "I2 should be between 0 and 1")
})

test_that("Predictive distribution is generated", {
  skip_if_not(stan_available(), "CmdStan not available")

  library(cmdstanr)

  test_data <- generate_meta_data(K = 10)
  model_file <- here::here("tests/models/stan/meta_analysis.stan")
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

  # Check theta_new exists
  theta_new <- fit$draws("theta_new")
  expect_equal(dim(theta_new)[3], 1)
})
