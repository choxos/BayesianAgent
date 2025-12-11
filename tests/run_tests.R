#!/usr/bin/env Rscript
# Run all BayesianAgent tests
# Usage: Rscript tests/run_tests.R [stan|jags|all]

args <- commandArgs(trailingOnly = TRUE)
test_type <- if (length(args) > 0) args[1] else "all"

# Load dependencies
library(testthat)
library(here)

# Source configuration
source(here("tests/conftest.R"))

cat("\n")
cat("========================================\n")
cat("BayesianAgent Test Suite\n")
cat("========================================\n")
cat("\n")

# Run tests based on type
results <- list()

if (test_type %in% c("stan", "all")) {
  if (stan_available()) {
    cat("Running Stan tests...\n")
    cat("----------------------------------------\n")
    results$stan <- test_dir(
      here("tests/integration/stan"),
      reporter = "summary"
    )
  } else {
    cat("Skipping Stan tests (CmdStan not available)\n")
  }
}

if (test_type %in% c("jags", "all")) {
  if (jags_available()) {
    cat("\nRunning JAGS tests...\n")
    cat("----------------------------------------\n")
    results$jags <- test_dir(
      here("tests/integration/jags"),
      reporter = "summary"
    )
  } else {
    cat("Skipping JAGS tests (JAGS not available)\n")
  }
}

# Summary
cat("\n")
cat("========================================\n")
cat("Test Summary\n")
cat("========================================\n")

total_passed <- 0
total_failed <- 0
total_skipped <- 0

for (name in names(results)) {
  r <- results[[name]]
  if (!is.null(r)) {
    df <- as.data.frame(r)
    passed <- sum(df$passed)
    failed <- sum(df$failed)
    skipped <- sum(df$skipped)

    total_passed <- total_passed + passed
    total_failed <- total_failed + failed
    total_skipped <- total_skipped + skipped

    cat(sprintf("%s: %d passed, %d failed, %d skipped\n",
                toupper(name), passed, failed, skipped))
  }
}

cat("----------------------------------------\n")
cat(sprintf("TOTAL: %d passed, %d failed, %d skipped\n",
            total_passed, total_failed, total_skipped))
cat("\n")

# Exit with appropriate code
if (total_failed > 0) {
  quit(status = 1)
} else {
  quit(status = 0)
}
