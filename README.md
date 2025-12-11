# BayesianAgent

A Claude Code agent ecosystem for creating, reviewing, and validating Bayesian statistical models in Stan, JAGS, and WinBUGS.

[![Stan Tests](https://github.com/choxos/BayesianAgent/actions/workflows/test-stan.yml/badge.svg)](https://github.com/choxos/BayesianAgent/actions/workflows/test-stan.yml)
[![JAGS Tests](https://github.com/choxos/BayesianAgent/actions/workflows/test-jags.yml/badge.svg)](https://github.com/choxos/BayesianAgent/actions/workflows/test-jags.yml)

## Features

- **Multi-language support**: Stan 2.37 (default), JAGS, and WinBUGS
- **Model creation**: Generate complete, runnable Bayesian models from natural language descriptions
- **Model review**: Analyze existing models for correctness, efficiency, and best practices
- **Automatic testing**: Run models with synthetic data to validate convergence
- **R integration**: Complete code for cmdstanr, R2jags, and R2WinBUGS

## Supported Model Types

| Model Type | Stan | JAGS | WinBUGS |
|------------|:----:|:----:|:-------:|
| Linear regression | ✓ | ✓ | ✓ |
| Logistic regression | ✓ | ✓ | ✓ |
| Hierarchical/Multilevel | ✓ | ✓ | ✓ |
| Time series (AR, ARMA) | ✓ | ✓ | ✓ |
| Survival analysis | ✓ | ✓ | ✓ |
| Meta-analysis | ✓ | ✓ | ✓ |

## Installation

### As a Claude Code Plugin

```bash
# Clone the repository
git clone https://github.com/choxos/BayesianAgent.git

# Install as Claude Code plugin
claude plugins install ./BayesianAgent
```

### Prerequisites for Testing

To run the integration tests, you need:

**For Stan models:**
```r
# Install cmdstanr
install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))

# Install CmdStan
library(cmdstanr)
check_cmdstan_toolchain()
install_cmdstan()
```

**For JAGS models:**
```bash
# macOS
brew install jags

# Ubuntu/Debian
sudo apt-get install jags

# Then in R:
install.packages("R2jags")
```

## Usage

### Creating a Model

Use the `/create-model` command or describe your model:

```
I need a hierarchical model for student test scores nested within schools.
The outcome is continuous, and I want to include student-level predictors
(age, socioeconomic status) and a school-level predictor (funding level).
Use Stan.
```

### Reviewing a Model

Use the `/review-model` command or paste your model:

```
Can you review this Stan model for errors?

data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real mu;
  real sigma;  // Missing constraint!
}
model {
  y ~ normal(mu, sigma);
}
```

### Running Diagnostics

Use the `/run-diagnostics` command to test a model:

```
Run diagnostics on my hierarchical model with synthetic data
```

## Project Structure

```
BayesianAgent/
├── .claude-plugin/
│   └── marketplace.json          # Plugin configuration
├── plugins/bayesian-modeling/
│   ├── agents/
│   │   ├── model-architect.md    # Orchestration agent
│   │   ├── stan-specialist.md    # Stan expert
│   │   ├── bugs-specialist.md    # BUGS/JAGS expert
│   │   ├── model-reviewer.md     # Review agent
│   │   └── test-runner.md        # Test execution agent
│   ├── commands/
│   │   ├── create-model.md       # Model creation workflow
│   │   ├── review-model.md       # Review workflow
│   │   └── run-diagnostics.md    # Diagnostics workflow
│   └── skills/
│       ├── stan-fundamentals/    # Stan syntax & patterns
│       ├── bugs-fundamentals/    # BUGS syntax & patterns
│       ├── hierarchical-models/  # Hierarchical model patterns
│       ├── regression-models/    # Regression patterns
│       ├── time-series-models/   # Time series patterns
│       ├── survival-models/      # Survival analysis patterns
│       ├── meta-analysis/        # Meta-analysis patterns
│       └── model-diagnostics/    # MCMC diagnostics
├── tests/
│   ├── conftest.R                # Test configuration
│   ├── run_tests.R               # Test runner script
│   ├── models/                   # Test model files
│   └── integration/              # Integration tests
└── .github/workflows/            # CI/CD pipelines
```

## Key Technical Notes

### Parameterization Differences

**This is the most common source of errors when working across languages:**

| Distribution | Stan | JAGS/WinBUGS |
|-------------|------|--------------|
| Normal | `normal(mu, sigma)` - SD | `dnorm(mu, tau)` - precision (tau = 1/sigma²) |
| Multivariate Normal | `multi_normal(mu, Sigma)` - covariance | `dmnorm(mu, Omega)` - precision matrix |

### Stan 2.37 Array Syntax

Use the modern array syntax:
```stan
// Correct (Stan 2.37)
array[N] real x;
array[J] vector[K] theta;

// Deprecated
real x[N];
```

### Non-Centered Parameterization

For hierarchical models with weak data:
```stan
parameters {
  vector[J] theta_raw;  // Standard normal
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  theta_raw ~ std_normal();
}
```

## Running Tests

```bash
# Run all tests
Rscript tests/run_tests.R

# Run Stan tests only
Rscript tests/run_tests.R stan

# Run JAGS tests only
Rscript tests/run_tests.R jags
```

## Convergence Criteria

Tests verify:
- **Rhat < 1.1** for all parameters
- **ESS > 100** for all parameters
- **No divergent transitions** (Stan)
- **Parameter recovery** within 90% credible interval

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Stan Development Team](https://mc-stan.org/) for Stan 2.37
- [JAGS](https://mcmc-jags.sourceforge.io/) for JAGS
- [WinBUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/) for WinBUGS
