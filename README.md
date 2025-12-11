# BayesianAgent

> **Bayesian Modeling Agent Ecosystem** — Create, review, and validate Stan, JAGS, and WinBUGS models with AI assistance

A comprehensive [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) plugin providing **5 specialized agents**, **8 knowledge skills**, and **3 workflow commands** for Bayesian statistical modeling.

## Overview

This plugin helps you create and review Bayesian models across three languages:

- **Stan 2.37** (default) — Modern HMC/NUTS sampling with cmdstanr
- **JAGS** — Cross-platform Gibbs sampling with R2jags
- **WinBUGS** — Classic BUGS implementation with R2WinBUGS

### Key Features

- **Model Creation** — Generate complete, runnable models from natural language descriptions
- **Model Review** — Analyze existing models for correctness, efficiency, and best practices
- **Automatic Testing** — Run models with synthetic data to validate convergence
- **R Integration** — Complete code for cmdstanr, R2jags, and R2WinBUGS
- **All Experience Levels** — Adjustable verbosity from beginner (extensive comments) to advanced (minimal)

## Quick Start

### Step 1: Add the Marketplace

Add this plugin marketplace to Claude Code:

```bash
/plugin marketplace add choxos/BayesianAgent
```

### Step 2: Install the Plugin

```bash
/plugin install bayesian-modeling
```

This loads the Bayesian modeling agents, commands, and skills into Claude's context.

## Usage

### Creating Models

Use the `/create-model` command or describe what you need:

```
I need a hierarchical model for patient outcomes nested within hospitals.
The outcome is continuous, with patient-level predictors (age, treatment)
and a hospital-level predictor (size). Use Stan.
```

The agent will:
1. Ask clarifying questions about your data and preferences
2. Generate complete Stan/JAGS code with appropriate priors
3. Provide R code for data preparation and model fitting
4. Include posterior predictive checks

### Reviewing Models

Use the `/review-model` command or paste your model:

```
Review this Stan model:

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

The reviewer checks:
- Syntax correctness
- Prior completeness
- Parameterization efficiency
- Common errors (SD vs precision, missing constraints)

### Running Diagnostics

Use the `/run-diagnostics` command:

```
Test my hierarchical model with synthetic data and report convergence
```

Reports include Rhat, ESS, divergences, and parameter recovery.

## Supported Model Types

| Model Type | Stan | JAGS | WinBUGS |
|------------|:----:|:----:|:-------:|
| Linear regression | ✓ | ✓ | ✓ |
| Logistic regression | ✓ | ✓ | ✓ |
| Hierarchical/Multilevel | ✓ | ✓ | ✓ |
| Time series (AR, ARMA) | ✓ | ✓ | ✓ |
| Survival analysis | ✓ | ✓ | ✓ |
| Meta-analysis | ✓ | ✓ | ✓ |

## Components

### Agents (5)

| Agent | Model | Purpose |
|-------|-------|---------|
| `model-architect` | Haiku | Routes requests to specialists |
| `stan-specialist` | Sonnet | Stan 2.37 expert with cmdstanr |
| `bugs-specialist` | Sonnet | BUGS/JAGS expert with precision parameterization |
| `model-reviewer` | Sonnet | Reviews models for correctness & efficiency |
| `test-runner` | Haiku | Executes models with synthetic data |

### Skills (8)

- **Language Skills**: `stan-fundamentals`, `bugs-fundamentals`
- **Model Skills**: `hierarchical-models`, `regression-models`, `time-series-models`, `survival-models`, `meta-analysis`
- **Diagnostics**: `model-diagnostics`

### Commands (3)

- `/create-model` — Interactive model creation workflow
- `/review-model` — Review existing models
- `/run-diagnostics` — Test model execution

## Critical: Parameterization Differences

**The most common source of errors when working across languages:**

| Distribution | Stan | JAGS/WinBUGS |
|-------------|------|--------------|
| Normal | `normal(mu, sigma)` — SD | `dnorm(mu, tau)` — precision (τ = 1/σ²) |
| Multivariate Normal | `multi_normal(mu, Sigma)` — covariance | `dmnorm(mu, Omega)` — precision matrix |

The agents automatically handle these differences when creating or converting models.

## Repository Structure

```
BayesianAgent/
├── .claude-plugin/
│   └── marketplace.json          # Plugin configuration
├── plugins/bayesian-modeling/
│   ├── agents/                   # 5 specialized agents
│   ├── commands/                 # 3 workflow commands
│   └── skills/                   # 8 knowledge skills
├── tests/
│   ├── models/                   # Stan and JAGS test models
│   └── integration/              # R integration tests
└── .github/workflows/            # CI/CD pipelines
```

## Running Tests Locally

If you want to verify the models work:

```bash
# Install R dependencies
Rscript -e "install.packages(c('testthat', 'here', 'cmdstanr', 'R2jags'))"

# Run all tests
Rscript tests/run_tests.R

# Run Stan tests only
Rscript tests/run_tests.R stan

# Run JAGS tests only
Rscript tests/run_tests.R jags
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Stan Development Team](https://mc-stan.org/) for Stan 2.37
- [JAGS](https://mcmc-jags.sourceforge.io/) for JAGS
- [Anthropic](https://www.anthropic.com/) for Claude Code
