# BayesianAgent

> **Bayesian Modeling Agent Ecosystem** — Create, review, and validate Stan, PyMC, JAGS, and WinBUGS models with AI assistance

A comprehensive [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) plugin providing **6 specialized agents**, **9 knowledge skills**, and **3 workflow commands** for Bayesian statistical modeling.

## Overview

This plugin helps you create and review Bayesian models across four languages:

- **Stan 2.37** (default for R) — Modern HMC/NUTS sampling with cmdstanr
- **PyMC 5** (default for Python) — Python-native Bayesian modeling with ArviZ
- **JAGS** — Cross-platform Gibbs sampling with R2jags
- **WinBUGS** — Classic BUGS implementation with R2WinBUGS

### Key Features

- **Model Creation** — Generate complete, runnable models from natural language descriptions
- **Model Review** — Analyze existing models for correctness, efficiency, and best practices
- **Automatic Testing** — Run models with synthetic data to validate convergence
- **R & Python Integration** — Complete code for cmdstanr, PyMC/ArviZ, R2jags, and R2WinBUGS
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

**For R users (Stan):**
```
I need a hierarchical model for patient outcomes nested within hospitals.
The outcome is continuous, with patient-level predictors (age, treatment)
and a hospital-level predictor (size). Use Stan.
```

**For Python users (PyMC):**
```
Create a Bayesian logistic regression in Python using PyMC.
I have binary outcomes and 3 predictor variables.
```

The agent will:
1. Ask clarifying questions about your data and preferences
2. Generate complete Stan/PyMC/JAGS code with appropriate priors
3. Provide integration code (R or Python) for model fitting
4. Include posterior predictive checks

### Reviewing Models

Use the `/review-model` command or paste your model. The agent auto-detects the language:

```python
# PyMC model to review
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 10)
    sigma = pm.HalfNormal("sigma", 1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
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

| Model Type | Stan | PyMC | JAGS | WinBUGS |
|------------|:----:|:----:|:----:|:-------:|
| Linear regression | ✓ | ✓ | ✓ | ✓ |
| Logistic regression | ✓ | ✓ | ✓ | ✓ |
| Hierarchical/Multilevel | ✓ | ✓ | ✓ | ✓ |
| Time series (AR, ARMA) | ✓ | ✓ | ✓ | ✓ |
| Survival analysis | ✓ | ✓ | ✓ | ✓ |
| Meta-analysis | ✓ | ✓ | ✓ | ✓ |

## Components

### Agents (6)

| Agent | Model | Purpose |
|-------|-------|---------|
| `model-architect` | Haiku | Routes requests to specialists |
| `stan-specialist` | Sonnet | Stan 2.37 expert with cmdstanr |
| `pymc-specialist` | Sonnet | PyMC 5 expert with ArviZ |
| `bugs-specialist` | Sonnet | BUGS/JAGS expert with precision parameterization |
| `model-reviewer` | Sonnet | Reviews models for correctness & efficiency |
| `test-runner` | Haiku | Executes models with synthetic data |

### Skills (9)

- **Language Skills**: `stan-fundamentals`, `pymc-fundamentals`, `bugs-fundamentals`
- **Model Skills**: `hierarchical-models`, `regression-models`, `time-series-models`, `survival-models`, `meta-analysis`
- **Diagnostics**: `model-diagnostics`

### Commands (3)

- `/create-model` — Interactive model creation workflow
- `/review-model` — Review existing models
- `/run-diagnostics` — Test model execution

## Critical: Parameterization Differences

**The most common source of errors when working across languages:**

| Distribution | Stan | PyMC | JAGS/WinBUGS |
|-------------|------|------|--------------|
| Normal | `normal(mu, sigma)` — SD | `Normal(mu, sigma)` — SD | `dnorm(mu, tau)` — precision (τ = 1/σ²) |
| Multivariate Normal | `multi_normal(mu, Sigma)` — cov | `MvNormal(mu, cov)` — cov | `dmnorm(mu, Omega)` — precision matrix |

**Note:** Stan and PyMC both use SD parameterization. Only BUGS/JAGS uses precision.

The agents automatically handle these differences when creating or converting models.

## Repository Structure

```
BayesianAgent/
├── .claude-plugin/
│   └── marketplace.json          # Plugin configuration
├── plugins/bayesian-modeling/
│   ├── agents/                   # 6 specialized agents
│   ├── commands/                 # 3 workflow commands
│   └── skills/                   # 9 knowledge skills
├── tests/
│   ├── models/                   # Stan, JAGS test models
│   └── integration/              # R and Python tests
└── .github/workflows/            # CI/CD pipelines
```

## Running Tests Locally

### R Tests (Stan/JAGS)

```bash
# Install R dependencies
Rscript -e "install.packages(c('testthat', 'here', 'cmdstanr', 'R2jags'))"

# Run all R tests
Rscript tests/run_tests.R

# Run Stan tests only
Rscript tests/run_tests.R stan

# Run JAGS tests only
Rscript tests/run_tests.R jags
```

### Python Tests (PyMC)

```bash
# Install Python dependencies
pip install pytest pymc arviz numpy

# Run PyMC tests
pytest tests/integration/pymc/ -v
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Stan Development Team](https://mc-stan.org/) for Stan 2.37
- [PyMC Developers](https://www.pymc.io/) for PyMC 5
- [JAGS](https://mcmc-jags.sourceforge.io/) for JAGS
- [Anthropic](https://www.anthropic.com/) for Claude Code
