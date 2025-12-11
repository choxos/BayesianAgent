"""
PyMC Test Configuration
BayesianAgent - PyMC Integration Tests
"""

import pytest
import numpy as np

# Test configuration
RHAT_THRESHOLD = 1.1
ESS_THRESHOLD = 100
TEST_CHAINS = 2
TEST_DRAWS = 200
TEST_TUNE = 200
TEST_SEED = 42


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "pymc: mark test as requiring PyMC")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture(scope="session")
def pymc_available():
    """Check if PyMC is available."""
    try:
        import pymc as pm
        import arviz as az
        return True
    except ImportError:
        return False


# ============================================================================
# Data Generators
# ============================================================================

@pytest.fixture
def regression_data():
    """Generate regression test data."""
    np.random.seed(TEST_SEED)
    N = 100
    K = 3

    # True parameters
    true_alpha = 2.0
    true_beta = np.array([0.5, -0.3, 0.8])
    true_sigma = 0.5

    # Generate data
    X = np.random.randn(N, K)
    y = true_alpha + X @ true_beta + np.random.randn(N) * true_sigma

    return {
        "data": {"N": N, "K": K, "X": X, "y": y},
        "true_values": {
            "alpha": true_alpha,
            "beta": true_beta,
            "sigma": true_sigma
        }
    }


@pytest.fixture
def hierarchical_data():
    """Generate hierarchical test data (Eight Schools style)."""
    np.random.seed(TEST_SEED)
    J = 8

    # True parameters
    true_mu = 5.0
    true_tau = 3.0
    true_theta = np.random.randn(J) * true_tau + true_mu
    true_sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)

    # Generate data
    y = np.random.randn(J) * true_sigma + true_theta

    return {
        "data": {"J": J, "y": y, "sigma": true_sigma},
        "true_values": {
            "mu": true_mu,
            "tau": true_tau,
            "theta": true_theta
        }
    }


@pytest.fixture
def meta_analysis_data():
    """Generate meta-analysis test data."""
    np.random.seed(TEST_SEED)
    K = 10

    # True parameters
    true_mu = 0.3
    true_tau = 0.15

    # Generate study effects
    theta = np.random.randn(K) * true_tau + true_mu
    se = np.random.uniform(0.05, 0.2, K)
    y = np.random.randn(K) * se + theta

    return {
        "data": {"K": K, "y": y, "se": se},
        "true_values": {
            "mu": true_mu,
            "tau": true_tau
        }
    }


# ============================================================================
# Diagnostic Helpers
# ============================================================================

def check_pymc_diagnostics(trace, true_values=None):
    """Check PyMC convergence diagnostics."""
    import arviz as az

    summary = az.summary(trace, hdi_prob=0.9)

    diagnostics = {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
    }

    diagnostics["converged"] = (
        diagnostics["max_rhat"] < RHAT_THRESHOLD and
        diagnostics["min_ess_bulk"] > ESS_THRESHOLD
    )

    # Parameter recovery
    if true_values:
        diagnostics["recovery"] = {}
        for param, true_val in true_values.items():
            if param in summary.index:
                row = summary.loc[param]
                if np.isscalar(true_val):
                    in_ci = row["hdi_5%"] <= true_val <= row["hdi_95%"]
                    diagnostics["recovery"][param] = {
                        "true": true_val,
                        "estimate": float(row["mean"]),
                        "in_90_ci": bool(in_ci)
                    }

    return diagnostics
