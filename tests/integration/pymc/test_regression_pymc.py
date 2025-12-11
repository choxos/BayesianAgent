"""
PyMC Regression Model Tests
BayesianAgent
"""

import pytest
import numpy as np

pytestmark = pytest.mark.pymc


@pytest.fixture
def pymc():
    """Import PyMC."""
    pytest.importorskip("pymc")
    import pymc as pm
    return pm


@pytest.fixture
def arviz():
    """Import ArviZ."""
    pytest.importorskip("arviz")
    import arviz as az
    return az


class TestLinearRegression:
    """Tests for linear regression model."""

    def test_model_samples(self, pymc, arviz, regression_data):
        """Test that linear regression model compiles and samples."""
        pm = pymc
        az = arviz
        data = regression_data["data"]

        with pm.Model() as model:
            # Priors
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=5, shape=data["K"])
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Expected value
            mu = alpha + pm.math.dot(data["X"], beta)

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data["y"])

            # Sample
            trace = pm.sample(
                draws=200,
                tune=200,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False
            )

        # Check convergence
        summary = az.summary(trace)
        max_rhat = summary["r_hat"].max()
        min_ess = summary["ess_bulk"].min()

        assert max_rhat < 1.1, f"Max Rhat {max_rhat} exceeds threshold"
        assert min_ess > 50, f"Min ESS {min_ess} too low"

    def test_parameter_recovery(self, pymc, arviz, regression_data):
        """Test that model recovers true parameters."""
        pm = pymc
        az = arviz
        data = regression_data["data"]
        true_values = regression_data["true_values"]

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=5, shape=data["K"])
            sigma = pm.HalfNormal("sigma", sigma=1)

            mu = alpha + pm.math.dot(data["X"], beta)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data["y"])

            trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False
            )

        summary = az.summary(trace, hdi_prob=0.9)

        # Check alpha recovery
        alpha_row = summary.loc["alpha"]
        assert alpha_row["hdi_5%"] <= true_values["alpha"] <= alpha_row["hdi_95%"], \
            f"True alpha {true_values['alpha']} not in CI [{alpha_row['hdi_5%']}, {alpha_row['hdi_95%']}]"

        # Check sigma recovery
        sigma_row = summary.loc["sigma"]
        assert sigma_row["hdi_5%"] <= true_values["sigma"] <= sigma_row["hdi_95%"], \
            f"True sigma {true_values['sigma']} not in CI [{sigma_row['hdi_5%']}, {sigma_row['hdi_95%']}]"

    def test_posterior_predictive(self, pymc, arviz, regression_data):
        """Test posterior predictive sampling."""
        pm = pymc
        az = arviz
        data = regression_data["data"]

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=5, shape=data["K"])
            sigma = pm.HalfNormal("sigma", sigma=1)

            mu = alpha + pm.math.dot(data["X"], beta)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data["y"])

            trace = pm.sample(
                draws=100,
                tune=100,
                chains=1,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False
            )

            # Posterior predictive
            ppc = pm.sample_posterior_predictive(trace, progressbar=False)

        assert "y_obs" in ppc.posterior_predictive
        assert ppc.posterior_predictive["y_obs"].shape[-1] == data["N"]
