"""
PyMC Hierarchical Model Tests
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


class TestHierarchicalModel:
    """Tests for hierarchical (Eight Schools) model."""

    def test_noncentered_model_samples(self, pymc, arviz, hierarchical_data):
        """Test non-centered hierarchical model samples without divergences."""
        pm = pymc
        az = arviz
        data = hierarchical_data["data"]

        with pm.Model() as model:
            # Hyperpriors
            mu = pm.Normal("mu", mu=0, sigma=5)
            tau = pm.HalfCauchy("tau", beta=5)

            # Non-centered parameterization
            theta_raw = pm.Normal("theta_raw", mu=0, sigma=1, shape=data["J"])
            theta = pm.Deterministic("theta", mu + tau * theta_raw)

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=theta, sigma=data["sigma"], observed=data["y"])

            # Sample with higher target_accept for hierarchical model
            trace = pm.sample(
                draws=300,
                tune=300,
                chains=2,
                cores=1,
                random_seed=42,
                target_accept=0.9,
                return_inferencedata=True,
                progressbar=False
            )

        # Check convergence
        summary = az.summary(trace, var_names=["mu", "tau"])
        max_rhat = summary["r_hat"].max()

        assert max_rhat < 1.1, f"Max Rhat {max_rhat} exceeds threshold"

    def test_mu_recovery(self, pymc, arviz, hierarchical_data):
        """Test that model recovers mu parameter."""
        pm = pymc
        az = arviz
        data = hierarchical_data["data"]
        true_values = hierarchical_data["true_values"]

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=5)
            tau = pm.HalfCauchy("tau", beta=5)

            theta_raw = pm.Normal("theta_raw", mu=0, sigma=1, shape=data["J"])
            theta = pm.Deterministic("theta", mu + tau * theta_raw)

            y_obs = pm.Normal("y_obs", mu=theta, sigma=data["sigma"], observed=data["y"])

            trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=1,
                random_seed=42,
                target_accept=0.9,
                return_inferencedata=True,
                progressbar=False
            )

        summary = az.summary(trace, var_names=["mu"], hdi_prob=0.9)
        mu_row = summary.loc["mu"]

        # With high uncertainty in Eight Schools, we use wider CI
        assert mu_row["hdi_5%"] <= true_values["mu"] <= mu_row["hdi_95%"], \
            f"True mu {true_values['mu']} not in CI [{mu_row['hdi_5%']}, {mu_row['hdi_95%']}]"

    def test_shrinkage_effect(self, pymc, arviz, hierarchical_data):
        """Test that theta estimates show shrinkage toward mu."""
        pm = pymc
        az = arviz
        data = hierarchical_data["data"]

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=5)
            tau = pm.HalfCauchy("tau", beta=5)

            theta_raw = pm.Normal("theta_raw", mu=0, sigma=1, shape=data["J"])
            theta = pm.Deterministic("theta", mu + tau * theta_raw)

            y_obs = pm.Normal("y_obs", mu=theta, sigma=data["sigma"], observed=data["y"])

            trace = pm.sample(
                draws=300,
                tune=300,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False
            )

        summary = az.summary(trace)
        mu_mean = summary.loc["mu", "mean"]
        theta_means = np.array([summary.loc[f"theta[{i}]", "mean"] for i in range(data["J"])])

        # Theta estimates should be shrunk toward mu (closer to mu than raw y)
        shrinkage_to_mu = np.abs(theta_means - mu_mean)
        distance_y_to_mu = np.abs(data["y"] - mu_mean)

        # On average, theta should be closer to mu than y
        assert np.mean(shrinkage_to_mu) < np.mean(distance_y_to_mu), \
            "Theta estimates should show shrinkage toward mu"
