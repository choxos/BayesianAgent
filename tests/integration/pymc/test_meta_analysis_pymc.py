"""
PyMC Meta-Analysis Model Tests
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


class TestMetaAnalysis:
    """Tests for random effects meta-analysis model."""

    def test_model_samples(self, pymc, arviz, meta_analysis_data):
        """Test that meta-analysis model samples correctly."""
        pm = pymc
        az = arviz
        data = meta_analysis_data["data"]

        with pm.Model() as model:
            # Hyperpriors
            mu = pm.Normal("mu", mu=0, sigma=1)
            tau = pm.HalfCauchy("tau", beta=0.5)

            # Study effects (non-centered)
            eta = pm.Normal("eta", mu=0, sigma=1, shape=data["K"])
            theta = pm.Deterministic("theta", mu + tau * eta)

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=theta, sigma=data["se"], observed=data["y"])

            # Sample
            trace = pm.sample(
                draws=300,
                tune=300,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False
            )

        # Check convergence
        summary = az.summary(trace, var_names=["mu", "tau"])
        max_rhat = summary["r_hat"].max()

        assert max_rhat < 1.1, f"Max Rhat {max_rhat} exceeds threshold"

    def test_overall_effect_recovery(self, pymc, arviz, meta_analysis_data):
        """Test that model recovers overall effect mu."""
        pm = pymc
        az = arviz
        data = meta_analysis_data["data"]
        true_values = meta_analysis_data["true_values"]

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            tau = pm.HalfCauchy("tau", beta=0.5)

            eta = pm.Normal("eta", mu=0, sigma=1, shape=data["K"])
            theta = pm.Deterministic("theta", mu + tau * eta)

            y_obs = pm.Normal("y_obs", mu=theta, sigma=data["se"], observed=data["y"])

            trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False
            )

        summary = az.summary(trace, var_names=["mu"], hdi_prob=0.9)
        mu_row = summary.loc["mu"]

        assert mu_row["hdi_5%"] <= true_values["mu"] <= mu_row["hdi_95%"], \
            f"True mu {true_values['mu']} not in CI [{mu_row['hdi_5%']}, {mu_row['hdi_95%']}]"

    def test_heterogeneity_estimation(self, pymc, arviz, meta_analysis_data):
        """Test that heterogeneity tau is estimated."""
        pm = pymc
        az = arviz
        data = meta_analysis_data["data"]

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            tau = pm.HalfCauchy("tau", beta=0.5)

            # I2 statistic
            I2 = pm.Deterministic("I2", tau**2 / (tau**2 + np.mean(data["se"]**2)))

            eta = pm.Normal("eta", mu=0, sigma=1, shape=data["K"])
            theta = pm.Deterministic("theta", mu + tau * eta)

            y_obs = pm.Normal("y_obs", mu=theta, sigma=data["se"], observed=data["y"])

            trace = pm.sample(
                draws=300,
                tune=300,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False
            )

        # Check I2 is between 0 and 1
        I2_samples = trace.posterior["I2"].values.flatten()
        assert np.all(I2_samples >= 0), "I2 should be non-negative"
        assert np.all(I2_samples <= 1), "I2 should be <= 1"

    def test_predictive_distribution(self, pymc, arviz, meta_analysis_data):
        """Test predictive distribution for new study."""
        pm = pymc
        az = arviz
        data = meta_analysis_data["data"]

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=0, sigma=1)
            tau = pm.HalfCauchy("tau", beta=0.5)

            # Predictive for new study
            theta_new = pm.Normal("theta_new", mu=mu, sigma=tau)

            eta = pm.Normal("eta", mu=0, sigma=1, shape=data["K"])
            theta = pm.Deterministic("theta", mu + tau * eta)

            y_obs = pm.Normal("y_obs", mu=theta, sigma=data["se"], observed=data["y"])

            trace = pm.sample(
                draws=300,
                tune=300,
                chains=2,
                cores=1,
                random_seed=42,
                return_inferencedata=True,
                progressbar=False
            )

        # theta_new should have wider distribution than mu (includes between-study variance)
        summary = az.summary(trace, var_names=["mu", "theta_new"])
        mu_sd = summary.loc["mu", "sd"]
        theta_new_sd = summary.loc["theta_new", "sd"]

        assert theta_new_sd >= mu_sd, \
            "Predictive distribution should have wider spread than overall effect"
