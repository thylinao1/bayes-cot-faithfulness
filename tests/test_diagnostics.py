"""Tests for the MCMC sampler-health gate (offline, no PyMC sampling)."""

from __future__ import annotations

import numpy as np
import pytest

from bayes_cot_faithfulness.diagnostics import (
    SamplerHealth,
    assert_sampler_healthy,
    raise_if_unhealthy,
)


def _healthy_idata():
    """Two well-mixed chains with all-zero divergences."""
    import arviz as az

    rng = np.random.default_rng(0)
    posterior = {"theta": rng.normal(0.0, 1.0, size=(2, 600))}
    sample_stats = {"diverging": np.zeros((2, 600), dtype=bool)}
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)


def _degenerate_idata():
    """Chains pinned to different constants -> large R-hat, tiny ESS."""
    import arviz as az

    rng = np.random.default_rng(1)
    jitter = rng.normal(0.0, 1e-6, size=(2, 600))
    posterior = {"theta": np.vstack([np.full(600, 0.0), np.full(600, 5.0)]) + jitter}
    return az.from_dict(posterior=posterior)


def _diverging_idata():
    """Well-mixed chains, but the sampler logged divergences."""
    import arviz as az

    rng = np.random.default_rng(2)
    posterior = {"theta": rng.normal(0.0, 1.0, size=(2, 600))}
    diverging = np.zeros((2, 600), dtype=bool)
    diverging[0, :17] = True  # 17 divergent transitions
    sample_stats = {"diverging": diverging}
    return az.from_dict(posterior=posterior, sample_stats=sample_stats)


# --------------------------------------------------------------------------- #
# assert_sampler_healthy
# --------------------------------------------------------------------------- #
def test_healthy_idata_reports_healthy() -> None:
    # Arrange
    idata = _healthy_idata()

    # Act
    health = assert_sampler_healthy(idata)

    # Assert
    assert isinstance(health, SamplerHealth)
    assert health.healthy
    assert health.max_rhat_seen < 1.01
    assert health.min_ess_seen >= 400.0
    assert health.n_divergences == 0


def test_degenerate_idata_reports_unhealthy() -> None:
    # Arrange: chains at different constants blow up R-hat.
    idata = _degenerate_idata()

    # Act
    health = assert_sampler_healthy(idata)

    # Assert
    assert not health.healthy
    assert health.max_rhat_seen > 1.01


def test_divergences_make_it_unhealthy() -> None:
    # Arrange: mixing is fine but divergences are present.
    idata = _diverging_idata()

    # Act
    health = assert_sampler_healthy(idata)

    # Assert
    assert health.n_divergences == 17
    assert not health.healthy


def test_missing_sample_stats_does_not_crash() -> None:
    # Arrange: no sample_stats group at all.
    idata = _degenerate_idata()

    # Act: divergences default to zero, no exception.
    health = assert_sampler_healthy(idata)

    # Assert
    assert health.n_divergences == 0


def test_low_ess_flags_unhealthy() -> None:
    # Arrange: a well-mixed but tiny chain has ESS well below the 400 default.
    import arviz as az

    rng = np.random.default_rng(3)
    idata = az.from_dict(posterior={"theta": rng.normal(0.0, 1.0, size=(2, 40))})

    # Act
    health = assert_sampler_healthy(idata, min_ess=400.0)

    # Assert
    assert health.min_ess_seen < 400.0
    assert not health.healthy


# --------------------------------------------------------------------------- #
# raise_if_unhealthy
# --------------------------------------------------------------------------- #
def test_raise_if_unhealthy_passes_healthy() -> None:
    # Arrange
    idata = _healthy_idata()

    # Act
    health = raise_if_unhealthy(idata)

    # Assert: returns the dataclass, raises nothing.
    assert health.healthy


def test_raise_if_unhealthy_raises_on_bad_run() -> None:
    # Arrange
    idata = _degenerate_idata()

    # Act / Assert
    with pytest.raises(ValueError, match="r-hat|rhat|R-hat"):
        raise_if_unhealthy(idata)
