# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import scipy
import math
import pytest
import numpy as np

from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism, log

def test_safe_log():
    assert np.isnan(log(-1)) == True
    assert np.isneginf(log(0)) == True
    assert log(1) == pytest.approx(0)


def reference_sf(t, sigma, p):
    def alpha(t):
        return scipy.stats.norm.cdf(-t/(1/sigma) - (1/sigma)/2)

    def oneminus_beta(t):
        return scipy.stats.norm.sf(t/(1/sigma) - (1/sigma)/2)

    # survival function (i.e., 1-cdf) for stdQ
    if t > 0:
        return p*oneminus_beta(t+math.log(1/p-(1-p)*math.exp(-t)/p))+(1-p)*alpha(t+math.log(1/p-(1-p)*math.exp(-t)/p))  # noqa: E501
    elif t > math.log(1-p):
        return p*oneminus_beta(math.log((math.exp(t)-(1-p))/p))+(1-p)*alpha(math.log((math.exp(t)-(1-p))/p))  # noqa: E501
    else:
        return 1


class TestPrivacyRandomVariable:
    def test_sf(self):
        p = 1e-2
        sigma = 1.0
        Q = PoissonSubsampledGaussianMechanism(p, sigma)

        t = [
            math.log(1-p) - 1,
            math.log(1-p),
            math.log(1-p)/2,
            0.0,
            1.0,
            100.0
        ]

        for t_i in t:
            assert 1 - Q.cdf(t_i) == pytest.approx(reference_sf(
                t=t_i, p=p, sigma=sigma))

    def test_normalised(self):
        p = 1e-2
        sigma = 1.0
        Q = PoissonSubsampledGaussianMechanism(p, sigma)

        t = np.linspace(-10.0, 10.0, 2000000, dtype=np.longdouble)
        dt = t[1] - t[0]

        t_L = t - dt/2.0
        t_R = t + dt/2.0

        pdf = Q.probability(t_L, t_R)
        assert pdf.sum() == pytest.approx(1.0, 1e-10)
