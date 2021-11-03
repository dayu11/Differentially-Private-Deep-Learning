# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.stats import binom, norm

from prv_accountant import composers
from prv_accountant.discrete_privacy_random_variable import DiscretePrivacyRandomVariable
from prv_accountant.domain import Domain


class TestFourier:
    def test_unity(self):
        domain = Domain(0, 100, 100)
        pmf = binom.pmf(domain.ts().astype(np.double), 100, 0.4)
        prv = DiscretePrivacyRandomVariable(pmf, domain)

        composer = composers.Fourier(prv)

        f_n = composer.compute_composition(1)

        assert_array_almost_equal(f_n.pmf, prv.pmf)

    def test_gaussian_analytical(self):
        """
        Test that the convolution of Gaussians gives the analytical solution
        """

        domain = Domain(-20, 20, 100000)
        pmf = norm.pdf(domain.ts(), 1, 1)*domain.dt()
        prv = DiscretePrivacyRandomVariable(pmf, domain)

        assert pmf.sum() == pytest.approx(1)

        composer = composers.Fourier(prv)

        f_n = composer.compute_composition(3)

        assert np.dot(f_n.pmf, domain.ts()) == pytest.approx(3, abs=1e-3)
