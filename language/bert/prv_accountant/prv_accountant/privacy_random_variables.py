# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
import numpy as np
from scipy import integrate
from numpy import exp, sqrt
from numpy import power as pow
from scipy.special import erfc

M_SQRT2 = sqrt(np.longdouble(2))
M_PI = np.pi


def log(x):
    valid = (x > 0)
    x_is_0 = (x == 0)
    return np.where(valid, np.log(np.where(valid, x, 1)), 
        np.where(x_is_0, -np.inf, np.nan))


class PrivacyRandomVariable(ABC):
    @abstractmethod
    def mean(self) -> float:
        pass

    def probability(self, a, b):
        return self.cdf(b) - self.cdf(a)

    @abstractmethod
    def pdf(self, t):
        pass

    @abstractmethod
    def cdf(self, t):
        pass


class PrivacyRandomVariableTruncated:
    def __init__(self, prv, t_min: float, t_max: float) -> None:
        self.prv = prv
        self.t_min = t_min
        self.t_max = t_max
        self.remaining_mass = self.prv.cdf(t_max) - self.prv.cdf(t_min)

    def mean(self) -> float:
        Ls = [self.t_min, -0.1, 0.1]
        Rs = [-0.1,  0.1, self.t_max]
        m = 0.0
        for L, R in zip(Ls, Rs):
            I, err = integrate.quad(self.cdf, L, R)

            m += (
                R*self.cdf(R) -
                L*self.cdf(L) -
                I
            )

        return m

    def probability(self, a, b):
        a = np.clip(a, self.t_min, self.t_max)
        b = np.clip(b, self.t_min, self.t_max)
        return self.prv.probability(a, b) / self.remaining_mass

    def pdf(self, t):
        return np.where(t < self.t_min, 0, np.where(t < self.t_max, self.prv.pdf(t)/self.remaining_mass, 0))

    def cdf(self, t):
        return np.where(t < self.t_min, 0, np.where(t < self.t_max, self.prv.cdf(t)/self.remaining_mass, 1))


class PoissonSubsampledGaussianMechanism(PrivacyRandomVariable):
    def __init__(self, sampling_probability: float, noise_multiplier: float) -> None:
        self.p = np.longdouble(sampling_probability)
        self.sigma = np.longdouble(noise_multiplier)

    def pdf(self, t):
        sigma = self.sigma
        p = self.p
        return np.where(t > 0, (
            (1.0/2.0) * M_SQRT2 * sigma *
            exp((
                -1.0/2.0*pow(sigma, 2)*pow(t, 2) - pow(sigma, 2)*t*log((p + exp(t) - 1)*exp(-t)/p) -
                1.0/2.0*pow(sigma, 2)*pow(log((p + exp(t) - 1)*exp(-t)/p), 2) + (3.0/2.0)*t - 1.0/8.0/pow(sigma, 2)
            )) /
            (sqrt(M_PI)*sqrt((p + exp(t) - 1)*exp(-t)/p)*(p + exp(t) - 1))
        ), np.where(t > log(1 - p), (
                (1.0/2.0) * M_SQRT2 * sigma *
                exp(-1.0/2.0*pow(sigma, 2)*pow(log((p + exp(t) - 1)/p), 2) + 2*t - 1.0/8.0/pow(sigma, 2)) /
                (sqrt(M_PI)*sqrt((p + exp(t) - 1)/p)*(p + exp(t) - 1))
            ), 0)
        )

    def cdf(self, t):
        sigma = self.sigma
        p = self.p
        z = np.where(t>0, log((p-1)/p + exp(t)/p), log((p-1)/p + exp(t)/p))
        return np.where(t > log(1 - p), (
                (1.0/2.0) * p * (-erfc(np.double((1.0/4.0)*M_SQRT2*(2*pow(sigma, 2)*z - 1)/sigma))) -
                1.0/2.0*(p - 1) * (-erfc(np.double((1.0/4.0)*M_SQRT2*(2*pow(sigma, 2)*z + 1)/sigma))) + 1
            ), 0.0)

    def mean(self):
        raise NotImplementedError("Mean computation not implemented")
