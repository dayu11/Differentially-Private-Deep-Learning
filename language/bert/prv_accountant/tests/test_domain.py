# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from prv_accountant.domain import Domain


class TestDomain:
    def test_create_aligned(self):
        d = Domain.create_aligned(-0.5, 0.5, 1)
        assert len(d) == 4
        assert d.t_min() == pytest.approx(-1)
        assert d.t_max() == pytest.approx(2)

        d = Domain.create_aligned(-0.5, 0.5, 0.5)
        assert len(d) == 4
        assert d.t_min() == pytest.approx(-0.5)
        assert d.t_max() == pytest.approx(1.0)

        d = Domain.create_aligned(-0.1, 0.5, 0.5)
        assert len(d) == 4
        assert d.t_min() == pytest.approx(-0.5)
        assert d.t_max() == pytest.approx(1.0)

        d = Domain.create_aligned(-0.6, 0.5, 0.5)
        assert len(d) == 4
        assert d.t_min() == pytest.approx(-1.0)

    def test_ts(self):
        d = Domain(-0.6, 0.5, 6)

        ts = d.ts()

        for ts_i, t_i in zip(ts, d):
            assert ts_i == pytest.approx(t_i)

    def test_construct_same(self):
        d1 = Domain(-0.6, 0.5, 6)
        d2 = Domain(d1.t_min(), d1.t_max(), len(d1))

        assert len(d1) == len(d2)
        assert d1.t_min() == pytest.approx(d2.t_min())
        assert d2.dt() == pytest.approx(d2.dt())

    def test_shift_right(self):
        d1 = Domain(-0.6, 0.5, 6)
        d2 = d1.shift_right(0.2)

        assert d2.t_min() == pytest.approx(-0.4)
        assert d2.t_max() == pytest.approx(0.7)
        assert len(d2) == len(d1)
        assert d2.dt() == pytest.approx(d1.dt())
