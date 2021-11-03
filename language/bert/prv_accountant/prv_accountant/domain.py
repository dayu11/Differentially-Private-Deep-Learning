# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
import numpy as np


class Domain:
    def __init__(self, t_min: float, t_max: float, size: int, shifts=0.0) -> None:
        """
        :param float t_min: Coordinate of the node farthest to the left
        :param float t_max: Coorindate of the node farthest to the right
        :param int size: Number of nodes in the domain
        :param float shift: Amount by which this domain has been shifted
        """
        if not isinstance(size, int):
            raise TypeError("`size` must be integer")
        if size % 2 != 0:
            raise ValueError("Must have an even size")
        self._t_min = t_min
        self._size = size
        self._dt = (t_max-t_min)/(size-1)
        self._shifts = shifts

    @abstractmethod
    def create_aligned(t_min: float, t_max: float, dt: float) -> "Domain":
        """
        Create a domain instance that is aligned with the origin.

        The actual domain might be slightly larger than [t_min, t_max]
        but it's guaranteed that the domain is smaller than [t_min-dt, t_max+dt]

        The domain will also be an even size which makes later computing the FFT easier

        :param float t_min: Lower point that will be in the domain
        :param float t_max: Upper point that will be in the domain
        :param float dt: Mesh size
        """
        t_min = np.floor(t_min/dt)*dt
        t_max = np.ceil(t_max/dt)*dt
        size = int((t_max-t_min)/dt) + 1
        if size % 2 == 1:
            size += 1
            t_max += dt
        return Domain(t_min, t_max, size)

    def shifts(self) -> float:
        """Sum of all shifts that were applied to this domain"""
        return self._shifts

    def shift_right(self, dt: float) -> "Domain":
        """Shift the domain right by `dt`"""
        return Domain(self.t_min()+dt, self.t_max()+dt, len(self), self.shifts() + dt)

    def shift_left(self, dt: float) -> "Domain":
        """Shift the domain left by `dt`"""
        return self.shift_right(-dt)

    def t(self, i: int) -> float:
        return self._t_min + i*self._dt

    def dt(self) -> float:
        return self._dt

    def t_min(self) -> float:
        return self._t_min

    def t_max(self) -> float:
        return self.t(self._size-1)

    def ts(self) -> np.ndarray:
        """Array of all node coordinates in the domain"""
        return np.linspace(self._t_min, self.t_max(), self._size, dtype=np.longdouble, endpoint=True)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, i: int) -> float:
        return self.t(i)

    def __repr__(self) -> str:
        return f"Domain(t_min={self.t_min()}, t_max={self.t_max()}, size={len(self)}, dt={self.dt()})"
