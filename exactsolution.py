import numpy as np


class Solution:
    """
    Exact solution for the coordinates from elements
    """

    def __init__(self, element):
        self.coord = element.coord
        self.c = element.c
        self.viscosity = element.viscosity
        self.exact_solution()

    def exact_solution(self):
        """
        Solution if its equal the one of the graph (u_2 last sin is y coordinate)
        """
        # Velocities
        self.u_1 = -1 * np.multiply(
            np.cos(2 * np.pi * self.coord[:, 0]), np.sin(2 * np.pi * self.coord[:, 1])
        ) + np.sin(2 * np.pi * self.coord[:, 1])
        self.u_2 = np.multiply(
            np.sin(2 * np.pi * self.coord[:, 0]), np.cos(2 * np.pi * self.coord[:, 1])
        ) - np.sin(2 * np.pi * self.coord[:, 0])
        # Partial derivatives:
        self.u_1_x = (
            2
            * np.pi
            * np.multiply(
                np.sin(2 * np.pi * self.coord[:, 0]),
                np.sin(2 * np.pi * self.coord[:, 1]),
            )
        )
        self.u_1_y = -2 * np.pi * np.multiply(
            np.cos(2 * np.pi * self.coord[:, 0]), np.cos(2 * np.pi * self.coord[:, 1])
        ) + 2 * np.pi * np.cos(2 * np.pi * self.coord[:, 1])
        self.u_2_x = 2 * np.pi * np.multiply(
            np.cos(2 * np.pi * self.coord[:, 0]), np.cos(2 * np.pi * self.coord[:, 1])
        ) - 2 * np.pi * np.cos(2 * np.pi * self.coord[:, 0])
        self.u_2_y = (
            -2
            * np.pi
            * np.multiply(
                np.sin(2 * np.pi * self.coord[:, 0]),
                np.sin(2 * np.pi * self.coord[:, 1]),
            )
        )
        # Partial derivatives 2nd order - Only for debugging:
        self.u_1_xx = (
            4
            * np.pi**2
            * np.multiply(
                np.cos(2 * np.pi * self.coord[:, 0]),
                np.sin(2 * np.pi * self.coord[:, 1]),
            )
        )
        self.u_1_yy = 4 * np.pi**2 * np.multiply(
            np.cos(2 * np.pi * self.coord[:, 0]), np.sin(2 * np.pi * self.coord[:, 1])
        ) - 4 * np.pi**2 * np.sin(2 * np.pi * self.coord[:, 1])
        self.u_2_xx = -4 * np.pi**2 * np.multiply(
            np.sin(2 * np.pi * self.coord[:, 0]), np.cos(2 * np.pi * self.coord[:, 1])
        ) + 4 * np.pi**2 * np.sin(2 * np.pi * self.coord[:, 0])
        self.u_2_yy = (
            -4
            * np.pi**2
            * np.multiply(
                np.sin(2 * np.pi * self.coord[:, 0]),
                np.cos(2 * np.pi * self.coord[:, 1]),
            )
        )

        # Pressure
        self.sol_p = (
            2
            * np.pi
            * (
                np.cos(2 * np.pi * self.coord[:, 1])
                - np.cos(2 * np.pi * self.coord[:, 0])
            )
        )

        # Partial derivative pressure:
        self.sol_p_x = 4 * np.pi**2 * np.sin(2 * np.pi * self.coord[:, 0])
        self.sol_p_y = -4 * np.pi**2 * np.sin(2 * np.pi * self.coord[:, 1])

        # RHSs TODO viscosity is constant (if it should change between nodes/elements this should be changed)
        self.f_1 = (
            self.c * self.u_1
            - 4
            * self.viscosity[0]
            * np.pi**2
            * np.multiply(
                np.sin(2 * np.pi * self.coord[:, 1]),
                2 * np.cos(2 * np.pi * self.coord[:, 0]) - 1,
            )
            + 4 * np.pi**2 * np.sin(2 * np.pi * self.coord[:, 0])
        )
        self.f_2 = (
            self.c * self.u_2
            + 4
            * self.viscosity[0]
            * np.pi**2
            * np.multiply(
                np.sin(2 * np.pi * self.coord[:, 0]),
                2 * np.cos(2 * np.pi * self.coord[:, 1]) - 1,
            )
            - 4 * np.pi**2 * np.sin(2 * np.pi * self.coord[:, 1])
        )
        self.g = np.zeros((len(self.coord),))
        # Check
        assert (
            sum(
                (
                    self.c * self.u_1
                    - self.viscosity[0] * (self.u_1_xx + self.u_1_yy)
                    + self.sol_p_x
                )
                - self.f_1
            )
            <= 1e-9
        )
        assert (
            sum(
                (
                    self.c * self.u_2
                    - self.viscosity[0] * (self.u_2_xx + self.u_2_yy)
                    + self.sol_p_y
                )
                - self.f_2
            )
            <= 1e-9
        )
        assert sum((self.u_1_x + self.u_2_y) - self.g) <= 1e-9
