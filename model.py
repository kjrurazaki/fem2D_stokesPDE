from stiffBuild import stiffBuild
from element import Element
from rhsBuild import build_rhs

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import display_results


class Model:
    def __init__(self, meshdir, c, delta, only_dirichlet, element_type):
        self.meshdir = meshdir
        self.c = c
        self.only_dirichlet = only_dirichlet
        self.delta = delta
        self.load_data()
        self.element = Element(self, element_type)
        self.update_model()
        self.lines_A = self.element.lines_A
        self.lines_B = self.element.lines_B
        self.element_type = self.element.element_type
        self.delete_auxiliary()
        # load element
        self.build_model()

    def load_data(self):
        """
        Define the 2D mesh and equation BC's and coefficients
        """
        _mesh = np.genfromtxt(self.meshdir + "/mesh.dat", dtype=int)
        self.coord = np.genfromtxt(self.meshdir + "/xy.dat")
        self.NeuNod = np.genfromtxt(self.meshdir + "/neunod.dat", dtype=int)
        self.DirNod = np.genfromtxt(self.meshdir + "/DirNod.dat", dtype=int)

        # Remove indices of elements
        self.triang = _mesh[:, 0:3]
        self.coord = self.coord[:, :]
        self.DirNod = self.DirNod.reshape(-1, 1)
        self.NeuNod = self.NeuNod.reshape(-1, 1)
        self.Diff = _mesh[:, 3].reshape(-1, 1).astype(float)  # Viscosity

    def build_model(self):
        """
        Build  stiffness matrix
        Should not have any modifier of the model
        """
        # build stiffness matrix (without BCs)
        _matrices = stiffBuild(self.element)
        self.elementalMat = _matrices[0]
        self.massMat = _matrices[1]
        self.divMat = _matrices[2]
        self.numdiffMat = _matrices[3]
        self.stiffMat = _matrices[4]

        # build rhs (with exact solution)
        self.rhs = build_rhs(self.element)

    def update_model(self):
        self.triang_pressure = self.element.triang_pressure
        self.triang_velocity = self.element.triang_velocity
        self.coord = self.element.coord

    def update_solution(self, values_velocity, values_pressure):
        self.uh = values_velocity
        self.p = values_pressure

    def update_rhs(self, rhs):
        self.rhs = rhs

    def delete_auxiliary(self):
        self.DirNod = None  # (changed Dirnod_pressure/velocity)
        self.Nodes = None  # (changed lines_A or lines_B)
        self.DirVal = None  # (changed for DirVal_pressure/velocity)
        self.NDir = None  # (changed for NDir_pressure/velocity)
        self.triang = None  # (changed for triang_pressure/velocity)

    def compute_residual(self):
        """
        Compare solution find with the true one
        Returns total residual and the one for each variable (u1, u2 and p)
        """
        u1_residual = self.uh[: self.lines_A, :] - self.element.solution.u_1
        u1_residual = [
            np.linalg.norm(u1_residual),
            np.linalg.norm(u1_residual / self.lines_A),
        ]
        u2_residual = (
            self.uh[self.lines_A : 2 * self.lines_A] - self.element.solution.u_2
        )
        u2_residual = [
            np.linalg.norm(u2_residual),
            np.linalg.norm(u2_residual / self.lines_A),
        ]
        p_residual = self.p - self.element.solution.sol_p[: self.lines_B]
        u2_residual = [
            np.linalg.norm(p_residual),
            np.linalg.norm(p_residual / self.lines_A),
        ]
        total_residual = u1_residual + u2_residual + p_residual
        return total_residual, u1_residual, u2_residual, p_residual

    def plot_boundary(self, save_name=None):
        """
        Plots boundary showing the nodes type (Dirichlet or Neumann)
        """
        # Extract x and y coordinates
        x_coords, y_coords = self.coord[:, 0], self.coord[:, 1]

        # Plot all points
        plt.scatter(x_coords, y_coords, color="grey", label="Points")

        # Plot and connect boundary nodes from the first array
        boundary_1_coords = self.coord[
            np.sort(self.element.DirNod_velocity.flatten(), axis=0) - 1
        ]
        x_boundary_1, y_boundary_1 = boundary_1_coords[:, 0], boundary_1_coords[:, 1]
        plt.scatter(x_boundary_1, y_boundary_1, color="red", label="Dirichlet boundary")
        # plt.plot(x_boundary_1, y_boundary_1, color='red')

        # Plot and connect boundary nodes from the second array
        if len(self.element.NeuNod) > 0:
            boundary_2_coords = self.coord[self.element.NeuNod.flatten() - 1]
            x_boundary_2, y_boundary_2 = (
                boundary_2_coords[:, 0],
                boundary_2_coords[:, 1],
            )
            plt.scatter(
                x_boundary_2, y_boundary_2, color="blue", label="Neumann boundary"
            )
            plt.plot(x_boundary_2, y_boundary_2, color="blue")

        # Add a legend to show which color represents which boundary
        plt.legend()
        if save_name is not None:
            self.save_fig(f"{save_name}")
        # Display the plot
        plt.draw()
        plt.pause(0.001)

    def plot_solution(self, title=True, save_name=None):
        fig, ax = plt.subplots(2, 2, figsize=[15, 8])
        display_results.plot_field_2D(
            self.coord[: self.lines_A, :],
            self.triang_velocity[:, :3],
            self.element.solution.u_1[: self.lines_A],
            nodal=False,
            ax=ax[0, 0],
            cmap="coolwarm",
        )
        display_results.plot_field_2D(
            self.coord[: self.lines_A, :],
            self.triang_velocity[:, :3],
            self.element.solution.u_2[: self.lines_A],
            nodal=False,
            ax=ax[0, 1],
            cmap="coolwarm",
        )
        display_results.plot_field_2D(
            self.coord[: self.lines_B, :],
            self.triang_pressure[:, :3],
            self.element.solution.sol_p[: self.lines_B],
            nodal=False,
            ax=ax[1, 0],
            cmap="coolwarm",
        )
        if title == True:
            fig.suptitle("True solution")
        if save_name is not None:
            self.save_fig(f"{save_name}")
        plt.draw()
        plt.pause(0.001)

    def plot_dirichlet(self, title=True, save_name=None):
        fig, ax = plt.subplots(2, 2, figsize=[15, 8])
        velocities = np.zeros((self.lines_A, 2))
        velocities[
            self.element.DirNod_velocity.flatten() - 1, 0
        ] = self.element.DirVal_velocity[:, 0]
        velocities[
            self.element.DirNod_velocity.flatten() - 1, 1
        ] = self.element.DirVal_velocity[:, 1]

        pressures = np.zeros((self.lines_B, 1))
        pressures[
            self.element.DirNod_pressure.flatten() - 1, 0
        ] = self.element.DirVal_pressure[:, 0]
        display_results.plot_field_2D(
            self.coord[:, :],
            self.triang_velocity[:, :3],
            velocities[:, 0],
            nodal=False,
            ax=ax[0, 0],
            cmap="coolwarm",
        )
        display_results.plot_field_2D(
            self.coord[:, :],
            self.triang_velocity[:, :3],
            velocities[:, 1],
            nodal=False,
            ax=ax[0, 1],
            cmap="coolwarm",
        )
        display_results.plot_field_2D(
            self.coord[: self.lines_B, :],
            self.triang_pressure[:, :3],
            pressures[:, 0],
            nodal=False,
            ax=ax[1, 0],
            cmap="coolwarm",
        )
        if title == True:
            fig.suptitle("Dirichlets original")
        if save_name is not None:
            self.save_fig(f"{save_name}")
        plt.draw()
        plt.pause(0.001)

    def plot_rhs(self, title=True, save_name=None):
        fig, ax = plt.subplots(2, 2, figsize=[15, 8])
        display_results.plot_field_2D(
            self.coord[: self.lines_A, :],
            self.triang_velocity[:, :3],
            self.rhs[: self.lines_A].flatten(),
            nodal=False,
            ax=ax[0, 0],
            cmap="coolwarm",
        )
        display_results.plot_field_2D(
            self.coord[: self.lines_A, :],
            self.triang_velocity[:, :3],
            self.rhs[self.lines_A : 2 * self.lines_A].flatten(),
            nodal=False,
            ax=ax[0, 1],
            cmap="coolwarm",
        )
        display_results.plot_field_2D(
            self.coord[: self.lines_B, :],
            self.triang_pressure[:, :3],
            self.rhs[2 * self.lines_A : 2 * self.lines_A + self.lines_B].flatten(),
            nodal=False,
            ax=ax[1, 0],
            cmap="coolwarm",
        )
        if title == True:
            fig.suptitle("rhs original")
        if save_name is not None:
            self.save_fig(f"{save_name}")
        plt.draw()
        plt.pause(0.001)

    def print_parameters(self):
        """
        Print parameters
        """
        print(
            self.Nelem,
            "\n",
            self.Nodes,
            "\n",
            self.NDir,
            "\n",
            self.NNeu,
            "\n",
            self.triang[0:5, :],
            "\n",
            self.coord[0:5, :],
            "\n",
            self.DirNod[0:5, :],
            "\n",
            self.DirVal[0:5, :],
            "\n",
            #   self.NeuNod[0:5, :], "\n",
            #   self.NeuVal[0:5, :], "\n",
            self.Diff[0:5, :],
            "\n",
        )

        print(
            self.Aloc[0:5, :],
            "\n",
            self.Bloc[0:5, :],
            "\n",
            self.Cloc[0:5, :],
            "\n",
            self.Area[0:5, :],
            "\n",
            self.h[0:5, :],
        )

        print(
            self.elementalMat[0:5, :],
            "\n",
            self.massMat[0:5, :],
            "\n",
            self.divMat[0:5, :],
            "\n",
            self.numdiffMat[0:5, :],
            "\n",
            self.stiffMat[0:5, :],
            "\n",
        )

    def save_fig(name):
        """
        Save the figure
        """
        plt.savefig(
            f"figures/fig_{name}.pdf",
            bbox_inches="tight",
            format="pdf",
        )
