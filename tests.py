import numpy as np
import math
from model import Model

from solver import gauss_quadrature_weights_and_points, quadrature_integral


def integral_tests():
    # Check integral
    xw = gauss_quadrature_weights_and_points(4)
    phi_values = np.array([]).reshape(-1, 2)  # store values of the phi function

    area_triangle = 1 / 2

    for x, y, weight in xw:
        phi_values = np.vstack((phi_values, [(1 - x - y) * x * y, weight]))

    assert (
        quadrature_integral(phi_values[:, 0], phi_values[:, 1]) * area_triangle
        - 1 / 120
    ) <= 1e9

    # Check integral
    xw = gauss_quadrature_weights_and_points(8)
    phi_values = np.array([]).reshape(-1, 2)  # store values of the phi function
    for x, y, weight in xw:
        phi_values = np.vstack(
            (phi_values, [(1 - x - y) * x * y * (1 - x - y), weight])
        )

    def integral_barycentric(a, b, c):
        """
        Computes the integra of phi^a * phi^b * phi^c
        Unitary area (|T| = 1)
        """
        return (
            2
            * math.factorial(a)
            * math.factorial(b)
            * math.factorial(c)
            / math.factorial(a + b + c + 2)
        )

    integral_phi = area_triangle * integral_barycentric(1, 1, 2)
    integral_quadrature = (
        quadrature_integral(phi_values[:, 0], phi_values[:, 1]) * area_triangle
    )

    assert (integral_quadrature - integral_phi) <= 1e9


def basis_functests():
    meshdir = "./Meshes/mesh0"

    method = "lifting"
    delta = None  # Numerical stabilization, None == No numerical stabilization
    c = 1
    # Elements type could be linear, bubble or p1-iso-p2
    model_stokes = Model(meshdir, c, delta, only_dirichlet=True, element_type="linear")
    iel = 20
    Aloc = model_stokes.element.Aloc[iel, :]
    Bloc = model_stokes.element.Bloc[iel, :]
    Cloc = model_stokes.element.Cloc[iel, :]
    triang = model_stokes.element.triang_velocity[iel, :3]
    for iloc in range(0, 3):
        iglob = triang[iloc] - 1
        x, y = model_stokes.element.coord[iglob, :]
        phi_value = Aloc + x * Bloc + y * Cloc
        print(iloc, phi_value)


def basis_functions_comp():
    meshdir = "./Meshes/mesh0"
    method = "lifting"
    delta = None  # Numerical stabilization, None == No numerical stabilization
    c = 1
    model_stokes = Model(meshdir, c, delta, only_dirichlet=True, element_type="linear")
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    area = np.zeros((model_stokes.element.Nelem, 1))
    a = np.zeros((model_stokes.element.Nelem, 3))
    b = np.zeros((model_stokes.element.Nelem, 3))
    c = np.zeros((model_stokes.element.Nelem, 3))
    for iel in range(0, model_stokes.element.Nelem):
        nodes = model_stokes.triang_velocity[iel, :3]
        p1 = model_stokes.coord[nodes[0] - 1, :].reshape(
            1, -1
        )  # coordinate of first node
        p2 = model_stokes.coord[nodes[1] - 1, :].reshape(
            1, -1
        )  # coordinate of second node
        p3 = model_stokes.coord[nodes[2] - 1, :].reshape(1, -1)
        V = np.array(
            [[1, p1[0, 0], p1[0, 1]], [1, p2[0, 0], p2[0, 1]], [1, p3[0, 0], p3[0, 1]]]
        )  # Vandermonde matrix
        area[iel] = np.linalg.det(V) / 2
        abc1 = np.linalg.solve(V, v1 * np.linalg.det(V))
        abc2 = np.linalg.solve(V, v2 * np.linalg.det(V))
        abc3 = np.linalg.solve(V, v3 * np.linalg.det(V))
        a[iel, :] = [abc1[0], abc2[0], abc3[0]]
        b[iel, :] = [abc1[1], abc2[1], abc3[1]]
        c[iel, :] = [abc1[2], abc2[2], abc3[2]]

    iel = 20
    Aloc = a[iel, :]
    Bloc = b[iel, :]
    Cloc = c[iel, :]
    triang = model_stokes.element.triang_velocity[iel, :3]
    for iloc in range(0, 3):
        iglob = triang[iloc] - 1
        x, y = model_stokes.element.coord[iglob, :]
        phi_value = Aloc + x * Bloc + y * Cloc
        print(iloc, phi_value)


def basis_functions_comp_2():
    meshdir = "./Meshes/mesh0"
    method = "lifting"
    delta = None  # Numerical stabilization, None == No numerical stabilization
    c = 1
    model_stokes = Model(meshdir, c, delta, only_dirichlet=True, element_type="linear")
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    area = np.zeros((model_stokes.element.Nelem, 1))
    a = np.zeros((model_stokes.element.Nelem, 3))
    b = np.zeros((model_stokes.element.Nelem, 3))
    c = np.zeros((model_stokes.element.Nelem, 3))
    for iel in range(0, model_stokes.element.Nelem):
        nodes = model_stokes.triang_velocity[iel, :3]
        p1 = model_stokes.coord[nodes[0] - 1, :].reshape(
            1, -1
        )  # coordinate of first node
        p2 = model_stokes.coord[nodes[1] - 1, :].reshape(
            1, -1
        )  # coordinate of second node
        p3 = model_stokes.coord[nodes[2] - 1, :].reshape(1, -1)
        V = np.array(
            [[1, p1[0, 0], p1[0, 1]], [1, p2[0, 0], p2[0, 1]], [1, p3[0, 0], p3[0, 1]]]
        )  # Vandermonde matrix
        area[iel] = np.linalg.det(V) / 2
        abc1 = np.linalg.solve(V, v1)
        abc2 = np.linalg.solve(V, v2)
        abc3 = np.linalg.solve(V, v3)
        a[iel, :] = [abc1[0], abc2[0], abc3[0]]
        b[iel, :] = [abc1[1], abc2[1], abc3[1]]
        c[iel, :] = [abc1[2], abc2[2], abc3[2]]

    iel = 20
    Aloc = a[iel, :]
    Bloc = b[iel, :]
    Cloc = c[iel, :]
    triang = model_stokes.element.triang_velocity[iel, :3]
    for iloc in range(0, 3):
        iglob = triang[iloc] - 1
        x, y = model_stokes.element.coord[iglob, :]
        phi_value = Aloc + x * Bloc + y * Cloc
        print(iloc, phi_value)


# Test with basis functions were built correctly
basis_functests()
basis_functions_comp()
basis_functions_comp_2()
