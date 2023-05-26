import numpy as np


def localBasis(model):
    """
    Build local P1 basis functions on triangles
    only the coefficients (b,c) multiplying x and y are built
    phi(x, y) = a + bx + cy
    Built based on page 84 from notes
    a_i = x_j * y_k - x_k * y_j (i -> j -> k -> i)
    b_i = y_j - y_k (i -> j -> k -> i) - convection (e.g. b_2 = y_3 - y_1)
    c_i = x_k - x_j (i -> j -> k -> i)
    """
    Aloc = np.zeros((model.Nelem, 3))
    Bloc = np.zeros((model.Nelem, 3))
    Cloc = np.zeros((model.Nelem, 3))
    B_K_inv = np.zeros((model.Nelem, 2, 2))  # Mapping matrix to reference triangle
    Area = np.zeros((model.Nelem, 1))
    h = np.zeros((model.Nelem, 1))
    for iel in range(0, model.Nelem):
        nodes = model.triang[iel, :3]
        p1 = model.coord[nodes[0] - 1, :].reshape(1, -1)  # coordinate of first node
        p2 = model.coord[nodes[1] - 1, :].reshape(1, -1)  # coordinate of second node
        p3 = model.coord[nodes[2] - 1, :].reshape(1, -1)  # coordinate of third node
        A = np.concatenate(
            (np.ones((3, 1)), (np.concatenate((p1, p2, p3), axis=0))), axis=1
        )
        DetA = np.linalg.det(A)
        # Area[iel] = abs(DetA/2)
        Area[iel] = DetA / 2
        h[iel] = np.max(np.array([norm(p1, p2), norm(p1, p3), norm(p2, p3)]))
        for inod in range(1, 4):
            n1 = mod_n(inod + 1, 3)
            n2 = mod_n(inod + 2, 3)
            Aloc[iel, inod - 1] = (
                model.coord[nodes[n1 - 1] - 1, 0] * model.coord[nodes[n2 - 1] - 1, 1]
                - model.coord[nodes[n2 - 1] - 1, 0] * model.coord[nodes[n1 - 1] - 1, 1]
            )
            Bloc[iel, inod - 1] = (
                model.coord[nodes[n1 - 1] - 1, 1] - model.coord[nodes[n2 - 1] - 1, 1]
            )
            Cloc[iel, inod - 1] = (
                model.coord[nodes[n2 - 1] - 1, 0] - model.coord[nodes[n1 - 1] - 1, 0]
            )
    return Aloc, Bloc, Cloc, B_K_inv, Area, h


def norm(p1, p2):
    """
    Vector norm of vector defined by points p1 and p2
    """
    return np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)


def mod_n(i, j):
    """
    define the recirculation  (i -> j -> k -> i) - convection (e.g. b_2 = y_3 - y_1)
    """
    aux = np.mod(i, j)
    if aux == 0:
        remainder = j
    else:
        remainder = aux
    return remainder


def phi_funcs(model, iel, x, y):
    """
    Basis functions for bubble element problem - returns x, y values
    """
    phi_1 = model.Aloc[iel, 0] + model.Bloc[iel, 0] * x + model.Cloc[iel, 0] * y
    phi_2 = model.Aloc[iel, 1] + model.Bloc[iel, 1] * x + model.Cloc[iel, 1] * y
    phi_3 = model.Aloc[iel, 2] + model.Bloc[iel, 2] * x + model.Cloc[iel, 2] * y
    phi_B = 27 * phi_1 * phi_2 * phi_3
    phi_B_x = 27 * (
        model.Bloc[iel, 0] * phi_2 * phi_3
        + model.Bloc[iel, 1] * phi_1 * phi_3
        + model.Bloc[iel, 2] * phi_1 * phi_2
    )
    phi_B_y = 27 * (
        model.Cloc[iel, 0] * phi_2 * phi_3
        + model.Cloc[iel, 1] * phi_1 * phi_3
        + model.Cloc[iel, 2] * phi_1 * phi_2
    )
    return phi_1, phi_2, phi_3, phi_B, phi_B_x, phi_B_y
