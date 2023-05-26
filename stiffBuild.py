import numpy as np
import math
from solver import gauss_quadrature_weights_and_points, find_key_for_value

from localBasis import phi_funcs
from solver import quadrature_integral


def elementalBuild(model):
    """
    Elemental matrix
    a^T_{ij} = int( grad(phi_i):grad(phi_j)dx
            = --- (b_i*b_j + c_i*c_j)
    Used to linear and p1-iso-p2 elements
    """
    elementalMat = np.zeros((2 * model.lines_A, 2 * model.lines_A))
    mult_elementalMat = np.zeros((2 * model.lines_A, 2 * model.lines_A))
    for iel in range(0, len(model.triang_velocity)):
        for iloc in range(0, 3):
            iglob = model.triang_velocity[iel, iloc]
            iglob_A2 = iglob + model.lines_A
            for jloc in range(0, 3):
                jglob = model.triang_velocity[iel, jloc]
                jglob_A2 = jglob + model.lines_A

                B = (
                    model.Bloc[iel, iloc] * model.Bloc[iel, jloc]
                    + model.Cloc[iel, iloc] * model.Cloc[iel, jloc]
                ) / (4 * model.Area[iel])

                elementalMat[iglob - 1, jglob - 1] = (
                    elementalMat[iglob - 1, jglob - 1] + model.viscosity[iel] * B
                )
                elementalMat[iglob_A2 - 1, jglob_A2 - 1] = elementalMat[
                    iglob - 1, jglob - 1
                ]
    return elementalMat


def elemental_bubbleBuild(model):
    """
    Elemental matrix for bubble element (central point added)
    a^T_{ij} = int( grad(phi_i):grad(phi_j)dx
            = --- (b_i*b_j + c_i*c_j)
    """
    elementalMat = np.zeros((2 * model.lines_A, 2 * model.lines_A))
    for iel in range(0, len(model.triang_velocity)):
        for iloc in range(0, 4):
            iglob = model.triang_velocity[iel, iloc]
            iglob_A2 = iglob + model.lines_A
            for jloc in range(0, 4):
                jglob = model.triang_velocity[iel, jloc]
                jglob_A2 = jglob + model.lines_A
                if iloc != 3 and jloc != 3:
                    elementalMat[iglob - 1, jglob - 1] = (
                        elementalMat[iglob - 1, jglob - 1]
                        + model.viscosity[iel]
                        * (
                            model.Bloc[iel, iloc] * model.Bloc[iel, jloc]
                            + model.Cloc[iel, iloc] * model.Cloc[iel, jloc]
                        )
                        / 4
                        / model.Area[iel]
                    )
                elif iloc != 3 and jloc == 3:
                    elementalMat[iglob - 1, jglob - 1] = (
                        elementalMat[iglob - 1, jglob - 1]
                        + model.viscosity[iel]
                        * (
                            model.Bloc[iel, iloc] * np.sum(model.Bloc[iel, :])
                            + model.Cloc[iel, iloc] * np.sum(model.Cloc[iel, :])
                        )
                        * 27
                        / 48
                        / model.Area[iel]
                    )
                elif jloc != 3 and iloc == 3:
                    elementalMat[iglob - 1, jglob - 1] = (
                        elementalMat[iglob - 1, jglob - 1]
                        + model.viscosity[iel]
                        * (
                            model.Bloc[iel, jloc] * np.sum(model.Bloc[iel, :])
                            + model.Cloc[iel, jloc] * np.sum(model.Cloc[iel, :])
                        )
                        * 27
                        / 48
                        / model.Area[iel]
                    )
                elif (
                    iloc == 3 and jloc == 3
                ):  # both basis functions are bubble functions
                    elementalMat[iglob - 1, jglob - 1] = (
                        elementalMat[iglob - 1, jglob - 1]
                        + model.viscosity[iel]
                        * (
                            np.sum(model.Bloc[iel, :] ** 2)
                            + np.sum(
                                [
                                    model.Bloc[iel, 0] * model.Bloc[iel, 1],
                                    model.Bloc[iel, 0] * model.Bloc[iel, 2],
                                    model.Bloc[iel, 1] * model.Bloc[iel, 2],
                                ]
                            )
                            + np.sum(model.Cloc[iel, :] ** 2)
                            + np.sum(
                                [
                                    model.Cloc[iel, 0] * model.Cloc[iel, 1],
                                    model.Cloc[iel, 0] * model.Cloc[iel, 2],
                                    model.Cloc[iel, 1] * model.Cloc[iel, 2],
                                ]
                            )
                        )
                        * 27**2
                        / 360
                        / model.Area[iel]
                    )
                elementalMat[iglob_A2 - 1, jglob_A2 - 1] = elementalMat[
                    iglob - 1, jglob - 1
                ]
    return elementalMat


def massBuild(model):
    """
    Mass matrix build
    """
    massMat = np.zeros((2 * model.lines_A, 2 * model.lines_A))
    for iel in range(0, len(model.triang_velocity)):
        for iloc in range(0, 3):
            iglob = model.triang_velocity[iel, iloc]
            iglob_A2 = iglob + model.lines_A
            for jloc in range(0, 3):
                jglob = model.triang_velocity[iel, jloc]
                jglob_A2 = jglob + model.lines_A
                if iglob == jglob:
                    massMat[iglob - 1, jglob - 1] = (
                        massMat[iglob - 1, jglob - 1] + model.Area[iel] / 6
                    )
                elif iglob != jglob:
                    massMat[iglob - 1, jglob - 1] = (
                        massMat[iglob - 1, jglob - 1] + model.Area[iel] / 12
                    )

                massMat[iglob_A2 - 1, jglob_A2 - 1] = massMat[iglob - 1, jglob - 1]
    return massMat


def mass_bubbleBuild(model):
    """
    Mass matrix build considering central nodes
    """
    massMat = np.zeros((2 * model.lines_A, 2 * model.lines_A))
    for iel in range(0, len(model.triang_velocity)):
        for iloc in range(0, 4):
            iglob = model.triang_velocity[iel, iloc]
            iglob_A2 = iglob + model.lines_A
            for jloc in range(0, 4):
                jglob = model.triang_velocity[iel, jloc]
                jglob_A2 = jglob + model.lines_A
                if iloc != 3 and jloc != 3:
                    if iglob == jglob:
                        massMat[iglob - 1, jglob - 1] = (
                            massMat[iglob - 1, jglob - 1] + model.Area[iel] / 6
                        )
                    elif iglob != jglob:
                        massMat[iglob - 1, jglob - 1] = (
                            massMat[iglob - 1, jglob - 1] + model.Area[iel] / 12
                        )
                elif iloc != 3 or jloc != 3:
                    massMat[iglob - 1, jglob - 1] = (
                        massMat[iglob - 1, jglob - 1] + 3 * model.Area[iel] / 20
                    )
                else:
                    massMat[iglob - 1, jglob - 1] = (
                        massMat[iglob - 1, jglob - 1] + 27**2 * model.Area[iel] / 2520
                    )
                massMat[iglob_A2 - 1, jglob_A2 - 1] = massMat[iglob - 1, jglob - 1]
    return massMat


def divBuild(model):
    """
    Divergence matrix build
    """
    divMat = np.zeros((model.lines_B, 2 * model.lines_A))
    for iel in range(0, len(model.triang_velocity)):
        for iloc in range(0, 3):
            iglob = model.triang_pressure[iel, iloc]
            for jloc in range(0, 3):
                jglob = model.triang_velocity[iel, jloc]
                jglob_A2 = jglob + model.lines_A
                divMat[iglob - 1, jglob - 1] = (
                    divMat[iglob - 1, jglob - 1] + model.Bloc[iel, jloc] / 6
                )
                divMat[iglob - 1, jglob_A2 - 1] = (
                    divMat[iglob - 1, jglob_A2 - 1] + model.Cloc[iel, jloc] / 6
                )
    return divMat


def div_bubbleBuild(model):
    """
    Divergence matrix build
    """
    divMat = np.zeros((model.lines_B, 2 * model.lines_A))
    for iel in range(0, len(model.triang_velocity)):
        for iloc in range(0, 3):
            iglob = model.triang_pressure[iel, iloc]
            for jloc in range(0, 4):
                jglob = model.triang_velocity[iel, jloc]
                jglob_A2 = jglob + model.lines_A
                if jloc != 3 and iloc != 3:
                    divMat[iglob - 1, jglob - 1] = (
                        divMat[iglob - 1, jglob - 1] + model.Bloc[iel, jloc] / 6
                    )
                    divMat[iglob - 1, jglob_A2 - 1] = (
                        divMat[iglob - 1, jglob_A2 - 1] + model.Cloc[iel, jloc] / 6
                    )
                elif jloc == 3 and iloc != 3:
                    divMat[iglob - 1, jglob - 1] = (
                        divMat[iglob - 1, jglob - 1]
                        + (
                            model.Bloc[iel, iloc]
                            + 2 * np.sum(model.Bloc[iel, np.delete([0, 1, 2], iloc)])
                        )
                        * 27
                        / 120
                    )
                    divMat[iglob - 1, jglob_A2 - 1] = (
                        divMat[iglob - 1, jglob_A2 - 1]
                        + (
                            model.Cloc[iel, iloc]
                            + 2 * np.sum(model.Cloc[iel, np.delete([0, 1, 2], iloc)])
                        )
                        * 27
                        / 120
                    )
    return divMat


def div_p1isop2Build(model):
    """
    Divergence matrix build for p1-iso-p2 element
    Note integrations are for linear polynomials
    """
    divMat = np.zeros((model.lines_B, 2 * model.lines_A))
    for iel in range(0, len(model.triang_velocity)):
        iel_pressure = math.floor(iel / 4)
        for iloc in range(0, 3):
            iglob = model.triang_pressure[iel_pressure, iloc]
            for jloc in range(0, 3):
                jglob = model.triang_velocity[iel, jloc]
                jglob_A2 = jglob + model.lines_A
                if (
                    iel + 1
                ) % 4 != 0:  # central smaller triangle in mesh h - no common nodes with mesh 2h
                    assert iel in model.triangles_relation[iel_pressure]
                    if (
                        iglob in model.triang_velocity[iel, :]
                    ):  # (model.coord[iglob - 1,:] == model.coord[jglob - 1,:]).all():
                        divMat[iglob - 1, jglob - 1] = (
                            divMat[iglob - 1, jglob - 1] + model.Bloc[iel, jloc] / 3
                        )
                        divMat[iglob - 1, jglob_A2 - 1] = (
                            divMat[iglob - 1, jglob_A2 - 1] + model.Cloc[iel, jloc] / 3
                        )
                    else:
                        divMat[iglob - 1, jglob - 1] = (
                            divMat[iglob - 1, jglob - 1] + model.Bloc[iel, jloc] / 12
                        )
                        divMat[iglob - 1, jglob_A2 - 1] = (
                            divMat[iglob - 1, jglob_A2 - 1] + model.Cloc[iel, jloc] / 12
                        )
                elif (
                    iel + 1
                ) % 4 == 0:  # central smaller triangle in mesh h - no common nodes with mesh 2h
                    divMat[iglob - 1, jglob - 1] = (
                        divMat[iglob - 1, jglob - 1] + model.Bloc[iel, jloc] / 6
                    )
                    divMat[iglob - 1, jglob_A2 - 1] = (
                        divMat[iglob - 1, jglob_A2 - 1] + model.Cloc[iel, jloc] / 6
                    )
    return divMat


def sup_stabilization(model):
    """
    Streamline upwind stabilization
    a^T_{ij} = delta * A_T # TODO update this stabilization description
    """
    numdiffMat = np.zeros((model.lines_B, model.lines_B))
    for iel in range(0, len(model.triang_velocity)):
        for iloc in range(0, 3):
            iglob = model.triang_pressure[iel, iloc]
            for jloc in range(0, 3):
                jglob = model.triang_velocity[iel, jloc]
                numdiffMat[iglob - 1, jglob - 1] = numdiffMat[
                    iglob - 1, jglob - 1
                ] + model.delta * (
                    model.Bloc[iel, iloc] * model.Bloc[iel, jloc]
                    + model.Cloc[iel, iloc] * model.Cloc[iel, jloc]
                ) / (
                    4 * model.Area[iel]
                )
    return numdiffMat


def stiffBuild(model):
    """
    Build stifness matrix
    """
    if (model.element_type == "linear") or (model.element_type == "linear_gls"):
        elementalMat = elementalBuild(model)
        massMat = massBuild(model)
        divMat = divBuild(model)

    elif model.element_type == "bubble":
        elementalMat = elemental_bubbleBuild(model)
        massMat = mass_bubbleBuild(model)
        divMat = div_bubbleBuild(model)

    elif model.element_type == "p1-iso-p2":
        elementalMat = elementalBuild(model)
        massMat = massBuild(model)
        divMat = div_p1isop2Build(model)

    AMat = model.c * massMat + elementalMat
    zeroMat = np.zeros((divMat.shape[0], divMat.shape[0]))
    stiffMat = np.concatenate([AMat, -1 * divMat.transpose()], axis=1)
    stiffMat = np.concatenate(
        [stiffMat, np.concatenate([-1 * divMat, zeroMat], axis=1)], axis=0
    )

    if model.delta is not None:
        numdiffMat = sup_stabilization(model)
        stiffMat[2 * model.lines_A :, 2 * model.lines_A :] = (
            stiffMat[2 * model.lines_A :, 2 * model.lines_A :] - numdiffMat
        )
    else:
        numdiffMat = np.array([]).reshape(1, -1)

    return elementalMat, massMat, divMat, numdiffMat, stiffMat
