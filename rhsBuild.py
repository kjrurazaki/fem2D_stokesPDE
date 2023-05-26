import numpy as np


def build_rhs(model):
    """
    Compute RHS giver f and g (from exact solution)
    """
    mode = "node"
    rhs = np.zeros((2 * model.lines_A + model.lines_B, 1))
    for iel in range(0, len(model.triang_velocity)):
        for iloc in range(0, model.triang_velocity.shape[1]):
            iglob = model.triang_velocity[iel, iloc]
            if mode == "node":
                f1 = model.solution.f_1[iglob - 1]
                f2 = model.solution.f_2[iglob - 1]
                g = model.solution.g[iglob - 1]
                if iloc < 3:
                    # f1 and # f2
                    rhs[iglob - 1, :] = rhs[iglob - 1, :] + model.Area[iel] * f1 / 3
                    rhs[iglob - 1 + model.lines_A, :] = (
                        rhs[iglob - 1 + model.lines_A, :] + model.Area[iel] * f2 / 3
                    )
                else:
                    # f1 and # f2
                    rhs[iglob - 1, :] = (
                        rhs[iglob - 1, :] + 9 * model.Area[iel] * f1 / 20
                    )
                    rhs[iglob - 1 + model.lines_A, :] = (
                        rhs[iglob - 1 + model.lines_A, :]
                        + 9 * model.Area[iel] * f2 / 20
                    )
                # g
                if iglob <= model.lines_B:
                    rhs[iglob - 1 + 2 * model.lines_A, :] = (
                        rhs[iglob - 1 + 2 * model.lines_A, :] + model.Area[iel] * g / 3
                    )
                    # g stabilization (GLS)
                    if model.delta is not None:
                        rhs[iglob - 1 + 2 * model.lines_A, :] = rhs[
                            iglob - 1 + 2 * model.lines_A, :
                        ] + model.delta * 1 / 6 * (
                            f1 * model.Bloc[iel, iloc] + f2 * model.Cloc[iel, iloc]
                        )
            if mode == "mean":
                f1_values = model.solution.f_1[model.triang_velocity[iel] - 1]
                f2_values = model.solution.f_2[model.triang_velocity[iel] - 1]
                g_values = model.solution.g[model.triang_velocity[iel] - 1]
                if iloc < 3:
                    # f1 and # f2
                    rhs[iglob - 1, :] = rhs[iglob - 1, :] + 1 / 3 * model.Area[
                        iel
                    ] * np.mean(f1_values)
                    rhs[iglob - 1 + model.lines_A, :] = rhs[
                        iglob - 1 + model.lines_A, :
                    ] + 1 / 3 * model.Area[iel] * np.mean(f2_values)
                else:
                    rhs[iglob - 1, :] = rhs[iglob - 1, :] + 9 / 20 * model.Area[
                        iel
                    ] * np.mean(f1_values)
                    rhs[iglob - 1 + model.lines_A, :] = rhs[
                        iglob - 1 + model.lines_A, :
                    ] + 9 / 29 * model.Area[iel] * np.mean(f2_values)
                # g
                if iglob <= model.lines_B:
                    rhs[iglob - 1 + 2 * model.lines_A, :] = rhs[
                        iglob - 1 + 2 * model.lines_A, :
                    ] + 1 / 3 * model.Area[iel] * np.mean(g_values)
                    # g stabilization (GLS)
                    if model.delta is not None:
                        rhs[iglob - 1 + 2 * model.lines_A, :] = rhs[
                            iglob - 1 + 2 * model.lines_A, :
                        ] + model.delta * 1 / 6 * (
                            np.mean(f1_values) * model.Bloc[iel, iloc]
                            + np.mean(f2_values) * model.Cloc[iel, iloc]
                        )
    # g is -int(gv)
    rhs[2 * model.lines_A :, 0] = -1 * rhs[2 * model.lines_A :, 0]

    return rhs
