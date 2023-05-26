"""
Some functions to print model properties to debug
"""


import numpy as np
import display_results
from matplotlib import pyplot as plt

def all_prints(model, dict_print):
    print_f = dict_print['print_f']
    print_rhs = dict_print['print_rhs']
    print_dirichlet = dict_print['print_dirichlet']
    print_solution = dict_print['print_solution']

    if print_f == 1:
                fig, ax = plt.subplots(2,2, figsize = [15, 8])
                display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                            model.triang_velocity[:, :3], 
                                            model.element.solution.f_1[:model.lines_A], 
                                            nodal = False, 
                                            ax = ax[0, 0],
                                            cmap = 'coolwarm')
                display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                            model.triang_velocity[:, :3], 
                                            model.element.solution.f_2[:model.lines_A], 
                                            nodal = False, 
                                            ax = ax[0, 1],
                                            cmap = 'coolwarm')
                display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                            model.triang_pressure[:, :3], 
                                            model.element.solution.g[:model.lines_A], 
                                            nodal = False, 
                                            ax = ax[1, 0],
                                            cmap = 'coolwarm')
                fig.suptitle('fs')
                plt.draw()
                plt.pause(0.001)
    
    if print_rhs == 1:
            fig, ax = plt.subplots(2,2, figsize = [15, 8])
            display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                          model.triang_velocity[:, :3], 
                                          model.rhs[:model.lines_A].flatten(), 
                                          nodal = False, 
                                          ax = ax[0,0],
                                          cmap = 'coolwarm')
            display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                          model.triang_velocity[:, :3], 
                                          model.rhs[model.lines_A : 2 * model.lines_A].flatten(), 
                                          nodal = False, 
                                          ax = ax[0,1],
                                          cmap = 'coolwarm')
            display_results.plot_field_2D(model.coord[:model.lines_B,:], 
                                          model.triang_pressure[:, :3], 
                                          model.rhs[2 * model.lines_A :].flatten(), 
                                          nodal = False, 
                                          ax = ax[1,0],
                                          cmap = 'coolwarm')
            fig.suptitle('rhs final')
            plt.draw()
            plt.pause(0.001)

    if print_dirichlet == 1:
            fig, ax = plt.subplots(2,2, figsize = [15, 8])
            rhs_dir = np.zeros((model.lines_A, 1))
            rhs_dir[model.boundary_nodes_velocity, 0] = model.rhs[model.boundary_nodes_velocity, 0]
            display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                          model.triang_velocity[:, :3], 
                                          rhs_dir.flatten(), 
                                          nodal = False, 
                                          ax = ax[0,0],
                                          cmap = 'coolwarm')
            
            rhs_dir = np.zeros((model.lines_A, 1))
            rhs_dir[model.boundary_nodes_velocity, 0] = model.rhs[[i + model.lines_A for i in model.boundary_nodes_velocity], 0]
            display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                          model.triang_velocity[:, :3], 
                                          rhs_dir.flatten(), 
                                          nodal = False, 
                                          ax = ax[0,1],
                                          cmap = 'coolwarm')

            rhs_dir = np.zeros((model.lines_B, 1))
            rhs_dir[model.boundary_nodes_pressure, 0] = model.rhs[[i + 2 * model.lines_A for i in model.boundary_nodes_pressure], 0]
            display_results.plot_field_2D(model.coord[:model.lines_B,:], 
                                          model.triang_pressure[:, :3], 
                                          rhs_dir.flatten(), 
                                          nodal = False, 
                                          ax = ax[1,0],
                                          cmap = 'coolwarm')
            fig.suptitle('Dirchlets')
            plt.draw()
            plt.pause(0.001)

    if print_solution == 1:
                fig, ax = plt.subplots(2,2, figsize = [15, 8])
                display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                            model.triang_velocity[:, :3], 
                                            model.uh[:model.lines_A].flatten(), 
                                            nodal = False, 
                                            ax = ax[0,0],
                                            cmap = 'coolwarm')
                display_results.plot_field_2D(model.coord[:model.lines_A,:], 
                                            model.triang_velocity[:, :3], 
                                            model.uh[model.lines_A : 2 * model.lines_A].flatten(), 
                                            nodal = False, 
                                            ax = ax[0,1],
                                            cmap = 'coolwarm')
                display_results.plot_field_2D(model.coord[:model.lines_B,:], 
                                              model.triang_pressure[:, :3], 
                                              model.p.flatten(), 
                                              nodal = False, 
                                              ax = ax[1,0],
                                              cmap = 'coolwarm')

                fig.suptitle('Solution')
                plt.draw()
                plt.pause(0.001)