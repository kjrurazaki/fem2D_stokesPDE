# Finite element code for 2D piecewise Linear Galerkin
# Extended Stokes PDE

from solver import gmres_solver
from imposeBC import imposeBC

from model import Model

import dill
import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix
import debug_print

def run_2D(model, method):
      # Impose BCs
      (stiffMat, rhs, u_dir_velocity, 
      u_dir_pressure, boundary_nodes_velocity, 
      boundary_nodes_pressure) = (imposeBC(model, method = method))

      # Convert the numpy array to a sparse matrix
      sparse_stiffMat = csc_matrix(stiffMat)

      # x = np.linalg.solve(stiffMat, rhs.reshape(-1,1))
      x, info, e = gmres_solver(sparse_stiffMat, rhs.reshape(-1,1))
      print(f'info:{info}')
      print(f'e:{e}')

      uh = x.reshape(-1, 1)

      # Residual
      residual = np.linalg.norm(rhs - np.matmul(stiffMat, uh))
      
      # Reshape solution for velocity
      num_rows = uh[:2 * model.lines_A].shape[0] // 2
      uh_reshaped = uh[:2 * model.lines_A].reshape(num_rows, 2)

      residual_bc = np.linalg.norm(u_dir_velocity[boundary_nodes_velocity, :2] - 
                                   uh_reshaped[boundary_nodes_velocity, :])
      residual_bc += np.linalg.norm(u_dir_pressure[boundary_nodes_pressure, 0] -
                                    uh[2 * model.lines_A:][boundary_nodes_pressure])

      return uh[:2 * model.lines_A], uh[2 * model.lines_A:], residual, residual_bc, rhs

if __name__ == '__main__':
      i_list = ['0', '1', '2', '3', '4']
      type_list = ['linear', 'linear_gls', 'bubble', 'p1-iso-p2'] 
      delta_list = [None, 0.001, 0.1, 0.5, 1, 10]
      c_list = [1, 10, 100]
      for i in i_list:
            for type in type_list:
                  for c in c_list:
                        for delta in delta_list:
                              if delta is not None and type != 'linear_gls':
                                    continue
                              elif delta is None and type == 'linear_gls':
                                    continue
                              else:
                                    meshdir = f"./Meshes/mesh{i}"
                                    method = 'lifting'
                                    # Elements type could be linear, bubble or p1-iso-p2
                                    model_stokes = Model(meshdir, 
                                                      c = c,
                                                      delta = delta,
                                                      only_dirichlet = True,
                                                      element_type = type)
                                    
                                    uh, p, residual, residual_boundary, rhs = run_2D(model_stokes, 
                                                                              method = method)
                                    
                                    # Print residuals
                                    print(f'Residual:{residual}')
                                    print(f'Residual BC: {residual_boundary}')

                                    model_stokes.update_rhs(rhs) # After applying boundary condition - to plot
                                    model_stokes.update_solution(uh, p)

                                    # Printing options for the run
                                    # dict_print = {
                                    # 'print_f' : 1,
                                    # 'print_rhs' : 1,
                                    # 'print_dirichlet' : 1,
                                    # 'print_neumann' : 0,
                                    # 'print_solution' : 1
                                    # }
                                    # debug_print.all_prints(model_stokes, dict_print)
                                    
                                    # with open(f'./Models/model_i{i}_type{type}_c{c}_delta{delta}.pkl', 'wb') as f:
                                    #       dill.dump(model_stokes, f)
                                    
                                    # Save results - TODO a more efficient way
                                    pd.DataFrame(model_stokes.coord).to_csv(f'./Models/coord_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.triang_velocity).to_csv(f'./Models/triangv_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.triang_pressure).to_csv(f'./Models/triangp_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.uh).to_csv(f'./Models/v_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.p).to_csv(f'./Models/p_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.element.solution.u_1).to_csv(f'./Models/v1sol_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.element.solution.u_2).to_csv(f'./Models/v2sol_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.element.solution.sol_p).to_csv(f'./Models/psol_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.element.DirNod_velocity).to_csv(f'./Models/DirNod_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    pd.DataFrame(model_stokes.element.h).to_csv(f'./Models/h_i{i}_type{type}_c{c}_delta{delta}.csv')
                                    
                                    # with open(f'./Models/model_i{i}_type{type}_c{c}_delta{delta}.pkl', 'wb') as f:
                                    #       dill.dump(model_stokes, f)

                                    # Residual compared to the true solution
                                    # (total_residual, 
                                    # u1_residual, 
                                    # u2_residual, 
                                    # p_residuals) = model_stokes.compute_residual()
                                    # print('Residual comparing true solution')
                                    # print(f'Total: {total_residual}')
                                    # print(f'u1: {u1_residual}')
                                    # print(f'u2: {u2_residual}')
                                    # print(f'p: {p_residuals}')

      # Wait to see graphics
      # input("Press [enter] to continue.")