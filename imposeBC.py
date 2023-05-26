import numpy as np

from matplotlib import pyplot as plt

import display_results

def imposeBC(model, method):
    '''
    Impose BCs
    '''
    stiffMat = model.stiffMat.copy()
    rhs = model.rhs
    # Impose Dirichlet
    (stiffMat, 
     rhs, 
     u_dir_velocity, 
     u_dir_pressure, 
     boundary_nodes_velocity, 
     boundary_nodes_pressure) = impose_dirichlet(model, stiffMat, rhs)

    # Impose Neumann boundary conditions
    if model.element.NNeu != 0:
        rhs = impose_neumann(model, rhs)
    return (stiffMat, rhs, u_dir_velocity, 
            u_dir_pressure, boundary_nodes_velocity, 
            boundary_nodes_pressure)

def impose_dirichlet(model, stiffMat, rhs):
    """
    Impose Dirichlet boundary conditions by lifting function
    """
    u_dir_velocity = np.zeros((model.lines_A, 2))
    index_nodes = list(range(0, model.lines_A))
    boundary_nodes_velocity = []

    u_dir_pressure = np.zeros((model.lines_B, 1))
    index_nodes_pressure = list(range(0, model.lines_B))
    boundary_nodes_pressure = []

    # Dirichlet nodes for velocity
    for idir in range(0, model.element.NDir_velocity):
        iglob = model.element.DirNod_velocity[idir][0]
        index_nodes.remove(iglob - 1)
        boundary_nodes_velocity.append(iglob - 1)
        u_dir_velocity[iglob - 1, 0] = model.element.DirVal_velocity[idir, 0]
        u_dir_velocity[iglob - 1, 1] = model.element.DirVal_velocity[idir, 1]

    for idir in range(0, model.element.NDir_pressure):
        iglob = model.element.DirNod_pressure[idir][0]
        index_nodes_pressure.remove(iglob - 1)
        boundary_nodes_pressure.append(iglob - 1)
        u_dir_pressure[iglob - 1, 0] = model.element.DirVal_pressure[idir, 0]

    u_dir = np.concatenate((u_dir_velocity[:, 0], u_dir_velocity[:, 1],  u_dir_pressure[:, 0]))
    dir_value = np.matmul(stiffMat, u_dir)
    rhs -= dir_value.reshape(-1, 1)

    for idir in range(0, model.element.NDir_velocity):
        iglob = model.element.DirNod_velocity[idir][0]
        stiffMat[:, iglob - 1] = 0
        stiffMat[iglob - 1, :] = 0
        stiffMat[iglob - 1, iglob - 1] = 1
        
        stiffMat[:, iglob - 1 + model.lines_A] = 0
        stiffMat[iglob - 1 + model.lines_A, :] = 0
        stiffMat[iglob - 1 + model.lines_A, iglob - 1 + model.lines_A] = 1

        rhs[iglob - 1, :] = model.element.DirVal_velocity[idir, 0]
        rhs[iglob - 1 + model.lines_A, :] = model.element.DirVal_velocity[idir, 1]
    
    for idir in range(0, model.element.NDir_pressure):
        iglob = model.element.DirNod_pressure[idir][0]
        stiffMat[:, iglob - 1 + 2 * model.lines_A] = 0
        stiffMat[iglob - 1 + 2 * model.lines_A, :] = 0
        stiffMat[iglob - 1 + 2 * model.lines_A, iglob - 1 + 2 * model.lines_A] =  -1
       
        rhs[iglob - 1 + 2 * model.lines_A, :] = -1 * model.element.DirVal_pressure[idir, 0]
    
    model.boundary_nodes_velocity = boundary_nodes_velocity
    model.boundary_nodes_pressure = boundary_nodes_pressure
    return stiffMat, rhs, u_dir_velocity, u_dir_pressure, boundary_nodes_velocity, boundary_nodes_pressure

def find_boundary_triangles(model, boundary_nodes):
    """
    Find boundary triangles for the Neumann integral
    """
    boundary_triangles = []
    # Iterate through the triangles
    for idx, triangle in enumerate(model.triang_velocity):
        # Count the number of nodes in the boundary
        common_nodes_count = np.sum(np.isin(triangle, boundary_nodes))

        # If there are at least two common nodes, the triangle is on the boundary
        if common_nodes_count >= 2:
            boundary_triangles.append(idx)
    return boundary_triangles

def impose_neumann(model, rhs):
    """
    Impose Neumann boundary condition
    """
    boundary_nodes = model.NeuNod.ravel()
    boundary_triangles = find_boundary_triangles(model, boundary_nodes)
    for ineu in range(0, model.NNeu):
        iglob = model.NeuNod[ineu][0]
        # Neumann side
        elements_boundary = np.intersect1d(np.where(model.triang == iglob)[0], 
                                           boundary_triangles)
        for iel in elements_boundary:
            boundary_loc = np.isin(model.triang[iel], boundary_nodes)
            iloc = np.where(model.triang[iel] == iglob)[0][0] # find the location of the node
            jloc = np.nonzero((model.triang[iel] != iglob) & boundary_loc)[0][0] # find the location of the other node in the boundary
            jglob = model.triang[iel, jloc]
            p1 = model.coord[model.triang[iel, iloc] - 1, :].reshape(1,-1) # coordinate of first node
            p2 = model.coord[model.triang[iel, jloc]  - 1, :].reshape(1,-1) # coordinate of second node
            
            # Calculate the length of the line segment
            length = np.linalg.norm(p2 - p1)
            rhs[iglob - 1, :] = rhs[iglob - 1, :] + length / 4 * (model.NeuVal[ineu, 0] +
                                                                   model.NeuVal[np.where(model.NeuNod == jglob)[0], 0])
            rhs[iglob - 1 + model.size_A1, :] = rhs[iglob - 1 + model.size_A1, :] + length / 4 * (model.NeuVal[ineu, 1] +
                                                                                                model.NeuVal[np.where(model.NeuNod == jglob)[0], 1])
    return rhs