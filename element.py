import numpy as np
from localBasis import localBasis
from exactsolution import Solution


class Element:
    """
    Builts element, returns mesh and basis functions for the mesh
    Element types:
        P1 (linear): Returns only P1 elements
        P1-Bubble: Returns mesh with center node
        P1-iso-P2: Returns refined mesh (h) and original mesh (2h)
    Returns:
        Mesh(es) coordinates, triangles, boundary nodes and conditions

    # TODO: Neumann boundary is not dealt when the mesh is refined
    """

    def __init__(self, model, type):
        self.coord = model.coord
        self.triang = model.triang
        self.NeuNod = model.NeuNod
        self.DirNod = model.DirNod
        self.element_type = type
        self.viscosity = model.Diff
        self.c = model.c
        self.delta = model.delta
        self.only_dirichlet = model.only_dirichlet

        if self.only_dirichlet == True:
            self.DirNod = np.concatenate([self.DirNod, self.NeuNod], axis=0)
            self.DirNod = np.sort(self.DirNod, axis=0)
            self.NeuNod = np.array([])
            self.NDir = len(self.DirNod)
            self.NNeu = len(self.NeuNod)
            self.boundary_nodes = self.DirNod
        else:
            self.boundary_nodes = np.concatenate([self.DirNod, self.NeuNod], axis=0)

        self.build_elements()
        self.delete_auxiliary()

    def build_elements(self):
        if (self.element_type == "linear") or (self.element_type == "linear_gls"):
            # Size identifiers
            self.Nodes = len(self.coord)
            self.NNeu = len(self.NeuNod)
            self.Nelem = len(self.triang)
            self.lines_A = len(self.coord)
            self.lines_B = len(self.coord)
            # Basis functions
            self.build_basis()
            # Exact solution
            self.solution = Solution(self)
            # Boundary nodes
            self.DirNod_pressure = self.DirNod
            self.NDir_pressure = len(self.DirNod_pressure)
            self.DirNod_velocity = self.DirNod
            self.NDir_velocity = len(self.DirNod_velocity)
            self.build_BC()
            self.triang_velocity = self.triang
            self.triang_pressure = self.triang

        if self.element_type == "bubble":
            # Exact solution
            self.solution = Solution(self)
            # Size identifiers
            self.Nodes = len(self.coord)
            self.NDir = len(self.DirNod)
            self.NNeu = len(self.NeuNod)
            self.Nelem = len(self.triang)
            # Basis functions
            self.build_basis()
            # Boundary nodes
            self.DirNod_pressure = self.DirNod
            self.NDir_pressure = len(self.DirNod_pressure)
            self.DirNod_velocity = self.DirNod
            self.NDir_velocity = len(self.DirNod_velocity)
            self.build_BC()
            # Add triangle centers (updates triangles and coordinates)
            self.add_triangle_centers()
            # Exact solution - Central nodes were added, no interfernce in BC and basis
            self.solution = Solution(self)
            # nodes of velocity
            self.lines_A = len(self.coord)
            self.triang_velocity = self.triang
            # nodes of pressure
            self.lines_B = self.Nodes
            self.triang_pressure = self.triang

        if self.element_type == "p1-iso-p2":
            # New boundary nodes are added in the check of "self.check_bn()"
            self.Nodes = len(self.coord)
            self.Nelem = len(self.triang)
            self.NDir = len(self.DirNod)
            self.NNeu = len(self.NeuNod)
            # Store 2h mesh boundary nodes for pressure
            self.DirNod_pressure = self.DirNod
            self.NDir_pressure = len(self.DirNod_pressure)
            # Triangles of pressure nodes
            self.triang_pressure = self.triang
            # Refinement
            self.refine_mesh()
            self.triang_velocity = self.triang
            # Size identifiers
            self.lines_A = len(self.coord)
            self.lines_B = self.Nodes
            self.Nelem_2h = self.Nelem
            self.Nelem = len(self.triang)

            # Exact solution
            self.solution = Solution(self)
            # Adds new nodes as boundary nodes (for velocity)
            self.check_bn()
            # Boundary conditions
            self.DirNod_velocity = self.DirNod
            self.NDir_velocity = len(self.DirNod_velocity)
            self.build_BC()
            # Basis functions
            self.build_basis()  # basis are built in the refined mesh

    def delete_auxiliary(self):
        """
        Making sure some values are not used in the code after modification for element types
        """
        self.DirNod = None  # (changed Dirnod_pressure/velocity)
        self.Nodes = None  # (changed lines_A or lines_B)
        self.DirVal = None  # (changed for DirVal_pressure/velocity)
        self.NDir = None  # (changed for NDir_pressure/velocity)
        self.boundary_nodes = None  # (should not be used)
        self.triang = None  # (changed for triang_pressure/velocity)

    def build_basis(self):
        """
        Calculate element area and elemental coeffients of basis functions
        (b,c)
        """
        self.Aloc, self.Bloc, self.Cloc, self.B_K_inv, self.Area, self.h = localBasis(
            self
        )

    def build_BC(self):
        """
        Set boundary conditions based on the exact solution
        Arrays DirVal_velocity has two columns for (u_x, u_y)
        Arrays DirVal_pressure has one columns for pressure
        Neumann nodes are x and y direction for velocity and pressure
        """
        # Dirichlet nodes for velocity
        self.DirVal_velocity = np.zeros((self.NDir_velocity, 2))
        for nod in range(0, self.NDir_velocity):
            global_node = self.DirNod_velocity[nod]
            self.DirVal_velocity[nod, 0] = self.solution.u_1[global_node - 1]
            self.DirVal_velocity[nod, 1] = self.solution.u_2[global_node - 1]

        # Dirichlet nodes for pressure
        self.DirVal_pressure = np.zeros((self.NDir_pressure, 1))
        for nod in range(0, self.NDir_pressure):
            global_node = self.DirNod_pressure[nod]
            self.DirVal_pressure[nod, 0] = self.solution.sol_p[global_node - 1]

        # Neumann nodes
        self.NeuVal = np.zeros((self.NNeu, 2))
        for nod in range(0, self.NNeu):
            global_node = self.NeuNod[nod]
            outer_normal = self.outer_normal(global_node)
            # TODO viscosity should not be hard coded to 1 (this is ok for the current problem)
            self.NeuVal[nod, 0] = (
                self.viscosity[0]
                * (
                    outer_normal[0] * self.solution.u_1_x[global_node - 1]
                    + outer_normal[1] * self.solution.u_1_y[global_node - 1]
                )
                - self.solution.sol_p[global_node - 1] * outer_normal[0]
            )
            self.NeuVal[nod, 1] = (
                self.viscosity[0]
                * (
                    outer_normal[0] * self.solution.u_2_x[global_node - 1]
                    + outer_normal[1] * self.solution.u_2_y[global_node - 1]
                )
                - self.solution.sol_p[global_node - 1] * outer_normal[1]
            )

    def check_bn(self):
        """
        Check if all the boundary nodes in the geometry are either defined as Neumann or Dirichlet
        Add all non defined boundaries as dirichlet
        """

        def find_boundary_nodes(self):
            boundary_nodes = []
            for i, coord in enumerate(self.coord):
                x, y = coord
                if (
                    x == self.coord[:, 0].min()
                    or x == self.coord[:, 0].max()
                    or y == self.coord[:, 1].min()
                    or y == self.coord[:, 1].max()
                ):
                    boundary_nodes.append(i)
            return np.array(boundary_nodes)

        boundary_nodes = np.sort(find_boundary_nodes(self))
        defined_bn = np.sort((self.boundary_nodes - 1).flatten())
        if self.only_dirichlet == True and not np.array_equal(
            boundary_nodes, defined_bn
        ):
            self.update_boundarynodes(boundary_nodes)
        defined_bn = np.sort((self.boundary_nodes - 1).flatten())
        assert np.array_equal(boundary_nodes, defined_bn)

    def update_boundarynodes(self, boundary_nodes):
        """
        P1-iso-P2 - used for it
        Need to update after mesh refinement (for velocity nodes)
        Adds all boundary nodes as Dirichlet boundaries for velocity
        """
        self.DirNod = np.array(boundary_nodes + 1).reshape(
            -1, 1
        )  # Nodes identification starts in 1
        self.boundary_nodes = np.concatenate(
            [self.DirNod, self.NeuNod.reshape(-1, 1)], axis=0
        )
        self.NDir = len(self.DirNod)

    def add_triangle_centers(self):
        """
        Function used to define Bubble element
        Returns:
         . Triangles with center node in the last position (4th column)
         . Coordinates of center nodes (at end of original coordinates)
        """
        updated_triang = []
        # Loop through each triangle
        for t in self.triang:
            # Calculate the center of the triangle (x_center, y_center)
            center = np.mean(self.coord[t], axis=0)

            # Add the center to the coordinates array
            self.coord = np.vstack((self.coord, center))

            # Add the center index (global) to the triangle identification
            updated_t = np.append(t, len(self.coord))
            updated_triang.append(updated_t)

        self.coord
        self.triang = np.array(updated_triang)

    def refine_mesh(self):
        """
        Refine mesh . adding triangles and coordinates
        For the new smaller triangles the viscosity is assumed from bigger original one
        Returns:
            . Coord - new points are added in the end of the coordinates
            . triang - new triangs - bigger one broken
            . triangs relation (bigger triangles to smaller corner triangles)
        """
        new_coord = self.coord.copy().tolist()
        new_triang = []
        edge_to_vertex = {}
        triangles_relation = {i: [] for i in range(0, self.Nelem)}
        viscosity_new = []

        def get_midpoint_vertex(v1, v2):
            if (v1, v2) in edge_to_vertex:
                return edge_to_vertex[(v1, v2)]
            elif (v2, v1) in edge_to_vertex:
                return edge_to_vertex[(v2, v1)]
            else:
                mid_point = (self.coord[v1 - 1] + self.coord[v2 - 1]) / 2
                new_index = len(new_coord) + 1
                new_coord.append(mid_point)
                edge_to_vertex[(v1, v2)] = new_index
                return new_index

        for i, tri in enumerate(self.triang):
            v0, v1, v2 = tri
            v3 = get_midpoint_vertex(v0, v1)
            v4 = get_midpoint_vertex(v1, v2)
            v5 = get_midpoint_vertex(v2, v0)

            refined_triangles = [
                [v0, v3, v5],
                [v1, v4, v3],
                [v2, v5, v4],
                [v3, v4, v5],
            ]

            for refined_tri in refined_triangles:
                new_triang.append(refined_tri)
                viscosity_new.append(self.viscosity[i])

            # Update node_to_triangles dictionary
            triangles_relation[i].extend([i * 4, i * 4 + 1, i * 4 + 2])

        self.triang = np.array(new_triang, dtype=int)
        self.coord = np.array(new_coord, dtype=float)
        self.triangles_relation = triangles_relation
        self.viscosity = np.array(viscosity_new)

    def outer_normal(self, node_index):
        """
        Returns the outer normal of the node
        """
        x = self.coord[node_index - 1, 0]
        y = self.coord[node_index - 1, 1]

        if x == min(x_coord for x_coord, _ in self.coord):
            return (-1, 0)  # Left side
        elif x == max(x_coord for x_coord, _ in self.coord):
            return (1, 0)  # Right side
        elif y == min(y_coord for _, y_coord in self.coord):
            return (0, -1)  # Bottom side
        elif y == max(y_coord for _, y_coord in self.coord):
            return (0, 1)  # Top side
        else:
            raise ValueError("Node is not on the boundary")
