from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Define the geometry for each magnet
def define_geometry(dimensions):
    magnet = Box(Point(0, 0, 0), Point(dimensions[0], dimensions[1], dimensions[2]))
    return magnet

# Define the boundary condition for each magnet
def define_boundary(magnet):
    boundary = CompiledSubDomain("on_boundary && near(x[0], side)", side = magnet.point(1)[0])
    return boundary

# Define the magnetization vector for each magnet
def define_magnetization(magnetization_direction, magnetization_magnitude, dimensions):
    magnetization = Constant(magnetization_magnitude) * \
                    Expression((magnetization_direction[0], magnetization_direction[1], magnetization_direction[2]), \
                              element = VectorElement("Lagrange", tetrahedron, 1), \
                              domain = define_geometry(dimensions))
    return magnetization

# Define the magnetic potential and field functions for each magnet
def define_potential_and_field(magnet, magnetization):
    V = FunctionSpace(magnet.mesh(), "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(magnetization, grad(v)) * dx
    U = Function(V)
    solve(a == L, U)
    B = project(grad(U), VectorFunctionSpace(magnet.mesh(), 'P', 1))
    return U, B

# Compute the magnetic force between the two magnets
def compute_force(magnet1, magnet2, magnetization1, magnetization2, distance):
    B1 = magnet1['B']
    B2 = magnet2['B']
    n = FacetNormal(magnet1['magnet'].mesh())
    force = -np.sum(assemble(dot(B1, n)*dot(B2, n)/distance**2 * ds(magnet1['boundary'])).get_local())
    return force

# Define the main function to simulate the magnetic force and field
def simulate_magnets():
    # Get the dimensions and magnetization of the first magnet
    dimensions1 = [float(input("Enter the length of the first magnet (in meters): ")),
                   float(input("Enter the width of the first magnet (in meters): ")),
                   float(input("Enter the height of the first magnet (in meters): "))]
    magnetization1_direction = [float(input("Enter the x-component of the magnetization direction for the first magnet: ")),
                                float(input("Enter the y-component of the magnetization direction for the first magnet: ")),
                                float(input("Enter the z-component of the magnetization direction for the first magnet: "))]
    magnetization1_magnitude = float(input("Enter the magnetization magnitude for the first magnet: "))

    # Get the dimensions and magnetization of the second magnet
    dimensions2 = [float(input("Enter the length of the second magnet (in meters): ")),
                   float(input("Enter the width of the second magnet (in meters): ")),
                   float(input("Enter the height of the second magnet (in meters): "))]
    magnetization2_direction = [float(input("Enter the x-component of the magnetization direction for the second magnet: ")),
                                float(input("Enter the y-component of the magnetization direction for the second magnet: ")),
                                float(input("Enter the z-component of the magnetization direction for the second magnet: "))]
    magnetization2_magnitude = float(input("Enter the magnetization magnitude for the
