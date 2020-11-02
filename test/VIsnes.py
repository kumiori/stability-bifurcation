# sys.path.append("../src/")
# from post_processing import compute_sig, local_project
import site
import sys




import sys
# from linsearch import LineSearch
# import solvers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# import mshr
import dolfin
from dolfin import MPI
import os
# import pandas as pd
import sympy
import numpy as np
# import post_processing as pp
import petsc4py
from functools import reduce
import ufl
petsc4py.init(sys.argv)
from petsc4py import PETSc
# from hashlib import md5
from pathlib import Path
import json
import hashlib
from copy import deepcopy
import mpi4py
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from slepc4py import SLEPc
from dolfin.cpp.log import log, LogLevel, set_log_level
dolfin.parameters["std_out_all_processes"] = False

# from dolfin import NonlinearProblem, derivative, \
#         TrialFunction, TestFunction, inner, assemble, sqrt, \
#         Constant, interpolate, RectangleMesh, Point

from dolfin import *
import yaml


set_log_level(LogLevel.INFO)
# form_compiler_parameters = {
#     "representation": "uflacs",
#     "quadrature_degree": 2,
#     "optimize": True,
#     "cpp_optimize": True,
# }

# timestepping_parameters = {"perturbation_choice": 'steepest',
#                             "savelag": 1,
#                             "outdir": '',
#                             'cont_rtol': 1e-5}
#                         # "perturbation_choice": 'steepest',               # admissible choices: steepest, first, #

# petsc_options_alpha_snes = {
#     "alpha_snes_type": "vinewtonrsls",
#     "alpha_snes_stol": 1e-5,
#     "alpha_snes_atol": 1e-5,
#     "alpha_snes_rtol": 1e-5,
#     "alpha_snes_max_it": 500,
#     "alpha_ksp_type": "preonly",
#     "alpha_pc_type": "lu"}

# petsc_options_u = {
#     "u_snes_type": "newtontr",
#     "u_snes_stol": 1e-6,
#     "u_snes_atol": 1e-6,
#     "u_snes_rtol": 1e-6,
#     "u_snes_max_it": 1000,
#     "u_snes_monitor": ''}

# alt_min_parameters = {"max_it": 300,
#                       "tol": 1.e-5,
#                       "solver_u": petsc_options_u,
#                       # either
#                       "solver_alpha": "snes",
#                       "solver_alpha_snes": petsc_options_alpha_snes
#                       # or
#                       # "solver_alpha": "tao",
#                      # "solver_alpha_tao": petsc_options_alpha_tao
#                      }

# parameters = {"alt_min": alt_min_parameters,
#                 # "solver_u": petsc_options_u,
#                 # "solver_alpha_tao": petsc_options_alpha_tao, "solver_alpha_snes": petsc_options_alpha_snes,
#                 "stability": stability_parameters,
#                 "time_stepping": timestepping_parameters,
#                 "material": {},
#                 "geometry": {},
#                 "experiment": {},
#                 "code": versions
#                 }

class EquilibriumSolver:
    """docstring for EquilibriumSolver"""
    def __init__(self, energy, model, state, bcs, parameters={}):
        super(EquilibriumSolver, self).__init__()
        self.energy = energy
        self.model = model
        self.state = state
        self.bcs = bcs

        self.u = state['u']
        self.alpha = state['alpha']

        self.bcs_u = bcs['u']
        self.bcs_alpha = bcs['alpha']

        self.V_alpha = self.alpha.function_space()

        comm = self.alpha.function_space().mesh().mpi_comm()
        self.comm = comm

        self.parameters = parameters

        if state_0:
            self.u_0 = state_0['alpha']
            self.alpha_0 = state_0['alpha']
        else:
            self.u_0 = self.u.copy(deepcopy=True)
            self.alpha_0 = self.alpha.copy(deepcopy=True)

        self.elasticity_solver = ElasticitySolver(energy, state, bcs['elastic'], parameters['elastic'])
        self.damage_solver = DamageSolver(energy, state, bcs['damage'], parameters['damage'])

    def solve(self):
        parameters = self.parameters
        it = 0
        err_alpha = 1
        u = self.u
        alpha = self.alpha
        alpha_old = alpha.copy(deepcopy=True)
        alpha_error = alpha.copy(deepcopy=True)
        criterion = 1

        alt_min_data = {
            "iterations": [],
            "alpha_error": [],
            "alpha_max": []}

        while criterion > parameters["tol"] and it < parameters["max_it"]:
            it = it + 1
            (u_it, u_reason) = self.solver_u.solve(self.problem_u, self.u.vector())
            (alpha_it, alpha_reason) = self.damage_solver.solve()

            # if parameters["solver_alpha"] == "snes":
            #     # self.set_solver_alpha_snes()
            #     # Solve the problem
            #     # import pdb; pdb.set_trace()
            #     # (alpha_it, alpha_reason) = 
                # (alpha_it, alpha_reason) = self.solver_alpha.solve(
                #     self.problem_alpha,
                #     self.alpha.vector())
                # del self.solver_alpha

            # elif parameters["solver_alpha"] == "tao":
            #     (alpha_it, alpha_reason) = self.solver_alpha.solve(
            #         self.problem_alpha,
            #         self.alpha.vector(),
            #         self.problem_alpha.lb.vector(),
            #         self.problem_alpha.ub.vector()
            #     )
            # irrev = alpha.vector()-self.problem_alpha.lb.vector()
            # if min(irrev[:]) >=0:
            #   ColorPrint.print_pass('')
            # else: 
            #   log(LogLevel.INFO,'Pointwise irrev {}'.format(' NOK'))


            alpha_error.vector()[:] = alpha.vector() - alpha_old.vector()
            # crit: energy norm
            err_alpha = abs(alpha_error.vector().max())
            # err_alpha = norm(alpha_error,'h1')
            criterion = err_alpha

            alt_min_data["iterations"].append(it)
            alt_min_data["alpha_error"].append(err_alpha)
            alt_min_data["alpha_max"].append(alpha.vector().max())

            ColorPrint.print_info(
                "iter {:2d}: alpha_error={:.4g}, alpha_max={:.4g}".format(
                    it,
                    err_alpha,
                    alpha.vector().max()
                )
            )

            # update
            alpha_old.assign(alpha)

    def update(self):
        self.problem_alpha.update_lower_bound()
        log(LogLevel.PROGRESS, 'Updated irreversibility')

        # if self.parameters["solver_alpha"] == "snes2":
        #     self.problem_alpha.lb.assign(self.alpha)
        #     self.problem_alpha.set_bounds(self.alpha, interpolate(Constant("2."), self.alpha.function_space()))
        #     print('Updated irreversibility')
        # else:
        #     self.problem_alpha.update_lb()
        #     print('Updated irreversibility')

class ElastcitityProblem:
    """docstring for ElastcitityProblem"""
    def __init__(self, arg):
        super(ElastcitityProblem, self).__init__()
        self.arg = arg

class ElasticitySolver:
    """docstring for ElasticitySolver"""
    def __init__(self, arg):
        super(ElasticitySolver, self).__init__()
        self.arg = arg

class DamageSolver:
    """docstring for DamageSolver"""
    def __init__(self, energy, state, bcs, parameters={}):
        super(DamageSolver, self).__init__()
        self.energy = energy
        self.state = state
        self.bcs = bcs
        self.parameters = parameters

        solver_type = parameters['type']

        if solver_type == 'TAO':
            self.solver = DamageTAO(energy, state, bcs, parameters)
            # self.problem = DamageProblemTAO(energy, state, bcs, parameters)
        elif solver_type == ['SNES']:
            self.problem = DamageProblemSNES(energy, state, bcs)
            self.solver = DamageSolverSNES(self.problem, parameters)
            # self.solver = DamageSolverSNES(energy, state, bcs, parameters)

        def solve(self):
            self.solver.solve()

    def solve(self):
        """
        Solve the damage problem for the current state.

        This method try to solve the problem using the first solver specified
        in the damage solvers parameters. If it fail, it will try to solve the
        problem using the second solver.
        """
        # for i in range(len(self.solvers)):
        #     try:
        #         # Set the current solver
        #         self.activate_solver(i)
        #         # Solve the problem
        #         self.subsolver.solve()
        #         # Set success to true
        #         success = True
        #         # Break if the resolution is ok
        #         break
        #     except:
        #         ColorPrint.print_warn(
        #                 "Damage solver 1 failed, trying with damage solver 2")
        #         continue


        try:
            self.solver.solve()
        except:
            log(LogLevel.WARNING,
                    "Damage solver failed, what's next?")
            raise RuntimeError("Damage solvers did not converge")
        # Check if the resolution is successfull
        # if not success:
# 
class DamageSolverSNES:
    """docstring for DamageSolverSNES"""
    def __init__(self, problem, parameters={}, lb=None):
        super(DamageSolverSNES, self).__init__()
        self.problem = problem
        self.energy = problem.energy
        self.state = problem.state
        alpha = problem.state['alpha']
        self.alpha_dvec = as_backend_type(alpha.vector())
        self.alpha_pvec = self.alpha_dvec.vec()

        self.bcs = problem.bcs
        self.parameters = parameters
        comm = alpha.function_space().mesh().mpi_comm()
        self.comm = comm
        V = alpha.function_space()

        solver = PETScSNESSolver()
        snes = solver.snes()
        # lb = self.alpha_init
        if lb == None: 
            lb=interpolate(Constant(0.), V)
        ub = interpolate(Constant(1.), V)

        # snes.setOptionsPrefix("alpha_")
        for option, value in self.parameters["snes"].items():
            PETScOptions.set(option, value)
            log(LogLevel.INFO, "Set: {} = {}".format(option,value))
        snes.setFromOptions()
        # import pdb; pdb.set_trace()

        (J, F, bcs_alpha) = (problem.J, problem.F, problem.bcs)
        # Create the SystemAssembler
        self.ass = SystemAssembler(J, F, bcs_alpha)
        # Intialise the residual
        self.b = self.init_residual()
        # Set the residual
        snes.setFunction(self.residual, self.b.vec())
        # Initialise the Jacobian
        self.A = self.init_jacobian()
        # Set the Jacobian
        snes.setJacobian(self.jacobian, self.A.mat())
        snes.ksp.setOperators(self.A.mat())
        # Set the bounds

        snes.setVariableBounds(lb.vector().vec(), ub.vector().vec()) # 

        self.solver = snes

    def update_x(self, x):
        """
        Given a PETSc Vec x, update the storage of our solution function alpha.
        """
        x.copy(self.alpha_pvec)
        self.alpha_dvec.update_ghost_values()


    def init_residual(self):
        # Get the state
        alpha = self.problem.state["alpha"]
        # Initialise b
        b = as_backend_type(
            Function(alpha.function_space()).vector()
            )
        return b

    def init_jacobian(self):
        A = PETScMatrix(self.comm)
        self.ass.init_global_tensor(A, Form(self.problem.J))
        return A

    def residual(self, snes, x, b):
        self.update_x(x)
        b_wrap = PETScVector(b)
        self.ass.assemble(b_wrap, self.alpha_dvec)

    def jacobian(self, snes, x, A, P):
        self.update_x(x)
        A_wrap = PETScMatrix(A)
        self.ass.assemble(A_wrap)

    def solve(self):
        # alpha = self.problem.state["alpha"]
        # self.solver.solve(self.problem,
        #             alpha.vector())
        #         # del self.solver_alpha

        alpha = self.problem.state["alpha"]
        # Need a copy for line searches etc. to work correctly.
        x = alpha.copy(deepcopy=True)
        xv = as_backend_type(x.vector()).vec()
        # Solve the problem
        self.solver.solve(None, xv)

class DamageProblemSNES(NonlinearProblem):
    """
    Class for the damage problem with an NonlinearVariationalProblem.
    """

    def __init__(self, energy, state, bcs=None):
        """
        Initializes the damage problem.

        Arguments:
            * energy
            * state
            * boundary conditions
        """
        # Initialize the NonlinearProblem
        NonlinearProblem.__init__(self)
        # Set the problem type
        self.type = "snes"
        # Store the boundary conditions
        self.bcs = bcs
        # Store the state
        self.state = state
        # Get state variables
        alpha = state["alpha"]
        # Get function space
        V = alpha.function_space()
        # Create trial function and test function
        alpha_v = TestFunction(V)
        dalpha = TrialFunction(V)
        # Determine the residual
        self.energy = energy
        self.F = derivative(energy, alpha, alpha_v)
        # Determine the Jacobian matrix
        self.J = derivative(self.F, alpha, dalpha)
        # Set the bound of the problem (converted to petsc vector)
        self.lb = alpha.copy(True).vector()
        self.ub = interpolate(Constant(1.), V).vector()

    def update_lower_bound(self):
        """
        Update the lower bounds.
        """
        # Get the damage variable
        alpha = self.state["alpha"]
        # Update the current bound values
        self.lb = alpha.copy(deepcopy = True).vector().vec()

def numerial_test(
    parameters,
    ell=0.05,
    nu=0.,
):
    # Create mesh and define function space
    Lx = 1; Ly = .1
    Lx = 1.; Ly = .1
    comm = MPI.comm_world
    mesh = RectangleMesh(Point(-Lx/2, 0.), Point(Lx/2, Ly), 50, 10)
    # Define Dirichlet boundaries
    outdir = '../test/output/VIsnes'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    with open('../parameters/form_compiler.yml') as f:
        form_compiler_parameters = yaml.load(f, Loader=yaml.FullLoader)

    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)
    # bottom.mark(mf, 3)
    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    alpha = Function(V_alpha)
    dalpha = TrialFunction(V_alpha)
    beta = TestFunction(V_alpha)
    ell = Constant(0.1) 
    # The energy
    def w(alpha):
        return alpha
    energy = (ell*inner(grad(alpha), grad(alpha)) + w(alpha)/ell)*dx

    # First directional derivative in alpha along the direction beta
    # dF = derivative(F,alpha,beta)
    # ddF = derivative(dF,alpha,dalpha)

    bc_l = DirichletBC(V_alpha,  Constant(0.0), left)
    bc_r = DirichletBC(V_alpha, Constant(1.0), right)
    bcs=[bc_l,bc_r]

    state = {'alpha': alpha}

    # log(LogLevel.INFO, 'Outdir = {}'.format(outdir))

    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_out.parameters["functions_share_mesh"] = True
    file_out.parameters["flush_output"] = True

    # problem_nl = NonlinearVariationalProblem(dF, alpha, bcs, J = ddF)
    # problem_nl.set_bounds(lb, ub)
    # set up the solver
    # solver_nl = NonlinearVariationalSolver(problem_nl)

    with open('../parameters/solvers_default.yml') as f:
        solver_parameters = yaml.load(f, Loader=yaml.FullLoader)

    solver = DamageSolver(energy, state, bcs, parameters=solver_parameters['damage'])
    # snes_solver_parameters_bounds = {"nonlinear_solver": "snes",
    #                           "snes_solver": {"linear_solver": "cg",
    #                                           "maximum_iterations": 100,
    #                                           "report": True,
    #                                           "line_search": "basic",
    #                                           "method":"vinewtonrsls",
    #                                           "absolute_tolerance":1e-6,
    #                                           "relative_tolerance":1e-6,
    #                                           "solution_tolerance":1e-6}}
    # solver_nl.parameters.update(snes_solver_parameters_bounds)
    # solver_nl.parameters.update({"nonlinear_solver": "snes"})
    # info(solver.parameters,True)
    # solve the problem
    solver.solve()
    plot(alpha)
    plt.savefig(os.path.join(outdir, 'alpha.pdf'))
    log(LogLevel.INFO, "Saved figure: {}".format(os.path.join(outdir, 'alpha.pdf')))


if __name__ == "__main__":

    # Parameters
    with open('../parameters/parameters.yml') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    # Choose the directory where the mesh is stored
    # mesh_dir = Path("mesh")
    # Choose directory to store the results
    # resu_dir = Path("results")
    # if MPI.comm_world.rank == 0:
    #     resu_dir.mkdir(parents=True, exist_ok=True)
    # Choose the name of the result files (without the extension)
    # resu_name = Path("tensile_test_beam")
    # Launch simulation
    numerial_test(parameters)

