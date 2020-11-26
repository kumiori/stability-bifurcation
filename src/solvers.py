from dolfin import norm, assemble, assemble_system, TestFunction, XDMFFile, TrialFunction, interpolate
from dolfin import PETScTAOSolver, PETScSNESSolver, OptimisationProblem, NonlinearProblem, PETScOptions, PETScVector, PETScMatrix, as_backend_type, Vector
from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
from dolfin import Constant, Expression
from utils import ColorPrint
from ufl import sqrt, inner, dot, conditional, derivative
import os
import site
import sys

from dolfin import Function, as_backend_type, SystemAssembler, Form



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


import yaml


def take_last(dic):
    new_dic = {}
    for key, value in dic.items():
        try:
            new_dic[key] = value[-1]
        except:
            new_dic[key] = value
    return new_dic

alt_min_parameters = {"max_it": 300,
                      "tol": 1.e-5,
                      "solver_alpha": "tao"
                     }

petsc_options_alpha_tao = {"tao_type": "gpcg",
                           "tao_ls_type": "gpcg",
                           "tao_gpcg_maxpgits": 50,
                           "tao_max_it": 300,
                           "tao_steptol": 1e-7,
                           "tao_gatol": 1e-4,
                           "tao_grtol": 1e-4,
                           "tao_gttol": 1e-4,
                           "tao_catol": 0.,
                           "tao_crtol": 0.,
                           "tao_ls_ftol": 1e-6,
                           "tao_ls_gtol": 1e-6,
                           "tao_ls_rtol": 1e-6,
                           "ksp_rtol": 1e-6,
                           "tao_ls_stepmin": 1e-8,  #
                           "tao_ls_stepmax": 1e6,  #
                           "pc_type": "bjacobi",
                           "tao_monitor": "",  # "tao_ls_type": "more-thuente"
                           # "ksp_type": "preonly"  # "tao_ls_type": "more-thuente"
                           }
petsc_options_alpha_snes = {
    "alpha_snes_type": "vinewtonrsls",
    "alpha_snes_stol": 1e-5,
    "alpha_snes_atol": 1e-5,
    "alpha_snes_rtol": 1e-5,
    "alpha_snes_max_it": 500,
    "alpha_ksp_type": "preonly",
    "alpha_pc_type": "lu"}

petsc_options_u = {
    "u_snes_type": "newtontr",
    "u_snes_stol": 1e-6,
    "u_snes_atol": 1e-6,
    "u_snes_rtol": 1e-6,
    "u_snes_max_it": 1000,
    "u_snes_monitor": ''}

default_parameters = {"alt_min": alt_min_parameters,
                      "solver_u": petsc_options_u,
                      "solver_alpha": "tao",
                      "tol": 1e-5,
                      "max_it": 50,
                      "solver_alpha_tao": petsc_options_alpha_tao}

class DamageProblemTAO(OptimisationProblem):
    def __init__(self, energy, alpha, bcs, lb=None, ub=None):

        OptimisationProblem.__init__(self)
        self.energy = energy
        self.alpha = alpha
        self.V = self.alpha.function_space()
        self.denergy = derivative(
            self.energy, self.alpha, TestFunction(self.V))
        self.ddenergy = derivative(
            self.denergy, self.alpha, TrialFunction(self.V))
        if lb == None:
            lb = interpolate(Constant("0."), self.V)
        if ub == None:
            ub = interpolate(Constant("2."), self.V)
        self.lb = lb
        self.ub = ub
        self.bcs = bcs
        self.update_lb()

    def f(self, x):
        """Function to be minimised"""
        self.alpha.vector()[:] = x
        return assemble(self.energy)

    def F(self, b, x):
        """Gradient (first derivative)"""
        self.alpha.vector()[:] = x
        assemble(self.denergy, b)

    def J(self, A, x):
        """Hessian (second derivative)"""
        self.alpha.vector()[:] = x
        assemble(self.ddenergy, A)

    def bc_apply(self):
        """Apply the bcs"""
        for bc in self.bcs:
            bc.apply(self.lb.vector())
            bc.apply(self.ub.vector())

    def update_lb(self, lb=None):
        """update the lower bound"""
        if lb:
            self.lb.assign(lb)
        else:
            self.lb.assign(self.alpha)
        self.bc_apply()

    def active_set_indicator_lb(self, tol=1.0e-5):
        clb = conditional(self.alpha > tol + self.lb, 0.0, 1.0)
        return clb

    def active_set_indicator_ub(self, tol=1.0e-5):
        cub = conditional(self.alpha < self.ub + tol, 0.0, 1.0)
        return cub

class AlternateMinimizationSolver(object):

    def __init__(self, energy, state, bcs, state_0=None, parameters=default_parameters, space=None):
        self.energy = energy
        self.u = state['u']
        self.alpha = state['alpha']
        self.bcs_u = bcs['elastic']
        self.bcs_alpha = bcs['damage']
        if state_0:
            self.u_init = state_0['u']
            self.alpha_init = state_0['alpha']
        else:
            self.u_init = self.u.copy(deepcopy=True)
            self.alpha_init = self.alpha.copy(deepcopy=True)

        self.V_alpha = self.alpha.function_space()
        # import pdb; pdb.set_trace()
        self.parameters = parameters
        self.problem_u = (energy, state, bcs['elastic'], parameters['elasticity'])
        self.set_solver_u()
        if self.parameters["solver_alpha"] == "snes":
            self.set_solver_alpha_snes()
        # elif self.parameters["solver_alpha"] == "snes2":
        #     self.set_solver_alpha_snes2()
        elif self.parameters["solver_alpha"] == "tao":
            self.set_solver_alpha_tao()

    def set_solver_u(self):
        for option, value in self.parameters["solver_u"].items():
            print("setting ", option,value)
            PETScOptions.set(option, value)
        solver = PETScSNESSolver()
        snes = solver.snes()
        snes.setOptionsPrefix("u_")
        snes.setType(self.parameters["solver_u"]["u_snes_type"])
        ksp = snes.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")

        # Namespace mismatch between petsc4py 3.7 and 3.9.
        if hasattr(pc, 'setFactorSolverType'):
            pc.setFactorSolverType("mumps")
        elif hasattr(pc, 'setFactorSolverPackage'):
            pc.setFactorSolverPackage('mumps')
        else:
            log(LogLevel.WARNING,'Could not configure preconditioner')
        solver.set_from_options()
        snes.setFromOptions()
        self.solver_u = solver

    # def set_solver_alpha_snes2(self):
    #     V = self.alpha.function_space()
    #     denergy = derivative(self.energy, self.alpha, TestFunction(V))
    #     ddenergy = derivative(denergy, self.alpha, TrialFunction(V))
    #     self.lb = self.alpha_init # interpolate(Constant("0."), V)
    #     ub = interpolate(Constant("1."), V)
    #     self.problem_alpha = NonlinearVariationalProblem(
    #         denergy, self.alpha, self.bcs_alpha, J=ddenergy)
    #     self.problem_alpha.set_bounds(self.lb, ub)
    #     self.problem_alpha.lb = self.lb
    #     # set up the solver
    #     solver = NonlinearVariationalSolver(self.problem_alpha)
        
    #     snes_solver_parameters_bounds = {"nonlinear_solver": "snes",
    #                                      "snes_solver": {"linear_solver": "mumps",
    #                                                      "maximum_iterations": 300,
    #                                                      "report": True,
    #                                                      "line_search": "basic",
    #                                                      "method": "vinewtonrsls",
    #                                                      "absolute_tolerance": 1e-5,
    #                                                      "relative_tolerance": 1e-5,
    #                                                      "solution_tolerance": 1e-5}}
    #     solver.parameters.update(snes_solver_parameters_bounds)
    #     #solver.solve()
    #     self.solver = solver

    def set_solver_alpha_tao(self):
        self.problem_alpha = DamageProblemTAO(
            self.energy, self.alpha, self.bcs_alpha, lb=self.alpha_init)
        solver = PETScTAOSolver()
        for option, value in self.parameters["solver_alpha_tao"].items():
            PETScOptions.set(option, value)
            # PETScOptions.set('help', 1)
            print('setting {} {}'.format(option, value))
        self.solver_alpha = solver

    def update(self):
        # if self.parameters["solver_alpha"] == "snes2":
        #     self.problem_alpha.lb.assign(self.alpha)
        #     self.problem_alpha.set_bounds(self.alpha, interpolate(Constant("2."), self.alpha.function_space()))
        #     print('Updated irreversibility')
        # else:
        self.problem_alpha.update_lb()
        print('Updated irreversibility')

    def solve(self):
        # initialization
        par = self.parameters
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
        # import pdb; pdb.set_trace()

        while criterion > self.parameters["tol"] and it < self.parameters["max_it"]:
            it = it + 1
            (u_it, u_reason) = self.solver_u.solve(
                self.problem_u, self.u.vector())

            # if self.parameters["solver_alpha"] == "snes2":
            #     self.set_solver_alpha_snes2()
            #     (alpha_it, alpha_reason) = self.solver.solve()

            if self.parameters["solver_alpha"] == "snes":
                self.set_solver_alpha_snes()
                import pdb; pdb.set_trace()
                (alpha_it, alpha_reason) = self.solver_alpha.solve(
                    self.problem_alpha,
                    self.alpha.vector(),
                    )
                del self.solver_alpha

            elif self.parameters["solver_alpha"] == "tao":
                (alpha_it, alpha_reason) = self.solver_alpha.solve(
                    self.problem_alpha,
                    self.alpha.vector(),
                    self.problem_alpha.lb.vector(),
                    self.problem_alpha.ub.vector()
                )
                irrev = alpha.vector()-self.problem_alpha.lb.vector()
                if min(irrev[:]) >=0:
                    log(LogLevel.INFO, '')
                else: 
                    log(LogLevel.WARNING,'Pointwise irrev {}'.format(' NOK'))
                    import pdb; pdb.set_trace()

            alpha_error.vector()[:] = alpha.vector() - alpha_old.vector()
            # crit: energy norm
            err_alpha = abs(alpha_error.vector().max())
            # err_alpha = norm(alpha_error,'h1')
            criterion = err_alpha

            alt_min_data["iterations"].append(it)
            alt_min_data["alpha_error"].append(err_alpha)
            alt_min_data["alpha_max"].append(alpha.vector().max())

            log(LogLevel.INFO,
                "Iteration #{:2d}: alpha_error={:.4g}, alpha_max={:.4g}".format(
                    it,
                    err_alpha,
                    alpha.vector().max()
                )
            )

            # update
            alpha_old.assign(alpha)

        return (take_last(alt_min_data), alt_min_data)

set_log_level(LogLevel.INFO)

class EquilibriumSolver:
    """docstring for EquilibriumSolver"""
    def __init__(self, energy, state, bcs, state_0 = {}, parameters={}):
        super(EquilibriumSolver, self).__init__()
        self.energy = energy
        self.state = state
        self.bcs = bcs

        self.u = state['u']
        self.alpha = state['alpha']

        self.bcs_u = bcs['elastic']
        self.bcs_alpha = bcs['damage']

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

        # import pdb; pdb.set_trace()

        self.elasticity = ElasticitySolver(energy, state, bcs['elastic'], parameters['elasticity'])
        self.damage = DamageSolver(energy, state, bcs['damage'], parameters['damage'])

    def solve(self, debugpath=''):
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
            "alpha_max": [],
            "energy": []}

        # log(LogLevel.WARNING,'self.damage.problem.lb[:]')
        # log(LogLevel.WARNING, '{}'.format(self.damage.problem.lb[:]))
        self.damage.solver.solver.setVariableBounds(self.damage.problem.lb.vec(),
            self.damage.problem.ub.vec())
        
        # import pdb; pdb.set_trace()
        if debugpath:
            file_am = XDMFFile(os.path.join(debugpath, "am_new.xdmf"))
        else:
            file_am = XDMFFile("/Users/kumiori/am_new.xdmf")
        file_am.parameters["functions_share_mesh"] = True
        file_am.parameters["flush_output"] = True

        while criterion > float(self.parameters['equilibrium']["tol"]) and it < self.parameters['equilibrium']["max_it"]:
            it = it + 1
            (u_it, u_reason) = self.elasticity.solve()
            (alpha_it, alpha_reason) = self.damage.solve()

            # if not np.all(alpha.vector()[:] >=self.damage.problem.lb):
            #     pd = np.where(alpha.vector()[:]-self.damage.problem.lb[:] < 0)[0]
            #     log(LogLevel.WARNING, 'Pointwise irreversibility issues on dofs {}'.format(pd))

            #     # log(LogLevel.WARNING, 'diff = {}'
            #     #     .format(self.alpha.vector()[pd]-self.alpha_old[pd]))
            #     log(LogLevel.WARNING, 'Continuing')

            alpha_error.vector()[:] = alpha.vector() - alpha_old.vector()
            # crit: energy norm
            err_alpha = abs(alpha_error.vector().max())
            # err_alpha = norm(alpha_error,'h1')
            criterion = err_alpha


            log(LogLevel.INFO,
                "   AM iter {:2d}: alpha_error={:.4g}, alpha_max={:.4g}, energy = {:.5g}".format(
                    it,
                    err_alpha,
                    alpha.vector().max(),
                    assemble(self.energy)
                )
            )

            # update
            alpha_old.assign(alpha)

            with file_am as file:
                file.write(alpha, it)
                file.write(u, it)

            print('Written AM step {}'.format(it))

        irrev = alpha.vector()-self.damage.problem.lb
        if min(irrev[:]) >=0:
            log(LogLevel.INFO,'')
        else:
            log(LogLevel.INFO,'Pointwise irrev {}'.format(' NOK'))
            import pdb; pdb.set_trace()

        alt_min_data["alpha_error"].append(err_alpha)
        alt_min_data["alpha_max"].append(alpha.vector().max())
        alt_min_data["iterations"].append(it)
        alt_min_data["energy"].append(assemble(self.energy))

        log(LogLevel.INFO,
                "AM converged in {} iterations, err = {}, energy = {}".format(it, err_alpha, assemble(self.energy)))

        return (alt_min_data, it)

    def update(self):
        self.damage.problem.update_lower_bound()
        log(LogLevel.PROGRESS, 'PROGRESS: Updated irreversibility')

        # if self.parameters["solver_alpha"] == "snes2":
        #     self.problem_alpha.lb.assign(self.alpha)
        #     self.problem_alpha.set_bounds(self.alpha, interpolate(Constant("2."), self.alpha.function_space()))
        #     print('Updated irreversibility')
        # else:
        #     self.problem_alpha.update_lb()
        #     print('Updated irreversibility')

class ElasticityProblem(NonlinearProblem):
    """docstring for ElastcitityProblem"""

    def __init__(self, energy, state, bcs):
        """
        Initialises the elasticity problem.

        Arguments:
            * energy
            * state
            * boundary conditions
        """
        # Initialize the parent
        NonlinearProblem.__init__(self)
        self.bcs = bcs
        self.state = state
        # import pdb; pdb.set_trace()
        u = state['u']

        V = u.function_space()
        v = TestFunction(V)
        du = TrialFunction(V)

        self.residual = derivative(energy, u, v)
        self.jacobian = derivative(self.residual, u, du)

    def F(self, b, x):
        """
        Compute F at current point x.
        This function is called at each interation of the solver.
        """
        assemble(self.residual, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)
        pass

    def J(self, A, x):
        """
        Compute J=F' at current point x.
        This function is called at each interation of the solver.
        """
        assemble(self.jacobian, tensor=A)
        for bc in self.bcs:
            bc.apply(A)
        pass

class ElasticitySolver:
    """docstring for ElasticitySolver"""
    def __init__(self, energy, state, bcs, parameters={}):
        super(ElasticitySolver, self).__init__()
        solver_name = 'elasticity'
        self.problem = ElasticityProblem(energy, state, bcs)
        # Set the solver
        self.solver = PETScSNESSolver()
        snes = self.solver.snes()

        prefix = "elasticity_"
        snes.setOptionsPrefix(prefix)

        for parameter, value in parameters.items():
            if value is not None:
                log(LogLevel.INFO, "Set: {} = {}".format(prefix + parameter, value)) 
                PETScOptions.set(prefix + parameter, value)
            else:
                log(LogLevel.INFO, "Set: {}".format(prefix + parameter)) 
                PETScOptions.set(prefix + parameter)

        snes.setFromOptions()

        snes.setFromOptions()
        # snes.view()
    def solve(self):
        # Get the problem
        log(LogLevel.INFO, '________________________ EQUILIBRIUM _________________________')
        log(LogLevel.INFO, "Solving elasticity")
        problem = self.problem
        # Get the vector
        u = problem.state["u"].vector()
        # Solve the problem
        self.solver.solve(problem, u)
        # import pdb; pdb.set_trace()
        return (self.solver.snes().getIterationNumber(), self.solver.snes().getConvergedReason())

class DamageSolver:
    """docstring for DamageSolver"""
    def __init__(self, energy, state, bcs, parameters={}):
        super(DamageSolver, self).__init__()
        self.energy = energy
        self.state = state
        self.bcs = bcs
        self.parameters = parameters

        solver_type = parameters['type']

        # FIX solver inspection
        if solver_type == ['TAO']:
            self.solver = DamageTAO(energy, state, bcs, parameters)
            # self.problem = DamageProblemTAO(energy, state, bcs, parameters)
        elif solver_type == ['SNES']:
            self.problem = DamageProblemSNES(energy, state, bcs)
            self.solver = DamageSolverSNES(self.problem, parameters)
            # self.solver = DamageSolverSNES(energy, state, bcs, parameters)

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
        #         log(LogLevel.INFO,
        #                 "Damage solver 1 failed, trying with damage solver 2")
        #         continue

        log(LogLevel.INFO, "Solving damage")

        try:
            self.solver.solve()
            return (self.solver.solver.getIterationNumber(), self.solver.solver.getConvergedReason())

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

        prefix = "damage_"
        # prefix = ""
        snes.setOptionsPrefix(prefix)
        for option, value in self.parameters["snes"].items():
            PETScOptions.set(prefix+option, value)
            log(LogLevel.INFO, "Set: {} = {}".format(prefix + option,value))
        snes.setFromOptions()
        # snes.view()
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

        snes.setVariableBounds(self.problem.lb.vec(),
            self.problem.ub.vec()) # 

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
        # self.lb = alpha.copy(deepcopy = True).vector().vec()
        self.lb = alpha.copy(deepcopy = True).vector()
        log(LogLevel.CRITICAL, '________________________ IRREV _________________________')
        log(LogLevel.CRITICAL, 'CRITICAL: Updated irreversibility')

