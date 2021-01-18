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

class EquilibriumAM:
    """docstring for EquilibriumAM"""
    def __init__(self, energy, state, bcs, state_0 = {}, parameters={}):
        super(EquilibriumAM, self).__init__()
        self.energy = energy
        self.state = state
        self.bcs = bcs

        OptDB = PETSc.Options()
        OptDB.view()
        OptDB.setFromOptions()
        # import pdb; pdb.set_trace()

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


        self.elasticity = ElasticitySolver(energy, state, bcs['elastic'], parameters['elasticity'])
        self.damage = DamageSolver(energy, state, bcs['damage'], parameters['damage'])

    def solve(self, debugpath=''):
        parameters = self.parameters
        it = 0
        # err_alpha = 1
        u = self.u
        alpha = self.alpha
        alpha_old = alpha.copy(deepcopy=True)
        alpha_diff = alpha.copy(deepcopy=True)
        criterion = 1

        alt_min_data = {
            "iterations": [],
            "alpha_error": [],
            "alpha_max": [],
            "energy": []}

        self.damage.solver.solver.setVariableBounds(self.damage.problem.lb.vec(),
            self.damage.problem.ub.vec())

        if debugpath:
            fname = 'am.xdmf'
            if os.path.exists(fname):
                os.remove(fname)

            file_am = XDMFFile(os.path.join(debugpath, fname))
            file_am.parameters["functions_share_mesh"] = True
            file_am.parameters["flush_output"] = True

        inactive_IS = self.damage.solver.inactive_set_indicator()
        Ealpha_vec = as_backend_type(assemble(self.damage.solver.Ealpha)).vec()
        Ealpha_residual = Ealpha_vec.getSubVector(inactive_IS)


        while criterion > float(self.parameters['equilibrium']["tol"]) and it < self.parameters['equilibrium']["max_it"]:
            it = it + 1
            (u_it, u_reason) = self.elasticity.solve()
            (alpha_it, alpha_reason) = self.damage.solve()


            alpha_diff.vector()[:] = alpha.vector() - alpha_old.vector()

            if self.parameters['equilibrium']['criterion'] == 'linf':
                criterion = abs(alpha_diff.vector().max())
            if self.parameters['equilibrium']['criterion'] == 'l2':
                criterion = norm(alpha_diff, 'l2')
            if self.parameters['equilibrium']['criterion'] == 'h1':
                criterion = norm(alpha_diff, 'h1')
            if self.parameters['equilibrium']['criterion'] == 'residual':
                criterion = Ealpha_residual.norm(2)

            log(LogLevel.INFO, 'linf {}'.format(abs(alpha_diff.vector().max())))
            log(LogLevel.INFO, 'l2   {}'.format(norm(alpha_diff, 'l2')))
            log(LogLevel.INFO, 'h1   {}'.format(norm(alpha_diff, 'h1')))
            log(LogLevel.INFO, 'res  {}'.format(Ealpha_residual.norm(2)))

            log(LogLevel.INFO,
                "Equilibrium solver iteration {:2d}: convergence criterion: {},\nalpha_error={:.3e} (tol={:.3e}), alpha_max={:.4g}, energy = {:.6e}".format(
                    it,
                    self.parameters['equilibrium']['criterion'],
                    criterion,
                    float(self.parameters['equilibrium']['tol']),
                    alpha.vector().max(),
                    assemble(self.energy)))

            # update
            alpha_old.assign(alpha)
            if debugpath:
                with file_am as file:
                    file.write(alpha, it)
                    file.write(u, it)

        irrev = alpha.vector()-self.damage.problem.lb
        if min(irrev[:]) >=0:
            log(LogLevel.INFO,'')
        else:
            log(LogLevel.INFO,'Pointwise irrev {}'.format(' NOK'))
            import pdb; pdb.set_trace()

        alt_min_data["alpha_error"].append(criterion)
        alt_min_data["alpha_max"].append(alpha.vector().max())
        alt_min_data["iterations"].append(it)
        alt_min_data["energy"].append(assemble(self.energy))

        log(LogLevel.INFO,
                "AM converged in {} iterations, err_alpha = {:.8e}, energy = {:.6e}".format(it, criterion, assemble(self.energy)))

        log(LogLevel.INFO,'')

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
        print(parameters)
        for parameter, value in parameters.items():
            log(LogLevel.INFO, "Set: {} = {}".format(prefix + parameter, value)) 
            PETScOptions.set(prefix + parameter, value)

        snes.setFromOptions()

    def solve(self):
        log(LogLevel.INFO, '________________________ EQUILIBRIUM _________________________')
        log(LogLevel.INFO, "Solving elasticity")
        problem = self.problem
        # Get the vector

        u = problem.state["u"].vector()
        self.solver.solve(problem, u)
        return (self.solver.snes().getIterationNumber(), self.solver.snes().getConvergedReason())

class DamageSolver:
    """Base class for damage solver, contains reference the damage problem and numerical solver (SNES)"""
    def __init__(self, energy, state, bcs, parameters={}):
        super(DamageSolver, self).__init__()
        self.energy = energy
        self.state = state
        self.bcs = bcs
        self.parameters = parameters

        solver_type = parameters['type']

        if solver_type == ['TAO']:
            self.solver = DamageTAO(energy, state, bcs, parameters)
            # self.problem = DamageProblemTAO(energy, state, bcs, parameters)
        elif solver_type == ['SNES']:
            self.problem = DamageProblemSNES(energy, state, bcs)
            self.solver = DamageSolverSNES(self.problem, parameters)

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

class DamageSolverSNES:
    """docstring for DamageSolverSNES"""
    def __init__(self, problem, parameters={}, lb=None):
        super(DamageSolverSNES, self).__init__()
        self.problem = problem
        self.energy = problem.energy
        self.state = problem.state
        self.alpha = problem.state['alpha']
        # self.alpha = problem.state['alpha']
        self.alpha_dvec = as_backend_type(self.alpha.vector())
        self.alpha_pvec = self.alpha_dvec.vec()

        self.bcs = problem.bcs
        self.parameters = parameters
        comm = self.alpha.function_space().mesh().mpi_comm()
        self.comm = comm
        V = self.alpha.function_space()
        self.V = V
        self.Ealpha = derivative(self.energy, self.alpha,
            dolfin.TestFunction(self.alpha.ufl_function_space()))
        self.dm = self.alpha.function_space().dofmap()
        solver = PETScSNESSolver()
        snes = solver.snes()
        # lb = self.alpha_init
        if lb == None: 
            lb=interpolate(Constant(0.), V)
        ub = interpolate(Constant(1.), V)

        prefix = "damage_"
        snes.setOptionsPrefix(prefix)
        for option, value in self.parameters["snes"].items():
            PETScOptions.set(prefix+option, value)
            # log(LogLevel.INFO, "PETScOptions.set({}, {})".format(prefix + option,value))
            log(LogLevel.INFO, "Set: {} = {}".format(prefix + option,value))

        snes.setFromOptions()

        # snes.view()

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

        # import pdb; pdb.set_trace()
        snes.solve(None, Function(V).vector().vec())

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

    def inactive_set_indicator(self, tol=1.0e-5):
        Ealpha = assemble(self.Ealpha)

        mask_grad = Ealpha[:] < tol
        mask_ub = self.alpha.vector()[:] < 1.-tol
        mask_lb = self.alpha.vector()[:] > self.problem.lb[:] + tol

        local_inactive_set_alpha = set(np.where(mask_ub == True)[0])    \
            & set(np.where(mask_grad == True)[0])                       \
            & set(np.where(mask_lb == True)[0])

        _set_alpha = [self.dm.local_to_global_index(k) for k in local_inactive_set_alpha]
        inactive_set_alpha = set(_set_alpha) | set(self.dm.dofs())

        index_set = petsc4py.PETSc.IS()
        index_set.createGeneral(list(inactive_set_alpha))  

        return index_set

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
        alpha = self.state["alpha"]
        self.lb = alpha.copy(deepcopy = True).vector()
        log(LogLevel.INFO, 'INFO: Updated irreversibility')


# -------------

# class EquilibriumNewton:
#     """docstring for EquilibriumNewton"""
#     def __init__(self, energy, state, bcs, z_0 = {}, parameters={}):
#         super(EquilibriumNewton, self).__init__()
#         self.energy = energy
#         self.state = state
#         self.bcs = bcs

#         OptDB = PETSc.Options()
#         OptDB.view()
#         OptDB.setFromOptions()

#         self.mesh = state['alpha'].function_space().mesh()
#         self.Z = dolfin.FunctionSpace(self.mesh, 
#             dolfin.MixedElement([state['u'].ufl_element(),
#                 state['alpha'].ufl_element()]))

#         self.z = Function(self.Z)


#         self.assigner = dolfin.FunctionAssigner(
#             self.Z,            # receiving space
#             [self.Z.sub(0), self.Z.sub(1)]) # assigning space

#         self.bcs = self.get_bcs(bcs)

#         comm = self.z.function_space().mesh().mpi_comm()
#         self.comm = comm

#         self.parameters = parameters

#         if z_0:
#             # self.u_0 = state_0['alpha']
#             self.alpha_0 = z_0['alpha']
#             self.u_0 = z_0['u']
#         else:
#             # self.u_0 = self.u.copy(deepcopy=True)
#             self.z_0 = self.z.copy(deepcopy=True)

#         # self.equilibrium = 
#         self.problem = EquilibriumProblemSNES(energy, state, self.bcs)
#         self.solver= EquilibriumSolverSNES(self.problem, parameters['newton'])
#         # self.solver = DamageSolverSNES(self.problem, parameters)

#     def get_bcs(self, bcs):
#         """
#         Construct the blocked u-DOF's for the stability problem
#         """

#         bcs_Z = []
#         zero = dolfin.Constant(0.0)

#         if self.state['u'].geometric_dimension()>1:
#             # vector
#             zeros = dolfin.Constant([0.0,]*self.state['u'].geometric_dimension())
#         elif self.state['u'].geometric_dimension()==1:
#             # scalar
#             zeros = dolfin.Constant(0.)

#         for bc in bcs['elastic']:
#             if hasattr(bc, 'sub_domain'):
#                 new_bc = dolfin.DirichletBC(self.Z.sub(0), zeros, bc.sub_domain, bc.method())
#             elif hasattr(bc, 'domain_args'):
#                 new_bc = dolfin.DirichletBC(self.Z.sub(0), zeros, bc.domain_args[0], bc.domain_args[1], bc.method())
#             else:
#                 raise RuntimeError("Couldn't find where bcs for displacement are applied")

#             bcs_Z.append(new_bc)

#         for bc in self.bcs['damage']:
#             if hasattr(bc, 'sub_domain'):
#                 new_bc = dolfin.DirichletBC(self.Z.sub(1), zero, bc.sub_domain, bc.method())
#             elif hasattr(bc, 'domain_args'):
#                 new_bc = dolfin.DirichletBC(self.Z.sub(1), zero, bc.domain_args[0], bc.domain_args[1], bc.method())
#             else:
#                 raise RuntimeError("Couldn't find where bcs for damage are applied")
#             bcs_Z.append(new_bc)



#         # for bc, space in zip([self.bcs['elastic'], self.bcs['damage']], [self.Z.sub.(0), self.Z.sub.(1)])

#         return bcs_Z

#     def solve(self, z_0={}, debugpath=''):
#         parameters = self.parameters
#         log(LogLevel.INFO, '________________________ EQUILIBRIUM _________________________')
#         log(LogLevel.INFO, "Solving equilibrium (Newton)")

#         if z_0:
#             self.alpha_0 = z_0['alpha']
#             self.u_0 = z_0['u']

#             dolfin.assign(self.problem.z.sub(0), self.u_0)
#             dolfin.assign(self.problem.z.sub(1), self.alpha_0)

#             # lb_alpha=self.alpha_0.copy(True).vector()

#             lb_u=interpolate(
#                 Constant([-1./dolfin.DOLFIN_EPS, -1./dolfin.DOLFIN_EPS]),
#                 self.Z.sub(0).collapse())

#             ub_alpha = interpolate(Constant(1.), self.Z.sub(1).collapse())

#             ub_u = interpolate(
#                 Constant([1./dolfin.DOLFIN_EPS, 1./dolfin.DOLFIN_EPS]),
#                 self.Z.sub(0).collapse())

#             lb, ub = Function(self.Z), Function(self.Z)

#             dolfin.assign(lb.sub(0), lb_u)
#             dolfin.assign(lb.sub(1), self.alpha_0)
#             dolfin.assign(ub.sub(0), ub_u)
#             dolfin.assign(ub.sub(1), ub_alpha)

#             self.problem.update_bounds(z={'lb': lb, 'ub': ub})

#         self.solver.snes.setVariableBounds(self.problem.lb.vec(),
#             self.problem.ub.vec())

#         inactive_IS = self.solver.inactive_set_indicator()

#         (z_it, z_reason) = self.solver.solve()

#         return (z_it, z_reason)

#     def update(self):
#         self.problem.update_lower_bound()
#         log(LogLevel.PROGRESS, 'PROGRESS: Updated irreversibility')

# class EquilibriumSolverSNES:
#     """docstring for EquilibriumSolverSNES"""
#     def __init__(self, problem, parameters={}):
#         super(EquilibriumSolverSNES, self).__init__()
#         self.problem = problem
#         self.energy = problem.energy
#         self.z = problem.z

#         self.z_dvec = as_backend_type(self.z.vector())
#         self.z_pvec = self.z_dvec.vec()

#         self.bcs = problem.bcs
#         self.parameters = parameters
#         comm = self.z.function_space().mesh().mpi_comm()
#         self.comm = comm
#         Z = self.z.function_space()
#         self.Z = Z
#         self.Ealpha = derivative(self.energy,
#             problem.alpha, 
#             TestFunction(problem.alpha.ufl_function_space()))

#         self.dm = self.z.function_space().dofmap()
#         solver = PETScSNESSolver()
#         snes = solver.snes()

#         prefix = "equilibrium_"
#         snes.setOptionsPrefix(prefix)
#         for option, value in self.parameters.items():
#             PETScOptions.set(prefix+option, value)
#             # log(LogLevel.INFO, "PETScOptions.set({}, {})".format(prefix + option,value))
#             log(LogLevel.INFO, "Set: {} = {}".format(prefix + option,value))

#         snes.setFromOptions()

#         # snes.view()

#         (J, F, bcs) = (problem.J, problem.F, problem.bcs)
#         # Create the SystemAssembler
#         # import pdb; pdb.set_trace()
#         self.ass = SystemAssembler(J, F, bcs)
#         # Intialise the residual
#         self.b = self.init_residual()
#         # Set the residual
#         snes.setFunction(self.residual, self.b.vec())
#         # Initialise the Jacobian
#         self.A = self.init_jacobian()
#         # Set the Jacobian
#         snes.setJacobian(self.jacobian, self.A.mat())
#         snes.ksp.setOperators(self.A.mat())

#         # Set the bounds
#         snes.setVariableBounds(self.problem.lb.vec(),
#             self.problem.ub.vec())

#         # import pdb; pdb.set_trace()

#         self.snes = snes

#     def update_x(self, x):
#         """
#         Given a PETSc Vec x, update the storage of our solution function z.
#         """
#         x.copy(self.z_pvec)
#         self.z_dvec.update_ghost_values()

#     def init_residual(self):
#         # Get the state
#         z = self.problem.z
#         # Initialise b
#         b = as_backend_type(
#             Function(z.function_space()).vector()
#             )
#         return b

#     def init_jacobian(self):
#         A = PETScMatrix(self.comm)
#         self.ass.init_global_tensor(A, Form(self.problem.J))
#         return A

#     def residual(self, snes, x, b):
#         self.update_x(x)
#         b_wrap = PETScVector(b)
#         self.ass.assemble(b_wrap, self.z_dvec)

#     def jacobian(self, snes, x, A, P):
#         self.update_x(x)
#         A_wrap = PETScMatrix(A)
#         self.ass.assemble(A_wrap)

#     def solve(self):
#         # alpha = self.problem.state["alpha"]
#         # self.solver.solve(self.problem,
#         #             alpha.vector())
#         #         # del self.solver_alpha

#         # alpha = self.problem.state["alpha"]
#         z = self.problem.z
#         # Need a copy for line searches etc. to work correctly.
#         x = z.copy(deepcopy=True)
#         xv = as_backend_type(x.vector()).vec()
#         # Solve the problem

#         self.snes.solve(None, xv)

#         return (self.snes.getIterationNumber(), self.snes.getConvergedReason())

#     def inactive_set_indicator(self, tol=1.0e-5):
#         Ealpha = assemble(self.Ealpha)
#         z = self.problem.z
#         # alpha = z.sub(1)

#         # mask_grad = Ealpha[:] < tol
#         mask_ub = z.vector()[:] < self.problem.ub - tol
#         mask_lb = z.vector()[:] > self.problem.lb[:] + tol

#         local_inactive_set_z = set(np.where(mask_ub == True)[0])  \
#             & set(np.where(mask_lb == True)[0])
#             # & set(np.where(mask_grad == True)[0])               \

#         _set_z = [self.dm.local_to_global_index(k) for k in local_inactive_set_z]
#         inactive_set_z = set(_set_z) | set(self.dm.dofs())

#         index_set = petsc4py.PETSc.IS()
#         index_set.createGeneral(list(inactive_set_z))  

#         return index_set

# class EquilibriumProblemSNES(NonlinearProblem):
#     """
#     Class for the Equilibrium problem with and NonlinearVariationalProblem.
#     """
#     def __init__(self, energy, state, bcs=None):
#         """
#         Initializes the damage problem.

#         Arguments:
#             * energy
#             * state
#             * boundary conditions
#         """
#         NonlinearProblem.__init__(self)
#         self.type = "snes"
#         self.bcs = bcs
#         self.mesh = state['alpha'].function_space().mesh()
#         self.Z = dolfin.FunctionSpace(self.mesh, 
#             dolfin.MixedElement([state['u'].ufl_element(),
#                 state['alpha'].ufl_element()]))

#         self.z = Function(self.Z)

#         self.alpha = state['alpha']
#         dolfin.assign(self.z.sub(0), state['u'])
#         dolfin.assign(self.z.sub(1), state['alpha'])


#         z = self.z
#         Z = self.Z

#         zv = TestFunction(Z)
#         dz = TrialFunction(Z)

#         self.energy = energy
#         self.F = derivative(energy, z, TestFunction(Z))

#         self.J = derivative(self.F, z, TrialFunction(Z))

#         self.lb, self.ub = self.setUpBounds()

#         pass

#     def setUpBounds(self):
#         lb, ub = Function(Z), Function(Z)

#         lb_u=interpolate(
#             Constant([-1./dolfin.DOLFIN_EPS, -1./dolfin.DOLFIN_EPS]),
#             Z.sub(0).collapse())


#         ub_alpha = interpolate(Constant(1.), Z.sub(1).collapse())
#         ub_u = interpolate(
#             Constant([1./dolfin.DOLFIN_EPS, 1./dolfin.DOLFIN_EPS]),
#             Z.sub(0).collapse())

#         dolfin.assign(lb.sub(0), lb_u)
#         dolfin.assign(lb.sub(1), self.alpha)

#         dolfin.assign(ub.sub(0), ub_u)
#         dolfin.assign(ub.sub(1), ub_alpha)

#         _lb = lb.vector()
#         _ub = ub.vector()

#         return _lb, _ub


#     def update_bounds(self, z):
#         lb = z['lb']
#         ub = z['ub']
#         self.lb = lb.copy(deepcopy = True).vector()
#         self.ub = ub.copy(deepcopy = True).vector()
#         log(LogLevel.INFO, 'INFO: Updated bounds')

#         return

#     def update_lower_bound(self):
#         """
#         Update the lower bounds.
#         """

#         z = self.z

#         self.lb = z.copy(deepcopy = True).vector()

#         log(LogLevel.INFO, 'INFO: Updated irreversibility')



