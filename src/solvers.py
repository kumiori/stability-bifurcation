from dolfin import norm, assemble, assemble_system, TestFunction, XDMFFile, TrialFunction, interpolate
from dolfin import PETScTAOSolver, PETScSNESSolver, OptimisationProblem, NonlinearProblem, PETScOptions, PETScVector, PETScMatrix, as_backend_type, Vector
from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
from dolfin import Constant, Expression
from utils import ColorPrint
from ufl import sqrt, inner, dot, conditional, derivative
import os


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

class ElasticityProblem(NonlinearProblem):

    def __init__(self, energy, u, bcs, nullspace=None):
        NonlinearProblem.__init__(self)
        self.z = u
        self.bcs = bcs
        self.nullspace=nullspace
        self.residual = derivative(
            energy, self.z, TestFunction(self.z.function_space()))
        self.jacobian = derivative(
            self.residual, self.z, TrialFunction(self.z.function_space()))

    def F(self, b, x):
        assemble(self.residual, tensor=b)
        [bc.apply(b, x) for bc in self.bcs]

    def J(self, A, x):
        assemble(self.jacobian, tensor=A)
        if self.nullspace:
            ColorPrint.print_info("Removing nullspace in 1st order problem")
            A.set_nullspace(self.nullspace)
        [bc.apply(A) for bc in self.bcs]

class AlternateMinimizationSolver(object):

    def __init__(self, energy, state, bcs, state_0=None, parameters=default_parameters, nullspace=None):
        self.energy = energy
        self.u = state[0]
        self.alpha = state[1]
        self.bcs_u = bcs[0]
        self.bcs_alpha = bcs[1]
        if state_0:
            self.u_init = state_0[0]
            self.alpha_init = state_0[1]
        else:
            self.u_init = self.u.copy(deepcopy=True)
            self.alpha_init = self.alpha.copy(deepcopy=True)

        self.V_alpha = self.alpha.function_space()
        # import pdb; pdb.set_trace()
        self.parameters = parameters
        self.problem_u = ElasticityProblem(self.energy, self.u, self.bcs_u, nullspace=nullspace)
        self.set_solver_u()
        if self.parameters["solver_alpha"] == "snes":
            self.set_solver_alpha_snes()
        elif self.parameters["solver_alpha"] == "snes2":
            self.set_solver_alpha_snes2()
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
            ColorPrint.print_warn('Could not configure preconditioner')
        solver.set_from_options()
        snes.setFromOptions()
        self.solver_u = solver

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
        if self.parameters["solver_alpha"] == "snes2":
            self.problem_alpha.lb.assign(self.alpha)
            self.problem_alpha.set_bounds(self.alpha, interpolate(Constant("2."), self.alpha.function_space()))
            print('Updated irreversibility')
        else:
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

        while criterion > par["tol"] and it < par["max_it"]:
            it = it + 1
            (u_it, u_reason) = self.solver_u.solve(
                self.problem_u, self.u.vector())

            if self.parameters["solver_alpha"] == "snes2":
                self.set_solver_alpha_snes2()
                (alpha_it, alpha_reason) = self.solver.solve()

            elif self.parameters["solver_alpha"] == "snes":
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
                    ColorPrint.print_pass('')
                else: ColorPrint.print_warn('Pointwise irrev {}'.format(' NOK'))


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

        return (take_last(alt_min_data), alt_min_data)




