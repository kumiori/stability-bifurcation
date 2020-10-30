from dolfin import norm, assemble, assemble_system, TestFunction, XDMFFile, TrialFunction, interpolate
from dolfin import PETScTAOSolver, PETScSNESSolver, OptimisationProblem, NonlinearProblem, PETScOptions, PETScVector, PETScMatrix, Vector
from dolfin import NonlinearVariationalProblem, NonlinearVariationalSolver
from dolfin import Constant, Expression, Function, Form
from dolfin import SystemAssembler, as_backend_type
from utils import ColorPrint
from ufl import sqrt, inner, dot, conditional, derivative
import os
from petsc4py import PETSc
from dolfin.cpp.log import log, LogLevel


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
    "alpha_pc_type": "lu",
    "alpha_linesearch_type": 'basic',
    'alpha_pc_factor_mat_solver_type': 'mumps'}

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


class DamageSolverSNES:
    """
    Class for the damage solver.
    """

    def __init__(self, energy, state, bcs, parameters={}):
        """
        Initializes the SNES damage solver.
        """
        # Set the solver name
        solver_name = "damage_snes"
        # Store the problem
        self.problem = DamageSNES(energy, state, bcs)
        # Get the damage variablex``xsnes
        self.alpha = self.problem.alpha
        # Get the vectors
        self.alpha_dvec = as_backend_type(self.alpha.vector())
        self.alpha_pvec = self.alpha_dvec.vec()
        # Create the solver
        comm = self.alpha.function_space().mesh().mpi_comm()
        self.comm = comm
        snes = PETSc.SNES().create(comm=comm)
        # Set the prefix
        prefix = "{}_".format(solver_name)
        snes.setOptionsPrefix(prefix)
        # Set the PETSc options from the parameters
        for parameter, value in parameters['solver_alpha_snes'].items():
            log(LogLevel.INFO, 'setting {}: {}'.format(parameter, value))
            PETScOptions.set(prefix + parameter, value)
        # Get the functions of the problem
        (J, F, bcs_alpha) = (self.problem.ddenergy, self.problem.denergy, self.problem.bcs)
        # Create the SystemAssembler
        self.ass = SystemAssembler(J, F, bcs_alpha)
        # Intialise the residual
        self.b = self.init_residual()
        # Set the residual
        snes.setFunction(self.residual, self.b.vec())
        # import pdb; pdb.set_trace()
        # Initialise the Jacobian
        self.A = self.init_jacobian()
        # Set the Jacobian
        snes.setJacobian(self.jacobian, self.A.mat())
        snes.ksp.setOperators(self.A.mat())
        # Set the bounds
        # import pdb; pdb.set_trace()

        snes.setVariableBounds(self.problem.lb.vec(), self.problem.ub.vec())
        # Update the parameters
        snes.setFromOptions()
        # Store the solver
        self.snes = snes

    def init_residual(self):
        # Get the state
        alpha = self.problem.alpha
        # Initialise b
        b = as_backend_type(Function(alpha.function_space()).vector())
        return b

    def init_jacobian(self):
        A = PETScMatrix(self.comm)
        # import pdb; pdb.set_trace()
        self.ass.init_global_tensor(A, Form(self.problem.ddenergy))
        return A

    def update_x(self, x):
        """
        Given a PETSc Vec x, update the storage of our solution function alpha.
        """
        x.copy(self.alpha_pvec)
        self.alpha_dvec.update_ghost_values()

    def residual(self, snes, x, b):
        self.update_x(x)
        b_wrap = PETScVector(b)
        self.ass.assemble(b_wrap, self.alpha_dvec)

    def jacobian(self, snes, x, A, P):
        self.update_x(x)
        A_wrap = PETScMatrix(A)
        self.ass.assemble(A_wrap)

    def solve(self):
        """
        Solve the damage problem in its current state.
        """
        # Display info
        log(LogLevel.INFO, 'Damage problem SNES')

        # Get the damage variable
        alpha = self.problem.alpha
        # Need a copy for line searches etc. to work correctly.
        x = alpha.copy(deepcopy=True)
        xv = as_backend_type(x.vector()).vec()
        # Solve the problem

        self.snes.solve(None, xv)

class DamageSNES(NonlinearProblem):

    def __init__(self, energy, alpha, bcs, lb=None, ub=None):

        NonlinearProblem.__init__(self)
        self.energy = energy
        self.alpha = alpha
        self.V = self.alpha.function_space()
        self.denergy = derivative(
            self.energy, self.alpha, TestFunction(self.V))
        self.ddenergy = derivative(
            self.denergy, self.alpha, TrialFunction(self.V))
        # import pdb; pdb.set_trace()

        self.lb = self.alpha.copy(True).vector()
        self.ub = interpolate(Constant(1.), self.V).vector()

        # if lb == None:
        #     lb = interpolate(Constant("0."), self.V)
        # if ub == None:
        #     ub = interpolate(Constant("1."), self.V)
        # self.lb = lb
        # self.ub = ub

        self.bcs = bcs
        self.b = PETScVector()
        self.A = PETScMatrix()
    # def form(self, A, b, x):
    #    pass

    def F(self, b, x):
     #       self.alpha.vector()[:] = x
        #assemble(self.denergy, tensor=b)
        #[bc.apply(b, x) for bc in self.bcs]
        assemble_system(self.ddenergy, self.denergy,
                        bcs=self.bcs, A_tensor=self.A, b_tensor=b)
        pass

    def J(self, A, x):
        #        self.alpha.vector()[:] = x
        #assemble(self.ddenergy, tensor=A)
        #[bc.apply(A) for bc in self.bcs]
        assemble_system(self.ddenergy, self.denergy,
                        bcs=self.bcs, A_tensor=A, b_tensor=self.b)
        pass

    def update_lb(self, lb=None):
        """update the lower bound"""
        # import pdb; pdb.set_trace()
        if lb:
            # self.lb.assign(lb)
            self.lb.vec()[:] = lb.vector()[:]
        else:
            # self.lb.assign(self.alpha)
            self.lb.vec()[:] = self.alpha.vector()[:]

    def active_set_indicator_lb(self, tol=1.0e-5):
        clb = conditional(self.alpha > tol + self.lb, 0.0, 1.0)
        return clb

    def active_set_indicator_ub(self, tol=1.0e-5):
        cub = conditional(self.alpha < self.ub + tol, 0.0, 1.0)
        return cub

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
        comm = self.alpha.function_space().mesh().mpi_comm()
        self.comm = comm

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
            self.solver_alpha = DamageSolverSNES(
                    self.energy, state[1], self.bcs_alpha, parameters)
            self.problem_alpha = self.solver_alpha.problem
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

    # def set_solver_alpha_snes(self):
    #     V = self.alpha.function_space()
    #     self.problem_alpha = DamageProblemSNES(
    #         self.energy, self.alpha, self.bcs_alpha, lb=self.alpha_init)

    #     # solver = PETScSNESSolver()
    #     # snes = solver.snes()
    #             # Create the solver
    #     comm = self.alpha.function_space().mesh().mpi_comm()
    #     # self.comm = comm
    #     snes = PETSc.SNES().create(comm=comm)

    #     lb = self.alpha_init
    #     ub = interpolate(Constant("1."), V)

    #     snes.setOptionsPrefix("alpha_")
    #     (J, F, bcs_alpha) = (self.problem_alpha.ddenergy,
    #                         self.problem_alpha.denergy,
    #                         self.problem_alpha.bcs)
 
    #     for option, value in self.parameters["solver_alpha_snes"].items():
    #         print("setting ", option,value)
    #         PETScOptions.set(option, value)

    #     snes.setFromOptions()

    #     self.ass = SystemAssembler(J, F, bcs_alpha)
    #     # Intialise the residual
    #     self.b = self.init_residual()
    #     # Set the residual
    #     snes.setFunction(self.residual, self.b.vec())
    #     # Initialise the Jacobian
    #     self.A = self.init_jacobian()
    #     # Set the Jacobian
    #     snes.setJacobian(self.jacobian, self.A.mat())
    #     snes.ksp.setOperators(self.A.mat())

    #     # import pdb; pdb.set_trace()
    #     snes.setVariableBounds(lb.vector().vec(), ub.vector().vec()) # 
    #     # self.solver_alpha = snes
    #     snes.setFromOptions()

    #     self.solver_alpha = snes

    def init_residual(self):
        # Get the state
        alpha = self.problem_alpha.alpha
        # Initialise b
        b = as_backend_type(
            Function(alpha.function_space()).vector()
            )
        return b

    def init_jacobian(self):
        A = PETScMatrix(self.comm)
        self.ass.init_global_tensor(A, Form(self.problem_alpha.ddenergy))
        return A

    def residual(self, snes, x, b):
        self.update_x(x)
        b_wrap = PETScVector(b)
        self.ass.assemble(b_wrap, self.alpha_dvec)

    def jacobian(self, snes, x, A, P):
        self.update_x(x)
        A_wrap = PETScMatrix(A)
        self.ass.assemble(A_wrap)


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

            # if self.parameters["solver_alpha"] == "snes2":
            #     # self.set_solver_alpha_snes2()
            #     (alpha_it, alpha_reason) = self.solver_alpha.solve()

            if self.parameters["solver_alpha"] == "snes":
                # self.set_solver_alpha_snes()
                # Solve the problem
                # import pdb; pdb.set_trace()
                # (alpha_it, alpha_reason) = 
                self.solver_alpha.solve()
                # (alpha_it, alpha_reason) = self.solver_alpha.solve(
                #     self.problem_alpha,
                #     self.alpha.vector())
                # del self.solver_alpha

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
                else: 
                    log(LogLevel.INFO,'Pointwise irrev {}'.format(' NOK'))


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




class DamageElasticitySolver:
    """
    Class for the damage elasticity solver.
    The resolution is done via an alternate minimization of the energy with
    respect to displacement and damage.
    """

    def __init__(self, energy, model, state, bcs, parameters={}):
        """
        Initializes the elasticity solver from the elasticity problem.
        """
        # Set the solver name
        self.solver_name = "damage_elasticity"
        # Store the model
        self.model = model
        # Store the parameters
        self.parameters = parameters
        # Set the elasticity solver
        self.elasticity_solver = ElasticitySolver(
                energy, state, bcs, parameters)
        # Set the damage solver
        self.damage_solver = DamageSolver(
                energy, state, bcs, parameters)

    def solve(self):
        """
        Solve the problem in its current state using an alternate minimization
        algorithm.
        """
        # Get the solver parameters
        pars = self.parameters[self.solver_name]
        # Get the model
        model = self.model
        # Get the state
        state = self.damage_solver.problem.state
        # Get the state variables
        alpha = state[1]
        # Update the bound of the damage problem
        self.damage_solver.problem.update_lower_bound()
        # Setup quantities for cnvergence criteria
        convergence = False
        i = 1
        damage_diss_dens_im1 = local_project(
            model.damage_dissipation_density(state), alpha.function_space())
        err_damage_diss_dens = Function(alpha.function_space())
        # Resolution loop
        while not convergence:
            # Raise error if max_it is reached
            if i == pars["max_it"]:
                template = "Alternate minimization did not converge in {}"
                message = template.format(i)
                raise RuntimeError(message)
            # Solve the elastic problem
            self.elasticity_solver.solve()
            # Solve the damage problem
            self.damage_solver.solve()
            # Compute damage error between two iterations of the AM
            damage_diss_dens = local_project(
                model.damage_dissipation_density(state),
                alpha.function_space())
            err_damage_diss_dens.assign(
                    damage_diss_dens - damage_diss_dens_im1,
                    )
            damage_energy_rerr = norm(
                err_damage_diss_dens,
                norm_type="L2",
                mesh=alpha.function_space().mesh()
                )
            # Update damage_diss_dens_im1
            damage_diss_dens_im1.assign(damage_diss_dens)
            # Check the criterion
            convergence = damage_energy_rerr < pars["alpha_rtol"]
            # Display some information
            message = "  INFO AM: Iteration {} -  Error {}"
            ColorPrint.print_info(
                message.format(i, damage_energy_rerr)
                )
            # Increment i
            i += 1


