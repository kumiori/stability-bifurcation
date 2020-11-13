# sys.path.append("../src/")
# from post_processing import compute_sig, local_project
import site
import sys


import pandas as pd

import sys
# from linsearch import LineSearch
# import solvers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mshr
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

        self.elasticity_solver = ElasticitySolver(energy, state, bcs['elastic'], parameters['elasticity'])
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

        while criterion > self.parameters["tol"] and it < self.parameters["max_it"]:
            it = it + 1
            (u_it, u_reason) = self.elasticity_solver.solve()
            (alpha_it, alpha_reason) = self.damage_solver.solve()

            irrev = alpha.vector()-self.damage_solver.problem.lb
            if min(irrev[:]) >= - self.damage_solver.parameters['snes']['snes_rtol']:
              log(LogLevel.INFO,'')
            else:
              log(LogLevel.INFO,'Pointwise irrev {}'.format(' NOK'))

            if not np.all(min(alpha.vector()[:] - self.damage_solver.problem.lb[:]) >= - self.damage_solver.parameters['snes']['snes_rtol']):
                pd = np.where(alpha.vector()[:]-self.damage_solver.problem.lb[:] 
                        < 0.)[0]
                log(LogLevel.WARNING, 'Pointwise irreversibility issues on dofs {}'.format(pd))
                log(LogLevel.WARNING, 'difference {}'.format(alpha.vector()[pd]-self.damage_solver.problem.lb[pd]))
                # import pdb; pdb.set_trace()

                log(LogLevel.WARNING, 'Continuing')

            alpha_error.vector()[:] = alpha.vector() - alpha_old.vector()
            # crit: energy norm
            err_alpha = abs(alpha_error.vector().max())
            # err_alpha = norm(alpha_error,'h1')
            criterion = err_alpha

            alt_min_data["iterations"].append(it)
            alt_min_data["alpha_error"].append(err_alpha)
            alt_min_data["alpha_max"].append(alpha.vector().max())

            log(LogLevel.INFO,
                "   AM iter {:2d}: alpha_error={:.4g}, alpha_max={:.4g}".format(
                    it,
                    err_alpha,
                    alpha.vector().max()
                )
            )

            # update
            alpha_old.assign(alpha)
        log(LogLevel.INFO,
                "AM converged in {} iterations, err = {}".format(it, err_alpha))

        return (alt_min_data, it)

    def update(self):
        self.damage_solver.problem.update_lower_bound()
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
        u = state["u"]

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
            log(LogLevel.INFO, "Set: {} = {}".format(prefix + parameter, value))
            PETScOptions.set(prefix + parameter, value)
        snes.setFromOptions()

    def solve(self):
        # Get the problem
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
        log(LogLevel.CRITICAL, 'CRITICAL: Updated irreversibility')


def numerical_test(
    user_parameters,
    ell=0.05,
    nu=0.,
):

    # Create mesh and define function space
    Lx = 1; Ly = .1
    Lx = 1.; Ly = .1
    n = 3
    comm = MPI.comm_world
    # mesh = RectangleMesh(Point(-Lx/2, 0.), Point(Lx/2, Ly), 50, 10)
    geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))
    mesh = mshr.generate_mesh(geom,  int(float(n * Lx / ell)))

    # Define Dirichlet boundaries
    outdir = '../test/output/test_firstorderevo'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
    meshf << mesh

    with open('../parameters/form_compiler.yml') as f:
        form_compiler_parameters = yaml.load(f, Loader=yaml.FullLoader)

    with open('../parameters/solvers_default.yml') as f:
        solver_parameters = yaml.load(f, Loader=yaml.FullLoader)

    with open('../parameters/tractionbar.yml') as f:
        material_parameters = yaml.load(f, Loader=yaml.FullLoader)['material']

    with open('../parameters/loading.yaml') as f:
        loading_parameters = yaml.load(f, Loader=yaml.FullLoader)['loading']

    Path(outdir).mkdir(parents=True, exist_ok=True)

    print('Outdir is: '+outdir)

    geometry_parameters = {'Lx': 1., 'n': 3}

    default_parameters = {'solver':{**solver_parameters},
        'compiler': {**form_compiler_parameters},
        'loading': {**loading_parameters},
        'material': {**material_parameters},
        'geometry': {**geometry_parameters}
        }

    default_parameters.update(user_parameters)
    # FIXME: Not nice
    parameters = default_parameters

    with open(os.path.join(outdir, 'parameters.yaml'), "w") as f:
        yaml.dump(parameters, f, default_flow_style=False)

    savelag = 1
    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    left_bottom_pt = dolfin.CompiledSubDomain("near(x[0],-Lx/2.) && near(x[1],-Ly/2.)", Lx=Lx, Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)

    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_u = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = Function(V_alpha)
    dalpha = TrialFunction(V_alpha)

    state = {'u': u, 'alpha': alpha}
    Z = dolfin.FunctionSpace(mesh, 
            dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)
    v, beta = dolfin.split(z)

    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u.sub(0), dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u.sub(0), ut, right),
             dolfin.DirichletBC(V_u, (0, 0), left_bottom_pt, method="pointwise")]

    bcs_alpha_l = DirichletBC(V_alpha,  Constant(0.0), left)
    bcs_alpha_r = DirichletBC(V_alpha, Constant(0.0), right)
    bcs_alpha =[bcs_alpha_l, bcs_alpha_r]
    # bcs_alpha = []

    bcs = {"damage": bcs_alpha, "elastic": bcs_u}

    # import pdb; pdb.set_trace()

    ell = parameters['material']['ell']

    # Problem definition

    k_ell = 1e-8
    a = (1 - alpha) ** 2. + k_ell
    w_1 = parameters['material']['sigma_D0'] ** 2 / parameters['material']['E']
    w = w_1 * alpha
    eps = sym(grad(u))
    lmbda0 = parameters['material']['E'] * parameters['material']['nu'] /(1. - parameters['material']['nu'])**2.
    mu0 = parameters['material']['E']/ 2. / (1.0 + parameters['material']['nu'])
    Wu = 1. / 2. * lmbda0 * tr(eps) ** 2 + mu0 * inner(eps, eps)

    energy = 1./2.* a * Wu * dx + w_1 *( alpha +  parameters['material']['ell']** 2.*alpha.dx(0)**2.)*dx

    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_out.parameters["functions_share_mesh"] = True
    file_out.parameters["flush_output"] = True

    with open('../parameters/solvers_default.yml') as f:
        solver_parameters = yaml.load(f, Loader=yaml.FullLoader)

    solver = EquilibriumSolver(energy, state, bcs, parameters=solver_parameters)

    load_steps = np.linspace(parameters['loading']['load_min'],
        parameters['loading']['load_max'],
        parameters['loading']['n_steps'])
    time_data = []
    time_data_pd = []
    spacetime = []

    for it, load in enumerate(load_steps):
        log(LogLevel.CRITICAL, 'CRITICAL: Solving load t = {:.2f}'.format(load))
        ut.t = load
        (time_data_i, am_iter) = solver.solve()
        solver.update()

        time_data_i["load"] = load
        # time_data_i["stable"] = stable

        time_data_i["elastic_energy"] = dolfin.assemble(
            1./2.* material_parameters['E']*a*eps**2. *dx)
        time_data_i["dissipated_energy"] = dolfin.assemble(
            (w + w_1 * material_parameters['ell'] ** 2. * alpha.dx(0)**2.)*dx)

        log(LogLevel.CRITICAL,
            "Load/time step {:.4g}: iteration: {:3d}, err_alpha={:.4g}".format(
                time_data_i["load"],
                time_data_i["iterations"][0],
                time_data_i["alpha_error"][0]))

        time_data.append(time_data_i)
        time_data_pd = pd.DataFrame(time_data)

        if np.mod(it, savelag) == 0:
            with file_out as f:
                f.write(alpha, load)
                f.write(u, load)
            with dolfin.XDMFFile(os.path.join(outdir, "output_postproc.xdmf")) as f:
                f.write_checkpoint(alpha, "alpha-{}".format(it), 0, append = True)
                log(LogLevel.PROGRESS, 'PROGRESS: written step {}'.format(it))

            time_data_pd.to_json(os.path.join(outdir, "time_data.json"))

        spacetime.append(get_trace(alpha))

    _spacetime = pd.DataFrame(spacetime)
    spacetime = _spacetime.fillna(0)
    mat = np.matrix(spacetime)
    plt.imshow(mat, cmap = 'Greys', vmin = 0., vmax = 1., aspect=.1)
    plt.colorbar()

    def format_space(x, pos, xresol = 100):
        return '$%1.1f$'%((-x+xresol/2)/xresol)

    def format_time(t, pos, xresol = 100):
        return '$%1.1f$'%((t-parameters['loading']['load_min'])/parameters['loading']['n_steps']*parameters['loading']['load_max'])

    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax = plt.gca()

    ax.yaxis.set_major_formatter(FuncFormatter(format_space))
    ax.xaxis.set_major_formatter(FuncFormatter(format_time))
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.savefig(os.path.join(outdir, "spacetime.pdf".format(load)), bbox_inches="tight")

    spacetime.to_json(os.path.join(outdir + "/spacetime.json"))
    plt.clf()

    plot(mesh)
    plt.savefig(os.path.join(outdir, 'mesh.pdf'))
    plt.clf()

    from matplotlib.ticker import FuncFormatter, MaxNLocator
    plot(alpha)
    plt.savefig(os.path.join(outdir, 'alpha.pdf'))
    plt.clf()
    log(LogLevel.INFO, "Saved figure: {}".format(os.path.join(outdir, 'alpha.pdf')))


    xs = np.linspace(-Lx/2., Lx/2., 100)
    profile = np.array([alpha(x, 0) for x in xs])

    plt.figure()
    plt.plot(xs, profile, marker='o')
    plt.plot(xs, np.array([u(x, 0) for x in xs]))
    # plt.ylim(0., 1.)
    plt.savefig(os.path.join(outdir, 'profile.pdf'))

    return time_data_pd, outdir

def get_trace(alpha, xresol = 100):
    X =alpha.function_space().tabulate_dof_coordinates()
    xs = np.linspace(min(X[:, 0]),max(X[:, 0]), xresol)
    alpha0 = [alpha(x, 0) for x in xs]

    return alpha0

if __name__ == "__main__":

    # Parameters
    with open('../parameters/tractionbar.yml') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    data, experiment = numerical_test(user_parameters = parameters)

    log(LogLevel.INFO, "Postprocess")
    with open(os.path.join(experiment, 'parameters.yaml')) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)


