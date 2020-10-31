# sys.path.append("../src/")
# from post_processing import compute_sig, local_project
import site
import sys




import sys
# from linsearch import LineSearch
from damage_elasticity_model import DamageElasticityModel1D
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
from dolfin.cpp.log import log, LogLevel
dolfin.parameters["std_out_all_processes"] = False



form_compiler_parameters = {
    "representation": "uflacs",
    "quadrature_degree": 2,
    "optimize": True,
    "cpp_optimize": True,
}

timestepping_parameters = {"perturbation_choice": 'steepest',
                            "savelag": 1,
                            "outdir": '',
                            'cont_rtol': 1e-5}
                        # "perturbation_choice": 'steepest',               # admissible choices: steepest, first, #

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

alt_min_parameters = {"max_it": 300,
                      "tol": 1.e-5,
                      "solver_u": petsc_options_u,
                      # either
                      "solver_alpha": "snes",
                      "solver_alpha_snes": petsc_options_alpha_snes
                      # or
                      # "solver_alpha": "tao",
                     # "solver_alpha_tao": petsc_options_alpha_tao
                     }

parameters = {"alt_min": alt_min_parameters,
                # "solver_u": petsc_options_u,
                # "solver_alpha_tao": petsc_options_alpha_tao, "solver_alpha_snes": petsc_options_alpha_snes,
                "stability": stability_parameters,
                "time_stepping": timestepping_parameters,
                "material": {},
                "geometry": {},
                "experiment": {},
                "code": versions
                }

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

	def solve(self)
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

		solver_type = parameters['solver_type']

		if solver_type = 'TAO':
			self.solver = DamageTAO(energy, state, bcs, parameters)
			# self.problem = DamageProblemTAO(energy, state, bcs, parameters)
		elif solver_type = 'SNES':
			self.solver = DamageSolverSNES(energy, state, bcs, parameters)
			self.problem = DamageProblemSNES(energy, state, bcs)

		def solve(self):

class DamageSolverSNES:
	"""docstring for DamageSolverSNES"""
	def __init__(self, energy, state, bcs, parameters={}, lb=None):
		super(DamageSolverSNES, self).__init__()
		self.energy = energy
		self.state = state
		alpha = state['alpha']
		self.bcs = bcs
		self.parameters = parameters
		V = alpha.function_space()

		solver = PETScSNESSolver()
		snes = solver.snes()
		# lb = self.alpha_init
		if lb == None: 
			lb=interpolate(Constant('0.', V))
        ub = interpolate(Constant("1."), V)

		snes.setOptionsPrefix("alpha_")
		for option, value in self.parameters["solver_alpha_snes"].items():
			PETScOptions.set(option, value)
			log(LogLevel.INFO, "Set ", option,value)
		snes.setFromOptions()
        snes.setVariableBounds(lb.vector().vec(), ub.vector().vec()) # 

		self.solver = snes

	def solve(self):
		self.solver(self.problem,
					self.alpha.vector())
				# del self.solver_alpha

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




def traction_test(
    ell=0.05,
    degree=1,
    n=3,
    nu=0.,
    load_min=0,
    load_max=2,
    loads=None,
    nsteps=20,
    Lx=1.,
    outdir="outdir",
    postfix='',
    savelag=1,
    sigma_D0=1.,
    continuation=False,
    checkstability=True,
    configString='',
):
    Lx = Lx
    load_min = load_min
    load_max = load_max
    nsteps = nsteps
    outdir = outdir
    loads=loads

    savelag = 1
    nu = dolfin.Constant(nu)
    ell = dolfin.Constant(ell)
    ell_e = ell_e
    E = dolfin.Constant(1.0)
    K = E.values()[0]/ell_e**2.
    sigma_D0 = E
    n = n
    h = max(ell.values()[0]/n, .005)
    cell_size = h
    continuation = continuation
    config = json.loads(configString) if configString != '' else ''

    cmd_parameters =  {
    'material': {
        "ell": ell.values()[0],
        "K": K,
        "E": E.values()[0],
        "nu": nu.values()[0],
        "sigma_D0": sigma_D0.values()[0]},
    'geometry': {
        'Lx': Lx,
        'n': n,
        },
    'experiment': {
        'test': test,
        'signature': ''
        },
    'stability': {
        'checkstability' : checkstability,
        'continuation' : continuation
        },
    'time_stepping': {
        'load_min': load_min,
        'load_max': load_max,
        'nsteps':  nsteps,
        'outdir': outdir,
        'postfix': postfix,
        'savelag': savelag},
    'alt_min': {}, "code": {}
    }



    for par in parameters: parameters[par].update(cmd_parameters[par])

    if config:
        for par in config: parameters[par].update(config[par])
    # else:

    # parameters['material']['ell_e'] = 

    Lx = parameters['geometry']['Lx']
    Ly = parameters['geometry']['Ly']
    ell = parameters['material']['ell']
    ell_e = parameters['material']['ell_e']

    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    fname="film"
    print(BASE_DIR)
    os.path.isfile(fname)

    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()

    Path(outdir).mkdir(parents=True, exist_ok=True)
    print('Outdir is: '+outdir)

    with open(os.path.join(outdir, 'rerun.sh'), 'w') as f:
        configuration = deepcopy(parameters)
        configuration['time_stepping'].pop('outdir')
        str(configuration).replace("\'True\'", "True").replace("\'False\'", "False")
        rerun_cmd = 'python3 {} --config="{}"'.format(os.path.basename(__file__), configuration)
        f.write(rerun_cmd)

    with open(os.path.join(outdir, 'parameters.pkl'), 'w') as f:
        json.dump(parameters, f)

    with open(os.path.join(outdir, 'signature.md5'), 'w') as f:
        f.write(signature)
    print(parameters)
    
 # ------------------------------------

    print('experiment = {}'.format(os.path.join('~/Documents/WIP/paper_stability_code', outdir)))
    mesh = dolfin.IntervalMesh(int(float(n * Lx / ell)), -Lx/2., Lx/2.)
    meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
    meshf << mesh

    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)
    # bottom.mark(mf, 3)
    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_u = dolfin.FunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = dolfin.Function(V_alpha, name="Damage")
    state = {'u':u, 'alpha': alpha}



    Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)

    v, beta = dolfin.split(z)

    # BCs (homogenous version needed for residual evaluation)
    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u, dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u, ut, right)]
    bcs_alpha = []
    bcs = {'elastic': bc_u, 'damage': bcs_alpha}

    log(LogLevel.INFO, 'Outdir = {}'.format(outdir))

	file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_out.parameters["functions_share_mesh"] = True
    file_out.parameters["flush_output"] = True
    file_con = dolfin.XDMFFile(os.path.join(outdir, "cont.xdmf"))
    file_con.parameters["functions_share_mesh"] = True
    file_con.parameters["flush_output"] = True
    file_eig = dolfin.XDMFFile(os.path.join(outdir, "modes.xdmf"))
    file_eig.parameters["functions_share_mesh"] = True
    file_eig.parameters["flush_output"] = True







