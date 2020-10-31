# sys.path.append("../src/")
# from post_processing import compute_sig, local_project
import site
import sys




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







