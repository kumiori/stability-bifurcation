import dolfin
import sys
from utils import ColorPrint
from ufl import sqrt, inner, dot, conditional, derivative, ge, le
import os
# from slepc_eigensolver import EigenSolver
from slepc4py import SLEPc
import ufl
from functools import reduce
import numpy as np
import petsc4py
from petsc4py import PETSc
from dolfin import MPI
import matplotlib.pyplot as plt
from dolfin.cpp.log import log, LogLevel, get_log_level
import mpi4py
import yaml 

from dolfin import derivative, assemble

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
log_level = get_log_level()

class EigenSolver(object):
    def __init__(self,
                 a_k,
                 u,
                 a_m=None,                                   # optional, for eigpb of the type (K-lambda M)x=0
                 bcs=None,
                 restricted_dofs_is=None,
                 slepc_options='',
                 option_prefix='eigen_',
                 comm=MPI.comm_world, 
                 slepc_eigensolver = None,
                 initial_guess = None
                ):
        self.comm = comm


        # import pdb; pdb.set_trace()
        self.slepc_options = slepc_options
        self.V = u.function_space()
        self.index_set_not_bc = None
        if type(bcs) == list:
            self.bcs = bcs
        elif type(bcs) == dolfin.fem.dirichletbc.DirichletBC:
            self.bcs = [bcs]
        else:
            self.bcs = None

        if type(a_k) == ufl.form.Form:
            # a form to be assembled
            self.K = dolfin.as_backend_type(assemble(a_k)).mat()
        elif type(a_k) == petsc4py.PETSc.Mat:
            # an assembled petsc matrix
            self.K = a_k

        if a_m is not None and type(a_m) == ufl.form.Form:
            self.M = dolfin.as_backend_type(assemble(a_m)).mat()
        elif a_m is not None and type(a_m) == petsc4py.PETSc.Mat:
            self.M = a_m

        # if bcs extract reduced matrices on dofs with no bcs
        if self.bcs:
            self.index_set_not_bc = self.get_interior_index_set(self.bcs, self.V)
        elif restricted_dofs_is:
            self.index_set_not_bc = restricted_dofs_is

        if self.index_set_not_bc is not None:
            try:
                self.K = self.K.createSubMatrix(self.index_set_not_bc, self.index_set_not_bc)
                if a_m:
                   self.M = self.M.createSubMatrix(self.index_set_not_bc, self.index_set_not_bc)
            except:
                self.K = self.K.getSubMatrix(self.index_set_not_bc, self.index_set_not_bc)
                if a_m:
                    self.M = self.M.getSubMatrix(self.index_set_not_bc, self.index_set_not_bc)
            self.projector = petsc4py.PETSc.Scatter()
            self.projector.create(
                vec_from=self.K.createVecRight(),
                is_from=None,
                vec_to=u.vector().vec(),
                is_to=self.index_set_not_bc
                )

        self.initial_guess = initial_guess

        # set up the eigensolver
        if slepc_eigensolver:
            self.E = slepc_eigensolver
        else:
            self.E = self.eigensolver_setup(prefix=option_prefix)

        if a_m:
            self.E.setOperators(self.K,self.M)
        else:
            self.E.setOperators(self.K)

    def get_interior_index_set(self, boundary_conditions, function_space):
        """Returns the index set with free dofs"""
        # Find dofs affected by boundary conditions
        bc_dofs = []
        for bc in boundary_conditions:
            bc_dofs.extend(bc.get_boundary_values().keys()) 
        ownership_range = function_space.dofmap().ownership_range()
        interior_dofs = [x for x in range(ownership_range[0], ownership_range[1]) if x not in bc_dofs]    
        # Create petsc4py.PETSc.IS object with interior degrees of freedom
        index_set = petsc4py.PETSc.IS()
        index_set.createGeneral(interior_dofs)  
        return index_set

    def eigensolver_setup(self, prefix=None):
        E = SLEPc.EPS()
        E.create()

        E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        if hasattr(self, 'M'):
            E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        else:
            E.setProblemType(SLEPc.EPS.ProblemType.HEP)
        # if self.initial_guess:
            # E.setInitialSpace(self.initial_guess)
        E.setWhichEigenpairs(E.Which.TARGET_REAL)
        E.setTarget(-.1)
        st = E.getST()
        st.setType('sinvert')
        st.setShift(-1.e-3)
        if prefix:
            E.setOptionsPrefix(prefix)

        self.set_options(self.slepc_options)
        E.setFromOptions()
        # E.view()

        return E

    def solve(self, n_eig):
        E = self.E
        E.setDimensions(n_eig)
        # self.set_options(self.slepc_options)
        E.setFromOptions()
        E.solve()
        # print info
        its = E.getIterationNumber()
        eps_type = E.getType()
        self.nev, ncv, mpd = E.getDimensions()
        tol, maxit = E.getTolerances()
        self.nconv = E.getConverged()
        log(LogLevel.INFO, "Solution method: {:s}, stopping condition: tol={:.4g}, maxit={:d}".format(eps_type,tol, maxit))
        log(LogLevel.INFO, "Number of converged/requested eigenvalues with {:d} iterations  : {:d}/{:d}".format(its,self.nconv,self.nev))
        return self.nconv, its

    def set_options(self, slepc_options):
        # import pdb; pdb.set_trace()   
        log(LogLevel.INFO, "---- Setting additional slepc options for eigen solver -----")
        prefix = 'eigen_'
        for (parameter, value) in slepc_options.items():
            if value is not None:
                log(LogLevel.DEBUG, "DEBUG: setting {} {}".format(prefix + parameter, value))
                dolfin.PETScOptions.set(prefix + parameter, value)
            else:
                log(LogLevel.DEBUG, "DEBUG: setting {}".format(prefix + parameter))
                dolfin.PETScOptions.set(prefix + parameter)
        log(LogLevel.INFO, "------------------------------------------------------------")

        # self.E.setFromOptions()

    def get_eigenpair(self,i):
        u_r = dolfin.Function(self.V)
        u_im = dolfin.Function(self.V)
        v_r, v_i = self.K.createVecs()
        eig = self.E.getEigenpair(i, v_r, v_i)
        err = self.E.computeError(i)
        if self.index_set_not_bc:
            self.projector.scatter(vec_from=v_r, vec_to=u_r.vector().vec())
            self.projector.scatter(vec_from=v_i, vec_to=u_im.vector().vec())
            u_r.vector().vec().ghostUpdate()
            u_im.vector().vec().ghostUpdate()
        return eig, u_r, u_im, err

    def get_eigenvalues(self,n):
        eigenvalues = [] 
        for i in range(n):
            eig, u_r, u_im, err = self.get_eigenpair(i)
            eigenvalues.append([eig.real, err])
        return np.array(eigenvalues)

    def get_eigenpairs(self,n):
        eigenvalues = [] 
        eigenvectors_real = [] 
        eigenvectors_im = [] 
        for i in range(n):
            eig, u_r, u_im, err = self.get_eigenpair(i)
            eigenvalues.append(eig)
            eigenvectors_real.append(u_r)
            eigenvectors_im.append(u_im)
        return np.array(eigenvalues), [eigenvectors_real, eigenvectors_im]

class StabilitySolver(object):
    """solves second order stability problem"""
    def __init__(self, energy, state, bcs, rayleigh=None, Hessian = None,
        nullspace=None, parameters=None):
        OptDB = PETSc.Options()
        OptDB.view()

        self.i = 0
        # self.parameters = self.default_parameters()
        # if parameters is not None: self.parameters.update(parameters)                                                         # for debug purposes
        
        self.u = state['u']
        self.alpha = state['alpha']
        self._u = dolfin.Vector(self.u.vector())
        self._alpha = dolfin.Vector(self.alpha.vector())
        self.mesh = state['alpha'].function_space().mesh()
        self.meshsize = (self.mesh.hmax()+self.mesh.hmax())/2.
        # self.Z = z.function_space()
        # self.z = z
        self.Z = dolfin.FunctionSpace(self.mesh, 
            dolfin.MixedElement([self.u.ufl_element(),self.alpha.ufl_element()]))
        self.z = dolfin.Function(self.Z)
        self.dm = self.Z.dofmap()
        self.ownership = self.Z.dofmap().ownership_range()

        self.assigner = dolfin.FunctionAssigner(
            self.Z,            # receiving space
            [self.u.function_space(), self.alpha.function_space()]) # assigning spaces

        dx = dolfin.Measure("dx", metadata=parameters['compiler'], domain=self.mesh)
        self.y = dolfin.Function(self.Z)
        # self.dz = 
        # self.Identity = dot(self.y, dolfin.TestFunction(self.Z))*dx
        # self.minmode = dolfin.Function(self.Z)

        with open('../parameters/stability.yaml') as f:
            self.stability_parameters = yaml.load(f, Loader=yaml.FullLoader)['stability']
        with open('../parameters/eigensolver.yml') as f:
            self.inertia_parameters = yaml.load(f, Loader=yaml.FullLoader)['inertia']
        with open('../parameters/eigensolver.yml') as f:
            self.eigen_parameters = yaml.load(f, Loader=yaml.FullLoader)['eigen']

        if 'eigen' in parameters:
            self.eigen_parameters.update(parameters['eigen'])
        if 'inertia' in parameters:
            self.inertia_parameters.update(parameters['inertia'])
        if 'stability' in parameters:
            self.stability_parameters.update(parameters['stability'])


        # self.Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement([state[0].ufl_element(), state[1].ufl_element()]))
        # self.z = dolfin.Function(self.Z)

        self.z_old = dolfin.Function(self.Z)
        zeta = dolfin.TestFunction(self.Z)
        v, beta = dolfin.split(zeta)

        cdm = dolfin.project(dolfin.CellDiameter(self.mesh)**2., dolfin.FunctionSpace(self.mesh, 'CG', 1))
        self.cellarea = dolfin.Function(self.z.function_space())
        self.cellarea.assign(cdm)

        self.ownership = self.Z.dofmap().ownership_range()
        # import pdb; pdb.set_trace()   

        self.assigner = dolfin.FunctionAssigner(
            self.Z,            # receiving space
            [self.u.function_space(), self.alpha.function_space()]) # assigning space

        dim = self.u.function_space().ufl_element().value_size()
        self.u_zero = dolfin.project(dolfin.Constant(0.), self.u.function_space()) if dim==1 else dolfin.project(dolfin.Constant([0.]*dim), self.u.function_space())
        self.a_one  = dolfin.project(dolfin.Constant(1.), self.alpha.function_space())

        Zu = self.Z.extract_sub_space([0])
        Za = self.Z.extract_sub_space([1])

        self.Xa = Za.collapse().tabulate_dof_coordinates()
        self.Xu = Zu.collapse().tabulate_dof_coordinates()
        (_, self.mapa) = Za.collapse(collapsed_dofs = True)
        (_, self.mapu) = Zu.collapse(collapsed_dofs = True)

        self.computed = []
        self.provided = []

        self.stable = ''
        self.negev = -1


        self.Ealpha = derivative(energy, self.alpha, dolfin.TestFunction(self.alpha.ufl_function_space()))
        self.energy = energy

        (z_u, z_a) = dolfin.split(self.z)
        energy = ufl.replace(energy, {self.u: z_u, self.alpha: z_a})
        # energy = ufl.replace(energy, {state['u']: z_u, state['alpha']: z_a})
        self.J = derivative(energy, self.z, dolfin.TestFunction(self.Z))
        self.H = derivative(self.J, self.z, dolfin.TrialFunction(self.Z))

        self.nullspace = nullspace

        if rayleigh:
            rP = rayleigh[0]
            rN = rayleigh[1]
            # rN = ufl.replace(rN, {state['u']: z_u, state['alpha']: z_a})
            # rP = ufl.replace(rP, {state['u']: z_u, state['alpha']: z_a})

            self.rayleigh = {'rP': rP, 'rN': rN}
            self.rP = derivative(rP, self.z, dolfin.TrialFunction(self.Z))
            self.rN = derivative(rN, self.z, dolfin.TrialFunction(self.Z))

            self.rP = derivative(derivative(rP, self.z, dolfin.TestFunction(self.Z)),
                                                              self.z, dolfin.TrialFunction(self.Z))
            self.rN = derivative(derivative(rN, self.z, dolfin.TestFunction(self.Z)),
                                                              self.z, dolfin.TrialFunction(self.Z))
        if Hessian:
            self.Hessian =  Hessian
            # self.Hessian =  derivative(derivative(Hessian, self.z, dolfin.TestFunction(self.Z)),
                                                              # self.z, dolfin.TrialFunction(self.Z))

        self.ownership_range = self.Z.dofmap().ownership_range()
        if len(bcs)>0:
            self.bcs = bcs
            self.bc_dofs = self.get_bc_dofs(bcs)
        else:
            self.bcs = None
            self.bc_dofs = set()

        self.perturbation_v = dolfin.Function(self.Z.sub(0).collapse())
        self.perturbation_beta = dolfin.Function(self.Z.sub(1).collapse())

    def normalise_eigen(self, v, beta, mode='none'):
        if mode=='none':
            return
        elif mode=='one':
            coef = beta.vector().norm('l2')
        elif mode=='max':
            coef = max(abs(beta.vector()[:]))
        # import pdb; pdb.set_trace()   
        coeff_glob = np.array(0.0,'d')
        comm.Allreduce(coef, coeff_glob, op=mpi4py.MPI.MAX)

        log(LogLevel.DEBUG, 'Normalising eigenvector mode={}'.format(mode))
        real = np.all(np.isfinite(v.vector()[:]))
        log(LogLevel.DEBUG, '{}: v vector real {}'.format(rank, real))
        # log(LogLevel.DEBUG, '{}: v vector {}'.format(rank, v.vector()[:]))

        real = np.all(np.isfinite(beta.vector()[:]))
        log(LogLevel.DEBUG, '{}: beta vector real {}'.format(rank, real))

        log(LogLevel.DEBUG, '{}: nonzero coeff {}'.format(rank, coef!=0))
        log(LogLevel.DEBUG, '{}: coeff {}'.format(rank, coef))

        log(LogLevel.DEBUG, '{}: coeff_glob {}'.format(rank, coeff_glob))

        vval = v.vector()[:]/coeff_glob
        bval = beta.vector()[:]/coeff_glob

        v.vector().set_local(vval)
        beta.vector().set_local(bval)

        # vec = dolfin.PETScVector(MPI.comm_self)
        # beta.gather(vec, 1, "float")
        # v.gather(vec, np.array(range(self.Z.sub(0).dim()), "intc"))

        return coeff_glob

    def save_matrix(self, Mat, name):
        if name[-3:]=='txt': viewer = PETSc.Viewer().createASCII(name, 'w')
        else: viewer = PETSc.Viewer().createBinary(name, 'w')
        Mat.view(viewer)
        print('saved matrix in ', name)

    def get_bc_dofs(self, boundary_conditions):
        """
        Construct the blocked u-DOF's for the stability problem
        """
        debug = False
        # Construct homogeneous BCs
        bcs_Z = []
        zero = dolfin.Constant(0.0)

        if self.u.geometric_dimension()>1:
            # vector
            zeros = dolfin.Constant([0.0,]*self.u.geometric_dimension())
        elif self.u.geometric_dimension()==1:
            # scalar
            zeros = dolfin.Constant(0.)

        for bc in self.bcs['elastic']:
            if hasattr(bc, 'sub_domain'):
                new_bc = dolfin.DirichletBC(self.Z.sub(0), zeros, bc.sub_domain, bc.method())
            elif hasattr(bc, 'domain_args'):
                new_bc = dolfin.DirichletBC(self.Z.sub(0), zeros, bc.domain_args[0], bc.domain_args[1], bc.method())
            else:
                raise RuntimeError("Couldn't find where bcs for displacement are applied")

            bcs_Z.append(new_bc)
        for bc in self.bcs['damage']:
            if hasattr(bc, 'sub_domain'):
                new_bc = dolfin.DirichletBC(self.Z.sub(1), zero, bc.sub_domain, bc.method())
            elif hasattr(bc, 'domain_args'):
                new_bc = dolfin.DirichletBC(self.Z.sub(1), zero, bc.domain_args[0], bc.domain_args[1], bc.method())
            else:
                raise RuntimeError("Couldn't find where bcs for damage are applied")
            bcs_Z.append(new_bc)

        # Locate the DOF's corresponding to the BC
        bc_keys = [set(bc.get_boundary_values().keys()) for bc in bcs_Z]
        dofmap = self.Z.dofmap()
        bc_keys_glob = []
        for bc_key in bc_keys:
            bc_key_global = []
            for x in bc_key:
                bc_key_global.append(dofmap.local_to_global_index(x))
            bc_keys_glob.append(set(bc_key_global))
        if bc_keys_glob:
            self.bc_dofs  = reduce(lambda x, y: x.union(y), bc_keys_glob)
        else:
            self.bc_dofs  = set()

        self.bcs_Z = bcs_Z
        return self.bc_dofs

    def is_elastic(self):
        # tol = 1e-6
        # elastic_dofs = ((z.vector() - z_old.vector()) < tol)

        Ealpha = assemble(self.Ealpha)
        vec = dolfin.PETScVector(MPI.comm_self)
        Ealpha.gather(vec, np.array(range(self.Z.sub(1).dim()), "intc"))

        return np.all(vec[:]>0)

    def get_inactive_set(self):
        gtol = self.stability_parameters['inactiveset_gatol']
        ubtol = self.stability_parameters['inactiveset_ubtol']

        self.inactivemarker1 = dolfin.Function(self.alpha.function_space())
        self.inactivemarker2 = dolfin.Function(self.alpha.function_space())
        self.inactivemarker3 = dolfin.Function(self.alpha.function_space())
        self.inactivemarker4 = dolfin.Function(self.alpha.function_space())

        Ealpha = assemble(self.Ealpha)
        # tol2 = .95
        mask = Ealpha[:] < gtol
        mask2 = self.alpha.vector()[:] < 1.-ubtol
        mask3 = self.alpha.vector()[:] > self.alpha_old[:]

        # set operations: intersection : A & B \equiv A \cap B
        inactive_set_alpha = set(np.where(mask2 == True)[0]) & set(np.where(mask == True)[0])
        
        # inactive_set_alpha = set(np.where(mask3 == True)[0])

        # inactive_set_alpha = set(np.where(mask == True)[0])
        log(LogLevel.INFO, 'Ealpha norm {}'.format(Ealpha.norm('l2')))
        log(LogLevel.INFO, 'Len Ealpha {}'.format(len(Ealpha[:])))
        log(LogLevel.INFO, 'Inactive set gradient tolerance {}'.format(self.stability_parameters['inactiveset_gatol']))
        log(LogLevel.INFO, 'Inactive set upper bound tolerance {}'.format(self.stability_parameters['inactiveset_ubtol']))
        log(LogLevel.INFO, 'Inactive set Ealpha #nodes {}'.format(len(set(np.where(mask == True)[0]))))
        log(LogLevel.INFO, 'Inactive set ub tol #nodes {}'.format(len(set(np.where(mask2 == True)[0]))))
        log(LogLevel.INFO, 'Inactive set a-a0 #nodes {}'.format(len(set(np.where(mask3 == True)[0]))))
        log(LogLevel.INFO, 'Inactive set Ea cap a<ub #nodes {}'.format(len(set(np.where(mask2 == True)[0]) & set(np.where(mask == True)[0]))))
        # local operation
        self.inactivemarker1.vector()[np.where(mask == True)[0]] = 1.
        self.inactivemarker2.vector()[np.where(mask2 == True)[0]] = 1.
        self.inactivemarker3.vector()[np.where(mask3 == True)[0]] = 1.
        self.inactivemarker4.vector()[list(inactive_set_alpha)] = 1.
        self.inactivemarker1.vector().vec().ghostUpdate()
        self.inactivemarker2.vector().vec().ghostUpdate()
        self.inactivemarker3.vector().vec().ghostUpdate()
        self.inactivemarker4.vector().vec().ghostUpdate()
        self.inactiveEalpha = self.Ealpha
        # import pdb; pdb.set_trace()   

        # from local subspace to local mixed space numbering
        local_inactive_set_alpha = [self.mapa[k] for k in inactive_set_alpha]
        # from local mixed space to global numbering
        global_set_alpha = [self.dm.local_to_global_index(k) for k in local_inactive_set_alpha]
        inactive_set = set(global_set_alpha) | set(self.Z.sub(0).dofmap().dofs())

        # log(LogLevel.CRITICAL, '{}: global_set_alpha {}'.format(rank, sorted(global_set_alpha)))
        # log(LogLevel.CRITICAL, '{}: len global_set_alpha {}'.format(rank, len(global_set_alpha)))
        # log(LogLevel.CRITICAL, '{}: local_inactive_set_alpha {}'.format(rank, sorted(local_inactive_set_alpha)))
        # log(LogLevel.DEBUG,    '{}: len local_inactive_set_alpha {}'.format(rank, len(local_inactive_set_alpha)))

        return inactive_set

    def reduce_Hessian(self, Hessian=None, restricted_dofs_is=None):
        if not Hessian: H = dolfin.as_backend_type(assemble(self.H)).mat()
        else: H = dolfin.as_backend_type(assemble(Hessian)).mat()
        if restricted_dofs_is is not None:
            try:
                H_reduced = H.createSubMatrix(restricted_dofs_is, restricted_dofs_is)
            except:
                H_reduced = H.getSubMatrix(restricted_dofs_is, restricted_dofs_is)
        return H_reduced

    def inertia_setup(self):
        self.pc = PETSc.PC().create(MPI.comm_world)
        prefix = "inertia_"
        if prefix:
            self.pc.setOptionsPrefix(prefix)
        self.pc.setFromOptions()

        for parameter, value in self.inertia_parameters.items():
            dolfin.PETScOptions.set(parameter, value)
            log(LogLevel.DEBUG, 'Setting up inertia solver: {}: {}'.format(prefix+parameter, value))

        dolfin.PETScOptions.set("inertia_ksp_type", "preonly")
        dolfin.PETScOptions.set("inertia_pc_type", "cholesky")
        dolfin.PETScOptions.set("inertia_pc_factor_mat_solver_type", "mumps")
        dolfin.PETScOptions.set("inertia_mat_mumps_icntl_24", 1)
        dolfin.PETScOptions.set("inertia_mat_mumps_icntl_13", 1)
        # dolfin.PETScOptions.set("inertia_eps_monitor", 1)

        self.pc.setFromOptions()
        # self.pc.view()

    def get_inertia(self, Mat = None, restricted_dof_is=None):
        if Mat == None:
            H = dolfin.as_backend_type(assemble(self.H)).mat()
        else:
            H = Mat
        if restricted_dof_is is not None:
            try:
                H = H.createSubMatrix(restricted_dof_is, restricted_dof_is)
            except:
                H = H.getSubMatrix(restricted_dof_is, restricted_dof_is)
        self.pc.setOperators(H)
        self.pc.setUp()
        # self.pc.view()
        # import pdb; pdb.set_trace()   
        Fm = self.pc.getFactorMatrix()
        # import pdb; pdb.set_trace()
        # myviewer = PETSc.Viewer().createASCII("test.txt", mode=PETSc.Viewer.Format.ASCII_COMMON,comm= PETSc.COMM_WORLD)
        (neg, zero, pos) = Fm.getInertia()
        log(LogLevel.INFO, "#Eigenvalues of E'': (%s [neg], %s [zero], %s [pos])" % (neg, zero, pos))
        if neg:
            self.stable = False
        else:
            self.stable = True
        return neg

    def is_compatible(self, bcs, v, homogeneous = False, tol=dolfin.DOLFIN_EPS_LARGE):
        V = v.function_space()
        v_array = v.vector()[:]
        ret = True
        _bc = dolfin.DirichletBC(bcs)
        # print(_bc.id())
        # print(bcs.id())
        # homogenize is in place
        # bcdofs = list(bcs.get_boundary_values().keys())
        # print(rank, bcdofs)
        _bc.homogenize()
        _bc.apply(v.vector())
        # import pdb; pdb.set_trace()
        # print(rank, v_array[:],v.vector()[:])
        ret = np.all(np.isclose(v_array[:],v.vector()[:]))
        return ret

    def solve(self, alpha_old):
        # debug = False
        self.alpha_old = alpha_old
        # postfix = 'seq' if size == 1 else 'mpi'
        self.H_norm = assemble(self.H).norm('frobenius')
        self.assigner.assign(self.z, [self.u, self.alpha])

        locnumbcs = np.array(len(self.bc_dofs))

        log(LogLevel.DEBUG, 'rank, {} : {}'.format(rank, locnumbcs))
        numbcs = np.array(0.,'d')
        comm.Reduce(locnumbcs, numbcs, op=mpi4py.MPI.SUM, root=0)

        if rank == 0:
            log(LogLevel.INFO, '#bc dofs = {}'.format(int(numbcs)))

        if self.is_elastic():
            log(LogLevel.INFO, 'Current state: elastic')
            self.stable = True
            self.negev = np.nan
            # return self.stable, 0
        else:
            log(LogLevel.INFO, 'Current state: inelastic')

        inactive_dofs = self.get_inactive_set()
        free_dofs = list(sorted(inactive_dofs - self.bc_dofs))

        index_set = petsc4py.PETSc.IS()
        index_set.createGeneral(free_dofs)

        if hasattr(self, 'rP') and hasattr(self, 'rN'):
            self.H2 = self.rP - self.rN

        if hasattr(self, 'Hessian'):
            log(LogLevel.INFO, 'Inertia: Using user-provided Hessian')
            self.H_reduced = self.reduce_Hessian(self.Hessian, restricted_dofs_is = index_set)
            log(LogLevel.INFO, 'H norm (provided) {}'.format(assemble(self.Hessian).norm('frobenius')))
            # import pdb; pdb.set_trace()

            log(LogLevel.INFO, 'H reduced norm (provided) {}'.format(self.H_reduced.norm(2)))
            self.Hessian_norm = assemble(self.Hessian).norm('frobenius')
        else:
            log(LogLevel.INFO, 'Inertnia: Using computed Hessian')
            self.H_reduced = self.reduce_Hessian(self.H, restricted_dofs_is = index_set)

        log(LogLevel.INFO, 'H norm (computed) {}'.format(assemble(self.H).norm('frobenius')))
        log(LogLevel.INFO, 'H reduced norm (computed) {}'.format(self.H_reduced.norm(2)))
        # typedef enum {NORM_1=0,NORM_2=1,NORM_FROBENIUS=2,NORM_INFINITY=3,NORM_1_AND_2=4} NormType;

        # self.Id_reduced = self.reduce_Hessian(self.Identity, restricted_dofs_is = index_set)

        stability_data_i = {
            'Hessian norm': assemble(self.H).norm('frobenius'),
            'Hessian_reduced norm': self.H_reduced.norm(2)
        }
        # typedef enum {NORM_1=0,NORM_2=1,NORM_FROBENIUS=2,NORM_INFINITY=3,NORM_1_AND_2=4} NormType;

        self.inertia_setup()

        # self.save_matrix(self.H_reduced, 'H-red-{}'.format(size))
        # self.save_matrix(self.H_reduced, 'H-red-{}.txt'.format(size))
        negev = self.get_inertia(self.H_reduced)
        # import pdb; pdb.set_trace()


        # if negev > 0:
            # import pdb; pdb.set_trace()
        if True:
            eigs = []
        # solve full eigenvalue problem

            # self.assigner.assign(self.y, [self.u, self.alpha])
            # assemble(derivative(self.Identity, self.y, TrialFunction(self.Z)))
            # Identity = assemble(derivative(self.Identity, self.y, dolfin.TrialFunction(self.Z)))

            eigen_tol = self.eigen_parameters['eig_rtol']


            if hasattr(self, 'H2'):
                log(LogLevel.INFO, 'Full eigenvalue: Using user-provided Rayleigh quotient')
                log(LogLevel.INFO, 'Norm provided {}'.format(assemble(self.H2).norm('frobenius')))
                eigen = EigenSolver(self.H2, self.z, restricted_dofs_is = index_set, slepc_options=self.eigen_parameters)

            elif hasattr(self, 'Hessian'):
                log(LogLevel.INFO, 'Full eigenvalue: Using user-provided Hessian')
                log(LogLevel.INFO, 'Norm provided {}'.format(assemble(self.Hessian).norm('frobenius')))
                eigen = EigenSolver(self.Hessian, self.z, restricted_dofs_is = index_set, slepc_options=self.eigen_parameters)

            else:
                log(LogLevel.INFO, 'Full eigenvalue: Using computed Hessian')
                # log(LogLevel.INFO, '{}'.format(self.eigen_parameters))
                if hasattr(self, 'minmode'):
                    log(LogLevel.INFO, 'init eigensolver')
                    eigen = EigenSolver(self.H, self.z, restricted_dofs_is = index_set, 
                        slepc_options=self.eigen_parameters,
                        initial_guess = self.minmode.vector().vec())
                else:
                    eigen = EigenSolver(self.H, self.z, restricted_dofs_is = index_set, 
                        slepc_options=self.eigen_parameters)
                # log(LogLevel.INFO, 'Full eigenvalue: Using computed Hessian and L2 identity')
                # eigen = EigenSolver(self.H, self.z, a_m=dolfin.as_backend_type(Identity).mat(), restricted_dofs_is = index_set, slepc_options=self.eigen_parameters)
            log(LogLevel.INFO, 'Norm computed {}'.format(assemble(self.H).norm('frobenius')))

            self.computed.append(assemble(self.H).norm('frobenius'))

            if hasattr(self, 'H2'): self.provided.append(assemble(self.H2).norm('frobenius'))

            maxmodes = self.stability_parameters['maxmodes']

            tol, maxit = eigen.E.getTolerances()
            log(LogLevel.INFO, "Eigensolver stopping condition: tol={:.4g}, maxit={:d}".format(tol, maxit))
            # import pdb; pdb.set_trace()

            nconv, it = eigen.solve(min(maxmodes, negev+1))

            if nconv == 0:
                log(LogLevel.WARNING, 'Eigensolver did not converge')
                self.stable = negev <= 0
                self.eigs = []
                self.mineig = np.nan
                return (self.stable, int(negev))

            eigs = eigen.get_eigenvalues(nconv)
            negconv = sum(eigs[:,0]<0)
            zeroconv = sum(abs(eigs[:,0])<3*float(self.eigen_parameters['eig_rtol']))
            # sanity check
            if nconv and negconv != negev:
                log(LogLevel.WARNING, 'Eigen solver found {} negative evs '.format(negconv))
            if nconv == 0 and negev>0:
                log(LogLevel.WARNING, 'Full eigensolver did not converge but inertia yields {} neg eigen'.format(negev))
                return
            log(LogLevel.WARNING, 'Eigen solver found {} (approx.) zero evs '.format(zeroconv))

            # eigen.save_eigenvectors(nconv)

            if nconv > 0:
                log(LogLevel.INFO, '')
                log(LogLevel.INFO, "i        k      err     ")
                log(LogLevel.INFO, "---------------------------")
                for (i, k) in enumerate(eigs):
                    log(LogLevel.INFO,  "%d %12e %12e" %(i, k[0], k[1]) )
                log(LogLevel.INFO, '')

            linsearch = []

            if negconv > 0:
                for n in range(negconv) if negconv < maxmodes else range(maxmodes):
                    log(LogLevel.INFO, 'Processing perturbation mode (neg eigenv) {}/{}'.format(n,maxmodes))
                    eig, u_r, u_im, err = eigen.get_eigenpair(n)
                    err2 = eigen.E.computeError(0, SLEPc.EPS.ErrorType.ABSOLUTE)
                    # import pdb; pdb.set_trace()
                    _z = dolfin.Vector(u_r.vector())
                    _H = assemble(self.H)
                    _J = assemble(self.J)
                    _lmbda = (_H*_z).inner(_z)/_z.norm('l2')
                    _j0 = _J.inner(_z)
                    v_n, beta_n = u_r.split(deepcopy=True)
                    log(LogLevel.INFO, 'Verify H*z z/||z|| = {:.5e}'.format(_lmbda))
                    log(LogLevel.INFO, 'Verify J*z = {:.5e}'.format(_j0))
                    # print(rank, [self.is_compatible(bc, u_r, homogeneous = True) for bc in self.bcs_Z])
                    norm_coeff = self.normalise_eigen(v_n, beta_n, mode='max')

                    log(LogLevel.DEBUG, '||vn||_l2 = {}'.format(dolfin.norm(v_n, 'l2')))
                    log(LogLevel.DEBUG, '||βn||_l2 = {}'.format(dolfin.norm(beta_n, 'l2')))
                    log(LogLevel.DEBUG, '||vn||_h1 = {}'.format(dolfin.norm(v_n, 'h1')))
                    log(LogLevel.DEBUG, '||βn||_h1 = {}'.format(dolfin.norm(beta_n, 'h1')))
                    # eig, u_r, u_im, err = eigen.get_eigenpair(n)

                    # order = self.stability_parameters['order']

                    linsearch.append({'n': n, 'lambda_n': eig.real,
                        'v_n': v_n, 'beta_n': beta_n, 'norm_coeff': norm_coeff})

            eig, u_r, u_im, err = eigen.get_eigenpair(0)
            self.eigs = eigs[:,0]
            self.mineig = eig.real
            self.minmode = u_r
            # self.stable = negev <= 0  # based on inertia

            self.negev = negev  # based on inertia

            modes = negconv if negconv < maxmodes else maxmodes
            # if eigs[0,0] < float(self.eigen_parameters['eig_rtol']):
            # if eigs[0,0] < 0:
            if eigs[0,0] < dolfin.DOLFIN_EPS:
                self.perturbation_v = linsearch[0]['v_n']
                self.perturbation_beta = linsearch[0]['beta_n']
                self.perturbations_v = [linsearch[n]['v_n'] for n in range(modes)]
                self.perturbations_beta = [linsearch[n]['beta_n'] for n in range(modes)]
                # self.hstar = linsearch[0]['hstar']
                # self.en_diff = linsearch[0]['en_diff']
                self.eigendata = linsearch

            self.i +=1
            log(LogLevel.INFO, '________________________ STABILITY _________________________')
            log(LogLevel.INFO, 'Negative eigenvalues (based on inertia) {}'.format(negev))
            log(LogLevel.INFO, 'Stable (counting neg. eigs) {}'.format(not (negev > 0)))
            log(LogLevel.INFO, 'Stable (Computing min. ev) {}'.format(eig.real > float(self.eigen_parameters['eig_rtol'])))
            log(LogLevel.INFO, 'Min eig {:.5e}'.format(self.mineig))
        
            self.stable = eig.real > float(self.eigen_parameters['eig_rtol'])  # based on eigenvalue

        return (self.stable, int(negev))

