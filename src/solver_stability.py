        # set operations: intersection : A & B \equiv A \cap B

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

        """
        Solver object for constrained eigenvalue pb of the type:
            - K z y = \\lmbda <z, y>, forall y \\in V_h(\\Omega')
            - K z y = \\lmbda <z, y>, forall y \\in {inactive constraints}

            where \\Omega' \\subseteq \\Omega
            where {inactive constraints} is a proper subset
        """
        self.comm = comm


        self.slepc_options = slepc_options
        self.V = u.function_space()
        self.u = u
        self.index_set = None

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

        if self.bcs:
            self.index_set = self.get_interior_index_set(self.bcs, self.V)

        if restricted_dofs_is:
            self.index_set = restricted_dofs_is

        self.K, self.M = self.restrictOperator(self.index_set)
        self.projector = self.createProjector(self.index_set)

        if slepc_eigensolver:
            self.E = slepc_eigensolver
        else:
            self.E = self.eigensolver_setup(prefix=option_prefix)

        if a_m:
            self.E.setOperators(self.K, self.M)
        else:
            self.E.setOperators(self.K)

    def restrictOperator(self, indexSet):
        if indexSet is None:
            return (self.K, self.M)
        try:
            K = self.K.createSubMatrix(indexSet, indexSet)
            # if a_m:
                # M = self.M.createSubMatrix(indexSet, indexSet)
        except:
            K = self.K.getSubMatrix(indexSet, indexSet)
            # if a_m:
                # M = self.M.getSubMatrix(indexSet, indexSet)

        if hasattr(self, 'M'):
            try:
                M = self.M.createSubMatrix(indexSet, indexSet)
            except:
                M = self.M.getSubMatrix(indexSet, indexSet)
        else:
            M = None

        return (K, M)

    def createProjector(self, indexSet):
        projector = petsc4py.PETSc.Scatter()
        projector.create(
            vec_from=self.K.createVecRight(),
            is_from=None,
            vec_to=self.u.vector().vec(),
            is_to=indexSet
            )

        return projector

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

        return E

    def solve(self, n_eig):
        E = self.E
        E.setDimensions(n_eig)
        E.setFromOptions()
        E.solve()
        its = E.getIterationNumber()
        eps_type = E.getType()
        self.nev, ncv, mpd = E.getDimensions()
        tol, maxit = E.getTolerances()
        self.nconv = E.getConverged()
        log(LogLevel.INFO, "Solution method: {:s}, stopping condition: tol={:.4g}, maxit={:d}".format(eps_type,tol, maxit))
        log(LogLevel.INFO, "Number of converged/requested eigenvalues with {:d} iterations  : {:d}/{:d}".format(its,self.nconv,self.nev))

        return self.nconv, its

    def set_options(self, slepc_options):
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

    def get_eigenpair(self,i):
        u_r = dolfin.Function(self.V)
        u_im = dolfin.Function(self.V)
        v_r, v_i = self.K.createVecs()
        eig = self.E.getEigenpair(i, v_r, v_i)
        err = self.E.computeError(i)
        if self.index_set:
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
    def __init__(self, energy, state, bcs,
        # rayleigh=None,
        Hessian = None,
        nullspace=None, parameters=None):
        """Solves second order stability problem
            - computes inertia
            - solves full eigenvalue problem

            * Parameters:
                - energy (form)
                - state (tuple)
                - bcs (list)

            Optional arguments: rayleigh (form), Hessian (form), 
            nullspace, parameters
        """
        OptDB = PETSc.Options()
        OptDB.view()

        self.i = 0

        self.u = state['u']
        self.alpha = state['alpha']
        self._u = dolfin.Vector(self.u.vector())
        self._alpha = dolfin.Vector(self.alpha.vector())
        self.mesh = state['alpha'].function_space().mesh()

        self.Z = dolfin.FunctionSpace(self.mesh, 
            dolfin.MixedElement([self.u.ufl_element(),self.alpha.ufl_element()]))
        self.z = dolfin.Function(self.Z)
        self.z_old = dolfin.Function(self.Z)
        zeta = dolfin.TestFunction(self.Z)
        v, beta = dolfin.split(zeta)

        self.dm = self.Z.dofmap()
        self.ownership = self.Z.dofmap().ownership_range()

        Zu = self.Z.extract_sub_space([0])
        Za = self.Z.extract_sub_space([1])

        self.Xa = Za.collapse().tabulate_dof_coordinates()
        self.Xu = Zu.collapse().tabulate_dof_coordinates()

        (_, self.mapa) = Za.collapse(collapsed_dofs = True)
        (_, self.mapu) = Zu.collapse(collapsed_dofs = True)

        self.assigner = dolfin.FunctionAssigner(
            self.Z,            # receiving space
            [self.u.function_space(), self.alpha.function_space()]) # assigning spaces

        self.parameters = self.setParameters(parameters)

        self.ownership = self.Z.dofmap().ownership_range()

        self.assigner = dolfin.FunctionAssigner(
            self.Z,            # receiving space
            [self.u.function_space(), self.alpha.function_space()]) # assigning space

        dim = self.u.function_space().ufl_element().value_size()

        self.stable = ''
        self.negev = -1

        self.Ealpha = derivative(energy, self.alpha, dolfin.TestFunction(self.alpha.ufl_function_space()))
        self.energy = energy

        (z_u, z_a) = dolfin.split(self.z)
        energy = ufl.replace(energy, {self.u: z_u, self.alpha: z_a})
        self.J = derivative(energy, self.z, dolfin.TestFunction(self.Z))
        self.H = derivative(self.J, self.z, dolfin.TrialFunction(self.Z))

        self.nullspace = nullspace

        if Hessian:
            self.Hessian =  Hessian

        self.ownership_range = self.Z.dofmap().ownership_range()
        if len(bcs)>0:
            self.bcs = bcs
            self.bc_dofs = self.get_bc_dofs(bcs)
        else:
            self.bcs = None
            self.bc_dofs = set()

        self.perturbation_v = dolfin.Function(self.Z.sub(0).collapse())
        self.perturbation_beta = dolfin.Function(self.Z.sub(1).collapse())

        self._Hessian = Hessian if Hessian.__class__ == ufl.form.Form else self.H

    def setParameters(self, parameters):

        with open('../parameters/stability.yaml') as f:
            stability_parameters = yaml.load(f, Loader=yaml.FullLoader)['stability']
        with open('../parameters/eigensolver.yml') as f:
            inertia_parameters = yaml.load(f, Loader=yaml.FullLoader)['inertia']
        with open('../parameters/eigensolver.yml') as f:
            eigen_parameters = yaml.load(f, Loader=yaml.FullLoader)['eigen']

        if 'eigen' in parameters:
            eigen_parameters.update(parameters['eigen'])
        if 'inertia' in parameters:
            inertia_parameters.update(parameters['inertia'])
        if 'stability' in parameters:
            stability_parameters.update(parameters['stability'])

        return {'eigen': eigen_parameters,
                'inertia': inertia_parameters,
                'stability': stability_parameters}


    def normalise_eigen(self, v, beta, mode='none'):
        if mode=='none':
            return
        elif mode=='one':
            coef = beta.vector().norm('l2')
        elif mode=='max':
            coef = max(abs(beta.vector()[:]))

        coeff_glob = np.array(0.0,'d')
        comm.Allreduce(coef, coeff_glob, op=mpi4py.MPI.MAX)

        log(LogLevel.DEBUG, 'Normalising eigenvector mode={}'.format(mode))
        real = np.all(np.isfinite(v.vector()[:]))
        log(LogLevel.DEBUG, '{}: v vector real {}'.format(rank, real))

        real = np.all(np.isfinite(beta.vector()[:]))

        log(LogLevel.DEBUG, '{}: beta vector real {}'.format(rank, real))
        log(LogLevel.DEBUG, '{}: coeff {}'.format(rank, coef))
        log(LogLevel.DEBUG, '{}: coeff_glob {}'.format(rank, coeff_glob))

        vval = v.vector()[:]/coeff_glob
        bval = beta.vector()[:]/coeff_glob

        v.vector().set_local(vval)
        beta.vector().set_local(bval)

        v.vector().vec().ghostUpdate()
        beta.vector().vec().ghostUpdate()

        return coeff_glob

    def save_matrix(self, Mat, name):
        if name[-3:]=='txt': viewer = PETSc.Viewer().createASCII(name, 'w')
        else: viewer = PETSc.Viewer().createBinary(name, 'w')
        Mat.view(viewer)
        print('saved matrix in ', name)

    def get_bc_dofs(self, boundary_conditions):
        """
        Construct the blocked u-DOF's for the stability problem
        Construct homogeneous BCs
        Locating DOFs corresponding to BCs
        """
        debug = False
        bcs_Z = []
        zero = dolfin.Constant(0.0)

        if self.u.geometric_dimension()>1:
            zeros = dolfin.Constant([0.0,]*self.u.geometric_dimension())
        elif self.u.geometric_dimension()==1:
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

        Ealpha = assemble(self.Ealpha)
        vec = dolfin.PETScVector(MPI.comm_self)
        Ealpha.gather(vec, np.array(range(self.Z.sub(1).dim()), "intc"))

        return np.all(vec[:]>0)

    def getInactiveSet(self):
        gtol = self.parameters['stability']['inactiveset_gatol']
        ubtol = self.parameters['stability']['inactiveset_ubtol']

        self.inactivemarker4 = dolfin.Function(self.alpha.function_space())

        Ealpha = assemble(self.Ealpha)
        mask = Ealpha[:] < gtol
        mask2 = self.alpha.vector()[:] < 1.-ubtol

        inactive_set_alpha = set(np.where(mask2 == True)[0]) & set(np.where(mask == True)[0])

        log(LogLevel.DEBUG, 'DEBUG: E\'(z)(0|β) norm {}'.format(Ealpha.norm('l2')))
        log(LogLevel.DEBUG, 'DEBUG: Inactive set E\'(z)(0|β) #nodes {}'.format(len(set(np.where(mask == True)[0]))))
        log(LogLevel.DEBUG, 'DEBUG: Inactive set ub tol #nodes {}'.format(len(set(np.where(mask2 == True)[0]))))
        log(LogLevel.DEBUG, 'DEBUG: Inactive set Ea cap a<ub #nodes {}'.format(len(set(np.where(mask2 == True)[0]) & set(np.where(mask == True)[0]))))

        self.inactivemarker4.vector()[list(inactive_set_alpha)] = 1.
        self.inactivemarker4.vector().vec().ghostUpdate()

        local_inactive_set_alpha = [self.mapa[k] for k in inactive_set_alpha]
        global_set_alpha = [self.dm.local_to_global_index(k) for k in local_inactive_set_alpha]
        inactive_set = set(global_set_alpha) | set(self.Z.sub(0).dofmap().dofs())

        return inactive_set

    def reduce_Hessian(self, Hessian=None, restricted_dofs_is=None):
        if isinstance(Hessian, ufl.form.Form):
            H = dolfin.as_backend_type(assemble(Hessian)).mat()
        elif isinstance(Hessian, petsc4py.PETSc.Mat):
            H = Hessian

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

        for parameter, value in self.parameters['inertia'].items():
            dolfin.PETScOptions.set(parameter, value)
            log(LogLevel.DEBUG, 'Setting up inertia solver: {}: {}'.format(prefix+parameter, value))

        dolfin.PETScOptions.set("inertia_ksp_type", "preonly")
        dolfin.PETScOptions.set("inertia_pc_type", "cholesky")
        dolfin.PETScOptions.set("inertia_pc_factor_mat_solver_type", "mumps")
        dolfin.PETScOptions.set("inertia_mat_mumps_icntl_24", 1)
        dolfin.PETScOptions.set("inertia_mat_mumps_icntl_13", 1)

        self.pc.setFromOptions()

    def get_inertia(self, Mat = None, restricted_dofs_is=None):

        if Mat == None:
            H = dolfin.as_backend_type(assemble(self.H)).mat()
        elif isinstance(self.H, ufl.form.Form):
            H = dolfin.as_backend_type(assemble(Mat)).mat()
        elif isinstance(Mat, petsc4py.PETSc.Mat):
            H = Mat

        if restricted_dofs_is is not None:
            H = self.reduce_Hessian(Mat, restricted_dofs_is)

        self.pc.setOperators(H)
        self.pc.setUp()

        Fm = self.pc.getFactorMatrix()
        (neg, zero, pos) = Fm.getInertia()

        log(LogLevel.INFO, "#Eigenvalues of E'': (%s [neg], %s [zero], %s [pos])" % (neg, zero, pos))

        return neg

    def getInitialGuess(self):
        if hasattr(self, 'minmode'):
            return self.minmode.vector().vec()
        return None

    def stabilityLog(self):

        log(LogLevel.INFO, 'H norm {}'.format(assemble(self.H).norm('frobenius')))
        log(LogLevel.INFO, 'H reduced norm {}'.format(self.H_reduced.norm(2)))
        log(LogLevel.INFO, "Eigensolver stopping condition: tol={:.4g}, maxit={:d}".format(tol, maxit))

        log(LogLevel.INFO, '________________________ STABILITY _________________________')
        log(LogLevel.INFO, 'Negative eigenvalues (based on inertia) {}'.format(negev))
        log(LogLevel.INFO, 'Stable (counting neg. eigs) {}'.format(not (negev > 0)))
        log(LogLevel.INFO, 'Stable (Computing min. ev) {}'.format(eig.real > float(self.parameters['eigen']['eig_rtol'])))
        log(LogLevel.INFO, 'Min eig {:.5e}'.format(self.mineig))

    def getFreeDofsIS(self, inactiveDofs):
        free_dofs = list(sorted(inactiveDofs - self.bc_dofs))

        index_set = petsc4py.PETSc.IS()
        index_set.createGeneral(free_dofs)

        return index_set

    def getNonPositiveCount(self, eigs):
        """
        Get the count of nonpositive eigenvalues bases on solution of full eigenproblem
        """
        negconv = sum(eigs[:,0]<0)
        zeroconv = sum(abs(eigs[:,0]) < 3*float(self.parameters['eigen']['eig_rtol']))

        return (negconv, zeroconv)

    def postprocEigs(self, eigs, eigen):
        nconv = len(eigs)

        if nconv == 0:
            log(LogLevel.WARNING, 'Eigensolver did not converge')
            self.stable = negev <= 0
            self.eigs = []
            self.mineig = np.nan
            # return (self.stable, int(negev))


        if nconv > 0:
            log(LogLevel.INFO, '')
            log(LogLevel.INFO, "i        k      err     ")
            log(LogLevel.INFO, "---------------------------")
            for (i, k) in enumerate(eigs):
                log(LogLevel.INFO,  "%d %12e %12e" %(i, k[0], k[1]) )
            log(LogLevel.INFO, '')

            _, self.minmode, _, _  = eigen.get_eigenpair(0)

        linsearch = []

        negconv, zeroconv = self.getNonPositiveCount(eigs)

        numModes = min(negconv, self.parameters['stability']['maxmodes'])

        if negconv > 0:
            for n in range(numModes):
                log(LogLevel.INFO, 'Processing perturbation mode (neg eigenv) {}/{}'.format(n,self.parameters['stability']['maxmodes']))
                eig, u_r, u_im, err = eigen.get_eigenpair(n)
                v_n, beta_n = u_r.split(deepcopy=True)
                norm_coeff = self.normalise_eigen(v_n, beta_n, mode='max')
                linsearch.append({'n': n, 'lambda_n': eig.real,
                    'v_n': v_n, 'beta_n': beta_n, 'norm_coeff': norm_coeff})

        return linsearch

    def sanityCheck(self, negev, eigs):
        """
        Sanity check comparing output from inertia (negev) and full eigenvalue problem (eigs)
        """
        nconv = len(eigs)
        negconv, zeroconv = self.getNonPositiveCount(eigs)

        if nconv and negconv != negev:
            log(LogLevel.WARNING, 'Eigen solver found {} negative evs '.format(negconv))
        if nconv == 0 and negev>0:
            log(LogLevel.WARNING, 'Full eigensolver did not converge but inertia yields {} neg eigen'.format(negev))
            return
        log(LogLevel.WARNING, 'Eigen solver found {} (approx.) zero evs '.format(zeroconv))

        pass

    def compileData(self, stabilityData, eigs):

        self.eigs = eigs[:,0]
        self.mineig = eigs[:,0][0].real
        # self.negev = negev  # based on inertia


        if eigs[0,0] < dolfin.DOLFIN_EPS:
            # self.minmode = stabilityData[0]['beta_n']
            self.perturbation_v = stabilityData[0]['v_n']
            self.perturbation_beta = stabilityData[0]['beta_n']
            self.perturbations_v = [mode['v_n'] for mode in stabilityData]
            self.perturbations_beta = [mode['beta_n'] for mode in stabilityData]

    def solve(self, alpha_old):
        """
        Solves second order problem returning stability flag and number of negative
        eigenvalues.
        """

        self.alpha_old = alpha_old
        self.assigner.assign(self.z, [self.u, self.alpha])

        if self.is_elastic():
            log(LogLevel.INFO, 'Current state: elastic')
        else:
            log(LogLevel.INFO, 'Current state: inelastic')

        inactive_dofs = self.getInactiveSet()
        index_set = self.getFreeDofsIS(inactive_dofs)

        self.inertia_setup()

        negev = self.get_inertia(
            self._Hessian,
            restricted_dofs_is=index_set)

        eigen = EigenSolver(self.H, self.z,
            restricted_dofs_is=index_set,
            slepc_options=self.parameters['eigen'],
            initial_guess=self.getInitialGuess())

        nconv, it = eigen.solve(min(self.parameters['stability']['maxmodes'], negev+1))

        eigs = eigen.get_eigenvalues(nconv)
        self.sanityCheck(negev, eigs)

        _stabilityData = self.postprocEigs(eigs, eigen)
        self.compileData(_stabilityData, eigs)

        self.i += 1

        self.stable = eigs[0, 0].real > float(self.parameters['eigen']['eig_rtol'])  # based on eigenvalue

        if self.is_elastic():
            self.stable = True

        return (self.stable, int(negev))

