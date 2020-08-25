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

import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class EigenSolver(object):
    def __init__(self,
                 a_k,
                 u,
                 a_m=None,                                   # optional, for eigpb of the type (K-lambda M)x=0
                 bcs=None,
                 restricted_dofs_is=None,
                 slepc_options={'eps_max_it':100},
                 option_prefix=None,
                 comm=MPI.comm_world, 
                 slepc_eigensolver = None
                ):
        self.comm = comm
        self.slepc_options = slepc_options
        if option_prefix:
            self.E.setOptionsPrefix(option_prefix)
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
            self.K = dolfin.as_backend_type(dolfin.assemble(a_k)).mat()
        elif type(a_k) == petsc4py.PETSc.Mat:
            # an assembled petsc matrix
            self.K = a_k

        if a_m is not None and type(a_m) == ufl.form.Form:
            self.M = dolfin.as_backend_type(dolfin.assemble(a_m)).mat()
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

        # set up the eigensolver
        if slepc_eigensolver:
            self.E = slepc_eigensolver
        else:
            self.E = self.eigensolver_setup()

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

    def eigensolver_setup(self):
        E = SLEPc.EPS()
        E.create()
        E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        E.setProblemType(SLEPc.EPS.ProblemType.HEP)
        E.setWhichEigenpairs(E.Which.TARGET_REAL)
        E.setTarget(-.1) 
        st = E.getST()
        st.setType('sinvert')
        st.setShift(-1.e-3)
        return E

    def solve(self, n_eig):
        E = self.E
        E.setDimensions(n_eig)
        self.set_options(self.slepc_options)
        E.setFromOptions()
        E.solve()
        # print info
        its = E.getIterationNumber()
        eps_type = E.getType()
        self.nev, ncv, mpd = E.getDimensions()
        tol, maxit = E.getTolerances()
        self.nconv = E.getConverged()
        print("Solution method: {:s}, stopping condition: tol={:.4g}, maxit={:d}".format(eps_type,tol, maxit))
        print("Number of converged/requested eigenvalues with {:d} iterations  : {:d}/{:d}".format(its,self.nconv,self.nev))
        return self.nconv, its

    def set_options(self,slepc_options):
        print("---- setting additional slepc options -----")
        for (opt, value) in slepc_options.items():
            print("    ",opt,":",value)
            dolfin.PETScOptions.set(opt,value) 
        print("-------------------------------------------")
        self.E.setFromOptions()

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
    def __init__(self, mesh, energy, state, bcs, z, rayleigh=None,
        nullspace=None, parameters=None):
        self.i = 0
        self.parameters = self.default_parameters()
        if parameters is not None: self.parameters.update(parameters)                                                         # for debug purposes
        self.u = state[0]
        self.alpha = state[1]
        self._u = dolfin.Vector(self.u.vector())
        self._alpha = dolfin.Vector(self.alpha.vector())
        self.meshsize = (mesh.hmax()+mesh.hmax())/2.
        self.mesh = mesh

        cdm = dolfin.project(dolfin.CellDiameter(self.mesh)**2., dolfin.FunctionSpace(self.mesh, 'CG', 1))
        self.cellarea = dolfin.Function(z.function_space())
        self.cellarea.assign(cdm)
        self.Z = z.function_space()
        self.ownership = self.Z.dofmap().ownership_range()
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
        self.z = z
        self.z_old = dolfin.Function(self.Z)
        zeta = dolfin.TestFunction(self.Z)
        v, beta = dolfin.split(zeta)

        self.Ealpha = dolfin.derivative(energy, self.alpha, dolfin.TestFunction(self.alpha.ufl_function_space()))
        self.energy = energy

        z_u, z_a = dolfin.split(self.z)
        energy = ufl.replace(energy, {self.u: z_u, self.alpha: z_a})
        self.J = dolfin.derivative(energy, self.z, dolfin.TestFunction(self.Z))
        self.H  = dolfin.derivative(self.J, self.z, dolfin.TrialFunction(self.Z))

        self.nullspace = nullspace

        if rayleigh:
            rP = rayleigh[0]
            rN = rayleigh[1]
            self.rP = dolfin.derivative(dolfin.derivative(rP, self.z, dolfin.TestFunction(self.Z)),
                                                              self.z, dolfin.TrialFunction(self.Z))
            self.rN = dolfin.derivative(dolfin.derivative(rN, self.z, dolfin.TestFunction(self.Z)),
                                                              self.z, dolfin.TrialFunction(self.Z))

        self.ownership_range = self.Z.dofmap().ownership_range()
        # import pdb; pdb.set_trace()
        if len(bcs)>0:
            self.bcs = bcs
            self.bc_dofs = self.get_bc_dofs(bcs)
        else:
            self.bcs = None
            self.bc_dofs = set()

        self.perturbation_v = dolfin.Function(self.Z.sub(0).collapse())
        self.perturbation_beta = dolfin.Function(self.Z.sub(1).collapse())

        dolfin.PETScOptions.set("ksp_type", "preonly")
        dolfin.PETScOptions.set("pc_type", "cholesky")
        dolfin.PETScOptions.set("pc_factor_mat_solver_type", "mumps")
        dolfin.PETScOptions.set("mat_mumps_icntl_24", 1)
        dolfin.PETScOptions.set("mat_mumps_icntl_13", 1)
        dolfin.PETScOptions.set("eps_monitor")

    def default_parameters(self):
        return {'order': 3,
                'eig_rtol': 1e-12,
                'projection': 'none',
                "maxmodes": 1,
                "inactiveset_atol": 1e-5
                }

    def normalise_eigen(self, v, beta, mode='none'):
        if mode=='none':
            return
        elif mode=='one':
            coef = beta.vector().norm('l2')
        elif mode=='max':
            coef = max(abs(beta.vector()[:]))

        vval = v.vector()[:]/coef
        bval = beta.vector()[:]/coef

        v.vector().set_local(vval)
        beta.vector().set_local(bval)

        return

    def linesearch(self, v_n, beta_n, m=3, mode=0):
        debug = False
        en0 = dolfin.assemble(self.energy)
        _u = self._u
        _alpha = self._alpha

        _u[:] = self.u.vector()[:]
        _alpha[:] = self.alpha.vector()[:]

        one = max(1., max(self.alpha.vector()[:]))

        if hasattr(self, 'bcs') and len(self.bcs[0])>0:
            assert np.all([self.is_compatible(bc, v_n, homogeneous = True) for bc in self.bcs[0]]), \
                'displacement test field is not kinematically admissible'

        # positive part
        mask = beta_n.vector()[:]>0.
        hp2 = (one-self.alpha.vector()[mask])/beta_n.vector()[mask]  if len(np.where(mask==True)[0])>0 else [np.inf]
        hp1 = (self.alpha_old.vector()[mask]-self.alpha.vector()[mask])/beta_n.vector()[mask]  if len(np.where(mask==True)[0])>0 else [-np.inf]
        hp = (max(hp1), min(hp2))

        # negative part
        mask = beta_n.vector()[:]<0.

        hn2 = (one-self.alpha.vector()[mask])/beta_n.vector()[mask] if len(np.where(mask==True)[0])>0 else [-np.inf]
        hn1 = (self.alpha_old.vector()[mask]-self.alpha.vector()[mask])/beta_n.vector()[mask]  if len(np.where(mask==True)[0])>0 else [np.inf]
        hn = (max(hn2), min(hn1))

        hmax = np.array(np.min([hp[1], hn[1]]))
        hmin = np.array(np.max([hp[0], hn[0]]))

        hmax_glob = np.array(0.0,'d')
        hmin_glob = np.array(0.0,'d')

        comm.Allreduce(hmax, hmax_glob, op=mpi4py.MPI.MIN)
        comm.Allreduce(hmin, hmin_glob, op=mpi4py.MPI.MAX)

        self.hmax = float(hmax_glob)
        self.hmin = float(hmin_glob)

        if self.hmin>0:
            ColorPrint.print_warn('Line search troubles: found hmin>0')
            # import pdb; pdb.set_trace()
            return 0., np.nan, (0., 0.), 0.
        if self.hmax==0 and self.hmin==0:
            ColorPrint.print_warn('Line search failed: found zero step size')
            # import pdb; pdb.set_trace()
            return 0., np.nan, (0., 0.), 0.
        if self.hmax < self.hmin:
            ColorPrint.print_warn('Line search failed: optimal h* not admissible')
            # import pdb; pdb.set_trace()
            return 0., np.nan, (0., 0.), 0.
            # get next perturbation mode

        en = []

        htest = np.linspace(self.hmin, self.hmax, m+1)

        for h in htest:
            uval = _u[:]     + h*v_n.vector()[:]
            aval = _alpha[:] + h*beta_n.vector()[:]

            if not np.all(aval - self.alpha_old.vector()[:] + dolfin.DOLFIN_EPS_LARGE >= 0.):
                print('damage test field doesn\'t verify sharp irrev from below')
                import pdb; pdb.set_trace()
            if not np.all(aval <= one):
                print('damage test field doesn\'t verify irrev from above')
                import pdb; pdb.set_trace()

            self.u.vector()[:] = uval
            self.alpha.vector()[:] = aval

            en.append(dolfin.assemble(self.energy)-en0)
            if debug and size == 1:
                ax2.plot(xs, [self.alpha(x, 0) for x in xs], label='$\\alpha+h \\beta_{{{}}}$, h={:.3f}'.format(mode, h), lw=.5, c='C1' if h>0 else 'C4')

        z = np.polyfit(htest, en, m)
        p = np.poly1d(z)

        if m==2:
            ColorPrint.print_info('Line search using quadratic interpolation')
            hstar = - z[1]/(2*z[0])
        else:
            ColorPrint.print_info('Line search using polynomial interpolation (order {})'.format(m))
            h = np.linspace(self.hmin, self.hmax, 100)
            hstar = h[np.argmin(p(h))]

        if hstar < self.hmin or hstar > self.hmax:
            ColorPrint.print_warn('Line search failed, h*={:3e} not in feasible interval'.format(hstar))
            return 0., np.nan

        ColorPrint.print_info('Line search h* = {:3f} in ({:.3f}, {:.3f}), h*/hmax {:3f}\
            '.format(hstar, self.hmin, self.hmax, hstar/self.hmax))
        ColorPrint.print_info('Line search approx =\n {}'.format(p))
        ColorPrint.print_info('h in ({:.5f},{:.5f})'.format(self.hmin,self.hmax))
        ColorPrint.print_warn('Line search estimate, relative energy variation={:.5f}%'.format((p(hstar))/en0*100))

        # restore solution
        self.u.vector()[:] = _u[:]
        self.alpha.vector()[:] = _alpha[:]

        return hstar, p(hstar)/en0, (self.hmin, self.hmax), en

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
        zeros = dolfin.Constant([0.0,]*self.u.geometric_dimension())
        for bc in self.bcs[0]:
            if hasattr(bc, 'sub_domain'):
                new_bc = dolfin.DirichletBC(self.Z.sub(0), zeros, bc.sub_domain, bc.method())
            elif hasattr(bc, 'domain_args'):
                new_bc = dolfin.DirichletBC(self.Z.sub(0), zeros, bc.domain_args[0], bc.domain_args[1], bc.method())
            else:
                raise RuntimeError("Couldn't find where bcs for displacement are applied")

            bcs_Z.append(new_bc)
        for bc in self.bcs[1]:
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

    def is_elastic(self, z, z_old):
        tol = 1e-6
        elastic_dofs = ((z.vector() - z_old.vector()) < tol)

        return np.all(elastic_dofs)

    def get_inactive_set(self):
        tol = self.parameters['inactiveset_atol']
        debug= False

        Ealpha = dolfin.assemble(self.Ealpha)
        vec = dolfin.PETScVector(MPI.comm_self)
        Ealpha.gather(vec, np.array(range(self.Z.sub(1).dim()), "intc"))

        if debug:
            print('len vec grad', len(vec[:]))


        mask = Ealpha[:]/self.cellarea.vector() < tol

        inactive_set_alpha = set(np.where(mask == True)[0])

        # from subspace to global numbering
        global_inactive_set_alpha = [self.mapa[k] for k in inactive_set_alpha]

        # add displacement dofs
        inactive_set = set(global_inactive_set_alpha) | set(self.Z.sub(0).dofmap().dofs())

        return inactive_set

    def reduce_Hessian(self, Hessian=None, restricted_dofs_is=None):
        if not Hessian: H = dolfin.as_backend_type(dolfin.assemble(self.H)).mat()
        else: H = dolfin.as_backend_type(dolfin.assemble(Hessian)).mat()
        if restricted_dofs_is is not None:
            try:
                H_reduced = H.createSubMatrix(restricted_dofs_is, restricted_dofs_is)
            except:
                H_reduced = H.getSubMatrix(restricted_dofs_is, restricted_dofs_is)
        return H_reduced

    def pc_setup(self):
        self.pc = PETSc.PC().create(MPI.comm_world)
        self.pc.setType("cholesky")
        if hasattr(self.pc, 'setFactorSolverType'):
            self.pc.setFactorSolverType("mumps")
        elif hasattr(self.pc, 'setFactorSolverPackage'):
            self.pc.setFactorSolverPackage('mumps')
        else:
            ColorPrint.print_warn('Could not configure preconditioner')

    def get_inertia(self, Mat = None, restricted_dof_is=None):
        if Mat == None:
            H = dolfin.as_backend_type(dolfin.assemble(self.H)).mat()
        else:
            H = Mat
        if restricted_dof_is is not None:
            try:
                H = H.createSubMatrix(restricted_dof_is, restricted_dof_is)
            except:
                H = H.getSubMatrix(restricted_dof_is, restricted_dof_is)
        self.pc.setOperators(H)
        self.pc.setUp()
        Fm = self.pc.getFactorMatrix()
        # myviewer = PETSc.Viewer().createASCII("test.txt", mode=PETSc.Viewer.Format.ASCII_COMMON,comm= PETSc.COMM_WORLD)
        (neg, zero, pos) = Fm.getInertia()
        ColorPrint.print_info("#Eigenvalues of E'': (%s [neg], %s [zero], %s [pos])" % (neg, zero, pos))
        if neg:
            self.stable = False
        else:
            self.stable = True
        return neg

    def project(self, beta, mode = 'none'):
        # import pdb; pdb.set_trace()
        if self.bcs: bc_a = self.bcs[1]
        if mode == 'truncate':
            mask = beta.vector()[:] < 0
            beta.vector()[np.where(mask == True)[0]] = 0
            if self.bcs:
                for bc in bc_a: bc.apply(beta.vector())
            return beta
        elif mode == 'shift':
            C = np.min(beta.vector()[:])
            beta.vector()[:] = beta.vector()[:] + abs(C)
            if self.bcs:
                for bc in bc_a: bc.apply(beta.vector())
            return beta
        elif mode == 'none':
            if self.bcs:
                for bc in bc_a: bc.apply(beta.vector())
            return beta
        else:
            return -1

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
        debug = False
        self.alpha_old = alpha_old
        postfix = 'seq' if size == 1 else 'mpi'

        locnumbcs = np.array(len(self.bc_dofs))
        if debug:
            print(rank, ': ', 'locnumbcs', locnumbcs)
        numbcs = np.array(0.,'d')
        comm.Reduce(locnumbcs, numbcs, op=mpi4py.MPI.SUM, root=0)

        if debug and rank == 0:
            print('#bc dofs = {}'.format(int(numbcs)))


        if not np.all(self.alpha.vector()[:] >=self.alpha_old.vector()[:]):
            pd = np.where(self.alpha.vector()[:]-self.alpha_old.vector()[:] < 0)[0]
            ColorPrint.print_warn('Pointwise irreversibility issues on dofs {}'.format(pd))
            ColorPrint.print_warn('diff = {}'
                .format(self.alpha.vector()[pd]-self.alpha_old.vector()[pd]))
            ColorPrint.print_warn('Continuing')

        self.assigner.assign(self.z_old, [self.u_zero, self.alpha_old])
        self.assigner.assign(self.z, [self.u_zero, self.alpha])

        if self.is_elastic(self.z, self.z_old):
            ColorPrint.print_pass('Current state: elastic')
            self.stable = True
            self.negev = np.nan
            return self.stable, 0
        else:
            ColorPrint.print_pass('Current state: not elastic')

        inactive_dofs = self.get_inactive_set()
        self.inactive_set = inactive_dofs

        free_dofs = list(sorted(inactive_dofs - self.bc_dofs))

        index_set = petsc4py.PETSc.IS()
        index_set.createGeneral(free_dofs)

        if hasattr(self, 'rP') and hasattr(self, 'rN'):
            self.H2 = self.rP-self.rN

        if hasattr(self, 'H2'):
            ColorPrint.print_pass('Inertia: Using user-provided Hessian')
            self.H_reduced = self.reduce_Hessian(self.H2, restricted_dofs_is = index_set)
        else:
            ColorPrint.print_pass('Inertia: Using computed Hessian')
            self.H_reduced = self.reduce_Hessian(self.H, restricted_dofs_is = index_set)

        self.pc_setup()

        negev = self.get_inertia(self.H_reduced)

        if negev > 0:
        # if True:
            eigs = []
        # solve full eigenvalue problem
            eigen_tol = self.parameters['eig_rtol']
            if hasattr(self, 'H2'):
                ColorPrint.print_pass('Full eig: Using user-provided Hessian')
                ColorPrint.print_pass('Norm provided {}'.format(dolfin.assemble(self.H2).norm('frobenius')))
                eigen = EigenSolver(self.H2, self.z, restricted_dofs_is = index_set, slepc_options={'eps_max_it':600, 'eps_tol': eigen_tol})
            else:
                ColorPrint.print_pass('Full eig: Using computed Hessian')
                eigen = EigenSolver(self.H, self.z, restricted_dofs_is = index_set, slepc_options={'eps_max_it':600, 'eps_tol': eigen_tol})
            ColorPrint.print_pass('Norm computed {}'.format(dolfin.assemble(self.H).norm('frobenius')))
            self.computed.append(dolfin.assemble(self.H).norm('frobenius'))
            if hasattr(self, 'H2'): self.provided.append(dolfin.assemble(self.H2).norm('frobenius'))

            maxmodes = self.parameters['maxmodes']

            nconv, it = eigen.solve(min(maxmodes, negev+7))

            if nconv == 0:
                ColorPrint.print_warn('Eigensolver did not converge')
                self.stable = negev <= 0
                self.eigs = []
                self.mineig = np.nan
                return (self.stable, int(negev))

            eigs = eigen.get_eigenvalues(nconv)
            negconv = sum(eigs[:,0]<0)
            # sanity check
            if nconv and negconv != negev:
                ColorPrint.print_bold('eigen solver found {} negative evs '.format(negconv))
            if nconv == 0 and negev>0:
                ColorPrint.print_bold('Full eigensolver did not converge but inertia yields {} neg eigen'.format(negev))
                return

            # eigen.save_eigenvectors(nconv)

            if nconv > 0:
                ColorPrint.print_pass('')
                ColorPrint.print_pass("i        k      err     ")
                ColorPrint.print_pass("---------------------------")
                for (i, k) in enumerate(eigs):
                    ColorPrint.print_pass( "%d %12e %12e" %(i, k[0], k[1]) )
                ColorPrint.print_pass('')

            linsearch = []

            if negconv > 0:
                for n in range(negconv) if negconv < maxmodes else range(maxmodes):
                    ColorPrint.print_pass('Perturbation mode {}'.format(n))
                    eig, u_r, u_im, err = eigen.get_eigenpair(n)
                    err2 = eigen.E.computeError(0, SLEPc.EPS.ErrorType.ABSOLUTE)
                    v_n, beta_n = u_r.split(deepcopy=True)
                    print(rank, [self.is_compatible(bc, u_r, homogeneous = True) for bc in self.bcs_Z])

                    if debug and size == 1:
                        plt.clf()
                        plt.colorbar(dolfin.plot(dot(v_n, v_n)**(.5)))
                        plt.savefig('data/vn-{}-{}.pdf'.format(rank, n))

                    self.normalise_eigen(v_n, beta_n, mode='max')
                    beta_n = self.project(beta_n, mode=self.parameters['projection'])
                    eig, u_r, u_im, err = eigen.get_eigenpair(n)

                    order = self.parameters['order']
                    h, en_diff, interval, energy = self.linesearch(v_n, beta_n, order, n)

                    linsearch.append({'n': n, 'lambda_n': eig.real,'hstar': h, 'en_diff': en_diff,
                        'v_n': v_n, 'beta_n': beta_n, 'order': order,
                        'interval': interval, 'energy': energy})

            eig, u_r, u_im, err = eigen.get_eigenpair(0)

            self.eigs = eigs[:,0]
            self.mineig = eig.real
            self.stable = negev <= 0  # based on inertia
            self.negev = negev  # based on inertia

            if eigs[0,0]<0:
                self.perturbation_v = linsearch[0]['v_n']
                self.perturbation_beta = linsearch[0]['beta_n']
                self.hstar = linsearch[0]['hstar']
                self.en_diff = linsearch[0]['en_diff']
                self.eigendata = linsearch

            self.i +=1

        return (self.stable, int(negev))

