import dolfin
import sys
from utils import ColorPrint
from ufl import sqrt, inner, dot, conditional, derivative, ge, le
import os
from slepc_eigensolver import EigenSolver
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

class MyEigen(EigenSolver):

    def eigensolver_setup(self):
        E = SLEPc.EPS()
        E.create()
        # E.setType(SLEPc.EPS.Type.ARNOLDI)
        E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
 # ARNOLDI          -
 # ARPACK           X 86
 # BLOPEX           X 86
 # BLZPACK          X 86
 # CISS 
 # FEAST            X 86
 # GD               X 56
 # JD               X petscerr 56
 # KRYLOVSCHUR      *
 # LANCZOS          -
 # LAPACK           - doesn't converge (cyl homog)
 # LOBPCG           X 56
 # POWER
 # PRIMME           X 86
 # RQCG             X petscerr 56
 # SUBSPACE
 # TRLAN            X 86
        E.setProblemType(SLEPc.EPS.ProblemType.HEP)
        E.setWhichEigenpairs(E.Which.TARGET_REAL)
        E.setTarget(-.1) 
        st = E.getST()
        st.setType('sinvert')
        st.setShift(-1.e-3)
        # E.view()
        return E

    def save_eigenvectors(self,n, file_name="output/modes.xdmf",save_imaginary=False):
        eigenvalues = [] 
        eigenvectors = []
        if file_name[-4::] == 'xdmf': 
            file = dolfin.XDMFFile(file_name)
            file.parameters["functions_share_mesh"] = True
            file.parameters["flush_output"] = True
        else:
            file = File(self.comm,file_name)
        for i in range(n):
            eig, w_r, w_im, err = self.get_eigenpair(i)
            # print('norm w_r {} {}'.format(i, w_r.vector().norm('l2')))
            v_r = w_r.sub(0)
            beta_r = w_r.sub(1)
            # print('norm v_r {} {}'.format(i, v_r.vector().norm('l2')))
            # print('norm b_r {} {}'.format(i, beta_r.vector().norm('l2')))
            v_r.rename("displacement mode real","displacement mode real")
            beta_r.rename("damage mode real","mode real")
            file.write(v_r, i)
            file.write(beta_r, i)
            if save_imaginary:
                v_im = w_im.sub(0)
                beta_im = w_im.sub(1)
                v_im.rename("displacement mode imaginary","displacement mode imaginary")
                beta_im.rename("damage mode imaginary","mode imaginary")
                file.write(v_im, i)
                file.write(beta_im, i)
        print('saved eigenmodes in in {}'.format(file_name))
        return file_name


class StabilitySolver(object):
    """solves second order stability problem"""
    def __init__(self, mesh, energy, state, bcs, z, rayleigh=None,
        nullspace=None, parameters=None):
        self.i = 0
        # import pdb; pdb.set_trace()
        self.parameters = self.default_parameters()
        if parameters is not None: self.parameters.update(parameters)                                                         # for debug purposes
        # self.file_debug = dolfin.XDMFFile(os.path.join('output', "debug.xdmf"))            # for debug purposes
        # self.file_debug.parameters["flush_output"] = True 
        self.u = state[0]
        self.alpha = state[1]
        self._u = dolfin.Vector(self.u.vector())
        self._alpha = dolfin.Vector(self.alpha.vector())
        self.meshsize = (mesh.hmax()+mesh.hmax())/2.
        self.mesh = mesh

        cdm = dolfin.project(dolfin.CellDiameter(self.mesh)**2., dolfin.FunctionSpace(self.mesh, 'CG', 1))
        self.cellarea = dolfin.Function(z.function_space())
        self.cellarea.assign(cdm)
        # self.Z = dolfin.FunctionSpace(mesh,
                    # dolfin.MixedElement([self.u.ufl_element(), self.alpha.ufl_element()]))
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
        # self.z = dolfin.Function(self.Z)
        self.z = z
        self.z_old = dolfin.Function(self.Z)
        zeta = dolfin.TestFunction(self.Z)
        v, beta = dolfin.split(zeta)

        self.Ealpha = dolfin.derivative(energy, self.alpha, dolfin.TestFunction(self.alpha.ufl_function_space()))
        # self.Ealpha = dolfin.derivative(energy, self.alpha, beta)
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
            self.bc_dofs = self.get_bc_dofs2(bcs)
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
        # dolfin.PETScOptions.set("eps_monitor")

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
        # if mode=='solution':
        #     coef1 = v.vector().norm('l2')/self.u.vector().norm('l2') \
        #          if self.u.vector().norm('l2')>0. else v.vector().norm('l2')
        #     coef2 = beta.vector().norm('l2')/self.alpha.vector().norm('l2') \
        #          if self.alpha.vector().norm('l2')>0. else beta.vector().norm('l2')
        elif mode=='one':
            # coef1 = v.vector().norm('l2')
            coef = beta.vector().norm('l2')
        elif mode=='max':
            # coef1 = v.vector().norm('l2')
            coef = max(abs(beta.vector()[:]))

        vval = v.vector()[:]/coef
        bval = beta.vector()[:]/coef

        v.vector().set_local(vval)
        beta.vector().set_local(bval)

        return

    def linesearch(self, v_n, beta_n, m=3, mode=0):
        debug = True
        en0 = dolfin.assemble(self.energy)
        _u = self._u
        _alpha = self._alpha

        # _u.set_local(self.u.vector()[:])
        # _alpha.set_local(self.alpha.vector()[:])

        _u[:] = self.u.vector()[:]
        _alpha[:] = self.alpha.vector()[:]
        tol = 5e-5

        one = max(1., max(self.alpha.vector()[:]))
        # tol = 0.
        # sanity check
        if hasattr(self, 'bcs') and len(self.bcs[0])>0:
            assert np.all([self.is_compatible2(bc, v_n, homogeneous = True) for bc in self.bcs[0]]), 'displacement test field is not kinematically admissible'

        if debug:
            print('||vn||_l2 = {}'.format(dolfin.norm(v_n, 'l2')))
            print('||βn||_l2 = {}'.format(dolfin.norm(beta_n, 'l2')))
            print('||vn||_h1 = {}'.format(dolfin.norm(v_n, 'h1')))
            print('||βn||_h1 = {}'.format(dolfin.norm(beta_n, 'h1')))
            # print(rank, mask)
            # print(rank, 'beta_n', beta_n.vector()[:])
            # print(rank, 'len(beta_n)', len(beta_n.vector()[:]))
            # print(rank, 'len(mask)', len(mask))

        # positive part
        mask = beta_n.vector()[:]>0.
        # if np.all(~mask==True):
        hp2 = (one-self.alpha.vector()[mask])/beta_n.vector()[mask]  if len(np.where(mask==True)[0])>0 else [np.inf]
        hp1 = (self.alpha_old.vector()[mask]-self.alpha.vector()[mask])/beta_n.vector()[mask]  if len(np.where(mask==True)[0])>0 else [-np.inf]
        hp = (max(hp1), min(hp2))

        # negative part
        mask = beta_n.vector()[:]<0.
        # print(rank, mask)
        # print(rank, 'len(beta_n)', len(beta_n.vector()[:]))
        # print(rank, 'len(mask)', len(mask))

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


        # np.set_printoptions(threshold=np.nan)

        if debug and size == 1:
            # ColorPrint.print_info('min(a + hmin beta)        = {:.3f}'.format(min(self.alpha.vector()[:] + self.hmin*beta_n.vector()[:])))
            # ColorPrint.print_info('min(a + hmax beta)        = {:.3f}'.format(min(self.alpha.vector()[:] + self.hmax*beta_n.vector()[:])))
            # ColorPrint.print_info('max(a + hmin beta)        = {:.3f}'.format(max(self.alpha.vector()[:] + self.hmin*beta_n.vector()[:])))
            # ColorPrint.print_info('max(a + hmax beta)        = {:.3f}'.format(max(self.alpha.vector()[:] + self.hmax*beta_n.vector()[:])))
            # rad = 1/2-dolfin.DOLFIN_EPS_LARGE
            self.alpha.set_allow_extrapolation(True)
            # rad = 1/2-.0000001
            # plt.plot([self.alpha(rad*np.cos(t), rad*np.sin(t)) for t in np.linspace(0,2*np.pi)],
            #     label='$\\alpha', lw=3, c='C1')
            # plt.plot([beta_n(rad*np.cos(t), rad*np.sin(t)) for t in np.linspace(0,2*np.pi)],
            #     label='$\\beta_n$', lw=3, c='C2')
            X =self.alpha.function_space().tabulate_dof_coordinates()
            plt.clf()
            xs = np.linspace(min(X[:, 0]),max(X[:, 0]), 300)
            plt.plot(xs, [self.alpha(x, 0) for x in xs], label='$\\alpha$', lw=3, c='C1')
            plt.plot(xs, [beta_n(x, 0) for x in xs], label='$\\beta_{{{}}}$'.format(mode), lw=3, c='C2')
            plt.axhline(one-1e-5, lw='.5', ls=':')
            plt.legend(loc='best')
            plt.savefig('data/prof.pdf')
            plt.close()

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

        # directional variations
        if debug and size == 1:
            fig = plt.figure(dpi=160)
            ax = fig.gca()
            fig2 = plt.figure(dpi=160)
            ax2 = fig2.gca()
            envsu = []
            maxvar = max(abs(self.hmin), abs(self.hmax))
            htest = np.linspace(-maxvar, maxvar, 2*m+1)
            # htest = np.linspace(-1, 1, 2*m+1)

            # directional variations. En vs uh
            for h in htest:
                uval = _u[:]     + h*v_n.vector()
                self.u.vector().set_local(uval)
                envsu.append(dolfin.assemble(self.energy)-en0)
            ax.plot(htest, envsu, label='E(u+h $v$, $\\alpha$)')

            ax.axvline(self.hmin, c='k')
            ax.axvline(self.hmax, c='k')
            # plt.savefig("en_vsu.png".format())

            # restore original state
            self.u.vector().set_local(_u[:])
            self.alpha.vector().set_local(_alpha[:])

            # directional variations. En vs ah
            envsa = []
            for h in htest:
                aval = _alpha[:] + h*beta_n.vector()
                self.alpha.vector().set_local(aval)
                envsa.append(dolfin.assemble(self.energy)-en0)
            ax.plot(htest, envsa, label='E(u, $\\alpha$+h $\\beta$)')

            # restore original state

        # self.u.vector()[:]  = _u[:]
        # self.alpha.vector()[:]  = _alpha[:]


        en = []

        htest = np.linspace(self.hmin, self.hmax, m+1)

        for h in htest:
            uval = _u[:]     + h*v_n.vector()[:]
            aval = _alpha[:] + h*beta_n.vector()[:]
            # import pdb; pdb.set_trace()
            # assert np.all(aval>= self.alpha_old.vector()[:]), 'damage test field doesn\'t verify irrev from below'
            # assert np.all(aval<= one), 'damage test field doesn\'t verify irrev from above'
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

        # restore original state
            # self.u.vector().set_local(_u[:])
            # self.alpha.vector().set_local(_alpha[:])


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
        # ColorPrint.print_warn('Line search estimate en={:3e}'.format(p(hstar)))
        ColorPrint.print_warn('Line search estimate, rel energy variation={:.5f}%'.format((p(hstar))/en0*100))

        if debug and size == 1:
            ColorPrint.print_info('1 - max(a + hstar beta)        = {:.3f}'.format(one - max(self.alpha.vector()[:] + hstar*beta_n.vector()[:])))

            ax2.plot(xs, [self.alpha(x, 0) for x in xs], label='$\\alpha$', lw=1, c='k')
            ax3 = ax2.twinx()
            ax3.axhline(0., lw=.5, c='k', ls='-')
            ax2.axhline(one-tol, lw='.5', ls=':')
            ax2.axhline(one, lw='.5', ls=':')
            ax3.plot(xs, [beta_n(x, 0) for x in xs], label='$\\beta_{{{}}}$'.format(mode), ls=':')
            ax2.plot(xs, [self.alpha_old(x, 0) for x in xs], label='$\\alpha_{old}$', c='k')
            ax2.axhline(0., ls=':')
            # ax2.set_xlim(0, 1.2*max([self.alpha(x, 0)+hstar*beta_n(x, 0) for x in xs]))
            ax2.plot(xs, [self.alpha(x, 0)+hstar*beta_n(x, 0) for x in xs],
                label='$\\alpha+\\bar h \\beta_{{{}}}$, \\bar h={:.3f}'.format(mode, hstar), lw=1, c='C2')
            ax2.legend(fontsize='small', bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
            ax3.legend(fontsize='small')
            fig2.savefig("data/prof-{:.3f}-{}.png".format(max(self.u.vector()), mode), bbox_inches="tight")
            plt.close(fig2)

            ax.axvline(hstar)
            ax.axvline(0., c='k', lw=.5, ls=':')
            ax.plot(np.linspace(self.hmin, self.hmax, 10), p(np.linspace(self.hmin, self.hmax, 10)),
                label='interp h star = {:.5e}'.format(hstar))
            ax.legend()
            fig.savefig("data/en-{:.3f}-{}.png".format(max(self.u.vector()), mode))
            plt.close(fig)


        # restore the solution
        self.u.vector()[:] = _u[:]
        self.alpha.vector()[:] = _alpha[:]
        dolfin.plot(self.alpha, cmap='hot')
        plt.savefig("data/alpha-{}.png".format(rank))

        return hstar, p(hstar)/en0, (self.hmin, self.hmax), en

    def save_matrix(self, Mat, name):
        if name[-3:]=='txt': viewer = PETSc.Viewer().createASCII(name, 'w')
        else: viewer = PETSc.Viewer().createBinary(name, 'w')
        Mat.view(viewer)
        print('saved matrix in ', name)

    def get_bc_dofs(self, boundary_conditions):
        """Returns the list of bc-constrained dofs in the mixed func space"""
        # FIXME: order is critical
        bcs_u = boundary_conditions[0]
        bcs_alpha = boundary_conditions[1]

        bc_Z = []
        bc_dofs = []
        dim = self.u.function_space().ufl_element().value_size()
        # FIXME: what if there are bcs only on a subspace of displacements?
        for bc in bcs_u:
            # boundary condition on a subspace of V_u = Z.sub(0)
            if bc.function_space().component():
                component = bc.function_space().component()[0]
                space = self.Z.sub(0).sub(component)
                U0 = dolfin.Constant(0.)
            # boundary condition on the entire space V_u = Z.sub(0)
            else:
                space = self.Z.sub(0)
                U0 = dolfin.Constant([0.]*dim)

            if hasattr(bc, 'sub_domain'):
                newbc = dolfin.DirichletBC(space, U0, bc.sub_domain, bc.method())
            elif hasattr(bc, 'domain_args'):
                newbc = dolfin.DirichletBC(space, U0, bc.domain_args[0], bc.domain_args[1], bc.method())
            # import pdb; pdb.set_trace()
            bc_Z.append(newbc)

        for bc in bcs_alpha:
            if hasattr(bc, 'sub_domain'):
                newbc = dolfin.DirichletBC(self.Z.sub(1), dolfin.Constant(0.), bc.sub_domain, bc.method())
            elif hasattr(bc, 'domain_args'):
                newbc = dolfin.DirichletBC(self.Z.sub(1), dolfin.Constant(0.), bc.domain_args[0], bc.domain_args[1], bc.method())
            bc_Z.append(newbc)

        self.bcs_Z = bc_Z
        # bc_dofs = [set(bc.get_boundary_values().keys()) for bc in bc_Z]
        # bc_dofs = reduce(lambda x,y: x|y, bc_dofs, set())
        # constrained_dofs = [x for x in range(self.ownership_range[0], self.ownership_range[1]) if x in bc_dofs]

        dofmap = self.Z.dofmap()
        bc_dofs = [set(bc.get_boundary_values().keys()) for bc in bc_Z]
        bc_dofs_glob = []
        print(rank, ': ', 'bc_dofs: ', bc_dofs)
        for bc in bc_dofs:
            _bc_glob = [dofmap.local_to_global_index(x) for x in bc]
            bc_dofs_glob.append(set(_bc_glob))

        # flatten
        # import pdb; pdb.set_trace()
        bc_dofs_glob = reduce(lambda x,y: x|y, bc_dofs_glob, set())
        bc_dofs = reduce(lambda x,y: x|y, bc_dofs, set())
        print(rank, ': ', 'ownership: ', self.ownership_range)
        print(rank, ': ', 'bc_dofs glob: ', bc_dofs_glob)
        print(rank, ': ', 'bc_dofs: ', bc_dofs)
        Xz = self.Z.tabulate_dof_coordinates()
        print(rank, ': bc dof coords', [list(Xz[dof]) for dof in list(sorted(bc_dofs))])
        # return set(constrained_dofs)
        return bc_dofs_glob

    def get_bc_dofs2(self, boundary_conditions):
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
        if debug:
            print(rank, ': ', 'ownership: ', self.ownership_range)
            print(rank, ': ', 'bc_keys loc ', bc_keys)
            print(rank, ': ', 'bc_dofs glob: ', bc_keys_glob)
            print(rank, ': ', 'bc_dofs: ', self.bc_dofs)
            print(rank, ': ', 'len bc_dofs: ', len(self.bc_dofs))
            # Xz = self.Z.tabulate_dof_coordinates()
            # print(rank, ': bc dof coords', [list(Xz[dof]) for dof in list(sorted(self.bc_dofs))])
        self.bcs_Z = bcs_Z
        return self.bc_dofs

    def is_elastic(self, z, z_old):
        tol = 1e-6
        elastic_dofs = ((z.vector() - z_old.vector()) < tol)

        return np.all(elastic_dofs)

    def get_inactive_set3(self):
        # returns list of dofs where constraints are inactive
        # global numbering wrt the mixed vector z \in Z
        tol = 1e-3
        debug= True

        # based on constraint: inactive set = {x: g(z)>0}

        self.z_old.vector()[:]=0.
        self.z.vector()[:] = .5
        dolfin.assign(self.z_old.sub(1), self.alpha_old)
        dolfin.assign(self.z.sub(1), self.alpha)
        # import pdb; pdb.set_trace()

        diff = self.z.vector()-self.z_old.vector()

        one = self.z.vector().copy()
        one[:]=1.

        vec = dolfin.PETScVector(MPI.comm_self)

        # non damaging dofs = elastic dofs
        diff.gather(vec, np.array(range(self.Z.dim()), "intc"))
        mask = vec < dolfin.DOLFIN_EPS_LARGE

        if debug:
            print('len vec damage', len(vec[:]))

        # fracture dofs
        diff2 = one - tol - self.z.vector()
        diff2.gather(vec, np.array(range(self.Z.dim()), "intc"))
        mask2 = vec < dolfin.DOLFIN_EPS_LARGE

        dofset = set(np.where(mask == True)[0])
        dofset2 = set(np.where(mask2 == True)[0])

        active_set = dofset | dofset2
        inactive_set = set(range(self.ownership[0], self.ownership[1]))-active_set

        # -----------------------

        Ealpha = dolfin.assemble(self.Ealpha)
        vec = dolfin.PETScVector(MPI.comm_self)
        Ealpha.gather(vec, np.array(range(self.Z.sub(1).dim()), "intc"))
        if debug:
            print('len vec grad', len(vec[:]))


        tol = self.parameters['inactiveset_atol']
        mask = Ealpha[:]/self.cellarea.vector() < tol
        # Ealpha[:]/self.cellarea.vector()[mask3]
        # mask3 = vec > 0.
        # tol = 1.5*self.cellarea
        # tol = 5.*self.cellarea.vector()
        # mask3 = vec > tol

        inactive_set_alpha = set(np.where(mask == True)[0])
        # active_set2 = set(np.where(mask3 == True)[0])
        # inactive_set2 = set(range(self.ownership[0], self.ownership[1]))-active_set2

        # from subspace to global numbering
        global_inactive_set = [self.mapa[k] for k in inactive_set_alpha]

        # add displacement dofs
        inactive_set2 = set(global_inactive_set) | set(self.Z.sub(0).dofmap().dofs())

        if debug:
            print('inactive set damage', len(inactive_set))
            print('inactive set gradie', len(inactive_set2))
            # print('tol', tol*(self.meshsize)**2.)
            # print('mesh**2', (self.meshsize)**2.)
            print('max Ealpha', max(Ealpha[:]))
            print('min Ealpha', min(Ealpha[:]))

            with open('test_mask.npz', 'wb') as f:
                np.save(f, mask)

            with open('test_cellarea.npz', 'wb') as f:
                np.save(f, Ealpha[:]/self.cellarea.vector()[:])

            with open('test_inactiveset.npz', 'wb') as f:
                np.save(f, inactive_set2)

            # with open('test_avgscale.npz', 'wb') as f:
            #     np.save(f, Ealpha[:]/(self.meshsize)**2.)

            # with open('test_carea.npz', 'wb') as f:
                # np.save(f, self.cellarea.vector()[:])

            # with open("test_maxJresc.txt", "a") as myfile:
                # myfile.write('{} '.format(max(Ealpha[:]/(self.cellarea.vector()[:]))))

        assert len(inactive_set) == len(inactive_set2)

        return inactive_set2

    def get_inactive_set4(self):
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
            ColorPrint.print_info('Unstable')
            self.stable = False
        else:
            ColorPrint.print_info('Stable')
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
        ret = True
        if type(bcs) == list:
            for bc in bcs:
                ret &= self.is_compatible(bc, v, homogeneous)
        elif type(bcs) == dolfin.fem.dirichletbc.DirichletBC:
            bcdofs = list(bcs.get_boundary_values().keys())
            if homogeneous:
                values = [0]*len(bcdofs)
            else:
                values = list(bcs.get_boundary_values().values())
            ret = np.all(np.isclose(v.vector()[bcdofs], values))
        return ret

    def is_compatible2(self, bcs, v, homogeneous = False, tol=dolfin.DOLFIN_EPS_LARGE):
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
        debug = True
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
            # import pdb; pdb.set_trace()

        self.assigner.assign(self.z_old, [self.u_zero, self.alpha_old])
        self.assigner.assign(self.z, [self.u_zero, self.alpha])

        if self.is_elastic(self.z, self.z_old):
            ColorPrint.print_pass('Current state: elastic')
            self.stable = True
            self.negev = np.nan
            return self.stable, 0
        else:
            ColorPrint.print_pass('Current state: not elastic')

        # import pdb; pdb.set_trace()
        inactive_dofs = self.get_inactive_set4()
        # inactive_dofs = self.get_inactive_set4()
        self.inactive_set = inactive_dofs
        if debug and rank == 0:
            print('#inactive dofs = {}'.format(len(inactive_dofs)))

        # free_dofs = list(sorted(inactive_dofs - self.bc_dofs))
        # import pdb; pdb.set_trace()
        free_dofs = list(sorted(inactive_dofs - self.bc_dofs))
        # free_dofs = sorted(set(range(self.ownership[0], self.ownership[1]))-self.bc_dofs)
        # free_dofs = sorted(set(range(self.ownership[0], self.ownership[1])))

        print(rank, len(free_dofs))
        print(rank, self.ownership)
        print(rank, self.Z.dim())

        index_set = petsc4py.PETSc.IS()
        index_set.createGeneral(free_dofs)

        # self.H
        if hasattr(self, 'rP') and hasattr(self, 'rN'):
            self.H2 = self.rP-self.rN

        # self.H_reduced = self.reduce_Hessian(restricted_dofs_is = index_set)

        if hasattr(self, 'H2'):
            ColorPrint.print_pass('Inertia: Using user-provided Hessian')
            self.H_reduced = self.reduce_Hessian(self.H2, restricted_dofs_is = index_set)
            # self.H_reduced = dolfin.as_backend_type(dolfin.assemble(self.H2)).mat()
            # self.save_matrix(self.H_reduced,
                            # 'data/Hessian-provided-{:d}.txt'.format(self.i))
        else:
            ColorPrint.print_pass('Inertia: Using computed Hessian')
            self.H_reduced = self.reduce_Hessian(self.H, restricted_dofs_is = index_set)
            # self.H_reduced = dolfin.as_backend_type(dolfin.assemble(self.H)).mat()

        # import pdb; pdb.set_trace()

        # self.save_matrix(self.H_reduced, 'data/Hessian-reduced.data')

        # self.save_matrix(self.H_reduced,
                        # 'data/Hessian-reduced-{:d}.txt'.format(self.i))

        self.pc_setup()
        # negev = self.get_inertia(dolfin.as_backend_type(dolfin.assemble(self.H)).mat(), restricted_dof_is = index_set)
        negev = self.get_inertia(self.H_reduced)


        # if negev > 0:
        if True:
            eigs = []
        # solve full eigenvalue problem
            eigen_tol = self.parameters['eig_rtol']
            if hasattr(self, 'H2'):
                ColorPrint.print_pass('Full eig: Using user-provided Hessian')
                ColorPrint.print_pass('Norm provided {}'.format(dolfin.assemble(self.H2).norm('frobenius')))
                eigen = MyEigen(self.H2, self.z, restricted_dofs_is = index_set, slepc_options={'eps_max_it':600, 'eps_tol': eigen_tol})
            else:
                ColorPrint.print_pass('Full eig: Using computed Hessian')
                eigen = MyEigen(self.H, self.z, restricted_dofs_is = index_set, slepc_options={'eps_max_it':600, 'eps_tol': eigen_tol})
            ColorPrint.print_pass('Norm computed {}'.format(dolfin.assemble(self.H).norm('frobenius')))
            self.computed.append(dolfin.assemble(self.H).norm('frobenius'))
            if hasattr(self, 'H2'): self.provided.append(dolfin.assemble(self.H2).norm('frobenius'))

            # Initialise with last step's (projected) perturbation direction
            # dolfin.assign(self.z.sub(0), self.perturbation_v)
            # dolfin.assign(self.z.sub(1), self.perturbation_beta)
            # _z_vec = dolfin.as_backend_type(self.z.vector()).vec()
            # _z_reduced = _z_vec.getSubVector(index_set)
            # eigen.E.setInitialSpace(_z_reduced)
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
            # assert sum(eigs[:,0]<0) == negev, 'Inertia/eigenvalues mismatch'
            if nconv == 0 and negev>0:
                ColorPrint.print_bold('Full eigensolver did not converge but inertia yields {} neg eigen'.format(negev))
                return

            eigen.save_eigenvectors(nconv)
            # eigen.save_eigenvectors(negconv)
            
            if nconv > 0:
                ColorPrint.print_pass('')
                ColorPrint.print_pass("i        k      err     ")
                ColorPrint.print_pass("---------------------------")
                for (i, k) in enumerate(eigs):
                    ColorPrint.print_pass( "%d %12e %12e" %(i, k[0], k[1]) )
                ColorPrint.print_pass('')

            linsearch = []

            if size == 1 and debug:
                for n in range(nconv):
                    eig, u_r, u_im, err = eigen.get_eigenpair(n)
                    err2 = eigen.E.computeError(0, SLEPc.EPS.ErrorType.ABSOLUTE)
                    v_n, beta_n = u_r.split(deepcopy=True)
                    plt.clf()
                    plt.colorbar(dolfin.plot(dot(v_n, v_n)**(.5)))
                    plt.savefig('data/vn-{}-{}.pdf'.format(rank,n))

            # import pdb; pdb.set_trace()
            if negconv > 0:
                for n in range(negconv) if negconv < maxmodes else range(maxmodes):
                    ColorPrint.print_pass('Perturbation mode {}'.format(n))
                    eig, u_r, u_im, err = eigen.get_eigenpair(n)
                    err2 = eigen.E.computeError(0, SLEPc.EPS.ErrorType.ABSOLUTE)
                    v_n, beta_n = u_r.split(deepcopy=True)
                    print(rank, [self.is_compatible2(bc, u_r, homogeneous = True) for bc in self.bcs_Z])

                    if debug and size == 1:
                        plt.clf()
                        plt.colorbar(dolfin.plot(dot(v_n, v_n)**(.5)))
                        plt.savefig('data/vn-{}-{}.pdf'.format(rank, n))
                    # import pdb;pdb.set_trace()
                    # assert np.all([self.is_compatible2(bc, v_n, homogeneous = True) for bc in self.bcs[0]])
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
            # self.stable = eig.real + eigen_tol > 0  # based on eigensolve
            self.stable = negev <= 0  # based on inertia
            self.negev = negev  # based on inertia
            # negev <= 0
            # if abs(self.mineig) < eigen_tol:
                # self.stable = True
            # hx = conditional(ge(proj_beta_0, 0), , np.inf)
            # self.stable = negev <= 0
            # import pdb; pdb.set_trace()
            if eigs[0,0]<0:
                # self.hstar, estimate = self.linesearch(v_0, beta_0)
                self.perturbation_v = linsearch[0]['v_n']
                self.perturbation_beta = linsearch[0]['beta_n']
                self.hstar = linsearch[0]['hstar']
                self.en_diff = linsearch[0]['en_diff']
                self.eigendata = linsearch

            self.i +=1

        return (self.stable, int(negev))

