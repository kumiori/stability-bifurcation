import dolfin
import numpy as np
from utils import ColorPrint

import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class LineSearch(object):
    """Given an energy, a state, and a perturbation, performs returns optimal step and """
    def __init__(self, energy, state):
        super(LineSearch, self).__init__()
        self.u_0 = dolfin.Vector(state['u'].vector())
        self.alpha_0 = dolfin.Vector(state['alpha'].vector())

        self.energy = energy

    def admissible_interval(self, alpha, alpha_old, beta_n):
        one = max(1., max(alpha.vector()[:]))
        self.upperbound = one
        self.lowerbound = alpha_old
        # if hasattr(self, 'bcs') and len(self.bcs[0])>0:
        #     assert np.all([self.is_compatible(bc, v_n, homogeneous = True) for bc in self.bcs[0]]), \
        #         'displacement test field is not kinematically admissible'

        # positive part
        mask = beta_n.vector()[:]>0.

        hp2 = (one-alpha.vector()[mask])/beta_n.vector()[mask]  if len(np.where(mask==True)[0])>0 else [np.inf]
        hp1 = (alpha_old.vector()[mask]-alpha.vector()[mask])/beta_n.vector()[mask]  if len(np.where(mask==True)[0])>0 else [-np.inf]
        hp = (max(hp1), min(hp2))

        # negative part
        mask = beta_n.vector()[:]<0.

        hn2 = (one-alpha.vector()[mask])/beta_n.vector()[mask] if len(np.where(mask==True)[0])>0 else [-np.inf]
        hn1 = (alpha_old.vector()[mask]-alpha.vector()[mask])/beta_n.vector()[mask]  if len(np.where(mask==True)[0])>0 else [np.inf]
        hn = (max(hn2), min(hn1))

        hmax = np.array(np.min([hp[1], hn[1]]))
        hmin = np.array(np.max([hp[0], hn[0]]))

        hmax_glob = np.array(0.0,'d')
        hmin_glob = np.array(0.0,'d')

        comm.Allreduce(hmax, hmax_glob, op=mpi4py.MPI.MIN)
        comm.Allreduce(hmin, hmin_glob, op=mpi4py.MPI.MAX)

        hmax = float(hmax_glob)
        hmin = float(hmin_glob)

        if hmin>0:
            ColorPrint.print_warn('Line search troubles: found hmin>0')
            return 0., np.nan, (0., 0.), 0.
        if hmax==0 and hmin==0:
            ColorPrint.print_warn('Line search failed: found zero step size')
            return 0., np.nan, (0., 0.), 0.
        if hmax < hmin:
            ColorPrint.print_warn('Line search failed: optimal h* not admissible')
            return 0., np.nan, (0., 0.), 0.
            # get next perturbation mode

        return hmin, hmax

    def search(self, state, v_n, beta_n, m=3, mode=0):
        # m: order of polynomial approximation
        # mode: index of mode, for display purposes
        debug = False

        en = []
        en0 = dolfin.assemble(self.energy)

        u_0 = self.u_0
        alpha_0 = self.alpha_0

        u = state['u']
        alpha = state['alpha']

        if 'alpha_old' in state.keys():
            alpha_old = state['alpha_old']

        u_0[:] = u.vector()[:]
        alpha_0[:] = alpha.vector()[:]

        (self.hmin, self.hmax) = self.admissible_interval(alpha, alpha_old, beta_n)

        htest = np.linspace(self.hmin, self.hmax, m+1)

        for h in htest:
            uval = u_0[:]     + h*v_n.vector()[:]
            aval = alpha_0[:] + h*beta_n.vector()[:]

            if not np.all(aval - alpha_old.vector()[:] + dolfin.DOLFIN_EPS_LARGE >= 0.):
                import pdb; pdb.set_trace()
                raise Exception('Damage test field doesn\'t verify sharp irrev from below')
            if not np.all(aval <= self.upperbound):
                import pdb; pdb.set_trace()
                raise Exception('Damage test field doesn\'t verify irrev from above')

            u.vector()[:] = uval
            alpha.vector()[:] = aval

            u.vector().vec().ghostUpdate()
            alpha.vector().vec().ghostUpdate()

            en.append(dolfin.assemble(self.energy)-en0)
            # if debug and size == 1:
                # ax2.plot(xs, [self.alpha(x, 0) for x in xs], label='$\\alpha+h \\beta_{{{}}}$, h={:.3f}'.format(mode, h), lw=.5, c='C1' if h>0 else 'C4')

        z = np.polyfit(htest, en, m)
        p = np.poly1d(z)
        # import pdb; pdb.set_trace()

        if m==2:
            ColorPrint.print_info('Line search using quadratic interpolation')
            h_opt = - z[1]/(2*z[0])
        else:
            ColorPrint.print_info('Line search using polynomial interpolation (order {})'.format(m))
            h = np.linspace(self.hmin, self.hmax, 100)
            h_opt = h[np.argmin(p(h))]

        if h_opt < self.hmin or h_opt > self.hmax:
            ColorPrint.print_warn('Line search failed, h_opt={:3e} not in feasible interval'.format(h_opt))
            return h_opt, self.hmin, self.hmax

        ColorPrint.print_info('Line search h_opt = {:3f} in ({:.3f}, {:.3f}), h_opt/hmax {:3f}\
            '.format(h_opt, self.hmin, self.hmax, h_opt/self.hmax))
        ColorPrint.print_info('Line search polynomial approximation =\n {}'.format(p))
        ColorPrint.print_info('h in ({:.5f},{:.5f})'.format(self.hmin,self.hmax))
        ColorPrint.print_warn('Line search estimate, relative energy variation={:.2f}%'.format((p(h_opt))/en0*100))

        # restore solution
        u.vector()[:] = u_0[:]
        alpha.vector()[:] = alpha_0[:]

        u.vector().vec().ghostUpdate()
        alpha.vector().vec().ghostUpdate()

        # return h_opt, p(h_opt)/en0, (self.hmin, self.hmax), en
        return h_opt, (self.hmin, self.hmax), en
