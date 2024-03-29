import dolfin
import numpy as np
# from utils import log
import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from dolfin.cpp.log import log, LogLevel, set_log_level


class LineSearch(object):
    """Given an energy, a state, and a perturbation, performs returns optimal step and """
    def __init__(self, energy, state):
        super(LineSearch, self).__init__()
        self.u_0 = dolfin.Vector(state['u'].vector())
        self.alpha_0 = dolfin.Vector(state['alpha'].vector())

        self.energy = energy

    def admissible_interval(self, alpha, alpha_old, beta):
        one = max(1., max(alpha.vector()[:]))
        self.upperbound = one
        self.lowerbound = alpha_old
        # if hasattr(self, 'bcs') and len(self.bcs[0])>0:
        #     assert np.all([self.is_compatible(bc, v, homogeneous = True) for bc in self.bcs[0]]), \
        #         'displacement test field is not kinematically admissible'

        # positive part
        mask = beta.vector()[:]>0.

        hp2 = (one-alpha.vector()[mask])/beta.vector()[mask]  if len(np.where(mask==True)[0])>0 else [np.inf]
        hp1 = (alpha_old.vector()[mask]-alpha.vector()[mask])/beta.vector()[mask]  if len(np.where(mask==True)[0])>0 else [-np.inf]
        hp = (max(hp1), min(hp2))

        # negative part
        mask = beta.vector()[:]<0.

        hn2 = (one-alpha.vector()[mask])/beta.vector()[mask] if len(np.where(mask==True)[0])>0 else [-np.inf]
        hn1 = (alpha_old.vector()[mask]-alpha.vector()[mask])/beta.vector()[mask]  if len(np.where(mask==True)[0])>0 else [np.inf]
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
            log(LogLevel.INFO, 'Line search troubles: found hmin>0')
            return (0., 0.)
        if hmax==0 and hmin==0:
            log(LogLevel.INFO, 'Line search failed: found zero step size')
            # import pdb; pdb.set_trace()
            return (0., 0.)
        if hmax < hmin:
            log(LogLevel.INFO, 'Line search failed: optimal h* not admissible')
            return (0., 0.)
            # get next perturbation mode

        return hmin, hmax

    def search(self, state, v, beta, m=3, mode=0):
        # m: order of polynomial approximation
        # mode: index of mode, for display purposes
        log(LogLevel.INFO,'')
        debug = False
        # FIXME: straighten interface: get rid of n dependence and put into a dictionary
        # v = perturbation['v']
        # beta = perturbation['beta']
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

        (self.hmin, self.hmax) = self.admissible_interval(alpha, alpha_old, beta)

        htest = np.linspace(self.hmin, self.hmax, m+1)


        if self.hmin == 0. and self.hmax == 0.:
            # import pdb; pdb.set_trace()   

            return 0., (0., 0.), [], 0
        else: 
            for h in htest:
                uval = u_0[:]     + h*v.vector()[:]
                aval = alpha_0[:] + h*beta.vector()[:]

                if not np.all(aval - alpha_old.vector()[:] + dolfin.DOLFIN_EPS_LARGE >= 0.):
                    raise Exception('Damage test field doesn\'t verify sharp irrev from below')
                if not np.all(aval <= self.upperbound):
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

            if m==2:
                log(LogLevel.INFO, 'Line search using quadratic interpolation')
                h_opt = - z[1]/(2*z[0])
            else:
                log(LogLevel.INFO, 'Line search using polynomial interpolation (order {})'.format(m))
                h = np.linspace(self.hmin, self.hmax, 100)
                h_opt = h[np.argmin(p(h))]

            if h_opt < self.hmin or h_opt > self.hmax:
                log(LogLevel.INFO, 'Line search failed, h_opt={:3e} not in feasible interval'.format(h_opt))
                return h_opt, (self.hmin, self.hmax), 0

        log(LogLevel.INFO, 'Line search h_opt = {:3f} in ({:.3f}, {:.3f})\
            '.format(h_opt, self.hmin, self.hmax))
        # log(LogLevel.INFO, 'Line search polynomial approximation =\n {}'.format(p))
        # log(LogLevel.INFO, 'h in ({:.5f},{:.5f})'.format(self.hmin,self.hmax))
        # log(LogLevel.INFO, 'p(h_opt) {:.5f}, en0 {:.5f}'.format(p(h_opt),en0))
        en_var = ((p(h_opt))/en0)
        log(LogLevel.INFO, 'Line search estimate, relative energy variation={:.3e}'.format(en_var))

        # restore solution
        u.vector()[:] = u_0[:]
        alpha.vector()[:] = alpha_0[:]

        u.vector().vec().ghostUpdate()
        alpha.vector().vec().ghostUpdate()

        # return h_opt, p(h_opt)/en0, (self.hmin, self.hmax), en
        return h_opt, (self.hmin, self.hmax), en, en_var
