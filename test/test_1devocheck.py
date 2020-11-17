# sys.path.append("../src/")
import sys
sys.path.append("../src/")
# from post_processing import compute_sig, local_project
import site
import sys

import pandas as pd

import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import mshr
import dolfin
from dolfin import MPI
import os
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

from dolfin.cpp.log import log, LogLevel, set_log_level
dolfin.parameters["std_out_all_processes"] = False

from solvers import EquilibriumSolver
from solver_stability import StabilitySolver
from linsearch import LineSearch

# from dolfin import NonlinearProblem, derivative, \
#         TrialFunction, TestFunction, inner, assemble, sqrt, \
#         Constant, interpolate, RectangleMesh, Point

from dolfin import *
import yaml

from utils import get_versions
code_parameters = get_versions()

set_log_level(LogLevel.DEBUG)




def numerical_test(
    user_parameters,
    ell=0.05,
    nu=0.,
):
    time_data = []
    time_data_pd = []
    spacetime = []
    lmbda_min_prev = 1e-6
    bifurcated = False
    bifurcation_loads = []
    save_current_bifurcation = False
    bifurc_i = 0
    bifurcation_loads = []

    # Create mesh and define function space
    geometry_parameters = {'Lx': 1., 'n': 5}

    # Define Dirichlet boundaries
    outdir = '../test/output/test_1dcheck'
    Path(outdir).mkdir(parents=True, exist_ok=True)


    with open('../parameters/form_compiler.yml') as f:
        form_compiler_parameters = yaml.load(f, Loader=yaml.FullLoader)

    with open('../parameters/solvers_default.yml') as f:
        solver_parameters = yaml.load(f, Loader=yaml.FullLoader)

    with open('../parameters/model1d.yaml') as f:
        material_parameters = yaml.load(f, Loader=yaml.FullLoader)['material']

    with open('../parameters/loading.yaml') as f:
        loading_parameters = yaml.load(f, Loader=yaml.FullLoader)['loading']

    with open('../parameters/stability.yaml') as f:
        stability_parameters = yaml.load(f, Loader=yaml.FullLoader)['stability']

    Path(outdir).mkdir(parents=True, exist_ok=True)

    print('Outdir is: '+outdir)

    default_parameters = {
        'code': {**code_parameters},
        'compiler': {**form_compiler_parameters},
        'geometry': {**geometry_parameters},
        'loading': {**loading_parameters},
        'material': {**material_parameters},
        'solver':{**solver_parameters},
        'stability': {**stability_parameters},
        }

    default_parameters.update(user_parameters)
    # FIXME: Not nice
    parameters = default_parameters

    with open(os.path.join(outdir, 'parameters.yaml'), "w") as f:
        yaml.dump(parameters, f, default_flow_style=False)

    Lx = parameters['geometry']['Lx']; 
    # Ly = parameters['geometry']['Ly']
    ell =  parameters['material']['ell']
    # comm = MPI.comm_world
    # geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))
    # import pdb; pdb.set_trace()
    # resolution = max(geometry_parameters['n'] * Lx / ell, 1/(Ly*10))
    # resolution = max(geometry_parameters['n'] * Lx / ell, 5/(Ly*10))
    resolution = 50
    mesh = dolfin.IntervalMesh(50, -Lx/2., Lx/2.)
    # mesh = dolfin.IntervalMesh(int(float(geometry_parameters['n'] * Lx / ell)), -Lx/2., Lx/2.)
    meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
    meshf << mesh
    plot(mesh)
    plt.savefig(os.path.join(outdir, "mesh.pdf"), bbox_inches='tight')

    savelag = 1
    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    # left_bottom_pt = dolfin.CompiledSubDomain("near(x[0],-Lx/2.) && near(x[1],-Ly/2.)", Lx=Lx, Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)

    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_u = dolfin.FunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = Function(V_alpha)
    dalpha = TrialFunction(V_alpha)
    alpha_old = dolfin.Function(V_alpha)
    alpha_bif = dolfin.Function(V_alpha)
    alpha_bif_old = dolfin.Function(V_alpha)


    state = {'u': u, 'alpha': alpha}
    Z = dolfin.FunctionSpace(mesh, 
            dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)
    z2 = dolfin.Function(Z)
    dz = dolfin.TestFunction(Z)
    v, beta = dolfin.split(z)

    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u, dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u, ut, right),
             # dolfin.DirichletBC(V_u, (0, 0), left_bottom_pt, method="pointwise")
             ]

    # bcs_alpha_l = DirichletBC(V_alpha,  Constant(0.0), left)
    # bcs_alpha_r = DirichletBC(V_alpha, Constant(0.0), right)
    # bcs_alpha =[bcs_alpha_l, bcs_alpha_r]
    bcs_alpha = []

    bcs = {"damage": bcs_alpha, "elastic": bcs_u}

    # import pdb; pdb.set_trace()

    ell = parameters['material']['ell']

    # Problem definition
    # Problem definition
    k_res = parameters['material']['k_res']
    a = (1 - alpha) ** 2. + k_res
    w_1 = parameters['material']['sigma_D0'] ** 2 / parameters['material']['E']
    w = w_1 * alpha
    eps = u.dx(0)
    # lmbda0 = parameters['material']['E'] * parameters['material']['nu'] /(1. - parameters['material']['nu'])**2.
    mu0 = parameters['material']['E']/ 2.
    Wu = mu0 * inner(eps, eps)

    energy = a * Wu * dx + w_1 *( alpha + \
            parameters['material']['ell']** 2.* alpha.dx(0)**2.)*dx

    def rP(z, z2):
        # w_1 = w(1)
        # z.split returns Coefficient
        # split(z) returns Indexed(Coefficient)
        # import pdb; pdb.set_trace()
        u, alpha = z.split()
        v, beta = dolfin.split(z2)
        sigma = lambda v: parameters['material']['E']*v.dx(0)
        # s = lambda alpha: a
        eps = u.dx(0)
        # eps = lambda v: v.dx(0)
        return (sqrt(a)*sigma(v) + diff(a, alpha)/sqrt(a)*sigma(u)*beta)* \
                (sqrt(a)*v.dx(0) + diff(a, alpha)/sqrt(a)*eps*beta) + \
                    2*w_1*ell ** 2 * beta.dx(0)*beta.dx(0)

    def rN(z, z2):
        u, alpha = z.split()
        v, beta = dolfin.split(z2)
        sigma = lambda v: parameters['material']['E']*v.dx(0)
        eps = u.dx(0)
        da = diff(a, alpha)
        dda = diff(diff(a, alpha), alpha)

        return -((dda - 2*da**2./a)*1./2.*inner(sigma(u), eps))*beta**2.

    # linear form
    rP = rP(z, z2)*dx
    rN = rN(z, z2)*dx

    (zu, za) = dolfin.split(z)
    mixener = ufl.replace(energy, {u: zu, alpha: za})
    Hessian = derivative(derivative(mixener, z, TestFunction(Z)), z, TrialFunction(Z))

    # import pdb; pdb.set_trace()

    eps_ = variable(eps)
    sigma = diff(a * Wu, eps_)

    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_out.parameters["functions_share_mesh"] = True
    file_out.parameters["flush_output"] = True
    file_postproc = dolfin.XDMFFile(os.path.join(outdir, "output_postproc.xdmf"))
    file_postproc.parameters["functions_share_mesh"] = True
    file_postproc.parameters["flush_output"] = True
    file_eig = dolfin.XDMFFile(os.path.join(outdir, "modes.xdmf"))
    file_eig.parameters["functions_share_mesh"] = True
    file_eig.parameters["flush_output"] = True
    file_bif = dolfin.XDMFFile(os.path.join(outdir, "bifurcation.xdmf"))
    file_bif.parameters["functions_share_mesh"] = True
    file_bif.parameters["flush_output"] = True
    file_bif_postproc = dolfin.XDMFFile(os.path.join(outdir, "bifurcation_postproc.xdmf"))
    file_bif_postproc.parameters["functions_share_mesh"] = True
    file_bif_postproc.parameters["flush_output"] = True


    solver = EquilibriumSolver(energy, state, bcs, parameters=parameters['solver'])
    stability = StabilitySolver(energy, state, bcs, parameters = parameters['stability'])
    # stability = StabilitySolver(energy, state, bcs, parameters = parameters['stability'], rayleigh=[rP, rN])
    # stability = StabilitySolver(energy, state, bcs, parameters = parameters['stability'], Hessian=Hessian)
    linesearch = LineSearch(energy, state)

    load_steps = np.linspace(parameters['loading']['load_min'],
        parameters['loading']['load_max'],
        parameters['loading']['n_steps'])
    xs = np.linspace(-parameters['geometry']['Lx']/2., parameters['geometry']['Lx']/2, 50)
    log(LogLevel.INFO, '====================== EVO ==========================')
    log(LogLevel.INFO, '{}'.format(parameters))
    for it, load in enumerate(load_steps):
        alpha_old.assign(alpha)
        log(LogLevel.CRITICAL, '====================== STEPPING ==========================')
        log(LogLevel.CRITICAL, 'CRITICAL: Solving load t = {:.2f}'.format(load))
        ut.t = load
        (time_data_i, am_iter) = solver.solve()

        # Second order stability conditions

        (stable, negev) = stability.solve(solver.damage.problem.lb)
        log(LogLevel.CRITICAL, 'Current state is{}stable'.format(' ' if stable else ' un'))

        # we postpone the update after the stability check
        solver.update()
        # if stable:
            # solver.update()
        # else:
            # perturbation_v    = stability.perturbation_v
            # perturbation_beta = stability.perturbation_beta
            # import pdb; pdb.set_trace()

            # h_opt, (hmin, hmax), energy_perturbations = linesearch.search(
            #     {'u': u, 'alpha':alpha, 'alpha_old':alpha_old},
            #     perturbation_v, perturbation_beta)

            # if h_opt != 0:
            #     save_current_bifurcation = True
            #     alpha_bif.assign(alpha)
            #     alpha_bif_old.assign(alpha_old)
            #     # admissible
            #     uval = u.vector()[:]     + h_opt * perturbation_v.vector()[:]
            #     aval = alpha.vector()[:] + h_opt * perturbation_beta.vector()[:]

            #     u.vector()[:] = uval
            #     alpha.vector()[:] = aval

            #     u.vector().vec().ghostUpdate()
            #     alpha.vector().vec().ghostUpdate()

            #     # import pdb; pdb.set_trace()
            #     (time_data_i, am_iter) = solver.solve()
            #     (stable, negev) = stability.solve(alpha_old)
            #     ColorPrint.print_pass('    Continuation iteration {}, current state is{}stable'.format(iteration, ' ' if stable else ' un'))
            #     iteration += 1

            # else:
            #     # warn
            #     ColorPrint.print_warn('WARNING: Found zero increment, we are stuck in the matrix')
                # log(LogLevel.WARNING, 'WARNING: Continuing load program')
                # break

        mineig = stability.mineig if hasattr(stability, 'mineig') else 0.0
        Deltav = (mineig-lmbda_min_prev) if hasattr(stability, 'eigs') else 0

        if (mineig + Deltav)*(lmbda_min_prev+dolfin.DOLFIN_EPS) < 0 and not bifurcated:
            bifurcated = True

            # save 3 bif modes
            log(LogLevel.CRITICAL, 'About to bifurcate load {} step {}'.format(load, it))
            bifurcation_loads.append(load)
            modes = np.where(stability.eigs < 0)[0]

            with dolfin.XDMFFile(os.path.join(outdir, "postproc.xdmf")) as file:
                leneigs = len(modes)
                maxmodes = min(3, leneigs)
                for n in range(maxmodes):
                    mode = dolfin.project(stability.linsearch[n]['beta_n'], V_alpha)
                    modename = 'beta-%d'%n
                    print(modename)
                    file.write_checkpoint(mode, modename, 0, append=True)

            bifurc_i += 1

        lmbda_min_prev = mineig if hasattr(stability, 'mineig') else 0.


        time_data_i["load"] = load
        time_data_i["alpha_max"] = max(alpha.vector()[:])
        time_data_i["elastic_energy"] = dolfin.assemble(
            1./2.* material_parameters['E']*a*eps**2. *dx)
        time_data_i["dissipated_energy"] = dolfin.assemble(
    # e1 = dolfin.Constant([1, 0])
            (w + w_1 * material_parameters['ell'] ** 2. * alpha.dx(0)**2.)*dx)
        time_data_i["stable"] = stability.stable
        time_data_i["# neg ev"] = stability.negev
        time_data_i["eigs"] = stability.eigs if hasattr(stability, 'eigs') else np.inf

        # snn = dolfin.dot(dolfin.dot(sigma, e1), e1)
        time_data_i["sigma"] = dolfin.assemble(a*mu0*eps*ds(1))

        log(LogLevel.INFO,
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
            with file_postproc as f:
                f.write_checkpoint(alpha, "alpha-{}".format(it), 0, append = True)
                log(LogLevel.PROGRESS, 'PROGRESS: written step {}'.format(it))

        if save_current_bifurcation == True:
            with file_eig as f:
                _v = dolfin.project(dolfin.Constant(h_opt)*perturbation_v, V_u)
                _beta = dolfin.project(dolfin.Constant(h_opt)*perturbation_beta, V_alpha)
                _v.rename('perturbation displacement', 'perturbation displacement')
                _beta.rename('perturbation damage', 'perturbation damage')
                # import pdb; pdb.set_trace()
                f.write(_v, load)
                f.write(_beta, load)
                f.write_checkpoint(_v, 'perturbation_v', 0, append=True)
                f.write_checkpoint(_beta, 'perturbation_beta', 0, append=True)

            time_data_pd.to_json(os.path.join(outdir, "time_data.json"))

        spacetime.append([alpha(x) for x in xs])


        if save_current_bifurcation:
            # modes = np.where(stability.eigs < 0)[0]

            time_data_i['h_opt'] = h_opt
            time_data_i['max_h'] = hmax
            time_data_i['min_h'] = hmin

            with file_bif_postproc as file:
                # leneigs = len(modes)
                # maxmodes = min(3, leneigs)
                beta0v = dolfin.project(stability.perturbation_beta, V_alpha)
                log(LogLevel.DEBUG, 'DEBUG: irrev {}'.format(alpha.vector()-alpha_old.vector()))
                file.write_checkpoint(beta0v, 'beta0', 0, append = True)
                file.write_checkpoint(alpha_bif_old, 'alpha-old', 0, append=True)
                file.write_checkpoint(alpha_bif, 'alpha-bif', 0, append=True)
                file.write_checkpoint(alpha, 'alpha', 0, append=True)

                np.save(os.path.join(outdir, 'energy_perturbations'), energy_perturbations, allow_pickle=True, fix_imports=True)

            with file_eig as file:
                _v = dolfin.project(dolfin.Constant(h_opt)*perturbation_v, V_u)
                _beta = dolfin.project(dolfin.Constant(h_opt)*perturbation_beta, V_alpha)
                _v.rename('perturbation displacement', 'perturbation displacement')
                _beta.rename('perturbation damage', 'perturbation damage')
                # import pdb; pdb.set_trace()
                f.write(_v, load)
                f.write(_beta, load)

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

    from matplotlib.ticker import FuncFormatter, MaxNLocator
    plot(alpha)
    plt.savefig(os.path.join(outdir, 'alpha.pdf'))
    log(LogLevel.INFO, "Saved figure: {}".format(os.path.join(outdir, 'alpha.pdf')))


    xs = np.linspace(-Lx/2., Lx/2., 100)
    profile = np.array([alpha(x) for x in xs])
    plt.figure()
    plt.plot(xs, profile, marker='o')
    plt.plot(xs, np.array([u(x) for x in xs]))
    # plt.ylim(0., 1.)
    plt.savefig(os.path.join(outdir, 'profile.pdf'))

    return time_data_pd, outdir


from test_firstorderevo import get_trace
# def get_trace(alpha, xresol = 100):
#     X =alpha.function_space().tabulate_dof_coordinates()
#     xs = np.linspace(min(X[:, 0]),max(X[:, 0]), xresol)
#     alpha0 = [alpha(x, 0) for x in xs]

#     return alpha0

if __name__ == "__main__":

    # Parameters
    with open('../parameters/tractionbar.yml') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    data, experiment = numerical_test(user_parameters = parameters)
    print(data)

    log(LogLevel.INFO, '________________________ VIZ _________________________')
    log(LogLevel.INFO, "Postprocess")
    import postprocess as pp

    with open(os.path.join(experiment, 'parameters.yaml')) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    lab = '\\ell={}, E={}, \\sigma_D = {}'.format(
        parameters['material']['ell'],
        parameters['material']['E'],
        parameters['material']['sigma_D0'])
    tc = (parameters['material']['sigma_D0']/parameters['material']['E'])**(.5)
    ell = parameters['material']['ell']
    # import pdb; pdb.set_trace()
    fig1, ax1 =pp.plot_energy(parameters, data, tc)
    # visuals.setspines2()
    plt.plot(data.load.values, 1-data.load.values**(-2.)*tc**2)

    # print(data['elastic_energy'])
    mu = parameters['material']['E']/2.
    elast_en = [1./2.*parameters['material']['E']*eps**2 for eps in data['load']]
    plt.plot(data['load'], elast_en, c='k', label='analytic')
    plt.axhline(1., c='k')
    plt.legend()

    plt.ylim(0, 1.)
    plt.title('${}$'.format(lab))

    fig1.savefig(os.path.join(experiment, "energy.pdf"), bbox_inches='tight')

    (fig2, ax1, ax2) =pp.plot_spectrum(parameters, data, tc)
    plt.legend(loc='lower left')
    # ax1.set_ylim(-1e-7, 2e-4)
    fig2.savefig(os.path.join(experiment, "spectrum.pdf"), bbox_inches='tight')


