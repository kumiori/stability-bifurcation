import sys
sys.path.append("../src/")
import numpy as np
import sympy
import pandas as pd
import os
import dolfin
import ufl
import matplotlib.pyplot as plt
import solvers
from damage_elasticity_model import DamagePrestrainedElasticityModel
from utils import ColorPrint, get_versions
from post_processing import make_figures, plot_global_data
# set_log_level(100)
dolfin.parameters["std_out_all_processes"] = False


dolfin.parameters["linear_algebra_backend"] = "PETSc"
from functools import reduce
from petsc4py import PETSc
import hashlib
from dolfin import MPI
from dolfin import *
import petsc4py
import post_processing as pp
from slepc_eigensolver import EigenSolver
from pathlib import Path
import json
from string import Template
from subprocess import Popen, PIPE, check_output
import numpy as np

from solver_stability import StabilitySolver
# from solver_stability_periodic import StabilitySolver
from time_stepping import TimeStepping
from copy import deepcopy
from linsearch import LineSearch

import os.path
import os

import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



form_compiler_parameters = {
    "representation": "uflacs",
    "quadrature_degree": 2,
    "optimize": True,
    "cpp_optimize": True,
}

dolfin.parameters["form_compiler"].update(form_compiler_parameters)
# np.set_printoptions(threshold=np.nan)
timestepping_parameters = {"perturbation_choice": 'steepest',
                            "savelag": 1,
                            "outdir": '',
                            'cont_rtol': 1e-5}
                        # "perturbation_choice": 'steepest',               # admissible choices: steepest, first, #

stability_parameters = {"order": 4,
                        "checkstability": True,
                        "continuation": False,
                        "projection": 'none',
                        'maxmodes': 5,
                        }

petsc_options_alpha_tao = {"tao_type": "gpcg",
                           "tao_ls_type": "gpcg",
                           "tao_gpcg_maxpgits": 50,
                           "tao_max_it": 300,
                           "tao_steptol": 1e-7,
                           "tao_gatol": 1e-5,
                           "tao_grtol": 1e-5,
                           "tao_gttol": 1e-5,
                           "tao_catol": 0.,
                           "tao_crtol": 0.,
                           "tao_ls_ftol": 1e-6,
                           "tao_ls_gtol": 1e-6,
                           "tao_ls_rtol": 1e-6,
                           "ksp_rtol": 1e-6,
                           "tao_ls_stepmin": 1e-8,  #
                           "tao_ls_stepmax": 1e6,  #
                           "pc_type": "bjacobi",
                           "tao_monitor": "",  # "tao_ls_type": "more-thuente"
                           # "ksp_type": "preonly"  # "tao_ls_type": "more-thuente"
                           }
# vinewtonrsls
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
                      "solver_alpha": "tao",
                      "solver_u": petsc_options_u,
                      # "solver_alpha_snes": petsc_options_alpha_snes
                     "solver_alpha_tao": petsc_options_alpha_tao
                     }

versions = get_versions()
versions.update({'filename': __file__})
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


# constants
ell = 0.1
Lx = 1
Ly = 0.1
load_min = 0.9
load_max = 1.1
nsteps = 10
outdir = "output"
savelag = 1
nu = dolfin.Constant(0.)
ell = dolfin.Constant(ell)
E0 = dolfin.Constant(1.0)
sigma_D0 = E0
n = 5



def traction_test(
    ell=0.05,
    ell_e=.1,
    degree=1,
    n=3,
    nu=0.,
    load_min=0,
    load_max=2,
    loads=None,
    nsteps=20,
    Lx=1.,
    Ly=0.1,
    outdir="outdir",
    postfix='',
    savelag=1,
    sigma_D0=1.,
    periodic=False,
    continuation=False,
    checkstability=True,
    configString='',
    test=True
):
    # constants
    # ell = ell
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
    # h = ell.values()[0]/n
    h = max(ell.values()[0]/n, .005)
    cell_size = h
    continuation = continuation
    isPeriodic = periodic
    config = json.loads(configString) if configString != '' else ''

    cmd_parameters =  {
    'material': {
        "ell": ell.values()[0],
        "ell_e": ell_e,
        "K": K,
        "E": E.values()[0],
        "nu": nu.values()[0],
        "sigma_D0": sigma_D0.values()[0]},
    'geometry': {
        'Lx': Lx,
        'Ly': Ly,
        'n': n,
        },
    'experiment': {
        'test': test,
        'periodic': isPeriodic,
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



    # --------------------

    for par in parameters: parameters[par].update(cmd_parameters[par])

    if config:
        # import pdb; pdb.set_trace()
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
    
    if parameters['experiment']['test'] == True: outdir += '-{}'.format(cmd_parameters['time_stepping']['postfix'])
    else: outdir += '-{}{}'.format(signature, cmd_parameters['time_stepping']['postfix'])
    
    parameters['time_stepping']['outdir']=outdir
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
    
# ----------------
    geometry_parameters = parameters['geometry']

    geom_signature = hashlib.md5(str(geometry_parameters).encode('utf-8')).hexdigest()
    # cmd_parameters['experiment']['signature']=signature
    meshsize = parameters['material']['ell']/parameters['geometry']['n']
    d={'Lx': parameters['geometry']['Lx'], 'Ly': parameters['geometry']['Ly'],
        'meshsize': meshsize}
    resolution = max(geometry_parameters['n'] * Lx / ell, 5/(Ly*10))
    resolution = 50

    comm = MPI.comm_world
    geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))
    mesh = mshr.generate_mesh(geom,  resolution)
    meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
    meshf << mesh
    plot(mesh)
    plt.savefig(os.path.join(outdir, "mesh.pdf"), bbox_inches='tight')

    print(meshfile)

    mesh_xdmf = dolfin.XDMFFile("meshes/%s-%s.xdmf"%(fname, geom_signature))
    mesh_xdmf.write(mesh)
    if rank == 0: 
        meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
        meshf << mesh



    V_u = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = dolfin.Function(V_alpha, name="Damage")

    bcs_alpha = []
    bcs_u = [DirichletBC(V_u, Constant((0., 0.)), 'on_boundary')]

    # left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    # right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    # bottom = dolfin.CompiledSubDomain("near(x[1],-Ly/2.)", Ly=Ly)
    # top = dolfin.CompiledSubDomain("near(x[1],Ly/2.)", Ly=Ly)

    # mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    # right.mark(mf, 1)
    # left.mark(mf, 2)
    # bottom.mark(mf, 3)
    state = {'u': u, 'alpha': alpha}
    bcs = {"damage": bcs_alpha, "elastic": bcs_u}

    Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)

    v, beta = dolfin.split(z)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)
    # ds = dolfin.Measure("ds", subdomain_data=mf)
    ds = dolfin.Measure("ds")

    # Files for output
    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_eig = dolfin.XDMFFile(os.path.join(outdir, "perturbations.xdmf"))
    file_con = dolfin.XDMFFile(os.path.join(outdir, "continuation.xdmf"))
    file_bif = dolfin.XDMFFile(os.path.join(outdir, "bifurcation_postproc.xdmf"))
    file_postproc = dolfin.XDMFFile(os.path.join(outdir, "postprocess.xdmf"))

    for f in [file_out, file_eig, file_con, file_bif, file_postproc]:
        f.parameters["functions_share_mesh"] = True
        f.parameters["flush_output"] = True

    # Problem definition
    k_res = parameters['material']['k_res']
    a = (1 - alpha) ** 2. + k_res
    w_1 = parameters['material']['sigma_D0'] ** 2 / parameters['material']['E']
    w = w_1 * alpha
    eps = sym(grad(u))
    eps0t=Expression([['t', 0.],[0.,'t']], t=0., degree=0)
    lmbda0 = parameters['material']['E'] * parameters['material']['nu'] /(1. - parameters['material']['nu'])**2.
    mu0 = parameters['material']['E']/ 2. / (1.0 + parameters['material']['nu'])
    nu = parameters['material']['nu']
    Wt = a*parameters['material']['E']*nu/(2*(1-nu**2.)) * tr(eps-eps0t)**2.                                \
        + a*parameters['material']['E']/(2.*(1+nu))*(inner(eps-eps0t, eps-eps0t))                           \
    + 1./2.*1./parameters['material']['ell_e']**2.*dot(u, u)

    # energy = Wt * dx + w_1 *( alpha + parameters['material']['ell']** 2.*inner(grad(alpha), grad(alpha)))*dx

    ell = parameters['material']['ell']
    ell_e = parameters['material']['ell_e']
    E = parameters['material']['E']

    def elastic_energy(u,alpha, E=E, nu=nu, ell_e=ell_e, eps0t=eps0t, k_res=k_res):
        a = (1 - alpha) ** 2. + k_res
        eps = sym(grad(u))
        Wt = a*E*nu/(2*(1-nu**2.)) * tr(eps-eps0t)**2.                                \
            + a*E/(2.*(1+nu))*(inner(eps-eps0t, eps-eps0t))                           \
            + 1./2.*1./ell_e**2.*dot(u, u)
        return Wt * dx 

    def dissipated_energy(alpha,w_1=w_1,ell=ell):
        return w_1 *( alpha + ell** 2.*inner(grad(alpha), grad(alpha)))*dx

    def total_energy(u, alpha, k_res=k_res, w_1=w_1, 
                            E=E, 
                            nu=nu, 
                            ell_e=ell_e,
                            ell=ell,
                            eps0t=eps0t):
        elastic_energy_ = elastic_energy(u,alpha, E=E, nu=nu, ell_e=ell_e, eps0t=eps0t, k_res=k_res)
        dissipated_energy_ = dissipated_energy(alpha,w_1=w_1,ell=ell)
        return elastic_energy_ + dissipated_energy_

    energy = total_energy(u,alpha)

    # -------------------


    solver = EquilibriumAM(energy, state, bcs, parameters=parameters)
    stability = StabilitySolver(energy, state, bcs, parameters = parameters, Hessian = Hessian)
    # import pdb; pdb.set_trace()
    # equilibrium = EquilibriumNewton(energy, state, bcs, parameters = parameters)
    # stability = StabilitySolver(energy, state, bcs, parameters = parameters['stability'], rayleigh= [rP, rN])
    linesearch = LineSearch(energy, state)



    load_steps = np.linspace(parameters['loading']['load_min'],
        parameters['loading']['load_max'],
        parameters['loading']['n_steps'])


    time_data = []
    time_data_pd = []
    spacetime = []
    lmbda_min_prev = 1e-6
    bifurcated = False
    bifurcation_loads = []
    save_current_bifurcation = False
    bifurc_i = 0
    alpha_bif = dolfin.Function(V_alpha)
    alpha_bif_old = dolfin.Function(V_alpha)
    bifurcation_loads = []
    to_remove = []

    from matplotlib import cm

    log(LogLevel.INFO, '{}'.format(parameters))
    for it, load in enumerate(load_steps):
        plt.clf()
        mineigs = []
        exhaust_modes = []

        log(LogLevel.CRITICAL, '====================== STEPPING ==========================')
        log(LogLevel.CRITICAL, 'CRITICAL: Solving load t = {:.2f}'.format(load))
        alpha_old.assign(alpha)
        eps0t.t = load
        # (time_data_i, am_iter) = solver.solve(outdir)
        (time_data_i, am_iter) = solver.solve()

        # Second order stability conditions
        (stable, negev) = stability.solve(solver.damage.problem.lb)

        log(LogLevel.CRITICAL, 'Current state is{}stable'.format(' ' if stable else ' un'))

        mineig = stability.mineig if hasattr(stability, 'mineig') else 0.0
        # log(LogLevel.INFO, 'INFO: lmbda min {}'.format(lmbda_min_prev))
        log(LogLevel.INFO, 'INFO: mineig {:.5e}'.format(mineig))
        Deltav = (mineig-lmbda_min_prev) if hasattr(stability, 'eigs') else 0

        if (mineig + Deltav)*(lmbda_min_prev+dolfin.DOLFIN_EPS) < 0 and not bifurcated:
            bifurcated = True

            # save 3 bif modes
            log(LogLevel.INFO, 'INFO: About to bifurcate load {:.3f} step {}'.format(load, step))
            bifurcation_loads.append(load)
            modes = np.where(stability.eigs < 0)[0]

            bifurc_i += 1

        lmbda_min_prev = mineig if hasattr(stability, 'mineig') else 0.

        # we postpone the update after the stability check
        if stable:
            solver.update()
            log(LogLevel.INFO,'    Current state is{}stable'.format(' ' if stable else ' un'))
        else:
            # Continuation
            iteration = 1
            mineigs.append(stability.mineig)

            while stable == False:
                log(LogLevel.INFO, 'Continuation iteration {}'.format(iteration))
                plt.close('all')
                pert = [(_v, _b) for _v, _b in zip(stability.perturbations_v, stability.perturbations_beta)]
                _nmodes = len(pert)
                en_vars = []
                h_opts = []
                hbounds = []
                en_perts = []

                for i,mode in enumerate(pert):
                    h_opt, bounds, enpert, en_var = linesearch.search(
                        {'u':u, 'alpha':alpha, 'alpha_old': alpha_old},
                        mode[0], mode[1])
                    h_opts.append(h_opt)
                    en_vars.append(en_var)
                    hbounds.append(bounds)
                    en_perts.append(enpert)
                # import pdb; pdb.set_trace()

                # if False:
                if rank == 0:
                    fig = plt.figure(dpi=80, facecolor='w', edgecolor='k')
                    plt.subplot(1, 4, 1)
                    plt.set_cmap('binary')
                    # dolfin.plot(mesh, alpha = 1.)
                    plt.colorbar(dolfin.plot(
                        project(stability.inactivemarker1, L2), alpha = 1., vmin=0., vmax=1.))
                    plt.title('derivative zero')
                    plt.subplot(1, 4, 2)
                    # dolfin.plot(mesh, alpha = .5)
                    plt.colorbar(dolfin.plot(
                        project(stability.inactivemarker2, L2), alpha = 1., vmin=0., vmax=1.))
                    plt.title('ub tolerance')
                    plt.subplot(1, 4, 3)
                    # dolfin.plot(mesh, alpha = .5)
                    plt.colorbar(dolfin.plot(
                        project(stability.inactivemarker3, L2), alpha = 1., vmin=0., vmax=1.))
                    plt.title('alpha-alpha_old')
                    plt.subplot(1, 4, 4)
                    # dolfin.plot(mesh, alpha = .5)
                    plt.colorbar(dolfin.plot(
                        project(stability.inactivemarker4, L2), alpha = 1., vmin=0., vmax=1.))
                    plt.title('intersec deriv, ub')
                    plt.savefig(os.path.join(outdir, "{:.3f}-inactivesets-{:d}.pdf".format(load, iteration)))

                    plt.set_cmap('hot')

                    for i,mode in enumerate(pert):
                        plt.subplot(2, _nmodes+1, i+2)
                        plt.axis('off')
                        plot(mode[1], cmap = cm.ocean)

                        # plt.title('mode {} $h^*$={:.3f}\n $\\lambda_{}$={:.3e} \n $\\Delta E$={:.3e}'
                        #     .format(i, h_opts[i], i, stability.eigs[i], en_vars[i]), fontsize= 15)

                        plt.title('mode {}'
                            .format(i), fontsize= 15)

                        plt.subplot(2, _nmodes+1, _nmodes+2+1+i)
                        plt.axis('off')
                        _pert_beta = mode[1]
                        _pert_v = mode[0]

                        if hbounds[i][0] == hbounds[i][1] == 0:
                            plt.plot(hbounds[i][0], 0)
                        else:
                            hs = np.linspace(hbounds[i][0], hbounds[i][1], 100)
                            z = np.polyfit(np.linspace(hbounds[i][0], hbounds[i][1],
                                len(en_perts[i])), en_perts[i], parameters['stability']['order'])
                            p = np.poly1d(z)
                            plt.plot(hs, p(hs), c='k')
                            plt.plot(np.linspace(hbounds[i][0], hbounds[i][1],
                                len(en_perts[i])), en_perts[i], marker='o', markersize=10, c='k')
                            # import pdb; pdb.set_trace()
                            plt.plot(hs, stability.eigs[i]*hs**2, c='r', lw=.3)
                            plt.axvline(h_opts[i], lw = .3, c='k')
                            plt.axvline(0, lw=2, c='k')
                        # plt.title('{}'.format(i))
                        plt.tight_layout(h_pad=1.5, pad=1.5)
                    # plt.legend()
                    plt.savefig(os.path.join(outdir, "{:.3f}-modes-{}.pdf".format(load, iteration)))
                    plt.close(fig)
                    plt.clf()
                    log(LogLevel.INFO, 'plotted modes')

                cont_data_pre = compile_continuation_data(state, energy)

                log(LogLevel.INFO, 'Estimated energy variation {:.3e}'.format(en_var))

                Ealpha = Function(V_alpha)
                Ealpha.vector()[:]=assemble(stability.inactiveEalpha)[:]
                Ealpha.rename('Ealpha-{}'.format(iteration), 'Ealpha-{}'.format(iteration))

                with file_ealpha as file:
                    file.write(Ealpha, load)

                save_current_bifurcation = True

                # pick the first of the non exhausted modes-

                opt_mode = 0
                # opt_mode = np.argmin(en_vars)
                log(LogLevel.INFO, 'Energy vars {}'.format(en_vars))
                log(LogLevel.INFO, 'Pick bifurcation mode {} out of {}'.format(opt_mode, len(en_vars)))
                h_opt = min(h_opts[opt_mode],1.e-2)
                perturbation_v    = stability.perturbations_v[opt_mode]
                perturbation_beta = stability.perturbations_beta[opt_mode]
                minmode = stability.minmode
                (perturbation_v, perturbation_beta) = minmode.split(deepcopy=True)
                # (perturbation_v, perturbation_beta) = stability.perturbation_v, stability.perturbation_beta

                def energy_1d(h):
                    #return assemble(energy_functional(u + h * perturbation_v, alpha + h * perturbation_beta))
                    u_ = Function(u.function_space())
                    alpha_ = Function(alpha.function_space())
                    u_.vector()[:] = u.vector()[:] + h * perturbation_v.vector()[:]
                    alpha_.vector()[:] = alpha.vector()[:] + h * perturbation_beta.vector()[:]
                    u_.vector().vec().ghostUpdate()
                    alpha_.vector().vec().ghostUpdate()
                    return assemble(total_energy(u_, alpha_))

                (hmin, hmax) = linesearch.admissible_interval(alpha, alpha_old, perturbation_beta)
                hs = np.linspace(hmin,hmax,20)
                energy_vals = np.array([energy_1d(h) for h in hs])
                stability.solve(solver.damage.problem.lb)

                Hzz = assemble(stability.H*minmode*minmode)
                Gz = assemble(stability.J*minmode)
                mineig_z = Hzz/assemble(dot(minmode,minmode)*dx)

                energy_vals_quad = energy_1d(0) + hs*Gz + hs**2*Hzz/2
                h_opt = hs[np.argmin(energy_vals)]
                print(h_opt)
                print("%%%%%%%%% ",mineig_z,"-",mineig)
                if rank == 0:
                    plt.figure()
                    plt.plot(hs,energy_vals, marker = 'o')
                    plt.plot(hs,energy_vals,label="exact")
                    plt.plot(hs,energy_vals_quad,label="quadratic approximation")
                    plt.legend()
                    plt.title("eig {:.4f} vs {:.4f} expected".format(mineig_z, mineig))
                    plt.axvline(h_opt)
                    # import pdb; pdb.set_trace()
                    plt.savefig(os.path.join(outdir, "energy1d-{:.3f}.pdf".format(load)))


                iteration += 1
                log(LogLevel.CRITICAL, 'Bifurcating')

                save_current_bifurcation = True
                alpha_bif.assign(alpha)
                alpha_bif_old.assign(alpha_old)

                # admissible perturbation
                uval = u.vector()[:]     + h_opt * perturbation_v.vector()[:]
                aval = alpha.vector()[:] + h_opt * perturbation_beta.vector()[:]

                u.vector()[:] = uval
                alpha.vector()[:] = aval
                u.vector().vec().ghostUpdate()
                alpha.vector().vec().ghostUpdate()

                log(LogLevel.INFO, 'min a+h_opt beta_{} = {}'.format(opt_mode, min(aval)))
                log(LogLevel.INFO, 'max a+h_opt beta_{} = {}'.format(opt_mode, max(aval)))
                log(LogLevel.INFO, 'Solving equilibrium from perturbed state')
                (time_data_i, am_iter) = solver.solve(outdir)

                log(LogLevel.INFO, 'Checking stability of new state')
                (stable, negev) = stability.solve(solver.damage.problem.lb)
                mineigs.append(stability.mineig)

                log(LogLevel.INFO, 'Continuation iteration {}, current state is{}stable'.format(iteration, ' ' if stable else ' un'))

                cont_data_post = compile_continuation_data(state, energy)
                DeltaE = (cont_data_post['energy']-cont_data_pre['energy'])
                relDeltaE = (cont_data_post['energy']-cont_data_pre['energy'])/cont_data_pre['energy']
                release = DeltaE < 0 and np.abs(DeltaE) > parameters['stability']['cont_rtol']

                log(LogLevel.INFO, 'Continuation: post energy {} - pre energy {}'.format(cont_data_post['energy'], cont_data_pre['energy']))
                log(LogLevel.INFO, 'Actual absolute energy variation Delta E = {:.7e}'.format(DeltaE))
                log(LogLevel.INFO, 'Actual relative energy variation relDelta E = {:.7e}'.format(relDeltaE))
                log(LogLevel.INFO, 'Iter {} mineigs = {}'.format(iteration, mineigs))

                if rank == 0:
                    plt.plot(mineigs, marker = 'o')
                    plt.axhline(0.)
                    plt.savefig(os.path.join(outdir, "mineigs-{:.3f}.pdf".format(load)))

                if abs(np.diff(mineigs)[-1]) > 1e-8:
                    log(LogLevel.INFO, 'Min eig change = {:.3e}'.format(np.diff(mineigs)[-1]))
                    log(LogLevel.INFO, 'Continuing perturbations')
                else:
                    log(LogLevel.INFO, 'Min eig change = {:.3e}'.format(np.diff(mineigs)[-1]))
                    log(LogLevel.CRITICAL, 'We are stuck in the matrix')
                    log(LogLevel.WARNING, 'Exploring next mode')
                    exhaust_modes.append(opt_mode)

            solver.update()
            log(LogLevel.INFO, 'bifurcation loads : {}'.format(bifurcation_loads))
            np.save(os.path.join(outdir, 'bifurcation_loads'), bifurcation_loads, allow_pickle=True, fix_imports=True)

            if save_current_bifurcation:
                time_data_i['h_opt'] = h_opt
                time_data_i['max_h'] = hbounds[opt_mode][1]
                time_data_i['min_h'] = hbounds[opt_mode][0]

                modes = np.where(stability.eigs < 0)[0]
                leneigs = len(modes)
                maxmodes = min(3, leneigs)

                with file_bif as file:
                    for n in range(len(pert)):
                        mode = dolfin.project(stability.perturbations_beta[n], V_alpha)
                        modename = 'beta-%d'%n
                        mode.rename(modename, modename)
                        log(LogLevel.INFO, 'Saved mode {}'.format(modename))
                        file.write(mode, load)

                np.save(os.path.join(outdir, 'energy_perturbations'), en_perts, allow_pickle=True, fix_imports=True)


                with file_eig as file:
                    _v = dolfin.project(dolfin.Constant(h_opt)*perturbation_v, V_u)
                    _beta = dolfin.project(dolfin.Constant(h_opt)*perturbation_beta, V_alpha)
                    _v.rename('perturbation displacement', 'perturbation displacement')
                    _beta.rename('perturbation damage', 'perturbation damage')
                    file.write(_v, load)
                    file.write(_beta, load)

        time_data_i["load"] = load
        time_data_i["stable"] = stable
        time_data_i["dissipated_energy"] = dolfin.assemble(
            model.damage_dissipation_density(alpha)*dx)
        time_data_i["foundation_energy"] = dolfin.assemble(
            1./2.*1/ell_e**2. * dot(u, u)*dx)
        time_data_i["membrane_energy"] = dolfin.assemble(
            model.elastic_energy_density(model.eps(u), alpha)*dx)
        time_data_i["elastic_energy"] = time_data_i["membrane_energy"]+time_data_i["foundation_energy"]
        time_data_i["eigs"] = stability.eigs if hasattr(stability, 'eigs') else np.inf
        time_data_i["stable"] = stability.stable
        time_data_i["# neg ev"] = stability.negev
        # import pdb; pdb.set_trace()

        _sigma = model.stress(model.eps(u), alpha)
        e1 = dolfin.Constant([1, 0])
        _snn = dolfin.dot(dolfin.dot(_sigma, e1), e1)
        time_data_i["sigma"] = 1/Ly * dolfin.assemble(_snn*model.ds(1))

        time_data_i["S(alpha)"] = dolfin.assemble(1./(model.a(alpha))*model.dx)
        time_data_i["A(alpha)"] = dolfin.assemble((model.a(alpha))*model.dx)
        time_data_i["avg_alpha"] = dolfin.assemble(alpha*model.dx)

        ColorPrint.print_pass(
            "Time step {:.4g}: it {:3d}, err_alpha={:.4g}".format(
                time_data_i["load"],
                time_data_i["iterations"],
                time_data_i["alpha_error"]))

        time_data.append(time_data_i)
        time_data_pd = pd.DataFrame(time_data)

        if np.mod(it, savelag) == 0:
            with file_out as f:
                f.write(alpha, load)
                f.write(u, load)
            # with file_out as f:
                f.write_checkpoint(alpha, "alpha-{}".format(it), 0, append = True)
                print('DEBUG: written step ', it)

        if save_current_bifurcation:
            # modes = np.where(stability.eigs < 0)[0]

            time_data_i['h_opt'] = h_opt
            time_data_i['max_h'] = hmax
            time_data_i['min_h'] = hmin

            with file_bif as file:
                # leneigs = len(modes)
                # maxmodes = min(3, leneigs)
                beta0v = dolfin.project(stability.perturbation_beta, V_alpha)
                print('DEBUG: irrev ', alpha.vector()-alpha_old.vector())
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
                file.write_checkpoint(_v, 'perturbation_v', 0, append=True)
                file.write_checkpoint(_beta, 'perturbation_beta', 0, append=True)

        time_data_pd.to_json(os.path.join(outdir, "time_data.json"))
        # user_postprocess_timestep(alpha, parameters, load, xresol = 100)

    plt.figure()
    dolfin.plot(alpha)
    plt.savefig(os.path.join(outdir, "alpha.png"))
    plt.figure()
    dolfin.plot(u, mode="displacement")
    plt.savefig(os.path.join(outdir, "u.png"))
    _nu = parameters['material']['nu']
    _E = parameters['material']['E']
    _w1 = parameters['material']['sigma_D0']**2. / parameters['material']['E']

    tc = np.sqrt(2*_w1/(_E*(1.-2.*_nu)*(1.+_nu)))
    if parameters['stability']['checkstability']=='True':
        pp.plot_spectrum(parameters, outdir, time_data_pd.sort_values('load'), tc)
    # plt.show()
    print(time_data_pd)
    return time_data_pd, outdir


if __name__ == "__main__":
    # Parameters
    with open('../parameters/film.yaml') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    data, experiment = numerical_test(user_parameters = parameters)
    print(data)

    log(LogLevel.INFO, '________________________ VIZ _________________________')
    log(LogLevel.INFO, "Postprocess")
    import postprocess as pp

    with open(os.path.join(experiment, 'parameters.yaml')) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    import mpi4py
    from dolfin import list_timings, TimingType, TimingClear

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        lab = '\\ell={}, ell_e={}, E={}, \\sigma_D = {}'.format(
            parameters['material']['ell'],
            parameters['material']['ell_e'],
            parameters['material']['E'],
            parameters['material']['sigma_D0'])
        tc = (parameters['material']['sigma_D0']/parameters['material']['E'])**(.5)
        tc = sqrt(2.)/2.
        ell = parameters['material']['ell']
        # import pdb; pdb.set_trace()
        fig1, ax1 =pp.plot_energy(parameters, data, tc)
        # from math import tanh
        # L = parameters['geometry']['R']
        # F_L = L/2. - tanh(L/2.)
        # elast_en = [1./2.*parameters['material']['E']*t**2*F_L for t in data['load']]
        # visuals.setspines2()
        # print(data['elastic_energy'])
        # mu = parameters['material']['E']/2.
        # elast_en = [1./2.*2.*mu*eps**2 for eps in data['load']]
        # Lx = 1.
        # Ly = .1
        # Omega = Lx*Ly
        # elast_en = [parameters['material']['E']*eps**2 for eps in data['load']]
        # plt.plot(data['load'], elast_en, c='k', label='analytic')
        # plt.axhline(parameters['geometry']['Ly'], c='k')
        # plt.legend()

        # plt.ylim(0, 1.)
        # plt.title('${}$'.format(lab))

        fig1.savefig(os.path.join(experiment, "energy.pdf"), bbox_inches='tight')

        (fig2, ax1, ax2) =pp.plot_spectrum(parameters, data, tc)
        plt.legend(loc='lower left')
        # ax2.set_ylim(-1e-7, 2e-4)
        fig2.savefig(os.path.join(experiment, "spectrum.pdf"), bbox_inches='tight')

        list_timings(TimingClear.keep, [TimingType.wall, TimingType.system])

        dump_timings_to_xml(os.path.join(experiment, "timings_avg_min_max.xml"), TimingClear.clear)


