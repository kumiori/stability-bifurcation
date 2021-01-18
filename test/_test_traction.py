# sys.path.append("../src/")
import sys
sys.path.append("../src/")
sys.path.append("../scripts/")
# from post_processing import compute_sig, local_project
import site
import sys

import pandas as pd

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import mshr
import dolfin
from dolfin import MPI
import os
import sympy
import numpy as np
# import post_processing as pp
import petsc4py
from functools import reduce
import ufl
from string import Template

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

from solvers import EquilibriumAM
# EquilibriumNewton
from solver_stability import StabilitySolver
from linsearch import LineSearch

from dolfin import *
import yaml
import mshr
from lib import create_output, compile_continuation_data, getDefaultParameters

from utils import get_versions
code_parameters = get_versions()

set_log_level(LogLevel.INFO)


def numerical_test(
    user_parameters
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
    # Define Dirichlet boundaries
    comm = MPI.comm_world

    default_parameters = getDefaultParameters()
    default_parameters.update(user_parameters)
    # FIXME: Not nice
    parameters = default_parameters
    parameters['code']['script'] = __file__
    # import pdb; pdb.set_trace()

    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
    outdir = '../output/traction/{}-{}CPU'.format(signature, size)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    log(LogLevel.INFO, 'INFO: Outdir is: '+outdir)
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    print(parameters['geometry'])
    d={'Lx': parameters['geometry']['Lx'],'Ly': parameters['geometry']['Ly'],
        'h': parameters['material']['ell']/parameters['geometry']['n']}

    geom_signature = hashlib.md5(str(d).encode('utf-8')).hexdigest()


    # --------------------------------------------------------
    # Mesh creation with gmsh
    Lx = parameters['geometry']['Lx']
    Ly = parameters['geometry']['Ly']
    n = parameters['geometry']['n']
    ell = parameters['material']['ell']
    fname = os.path.join('../meshes', 'strip-{}'.format(geom_signature))

    resolution = max(parameters['geometry']['n'] * Lx / ell, 5/(Ly*10))
    resolution = 100

    geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))
    # mesh = mshr.generate_mesh(geom, n * int(float(Lx / ell)))
    mesh = mshr.generate_mesh(geom, resolution)

    log(LogLevel.INFO, 'Number of dofs: {}'.format(mesh.num_vertices()*(1+parameters['general']['dim'])))
    if size == 1:
        meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
        plot(mesh)
        plt.savefig(os.path.join(outdir, "mesh.pdf"), bbox_inches='tight')

    with open(os.path.join(outdir, 'parameters.yaml'), "w") as f:
        yaml.dump(parameters, f, default_flow_style=False)

    Lx = parameters['geometry']['Lx']
    ell =  parameters['material']['ell']
    savelag = 1
    # mf = dolfin.MeshFunction("size_t", mesh, 1, 0)

    # Function Spaces
    V_u = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    L2 = dolfin.FunctionSpace(mesh, "DG", 0)
    u = dolfin.Function(V_u, name="Total displacement")
    u.rename('u', 'u')
    alpha = Function(V_alpha)
    alpha_old = dolfin.Function(alpha.function_space())
    alpha.rename('alpha', 'alpha')
    dalpha = TrialFunction(V_alpha)
    alpha_bif = dolfin.Function(V_alpha)
    alpha_bif_old = dolfin.Function(V_alpha)


    state = {'u': u, 'alpha': alpha}
    Z = dolfin.FunctionSpace(mesh,
            dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)
    v, beta = dolfin.split(z)
    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    bottom = dolfin.CompiledSubDomain("near(x[1],-Ly/2.)", Ly=Ly)
    top = dolfin.CompiledSubDomain("near(x[1],Ly/2.)", Ly=Ly)
    left_bottom_pt = dolfin.CompiledSubDomain("near(x[0],-Lx/2.) && near(x[1],-Ly/2.)", Lx=Lx, Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)
    bottom.mark(mf, 3)
    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u.sub(0), dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u.sub(0), ut, right),
             dolfin.DirichletBC(V_u, (0, 0), left_bottom_pt, method="pointwise")]
    bcs_alpha = []

    bcs = {"damage": bcs_alpha, "elastic": bcs_u}

    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=parameters['compiler'], domain=mesh)

    ell = parameters['material']['ell']

    # -----------------------
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
    sigma0 = lmbda0 * tr(eps)*dolfin.Identity(parameters['general']['dim']) + 2*mu0*eps
    e1 = Constant((1., 0))
    _sigma = ((1 - alpha) ** 2. + k_res)*sigma0
    _snn = dolfin.dot(dolfin.dot(_sigma, e1), e1)

    # -------------------

    ell = parameters['material']['ell']
    E = parameters['material']['E']

    def elastic_energy(u,alpha, E=E, nu=nu, eps0t=eps0t, k_res=k_res):
        a = (1 - alpha) ** 2. + k_res
        eps = sym(grad(u))
        Wt = a*E*nu/(2*(1-nu**2.)) * tr(eps)**2.                                \
            + a*E/(2.*(1+nu))*(inner(eps, eps))
        return Wt * dx 

    def dissipated_energy(alpha,w_1=w_1,ell=ell):
        return w_1 *( alpha + ell** 2.*inner(grad(alpha), grad(alpha)))*dx

    def total_energy(u, alpha, k_res=k_res, w_1=w_1, 
                            E=E, 
                            nu=nu, 
                            ell=ell,
                            eps0t=eps0t):
        elastic_energy_ = elastic_energy(u,alpha, E=E, nu=nu, eps0t=eps0t, k_res=k_res)
        dissipated_energy_ = dissipated_energy(alpha,w_1=w_1,ell=ell)
        return elastic_energy_ + dissipated_energy_

    energy = total_energy(u,alpha)

    # Hessian = derivative(derivative(Wppt*dx, z, TestFunction(Z)), z, TrialFunction(Z))

    def create_output(outdir):
        file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
        file_out.parameters["functions_share_mesh"] = True
        file_out.parameters["flush_output"] = True
        file_postproc = dolfin.XDMFFile(os.path.join(outdir, "postprocess.xdmf"))
        file_postproc.parameters["functions_share_mesh"] = True
        file_postproc.parameters["flush_output"] = True
        file_eig = dolfin.XDMFFile(os.path.join(outdir, "perturbations.xdmf"))
        file_eig.parameters["functions_share_mesh"] = True
        file_eig.parameters["flush_output"] = True
        file_bif = dolfin.XDMFFile(os.path.join(outdir, "bifurcation.xdmf"))
        file_bif.parameters["functions_share_mesh"] = True
        file_bif.parameters["flush_output"] = True
        file_bif_postproc = dolfin.XDMFFile(os.path.join(outdir, "bifurcation_postproc.xdmf"))
        file_bif_postproc.parameters["functions_share_mesh"] = True
        file_bif_postproc.parameters["flush_output"] = True
        file_ealpha = dolfin.XDMFFile(os.path.join(outdir, "elapha.xdmf"))
        file_ealpha.parameters["functions_share_mesh"] = True
        file_ealpha.parameters["flush_output"] = True

        files = {'output': file_out, 
                 'postproc': file_postproc,
                 'eigen': file_eig,
                 'bifurcation': file_bif,
                 'ealpha': file_ealpha}

        return files


    files = create_output(outdir)

    solver = EquilibriumAM(energy, state, bcs, parameters=parameters)
    stability = StabilitySolver(energy, state, bcs, parameters = parameters)
    linesearch = LineSearch(energy, state)

    load_steps = np.linspace(parameters['loading']['load_min'],
        parameters['loading']['load_max'],
        parameters['loading']['n_steps'])

    tc = (parameters['material']['sigma_D0']/parameters['material']['E'])**(.5)

    _eps = 1e-3
    load_steps = [0., tc-_eps, tc+_eps]


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

    perturb = False
    from matplotlib import cm

    log(LogLevel.INFO, '{}'.format(parameters))
    for step, load in enumerate(load_steps):
        plt.clf()
        mineigs = []
        exhaust_modes = []

        log(LogLevel.CRITICAL, '====================== STEPPING ==========================')
        log(LogLevel.CRITICAL, 'CRITICAL: Solving load t = {:.2f}'.format(load))
        alpha_old.assign(alpha)
        ut.t = load
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

                if rank == 0:
                    fig = plt.figure(dpi=80, facecolor='w', edgecolor='k')
                    plt.subplot(1, 1, 1)
                    plt.set_cmap('binary')
                    # # dolfin.plot(mesh, alpha = 1.)
                    # dolfin.plot(
                    #     project(stability.inactivemarker1, L2), alpha = 1., vmin=0., vmax=1.)
                    # plt.title('derivative zero')
                    # plt.subplot(1, 4, 2)
                    # # dolfin.plot(mesh, alpha = .5)
                    # dolfin.plot(
                    #     project(stability.inactivemarker2, L2), alpha = 1., vmin=0., vmax=1.)
                    # plt.title('ub tolerance')
                    # plt.subplot(1, 4, 3)
                    # # dolfin.plot(mesh, alpha = .5)
                    # dolfin.plot(
                    #     project(stability.inactivemarker3, L2), alpha = 1., vmin=0., vmax=1.)
                    # plt.title('alpha-alpha_old')
                    # plt.subplot(1, 4, 4)
                    # dolfin.plot(mesh, alpha = .5)
                    dolfin.plot(
                        project(stability.inactivemarker4, L2), alpha = 1., vmin=0., vmax=1.)
                    plt.title('intersec deriv, ub')
                    plt.savefig(os.path.join(outdir, "inactivesets-{:.3f}-{:d}.pdf".format(load, iteration)))


                    plt.set_cmap('hot')

                    fig = plt.figure(dpi=80, facecolor='w', edgecolor='k')

                    for i,mode in enumerate(pert):
                        plt.subplot(2, _nmodes+1, i+2)
                        plt.axis('off')
                        plot(mode[1], cmap = cm.ocean)

                        plt.title('mode {} $h^*$={:.3f}\n $\\lambda_{}$={:.3e} \n $\\Delta E$={:.3e}'
                            .format(i, h_opts[i], i, stability.eigs[i], en_vars[i]), fontsize= 15)

                        # plt.title('mode {}'
                        #     .format(i), fontsize= 15)

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
                    plt.savefig(os.path.join(outdir, "modes-{:.3f}-{}.pdf".format(load, iteration)))
                    plt.close(fig)
                    plt.clf()
                    log(LogLevel.INFO, 'plotted modes')

                cont_data_pre = compile_continuation_data(state, energy)

                log(LogLevel.INFO, 'Estimated energy variation {:.3e}'.format(en_var))

                # Ealpha = Function(V_alpha)
                # Ealpha.vector()[:]=assemble(stability.inactiveEalpha)[:]
                # Ealpha.rename('Ealpha-{}'.format(iteration), 'Ealpha-{}'.format(iteration))

                # with files['ealpha'] as file:
                    # file.write(Ealpha, load)

                save_current_bifurcation = True

                # pick the first of the non exhausted modes-
                non_zero_h = np.where(abs(np.array(h_opts)) > DOLFIN_EPS)[0]
                log(LogLevel.INFO, 'Nonzero h {}'.format(non_zero_h))
                avail_modes = set(non_zero_h)-set(exhaust_modes)

                opt_mode = 0
                # opt_mode = np.argmin(en_vars)
                log(LogLevel.INFO, 'Energy vars {}'.format(en_vars))
                log(LogLevel.INFO, 'Pick bifurcation mode {} out of {}'.format(opt_mode, len(en_vars)))
                # h_opt = min(h_opts[opt_mode],1.e-2)
                h_opt = h_opts[opt_mode]
                perturbation_v    = stability.perturbations_v[opt_mode]
                perturbation_beta = stability.perturbations_beta[opt_mode]
                minmode = stability.minmode
                # (perturbation_v, perturbation_beta) = minmode.split(deepcopy=True)
                (perturbation_v, perturbation_beta) = stability.perturbation_v, stability.perturbation_beta

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
                # h_opt = hs[np.argmin(energy_vals)]
                print('computed h_opt {}'.format(hs[np.argmin(energy_vals)]))
                print("%%%%%%%%% ",mineig_z,"-",mineig)

                if rank == 0:
                    plt.figure()
                    # plt.plot(hs,energy_vals, marker = 'o')
                    plt.plot(hs,energy_vals, marker = 'o', label="exact")
                    plt.plot(hs,energy_vals_quad, label="quadratic approximation")
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
                # (time_data_i, am_iter) = solver.solve()
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
                    plt.figure()
                    plt.plot(mineigs, marker = 'o')
                    plt.axhline(0.)
                    plt.savefig(os.path.join(outdir, "mineigs-{:.3f}.pdf".format(load)))

                # continuation criterion
                if abs(np.diff(mineigs)[-1]) > 1e-10:
                    log(LogLevel.INFO, 'Min eig change = {:.3e}'.format(np.diff(mineigs)[-1]))
                    log(LogLevel.INFO, 'Continuing perturbations')
                else:
                    log(LogLevel.INFO, 'Min eig change = {:.3e}'.format(np.diff(mineigs)[-1]))
                    log(LogLevel.CRITICAL, 'We are stuck in the matrix')
                    log(LogLevel.WARNING, 'Exploring next mode')
                    exhaust_modes.append(opt_mode)
                    # import pdb; pdb.set_trace()
                    log(LogLevel.WARNING, 'Continuing load program')
                    break
                    #
                # if not release:
                    # log(LogLevel.CRITICAL, 'Small nergy release , we are stuck in the matrix')
                    # log(LogLevel.CRITICAL, 'No decrease in energy, we are stuck in the matrix')
                    # log(LogLevel.WARNING, 'Continuing load program')
                    # import pdb; pdb.set_trace()
                    # break
                # else:
                #     # warn
                #     log(LogLevel.CRITICAL, 'Found zero increment, we are stuck in the matrix')
                #     log(LogLevel.WARNING, 'Exploring next mode')
                #     exhaust_modes.append(opt_mode)
                #     import pdb; pdb.set_trace()
                #     # log(LogLevel.WARNING, 'Continuing load program')
                #     # break

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

                with files['bifurcation'] as file:
                    for n in range(len(pert)):
                        mode = dolfin.project(stability.perturbations_beta[n], V_alpha)
                        modename = 'beta-%d'%n
                        mode.rename(modename, modename)
                        log(LogLevel.INFO, 'Saved mode {}'.format(modename))
                        file.write(mode, load)

                # with files['file_bif_postproc'] as file:
                    # leneigs = len(modes)
                    # maxmodes = min(3, leneigs)
                    # beta0v = dolfin.project(stability.perturbation_beta, V_alpha)
                    # log(LogLevel.DEBUG, 'DEBUG: irrev {}'.format(alpha.vector()-alpha_old.vector()))
                    # file.write_checkpoint(beta0v, 'beta0', 0, append = True)
                    # file.write_checkpoint(alpha_bif_old, 'alpha-old', 0, append=True)
                    # file.write_checkpoint(alpha_bif, 'alpha-bif', 0, append=True)
                    # file.write_checkpoint(alpha, 'alpha', 0, append=True)

                np.save(os.path.join(outdir, 'energy_perturbations'), en_perts, allow_pickle=True, fix_imports=True)


                with files['eigen'] as file:
                    _v = dolfin.project(dolfin.Constant(h_opt)*perturbation_v, V_u)
                    _beta = dolfin.project(dolfin.Constant(h_opt)*perturbation_beta, V_alpha)
                    _v.rename('perturbation displacement', 'perturbation displacement')
                    _beta.rename('perturbation damage', 'perturbation damage')
                    file.write(_v, load)
                    file.write(_beta, load)

                # save_current_bifurcation = False

        time_data_i["load"] = load
        time_data_i["alpha_max"] = max(alpha.vector()[:])
        time_data_i["elastic_energy"] = dolfin.assemble(
            elastic_energy(u,alpha, E=E, nu=nu, eps0t=eps0t, k_res=k_res))
        time_data_i["dissipated_energy"] = dolfin.assemble(
            (w + w_1 * parameters['material']['ell'] ** 2. * inner(grad(alpha), grad(alpha)))*dx)
        time_data_i["stable"] = stability.stable
        time_data_i["# neg ev"] = stability.negev
        time_data_i["eigs"] = stability.eigs if hasattr(stability, 'eigs') else np.inf
        time_data_i["sigma"] = 1/Ly * dolfin.assemble(_snn*ds(1))
        # import pdb; pdb.set_trace()

        log(LogLevel.INFO,
            "Load/time step {:.4g}: converged in iterations: {:3d}, err_alpha={:.4e}".format(
                time_data_i["load"],
                time_data_i["iterations"][0],
                time_data_i["alpha_error"][0]))

        time_data.append(time_data_i)
        time_data_pd = pd.DataFrame(time_data)

        with files['output'] as file:
            file.write(alpha, load)
            file.write(u, load)

        with files['postproc'] as file:
            file.write_checkpoint(alpha, "alpha-{}".format(step), step, append = True)
            file.write_checkpoint(u, "u-{}".format(step), step, append = True)
            log(LogLevel.INFO, 'INFO: written postprocessing step {}'.format(step))

        time_data_pd.to_json(os.path.join(outdir, "time_data.json"))

        if rank == 0:
            plt.close('all')

            fig = plt.figure()
            for i,d in enumerate(time_data_pd['eigs']):
                # if d is not (np.inf or np.nan or float('inf')):
                if np.isfinite(d).all():
                    # import pdb; pdb.set_trace()
                    lend = len(d) if isinstance(d, np.ndarray) else 1
                    plt.scatter([(time_data_pd['load'].values)[i]]*lend, d,
                               c=np.where(np.array(d)<0., 'red', 'black'))

            plt.axhline(0, c='k', lw=2.)
            plt.xlabel('t')

            log(LogLevel.INFO, 'Spectrum bifurcation loads : {}'.format(bifurcation_loads))
            plt.xticks(list(plt.xticks()[0]) + bifurcation_loads)
            [plt.axvline(bif, lw=2, c='k') for bif in bifurcation_loads]
            plt.savefig(os.path.join(outdir, "spectrum.pdf"), bbox_inches='tight')
        # plt.plot()

    return time_data_pd, outdir

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", "-p", type=str, default=None)
    args, unknown = parser.parse_known_args()

    # Parameters
    if args.parameters:
        print('loading {}'.format(args.parameters))
        with open(args.parameters) as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)
    else:
        # with open('../parameters/bar_short.yaml') as f:
        with open('../parameters/bar_long.yaml') as f:
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
        lab = '\\ell={}, E={}, \\sigma_D = {}'.format(
            parameters['material']['ell'],
            parameters['material']['E'],
            parameters['material']['sigma_D0'])
        tc = (parameters['material']['sigma_D0']/parameters['material']['E'])**(.5)
        # tc = sqrt(2.)/2.
        ell = parameters['material']['ell']

        fig1, ax1 =pp.plot_energy(parameters, data, tc)

        fig1.savefig(os.path.join(experiment, "energy.pdf"), bbox_inches='tight')

        (fig2, ax1, ax2) =pp.plot_spectrum(parameters, data, tc)
        plt.legend(loc='lower left')
        # ax2.set_ylim(-1e-7, 2e-4)
        fig2.savefig(os.path.join(experiment, "spectrum.pdf"), bbox_inches='tight')

        list_timings(TimingClear.keep, [TimingType.wall, TimingType.system])

        dump_timings_to_xml(os.path.join(experiment, "timings_avg_min_max.xml"), TimingClear.clear)



