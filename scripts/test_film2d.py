# Generic imports
from utils import get_versions, ColorPrint
from linsearch import LineSearch
from solver_stability import StabilitySolver
from solvers import EquilibriumAM, EquilibriumNewton
from dolfin.cpp.log import log, set_log_level, LogLevel
from petsc4py import PETSc
import petsc4py
import mpi4py
import ufl
from dolfin import MPI
import dolfin
from string import Template
from functools import reduce
import numpy as np
import sympy
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import pandas as pd
import yaml
import subprocess
import site
import sys
sys.path.append("../src/")
matplotlib.use('Agg')


# FEniCS and PETSc related imports

# Imports from our module

petsc4py.init(sys.argv)
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

set_log_level(LogLevel.ERROR)
if MPI.rank(comm) == 0:
    set_log_level(LogLevel.INFO)

try:
    code_parameters = get_versions()
    ColorPrint.print_info(f"VERSIONS:::: {code_parameters}")
except:
    code_parameters = None


def compile_continuation_data(state, energy):
    continuation_data_i = {}
    continuation_data_i["energy"] = assemble(energy)
    return continuation_data_i


def getDefaultParameters():

    with open('../parameters/form_compiler.yml') as f:
        form_compiler_parameters = yaml.load(f, Loader=yaml.FullLoader)
    with open('../parameters/solvers_default.yml') as f:
        equilibrium_parameters = yaml.load(
            f, Loader=yaml.FullLoader)['equilibrium']
    with open('../parameters/solvers_default.yml') as f:
        damage_parameters = yaml.load(f, Loader=yaml.FullLoader)['damage']
    with open('../parameters/solvers_default.yml') as f:
        elasticity_parameters = yaml.load(
            f, Loader=yaml.FullLoader)['elasticity']
    with open('../parameters/film2d.yaml') as f:
        material_parameters = yaml.load(f, Loader=yaml.FullLoader)['material']
    with open('../parameters/film2d.yaml') as f:
        newton_parameters = yaml.load(f, Loader=yaml.FullLoader)['newton']
    with open('../parameters/loading.yaml') as f:
        loading_parameters = yaml.load(f, Loader=yaml.FullLoader)['loading']
    with open('../parameters/stability.yaml') as f:
        stability_parameters = yaml.load(
            f, Loader=yaml.FullLoader)['stability']
    with open('../parameters/stability.yaml') as f:
        inertia_parameters = yaml.load(f, Loader=yaml.FullLoader)['inertia']
    with open('../parameters/stability.yaml') as f:
        eigen_parameters = yaml.load(f, Loader=yaml.FullLoader)['eigen']

    default_parameters = {
        'code': None,
        'compiler': {**form_compiler_parameters},
        'eigen': {**eigen_parameters},
        # 'geometry': {**geometry_parameters},
        'inertia': {**inertia_parameters},
        'loading': {**loading_parameters},
        'material': {**material_parameters},
        'newton': {**newton_parameters},
        'equilibrium': {**equilibrium_parameters},
        'damage': {**damage_parameters},
        'elasticity': {**elasticity_parameters},
        'stability': {**stability_parameters},
    }

    if code_parameters is not None:
        default_parameters["code"] = {**code_parameters}

    return default_parameters


def import_mesh(mesh_name, mesh_parameters,
                mesh_path='../meshes',
                mesh_template_file="../scripts/templates/circle_template.geo"):
    geom_signature = hashlib.md5(
        str(mesh_parameters).encode('utf-8')).hexdigest()
    fname = os.path.join(mesh_path, f"{mesh_name}-{geom_signature}")

    if os.path.isfile(f'{fname}.xdmf'):
        ColorPrint.print_info(f"Meshfile {fname} exists")
    else:
        ColorPrint.print_info("Creating meshfile: %s" % fname)
        ColorPrint.print_info("INFO: parameters: %s" % mesh_parameters)
        if rank == 0:
            Path(mesh_path).mkdir(parents=True, exist_ok=True)
            mesh_template = open(mesh_template_file)
            src = Template(mesh_template.read())
            geofile = src.substitute(mesh_parameters)

            with open(fname+".geo", 'w') as f:
                f.write(geofile)

            cmd1 = 'gmsh {}.geo -2 -o {}.msh'.format(fname, fname)
            cmd2 = 'meshio-convert -i gmsh {}.msh {}.xdmf --prune-z-0'.format(
                fname, fname)
            # meshio-convert -> xdmf
            ColorPrint.print_info(cmd1)
            ColorPrint.print_info(cmd2)
            subprocess.call(cmd1, shell=True)
            subprocess.call(cmd2, shell=True)
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(f"{fname}.xdmf") as infile:
        infile.read(mesh)
    ColorPrint.print_info(fname)
    ColorPrint.print_info('Number of dofs: {}'.format(
        mesh.num_vertices()*(1+parameters['general']['dim'])))
    return mesh


def numerical_test(user_parameters):
    time_data = []
    time_data_pd = []
    spacetime = []
    lmbda_min_prev = 1e-6
    bifurcated = False
    bifurcation_loads = []
    save_current_bifurcation = False
    bifurc_i = 0
    bifurcation_loads = []
    savelag = 1
    comm = MPI.comm_world

    default_parameters = getDefaultParameters()
    default_parameters.update(user_parameters)
    parameters = default_parameters
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
    outdir = f'../output/film2d/{signature}-{size}CPU'
    if rank == 0:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(outdir, 'parameters.yaml'), "w") as f:
            yaml.dump(parameters, f, default_flow_style=False)
    ColorPrint.print_info(f'{parameters}')
    ColorPrint.print_info(f"INFO: Outdir is: {outdir}")
    R = parameters['geometry']['R']
    mesh_size = parameters['material']['ell']/parameters['geometry']['n']
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    mesh_parameters = {'rad': R, 'h': mesh_size}
    mesh_name = "circle"
    mesh = import_mesh(mesh_name, mesh_parameters)

    ds = dolfin.Measure("ds", metadata=parameters['compiler'], domain=mesh)
    dx = dolfin.Measure("dx", metadata=parameters['compiler'], domain=mesh)

    # Function Spaces
    V_u = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    L2 = dolfin.FunctionSpace(mesh, "DG", 0)
    Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement(
        [V_u.ufl_element(), V_alpha.ufl_element()]))

    u = dolfin.Function(V_u, name="Total displacement")
    u.rename('u', 'u')
    alpha = dolfin.Function(V_alpha)
    alpha_old = dolfin.Function(alpha.function_space())
    alpha.rename('alpha', 'alpha')
    dalpha = dolfin.TrialFunction(V_alpha)
    alpha_bif = dolfin.Function(V_alpha)
    alpha_bif_old = dolfin.Function(V_alpha)

    state = {'u': u, 'alpha': alpha}
    z = dolfin.Function(Z)
    v, beta = dolfin.split(z)

    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = []
    bcs_alpha = []

    bcs = {"damage": bcs_alpha, "elastic": bcs_u}

    # -----------------------
    # Problem definition
    k_res = parameters['material']['k_res']
    w_1 = parameters['material']['sigma_D0'] ** 2 / parameters['material']['E']
    eps0t = dolfin.Expression([['t', 0.], [0., 't']], t=0., degree=0)
    lmbda0 = parameters['material']['E'] * parameters['material']['nu'] / \
        (1. - parameters['material']['nu'])**2.
    mu0 = parameters['material']['E'] / 2. / \
        (1.0 + parameters['material']['nu'])
    nu = parameters['material']['nu']
    ell = parameters['material']['ell']
    ell_e = parameters['material']['ell_e']
    E = parameters['material']['E']

    def elastic_energy(u, alpha, E=E, nu=nu, ell_e=ell_e, eps0t=eps0t, k_res=k_res):
        a = (1 - alpha) ** 2. + k_res
        eps = ufl.sym(ufl.grad(u))
        Wt = a*E*nu/(2*(1-nu**2.)) * ufl.tr(eps-eps0t)**2.                                \
            + a*E/(2.*(1+nu))*(ufl.inner(eps-eps0t, eps-eps0t))                           \
            + 1./2.*1./ell_e**2.*ufl.dot(u, u)
        return Wt * dx

    def dissipated_energy(alpha, w_1=w_1, ell=ell):
        return w_1 * (alpha + ell ** 2.*ufl.inner(ufl.grad(alpha), ufl.grad(alpha)))*dx

    def total_energy(u, alpha, k_res=k_res, w_1=w_1,
                     E=E,
                     nu=nu,
                     ell_e=ell_e,
                     ell=ell,
                     eps0t=eps0t):
        elastic_energy_ = elastic_energy(
            u, alpha, E=E, nu=nu, ell_e=ell_e, eps0t=eps0t, k_res=k_res)
        dissipated_energy_ = dissipated_energy(alpha, w_1=w_1, ell=ell)
        return elastic_energy_ + dissipated_energy_

    energy = total_energy(u, alpha)
    residual = dolfin.derivative(energy, z, dolfin.TestFunction(Z))
    hessian = dolfin.derivative(residual, z, dolfin.TrialFunction(Z))

    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_postproc = dolfin.XDMFFile(os.path.join(outdir, "postprocess.xdmf"))
    file_eig = dolfin.XDMFFile(os.path.join(outdir, "perturbations.xdmf"))
    file_bif = dolfin.XDMFFile(os.path.join(outdir, "bifurcation.xdmf"))
    file_bif_postproc = dolfin.XDMFFile(
        os.path.join(outdir, "bifurcation_postproc.xdmf"))
    file_ealpha = dolfin.XDMFFile(os.path.join(outdir, "elapha.xdmf"))
    for f in [file_out, file_postproc, file_eig, file_bif, file_bif_postproc, file_ealpha]:
        f.parameters["functions_share_mesh"] = True
        f.parameters["flush_output"] = True

    solver = EquilibriumAM(energy, state, bcs, parameters=parameters)
    equilibrium = EquilibriumNewton(energy, state, bcs, parameters=parameters)
    stability = StabilitySolver(energy, state, bcs, parameters=parameters)
    # stability = StabilitySolver(energy, state, bcs, parameters = parameters['stability'], rayleigh= [rP, rN])
    linesearch = LineSearch(energy, state)

    load_steps = np.linspace(parameters['loading']['load_min'],
                             parameters['loading']['load_max'],
                             parameters['loading']['n_steps'])

    time_data = []
    time_data_pd = []
    lmbda_min_prev = 1e-6
    bifurcated = False
    bifurcation_loads = []
    save_current_bifurcation = False
    bifurc_i = 0
    alpha_bif = dolfin.Function(V_alpha)
    alpha_bif_old = dolfin.Function(V_alpha)
    bifurcation_loads = []

    for step, load in enumerate(load_steps):
        plt.clf()
        mineigs = []
        exhaust_modes = []

        ColorPrint.info_bold(
            '====================== STEPPING ==========================')
        olorPrint.info_bold('CRITICAL: Solving load t = {:.2f}'.format(load))
        alpha_old.assign(alpha)
        eps0t.t = load
        # (time_data_i, am_iter) = solver.solve(outdir)
        (time_data_i, am_iter) = solver.solve()

        # Second order stability conditions
        (stable, negev) = stability.solve(solver.damage.problem.lb)
        mineig = stability.mineig if hasattr(stability, 'mineig') else 0.0

        ColorPrint.info_bold(
            'Current state is{}stable'.format(' ' if stable else ' un'))
        ColorPrint.print_info('INFO: mineig {:.5e}'.format(mineig))
        # we postpone the update after the stability check
        if stable:
            solver.update()
            ColorPrint.print_info(
                '    Current state is{}stable'.format(' ' if stable else ' un'))
        else:
            # Continuation
            iteration = 1
            mineigs.append(stability.mineig)
            bifurcation_loads.append(load)
            modes = np.where(stability.eigs < 0)[0]

            while stable == False:
                ColorPrint.print_info(
                    'Perturbation iteration {}'.format(iteration))
                plt.close('all')
                pert = [(_v, _b) for _v, _b in zip(
                    stability.perturbations_v, stability.perturbations_beta)]
                _nmodes = len(pert)
                en_vars = []
                h_opts = []
                hbounds = []
                en_perts = []

                for i, mode in enumerate(pert):
                    h_opt, bounds, enpert, en_var = linesearch.search(
                        {'u': u, 'alpha': alpha, 'alpha_old': alpha_old},
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
                        project(stability.inactivemarker1, L2), alpha=1., vmin=0., vmax=1.))
                    plt.title('derivative zero')
                    plt.subplot(1, 4, 2)
                    # dolfin.plot(mesh, alpha = .5)
                    plt.colorbar(dolfin.plot(
                        project(stability.inactivemarker2, L2), alpha=1., vmin=0., vmax=1.))
                    plt.title('ub tolerance')
                    plt.subplot(1, 4, 3)
                    # dolfin.plot(mesh, alpha = .5)
                    plt.colorbar(dolfin.plot(
                        project(stability.inactivemarker3, L2), alpha=1., vmin=0., vmax=1.))
                    plt.title('alpha-alpha_old')
                    plt.subplot(1, 4, 4)
                    # dolfin.plot(mesh, alpha = .5)
                    plt.colorbar(dolfin.plot(
                        project(stability.inactivemarker4, L2), alpha=1., vmin=0., vmax=1.))
                    plt.title('intersec deriv, ub')
                    plt.savefig(os.path.join(
                        outdir, "{:.3f}-inactivesets-{:d}.pdf".format(load, iteration)))
                    plt.set_cmap('hot')

                    for i, mode in enumerate(pert):
                        plt.subplot(2, _nmodes+1, i+2)
                        plt.axis('off')
                        dolfin.plot(mode[1], cmap=cm.ocean)

                        plt.title('mode {}'
                                  .format(i), fontsize=15)

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
                            plt.axvline(h_opts[i], lw=.3, c='k')
                            plt.axvline(0, lw=2, c='k')
                        # plt.title('{}'.format(i))
                        plt.tight_layout(h_pad=1.5, pad=1.5)
                    # plt.legend()
                    plt.savefig(os.path.join(
                        outdir, "{:.3f}-modes-{}.pdf".format(load, iteration)))
                    plt.close(fig)
                    plt.clf()
                    ColorPrint.print_info('plotted modes')

                cont_data_pre = compile_continuation_data(state, energy)

                ColorPrint.print_info(
                    'Estimated energy variation {:.3e}'.format(en_var))

                Ealpha = Function(V_alpha)
                Ealpha.vector()[:] = assemble(stability.inactiveEalpha)[:]
                Ealpha.rename('Ealpha-{}'.format(iteration),
                              'Ealpha-{}'.format(iteration))

                with file_ealpha as file:
                    file.write(Ealpha, load)

                save_current_bifurcation = True

                # pick the first of the non exhausted modes-
                non_zero_h = np.where(abs(np.array(h_opts)) > DOLFIN_EPS)[0]
                ColorPrint.print_info('Nonzero h {}'.format(non_zero_h))
                # opt_mode = list(set(range(_nmodes))-set(exhaust_modes))[0]
                avail_modes = set(non_zero_h)-set(exhaust_modes)

                opt_mode = 0
                # opt_mode = np.argmin(en_vars)
                ColorPrint.print_info('Energy vars {}'.format(en_vars))
                ColorPrint.print_info(
                    'Pick bifurcation mode {} out of {}'.format(opt_mode, len(en_vars)))
                h_opt = min(h_opts[opt_mode], 1.e-2)
                perturbation_v = stability.perturbations_v[opt_mode]
                perturbation_beta = stability.perturbations_beta[opt_mode]
                (perturbation_v, perturbation_beta) = stability.perturbation_v, stability.perturbation_beta

                def energy_1d(h):
                    # return assemble(energy_functional(u + h * perturbation_v, alpha + h * perturbation_beta))
                    u_ = Function(u.function_space())
                    alpha_ = Function(alpha.function_space())
                    u_.vector()[:] = u.vector()[:] + h * \
                        perturbation_v.vector()[:]
                    alpha_.vector()[:] = alpha.vector()[:] + \
                        h * perturbation_beta.vector()[:]
                    u_.vector().vec().ghostUpdate()
                    alpha_.vector().vec().ghostUpdate()
                    return assemble(total_energy(u_, alpha_))

                (hmin, hmax) = linesearch.admissible_interval(
                    alpha, alpha_old, perturbation_beta)
                hs = np.linspace(hmin, hmax, 20)
                energy_vals = np.array([energy_1d(h) for h in hs])
                stability.solve(solver.damage.problem.lb)

                Hzz = assemble(stability.H*minmode*minmode)
                Gz = assemble(stability.J*minmode)
                mineig_z = Hzz/assemble(dot(minmode, minmode)*dx)

                energy_vals_quad = energy_1d(0) + hs*Gz + hs**2*Hzz/2

                h_opt = hs[np.argmin(energy_vals)]

                if rank == 0:
                    plt.figure()
                    plt.plot(hs, energy_vals, marker='o')
                    plt.plot(hs, energy_vals, label="exact")
                    plt.plot(hs, energy_vals_quad,
                             label="quadratic approximation")
                    plt.legend()
                    plt.title("eig {:.4f} vs {:.4f} expected".format(
                        mineig_z, mineig))
                    plt.axvline(h_opt)
                    # import pdb; pdb.set_trace()
                    plt.savefig(os.path.join(
                        outdir, "energy1d-{:.3f}.pdf".format(load)))

                # the following should always be true by now.

                iteration += 1
                log(LogLevel.CRITICAL, 'Bifurcating')

                save_current_bifurcation = True
                alpha_bif.assign(alpha)
                alpha_bif_old.assign(alpha_old)

                # admissible perturbation
                uval = u.vector()[:] + h_opt * perturbation_v.vector()[:]
                aval = alpha.vector()[:] + h_opt * \
                    perturbation_beta.vector()[:]

                u.vector()[:] = uval
                alpha.vector()[:] = aval
                u.vector().vec().ghostUpdate()
                alpha.vector().vec().ghostUpdate()

                ColorPrint.print_info(
                    'min a+h_opt beta_{} = {}'.format(opt_mode, min(aval)))
                ColorPrint.print_info(
                    'max a+h_opt beta_{} = {}'.format(opt_mode, max(aval)))
                ColorPrint.print_info(
                    'Solving equilibrium from perturbed state')
                (time_data_i, am_iter) = solver.solve(outdir)
                # (time_data_i, am_iter) = solver.solve()
                ColorPrint.print_info('Checking stability of new state')
                (stable, negev) = stability.solve(solver.damage.problem.lb)
                mineigs.append(stability.mineig)

                ColorPrint.print_info('Continuation iteration {}, current state is{}stable'.format(
                    iteration, ' ' if stable else ' un'))

                cont_data_post = compile_continuation_data(state, energy)
                DeltaE = (cont_data_post['energy']-cont_data_pre['energy'])
                relDeltaE = (
                    cont_data_post['energy']-cont_data_pre['energy'])/cont_data_pre['energy']
                release = DeltaE < 0 and np.abs(
                    DeltaE) > parameters['stability']['cont_rtol']

                ColorPrint.print_info('Continuation: post energy {} - pre energy {}'.format(
                    cont_data_post['energy'], cont_data_pre['energy']))
                ColorPrint.print_info(
                    'Actual absolute energy variation Delta E = {:.7e}'.format(DeltaE))
                ColorPrint.print_info(
                    'Actual relative energy variation relDelta E = {:.7e}'.format(relDeltaE))
                ColorPrint.print_info(
                    'Iter {} mineigs = {}'.format(iteration, mineigs))

                if rank == 0:
                    plt.plot(mineigs, marker='o')
                    plt.axhline(0.)
                    plt.savefig(os.path.join(
                        outdir, "mineigs-{:.3f}.pdf".format(load)))

                # continuation criterion
                if abs(np.diff(mineigs)[-1]) > 1e-8:
                    ColorPrint.print_info(
                        'Min eig change = {:.3e}'.format(np.diff(mineigs)[-1]))
                    ColorPrint.print_info('Continuing perturbations')
                else:
                    ColorPrint.print_info(
                        'Min eig change = {:.3e}'.format(np.diff(mineigs)[-1]))
                    log(LogLevel.CRITICAL, 'We are stuck in the matrix')
                    log(LogLevel.WARNING, 'Exploring next mode')
                    exhaust_modes.append(opt_mode)

            solver.update()
            ColorPrint.print_info(
                'bifurcation loads : {}'.format(bifurcation_loads))
            np.save(os.path.join(outdir, 'bifurcation_loads'),
                    bifurcation_loads, allow_pickle=True, fix_imports=True)

            if save_current_bifurcation:
                time_data_i['h_opt'] = h_opt
                time_data_i['max_h'] = hbounds[opt_mode][1]
                time_data_i['min_h'] = hbounds[opt_mode][0]

                modes = np.where(stability.eigs < 0)[0]
                leneigs = len(modes)
                maxmodes = min(3, leneigs)

                with file_bif as file:
                    for n in range(len(pert)):
                        mode = dolfin.project(
                            stability.perturbations_beta[n], V_alpha)
                        modename = 'beta-%d' % n
                        mode.rename(modename, modename)
                        ColorPrint.print_info('Saved mode {}'.format(modename))
                        file.write(mode, load)

                np.save(os.path.join(outdir, 'energy_perturbations'),
                        en_perts, allow_pickle=True, fix_imports=True)

                with file_eig as file:
                    _v = dolfin.project(dolfin.Constant(
                        h_opt)*perturbation_v, V_u)
                    _beta = dolfin.project(dolfin.Constant(
                        h_opt)*perturbation_beta, V_alpha)
                    _v.rename('perturbation displacement',
                              'perturbation displacement')
                    _beta.rename('perturbation damage', 'perturbation damage')
                    file.write(_v, load)
                    file.write(_beta, load)

                # save_current_bifurcation = False

        time_data_i["load"] = load
        time_data_i["alpha_max"] = max(alpha.vector()[:])
        time_data_i["elastic_energy"] = dolfin.assemble(
            elastic_energy(u, alpha))
        time_data_i["dissipated_energy"] = dolfin.assemble(
            dissipated_energy(alpha))
        time_data_i["stable"] = stability.stable
        time_data_i["# neg ev"] = stability.negev
        time_data_i["eigs"] = stability.eigs if hasattr(
            stability, 'eigs') else np.inf

        ColorPrint.print_info(
            "Load/time step {:.4g}: converged in iterations: {:3d}, err_alpha={:.4e}".format(
                time_data_i["load"],
                time_data_i["iterations"][0],
                time_data_i["alpha_error"][0]))

        time_data.append(time_data_i)
        time_data_pd = pd.DataFrame(time_data)

        with file_out as file:
            file.write(alpha, load)
            file.write(u, load)

        with file_postproc as file:
            file.write_checkpoint(
                alpha, "alpha-{}".format(step), step, append=True)
            file.write_checkpoint(u, "u-{}".format(step), step, append=True)
            ColorPrint.print_info(
                'INFO: written postprocessing step {}'.format(step))

        time_data_pd.to_json(os.path.join(outdir, "time_data.json"))

        if rank == 0:
            plt.clf()
            dolfin.plot(alpha)
            plt.savefig(os.path.join(outdir, 'alpha.pdf'))
            ColorPrint.print_info("Saved figure: {}".format(
                os.path.join(outdir, 'alpha.pdf')))
            plt.close('all')

            fig = plt.figure()
            for i, d in enumerate(time_data_pd['eigs']):
                # if d is not (np.inf or np.nan or float('inf')):
                if np.isfinite(d).all():
                    lend = len(d) if isinstance(d, np.ndarray) else 1
                    plt.scatter([(time_data_pd['load'].values)[i]]*lend, d,
                                c=np.where(np.array(d) < 0., 'red', 'black'))

            plt.axhline(0, c='k', lw=2.)
            plt.xlabel('t')
            # [plt.axvline(b) for b in bifurcation_loads]
            # import pdb; pdb.set_trace()
            ColorPrint.print_info(
                'Spectrum bifurcation loads : {}'.format(bifurcation_loads))
            plt.xticks(list(plt.xticks()[0]) + bifurcation_loads)
            [plt.axvline(bif, lw=2, c='k') for bif in bifurcation_loads]
            plt.savefig(os.path.join(outdir, "spectrum.pdf"),
                        bbox_inches='tight')

    return time_data_pd, outdir


if __name__ == "__main__":
    import postprocess as pp
    from dolfin import list_timings, TimingType, TimingClear, dump_timings_to_xml
    import mpi4py
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parameters
    with open('../parameters/film2d.yaml') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    data, experiment = numerical_test(user_parameters=parameters)
    ColorPrint.print_info(data)

    ColorPrint.print_info(
        '________________________ VIZ _________________________')
    ColorPrint.print_info("Postprocess")

    with open(os.path.join(experiment, 'parameters.yaml')) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    if rank == 0:
        lab = '\\ell={}, ell_e={}, E={}, \\sigma_D = {}'.format(
            parameters['material']['ell'],
            parameters['material']['ell_e'],
            parameters['material']['E'],
            parameters['material']['sigma_D0'])
        tc = (parameters['material']['sigma_D0'] /
              parameters['material']['E'])**(.5)
        tc = sqrt(2.)/2.
        ell = parameters['material']['ell']

        fig1, ax1 = pp.plot_energy(parameters, data, tc)
        fig1.savefig(os.path.join(experiment, "energy.pdf"),
                     bbox_inches='tight')

        (fig2, ax1, ax2) = pp.plot_spectrum(parameters, data, tc)
        plt.legend(loc='lower left')
        fig2.savefig(os.path.join(experiment, "spectrum.pdf"),
                     bbox_inches='tight')

        list_timings(TimingClear.keep, [TimingType.wall, TimingType.system])

        dump_timings_to_xml(os.path.join(
            experiment, "timings_avg_min_max.xml"), TimingClear.clear)
