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
from string import Template

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

set_log_level(LogLevel.INFO)

def compile_continuation_data(state, energy):
    continuation_data_i = {}
    continuation_data_i["energy"] = assemble(energy)
    return continuation_data_i


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
    # geometry_parameters = {'R': 1., 'n': 5}

    # Define Dirichlet boundaries
    comm = MPI.comm_world

    with open('../parameters/form_compiler.yml') as f:
        form_compiler_parameters = yaml.load(f, Loader=yaml.FullLoader)
    with open('../parameters/solvers_default.yml') as f:
        solver_parameters = yaml.load(f, Loader=yaml.FullLoader)
    with open('../parameters/solvers_default.yml') as f:
        damage_parameters = yaml.load(f, Loader=yaml.FullLoader)['damage']
    with open('../parameters/solvers_default.yml') as f:
        elasticity_parameters = yaml.load(f, Loader=yaml.FullLoader)['elasticity']
    with open('../parameters/film.yaml') as f:
        material_parameters = yaml.load(f, Loader=yaml.FullLoader)['material']
    with open('../parameters/loading.yaml') as f:
        loading_parameters = yaml.load(f, Loader=yaml.FullLoader)['loading']
    with open('../parameters/stability.yaml') as f:
        stability_parameters = yaml.load(f, Loader=yaml.FullLoader)['stability']
    with open('../parameters/stability.yaml') as f:
        inertia_parameters = yaml.load(f, Loader=yaml.FullLoader)['inertia']
    with open('../parameters/stability.yaml') as f:
        eigen_parameters = yaml.load(f, Loader=yaml.FullLoader)['eigen']

    # import pdb; pdb.set_trace()

    default_parameters = {
        'code': {**code_parameters},
        'compiler': {**form_compiler_parameters},
        'eigen': {**eigen_parameters},
        # 'geometry': {**geometry_parameters},
        'inertia': {**inertia_parameters},
        'loading': {**loading_parameters},
        'material': {**material_parameters},
        'solver':{**solver_parameters},
        'stability': {**stability_parameters},
        'elasticity': {**elasticity_parameters},
        'damage': {**damage_parameters},
        }

    default_parameters.update(user_parameters)
    # FIXME: Not nice
    parameters = default_parameters
    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
    outdir = '../test/output/film2d/{}'.format(signature)
    # outdir = '../test/output/test_film2d-init-{}'.format(signature)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    log(LogLevel.INFO, 'INFO: Outdir is: '+outdir)

    R = parameters['geometry']['R']

#   Mesh1 ==================================
    # geom = mshr.Circle(dolfin.Point(0., 0.), R)
    # resolution = max(geometry_parameters['n'] * Lx / ell, 1/(Ly*10))
    # resolution = max(parameters['geometry']['n'] * R / ell, (10/R))
    # resolution = 10
    # log(LogLevel.INFO, 'mesh resolution {}'.format(resolution))
    # mesh = mshr.generate_mesh(geom,  resolution)
#   Mesh2 ==================================
    # mesh = dolfin.Mesh("../scripts/meshes/diskR3-mesh.xml")
#   Mesh3 ==================================
    # fname = os.path.join('../meshes', 'circle-init-{}'.format(geom_signature))
    

    d={'rad': parameters['geometry']['R'],
        'h': parameters['material']['ell']/parameters['geometry']['n']}
    geom_signature = hashlib.md5(str(d).encode('utf-8')).hexdigest()
    fname = os.path.join('../meshes', 'circle-{}'.format(geom_signature))
    if os.path.isfile(fname+'.xml'):
        log(LogLevel.INFO, "Meshfile {} exists".format(fname))
        mesh = dolfin.Mesh("{}.xml".format(fname))
    else:
        log(LogLevel.INFO, "Creating meshfile: %s"%fname)
        log(LogLevel.INFO, "DEBUG: parameters: %s"%d)

        # mesh_template = open('../scripts/templates/circle_template_init.geo')
        if MPI.rank(MPI.comm_world) == 0:

            mesh_template = open('../scripts/templates/circle_template.geo')

            src = Template(mesh_template.read())
            geofile = src.substitute(d)

            with open(fname+".geo", 'w') as f:
                f.write(geofile)

            cmd1 = 'gmsh {}.geo -2 -o {}.msh'.format(fname, fname)
            cmd2 = 'dolfin-convert -i gmsh {}.msh {}.xml'.format(fname, fname)
            
            log(LogLevel.INFO, 'Unable to handle mesh generation at the moment, please generate the mesh and test again.')
            log(LogLevel.INFO, cmd1)
            log(LogLevel.INFO, cmd2)
            sys.exit()
            log(LogLevel.INFO, check_output([cmd1], shell=True))  # run in shell mode in case you are not run in terminal
            Popen([cmd2], stdout=PIPE, shell=True).communicate()

        mesh = Mesh('{}.xml'.format(fname))
        mesh_xdmf = XDMFFile("{}.xdmf".format(fname))
        mesh_xdmf.write(mesh)

    log(LogLevel.INFO, fname)

    # boundary_meshfunction = dolfin.MeshFunction("size_t", mesh, "{}_facet_region.xml".format(fname))
    # cells_meshfunction = dolfin.MeshFunction("size_t", mesh, "{}_physical_region.xml".format(fname))

    # mesh_xdmf = dolfin.XDMFFile("meshes/%s-%s.xdmf"%(fname, geom_signature))
    # mesh_xdmf.write(mesh)
    # if rank == 0: 
    #     meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
    #     meshf << mesh
    log(LogLevel.WARNING, 'porcoilclero')

    if size == 1:
        meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
        meshf << mesh
        plot(mesh)
        plt.savefig(os.path.join(outdir, "mesh.pdf"), bbox_inches='tight')

    log(LogLevel.INFO, 'num vertices {}'.format(mesh.num_vertices()))
    
    log(LogLevel.INFO, 'Number of dofs: {}'.format(mesh.num_vertices()*(1+parameters['general']['dim'])))
    # import pdb; pdb.set_trace()
    
    with open(os.path.join(outdir, 'parameters.yaml'), "w") as f:
        yaml.dump(parameters, f, default_flow_style=False)

    R = parameters['geometry']['R']
    ell =  parameters['material']['ell']

    savelag = 1
    # left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    # right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    # left_bottom_pt = dolfin.CompiledSubDomain("near(x[0],-Lx/2.) && near(x[1],-Ly/2.)", Lx=Lx, Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    # right.mark(mf, 1)
    # left.mark(mf, 2)

    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_u = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
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

    # ut = dolfin.Expression("t", t=0.0, degree=0)
    # bcs_u = [dolfin.DirichletBC(V_u, dolfin.Constant((0,0)), left),
             # dolfin.DirichletBC(V_u, dolfin.Constant((0,0)), right),
             # dolfin.DirichletBC(V_u, (0, 0), left_bottom_pt, method="pointwise")
             # ]
    bcs_u = [dolfin.DirichletBC(V_u, dolfin.Constant((0., 0.)), 'on_boundary')]
    # bcs_u = []
    # bcs_alpha_l = DirichletBC(V_alpha,  Constant(0.0), left)
    # bcs_alpha_r = DirichletBC(V_alpha, Constant(0.0), right)
    # bcs_alpha =[bcs_alpha_l, bcs_alpha_r]
    # bcs_alpha = [DirichletBC(V_alpha, Constant(0.0), 'on_boundary')]
    bcs_alpha = []
    # bcs_alpha = [DirichletBC(V_alpha, Constant(1.), boundary_meshfunction, 101)]

    bcs = {"damage": bcs_alpha, "elastic": bcs_u}


    ell = parameters['material']['ell']

    # Problem definition
    # Problem definition
    k_res = parameters['material']['k_res']
    a = (1 - alpha) ** 2. + k_res
    w_1 = parameters['material']['sigma_D0'] ** 2 / parameters['material']['E']
    w = w_1 * alpha
    eps = sym(grad(u))
    eps0t=Expression([['t', 0.],[0.,'t']], t=0., degree=0)
    X0 = parameters['geometry']['R']
    lmbda0 = parameters['material']['E'] * parameters['material']['nu'] /(1. - parameters['material']['nu'])**2.
    mu0 = parameters['material']['E']/ 2. / (1.0 + parameters['material']['nu'])
    nu = parameters['material']['nu']
    # Wt = a*parameters['material']['E']*nu/(2*(1-nu**2.)) * tr(eps-eps0t)**2.                                \
    #     + a*parameters['material']['E']*(inner(eps-eps0t, eps-eps0t)/(2.*(1+nu)))                           \
    # + 1./2.*1./parameters['material']['ell_e']**2.*dot(u, u)
    # effective coeff of trace  = lmbda mu / (lmbda + 2 mu) = nu/(1-nu)

    # Wt = a* lmbda0/(lmbda0+ 2.*mu0) * tr(eps-eps0t)**2.                             \
    #     + a*2.*mu0*(inner(eps-eps0t, eps-eps0t))                           \

    Wt = a*parameters['material']['E']*nu/((1+nu)*(1-2*nu)) * tr(eps-eps0t)**2.                             \
        + a*parameters['material']['E']*(inner(eps-eps0t, eps-eps0t)/(2.*(1+nu)))                           \
        + 1./2.*X0**2./parameters['material']['ell_e']**2.*dot(u, u)

    energy = Wt * dx + w_1 *( alpha + (parameters['material']['ell']/X0** 2.)*inner(grad(alpha), grad(alpha)))*dx

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

    solver = EquilibriumSolver(energy, state, bcs, parameters=parameters)
    stability = StabilitySolver(energy, state, bcs, parameters = parameters)
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
    perturb = False
    _file = dolfin.XDMFFile(os.path.join(outdir, "test.xdmf"))
    log(LogLevel.INFO, '{}'.format(parameters))
    log(LogLevel.INFO, 'ell/X0 = {}, ell/ell_e = {}, ell_e/X0 = {}'.format(
        parameters['material']['ell']/X0,
        parameters['material']['ell']/parameters['material']['ell_e'],
        parameters['material']['ell_e']/parameters['geometry']['R']))

    break_after_bifurcation = False
    file_debug = dolfin.XDMFFile('/Users/kumiori/debugalpha.xdmf')
    file_debug.parameters["functions_share_mesh"] = True
    file_debug.parameters["flush_output"] = True

    from matplotlib import cm

    for step, load in enumerate(load_steps):
        log(LogLevel.INFO, '==========================================================')
        log(LogLevel.INFO, '====================== STEPPING ==========================')
        log(LogLevel.INFO, 'INFO: Solving load t = {:.2f}'.format(load))
        alpha_old.assign(alpha)
        eps0t.t = load
        (time_data_i, am_iter) = solver.solve(debugpath=outdir)

        # Second order stability conditions

        (stable, negev) = stability.solve(solver.damage.problem.lb)
        log(LogLevel.INFO, 'Current state is{}stable'.format(' ' if stable else ' un'))

        mineig = stability.mineig if hasattr(stability, 'mineig') else 0.0
        log(LogLevel.INFO, 'INFO: lmbda min {}'.format(lmbda_min_prev))
        log(LogLevel.INFO, 'INFO: mineig {}'.format(mineig))
        Deltav = (mineig-lmbda_min_prev) if hasattr(stability, 'eigs') else 0


        if (mineig + Deltav)*(lmbda_min_prev+dolfin.DOLFIN_EPS) < 0 and not bifurcated:
            bifurcated = True
            # save 3 bif modes
            log(LogLevel.INFO, 'INFO: About to bifurcate load {} step {}'.format(load, step))
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
            #     save_current_bifurcation = True
            #     save_stability = stability
            iteration = 1
            while stable == False:
                log(LogLevel.CRITICAL, 'Continuation iteration {}'.format(iteration))
            #         # linesearch

                cont_data_pre = compile_continuation_data(state, energy)
                perturbation_v    = stability.perturbation_v
                perturbation_beta = stability.perturbation_beta
                perturbation_v    = stability.perturbations_v[0]
                perturbation_beta = stability.perturbations_beta[0]
                log(LogLevel.INFO,' Perturbations')
                # log(LogLevel.INFO, '{}'.format(stability.   perturbations_beta))

                pert = [(_v, _b) for _v, _b in zip(stability.perturbations_v, stability.perturbations_beta)]

                for n in range(len(pert)):
                    linesearch.search({'u':u, 'alpha':alpha, 'alpha_old': alpha_old},
                    pert[n][0], pert[n][1])

                if size == 1:
                    fig = plt.figure(figsize=(4, 1.5), dpi=180,)
                    _nmodes = len(pert)
                    for mode in range(_nmodes):
                        plt.subplot(2, int(_nmodes/2)+_nmodes%2, mode+1)
                        ax = plt.gca()
                        plt.axis('off')
                        plot(stability.perturbations_beta[mode], vmin=-1., vmax=1.)
                        # ax.set_title('mode: '.format(mode))
                    fig.savefig(os.path.join(outdir, "modes-{:.3f}.pdf".format(load)), bbox_inches="tight")

                    plt.close()

                    fig = plt.figure(figsize=((_nmodes+1)*3, 3), dpi=80, facecolor='w', edgecolor='k')
                    fig.suptitle('Load {:3f}'.format(load), fontsize=16)
                    plt.subplot(2, _nmodes+1, 1)
                    plt.title('$\\alpha$ (max = {:2.2f})'.format(max(alpha.vector()[:])))
                    plt.set_cmap('coolwarm')
                    plt.axis('off')
                    plot(alpha, vmin=0., vmax=1.)

                    plt.set_cmap('hot')

                    for i,mode in enumerate(pert):
                        plt.subplot(2, _nmodes+1, i+2)
                        plt.axis('off')
                        plot(mode[1], cmap = cm.ocean, rowspan=2)

                        h_opt, bounds, energy_perturbations = linesearch.search(
                            {'u':u, 'alpha':alpha, 'alpha_old': alpha_old},
                            mode[0], mode[1])
                        # import pdb; pdb.set_trace()
                        # plt.title('mode {}\n$\\lambda_{{{}}}={:.1e},$\n$h_opt$={:.3f}'.format(
                            # i, i, stability.eigs[i], h_opt))
                        # print('plot mode {}'.format(i))
                        # plt.tight_layout(h_pad=0.0, pad=1.5)
                        # plt.savefig(os.path.join(outdir, "modes-{:3.4f}.png".format(load)))

                    for i,mode in enumerate(pert):
                        plt.subplot(2, _nmodes+1, _nmodes+2+1+i)
                        plt.axis('off')
                        _pert_beta = mode[1]
                        _pert_v = mode[0]
                        h_opt, bounds, energy_perturbations = linesearch.search(
                            {'u':u, 'alpha':alpha, 'alpha_old': alpha_old},
                            mode[0], mode[1])
                        # bounds = mode['interval']
                        # import pdb; pdb.set_trace()
                        if bounds[0] == bounds[1] == 0:
                            plt.plot(bounds[0], 0)
                        else:
                            hs = np.linspace(bounds[0], bounds[1], 100)
                            z = np.polyfit(np.linspace(bounds[0], bounds[1],
                                len(energy_perturbations)), energy_perturbations, parameters['stability']['order'])
                            p = np.poly1d(z)
                            plt.plot(hs, p(hs), c='k')
                            plt.plot(np.linspace(bounds[0], bounds[1],
                                len(energy_perturbations)), energy_perturbations, marker='o', c='k')
                            # plt.axvline(mode['hstar'])
                            plt.axvline(0, lw=.5, c='k')
                        # plt.title('{}'.format(i))
                        plt.tight_layout(h_pad=1.5, pad=1.5)
                    # plt.legend()
                    plt.savefig(os.path.join(outdir, "modes-{:3.4f}.pdf".format(load)))
                    plt.close(fig)
                    plt.clf()
                    log(LogLevel.INFO, 'INFO: plotted modes')

                h_opt, (hmin, hmax), energy_perturbations = linesearch.search(
                    {'u':u, 'alpha':alpha, 'alpha_old': alpha_old},
                    perturbation_v, perturbation_beta)
                log(LogLevel.INFO,'h_opt {}, (hmin, hmax) {}, energy_perturbations {}'.format(h_opt, (hmin, hmax), energy_perturbations))

                #         # stable = True

                if h_opt != 0:
                    log(LogLevel.INFO, '    Bifurcating')
                    save_current_bifurcation = True
                    alpha_bif.assign(alpha)
                    alpha_bif_old.assign(alpha_old)

                    # # admissible
                    uval = u.vector()[:]     + h_opt * perturbation_v.vector()[:]
                    aval = alpha.vector()[:] + h_opt * perturbation_beta.vector()[:]

                    u.vector()[:] = uval
                    alpha.vector()[:] = aval

                    u.vector().vec().ghostUpdate()
                    alpha.vector().vec().ghostUpdate()


                    file_debug.write_checkpoint(alpha, 'alpha_new', 0, append = True)
                    alpha.rename('alpha_new', 'alpha_new')
                    file_debug.write(alpha)
                    __v = alpha-alpha_old
                    _v = dolfin.project(__v, V_alpha)
                    file_debug.write(_v)
                    alpha.rename('alpha', 'alpha')

                    plt.clf()
                    plt.colorbar(dolfin.plot(__v, cmap = cm.binary))
                    plt.savefig('/Users/kumiori/apert-aold-{}-new-{}.pdf'.format(step, iteration))

                    # import pdb; pdb.set_trace()

                    (time_data_i, am_iter) = solver.solve(debugpath=outdir)

                    file_debug.write_checkpoint(alpha, 'alpha_new_post', 0, append = True)
                    alpha.rename('alpha_new_post', 'alpha_new_post')
                    file_debug.write(alpha)
                    alpha.rename('alpha', 'alpha')

                    (stable, negev) = stability.solve(solver.damage.problem.lb)
                    log(LogLevel.INFO, '    Continuation iteration {}, current state is{}stable'.format(iteration, ' ' if stable else ' un'))
                    iteration += 1
                    cont_data_post = compile_continuation_data(state, energy)

                    criterion = (cont_data_post['energy']-cont_data_pre['energy'])/cont_data_pre['energy'] < parameters['stability']['cont_rtol']
                    log(LogLevel.INFO, 'INFO: Continuation criterion {}'.format(criterion))
                else:
                    # warn
                    log(LogLevel.WARNING, 'Found zero increment, we are stuck in the matrix')
                    log(LogLevel.WARNING, 'Continuing load program')
                    import pdb; pdb.set_trace()
                    break

            # solver.update()

            # if save_current_bifurcation:
            #     import pdb; pdb.set_trace()
            #     time_data_i['h_opt'] = h_opt
            #     time_data_i['max_h'] = hmax
            #     time_data_i['min_h'] = hmin

            #     modes = np.where(save_stability.eigs < 0)[0]
            #     leneigs = len(modes)
            #     maxmodes = min(3, leneigs)

            #     with file_bif as file:
            #         for n in range(maxmodes):
            #             mode = dolfin.project(save_stability.linsearch[n]['beta_n'], V_alpha)
            #             modename = 'beta-%d'%n
            #             mode.rename(modename, modename)
            #             log(LogLevel.INFO, 'Saved mode {}'.format(modename))
            #             file.write(mode, step)

            #     with file_bif_postproc as file:
            #         # leneigs = len(modes)
            #         # maxmodes = min(3, leneigs)
            #         beta0v = dolfin.project(save_stability.perturbation_beta, V_alpha)
            #         log(LogLevel.DEBUG, 'DEBUG: irrev {}'.format(alpha.vector()-alpha_old.vector()))
            #         file.write_checkpoint(beta0v, 'beta0', 0, append = True)
            #         file.write_checkpoint(alpha_bif_old, 'alpha-old', 0, append=True)
            #         file.write_checkpoint(alpha_bif, 'alpha-bif', 0, append=True)
            #         file.write_checkpoint(alpha, 'alpha', 0, append=True)

            #     # np.save(os.path.join(outdir, 'energy_perturbations-{}'.format(step)), energy_perturbations, allow_pickle=True, fix_imports=True)

            #     if size == 1:
            #         dolfin.plot(perturbation_beta)
            #         plt.savefig(os.path.join(outdir, "pert_beta.pdf"))

            #     with file_eig as file:
            #         _v = dolfin.project(dolfin.Constant(h_opt)*perturbation_v, V_u)
            #         _beta = dolfin.project(dolfin.Constant(h_opt)*perturbation_beta, V_alpha)
            #         _v.rename('perturbation displacement', 'perturbation displacement')
            #         _beta.rename('perturbation damage', 'perturbation damage')
            #         file.write(_v, load)
            #         file.write(_beta, load)

            #         break_after_bifurcation = True

        time_data_i["load"] = load
        time_data_i["alpha_max"] = max(alpha.vector()[:])
        time_data_i["elastic_energy"] = dolfin.assemble(
            a * Wt * dx)
        time_data_i["dissipated_energy"] = dolfin.assemble(
            (w + w_1 * material_parameters['ell'] ** 2. * inner(grad(alpha), grad(alpha)))*dx)
        time_data_i["stable"] = stability.stable
        time_data_i["# neg ev"] = stability.negev
        time_data_i["eigs"] = stability.eigs if hasattr(stability, 'eigs') else np.inf

        log(LogLevel.INFO,
            "Load/time step {:.4g}: converged in iterations: {:3d}, err_alpha={:.4e}".format(
                time_data_i["load"],
                time_data_i["iterations"][0],
                time_data_i["alpha_error"][0]))

        time_data.append(time_data_i)
        time_data_pd = pd.DataFrame(time_data)
        # import pdb; pdb.set_trace()

        # plt.figure()
        # plt1 = time_data_pd.plot(
        #     x=time_data_pd["load"],
        #     y=time_data_pd["iterations"],
        #     marker=".",
        #     logy=True,
        #     logx=False,
        # )
        # plt.savefig(os.path.join(outdir, "plot_err.pdf"))

        with file_out as file:
            file.write(alpha, load)
            file.write(u, load)

        with file_postproc as file:
            file.write_checkpoint(alpha, "alpha-{}".format(step), step, append = True)
            file.write_checkpoint(u, "u-{}".format(step), step, append = True)
            log(LogLevel.INFO, 'INFO: written postprocessing step {}'.format(step))

        if break_after_bifurcation:
            log(LogLevel.CRITICAL, '    Bifurcarted. Breaking here')
            break
            # step = it
            # for s in range(step):
                # log(LogLevel.INFO, 'INFO: reading step {}'.format(s))
                # f.read_checkpoint(alpha, "alpha-{}".format(s))
            # log(LogLevel.INFO, 'INFO: read step {}'.format(step))
                # f.close()
            # import pdb; pdb.set_trace()

        # spacetime.append(get_trace(alpha))


        time_data_pd.to_json(os.path.join(outdir, "time_data.json"))

    if size == 1:
        def format_space(x, pos, xresol = 100):
            return '$%1.1f$'%((-x+xresol/2)/xresol)

        def format_time(t, pos, xresol = 100):
            return '$%1.1f$'%((t-parameters['loading']['load_min'])/parameters['loading']['n_steps']*parameters['loading']['load_max'])

        from matplotlib.ticker import FuncFormatter, MaxNLocator

        # ax = plt.gca()

        # ax.yaxis.set_major_formatter(FuncFormatter(format_space))
        # ax.xaxis.set_major_formatter(FuncFormatter(format_time))

        plot(alpha)
        plt.savefig(os.path.join(outdir, 'alpha.pdf'))
        log(LogLevel.INFO, "Saved figure: {}".format(os.path.join(outdir, 'alpha.pdf')))


    return time_data_pd, outdir


if __name__ == "__main__":

    # Parameters
    with open('../parameters/film2d.yaml') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    data, experiment = numerical_test(user_parameters = parameters)
    print(data)

    log(LogLevel.INFO, '________________________ VIZ _________________________')
    log(LogLevel.INFO, "Postprocess")
    import postprocess as pp

    with open(os.path.join(experiment, 'parameters.yaml')) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    import mpi4py

    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        lab = '\\ell={}, ell_e={}, E={}, \\sigma_D = {}'.format(
            parameters['material']['ell'],
            parameters['material']['ell_e'],
            parameters['material']['E'],
            parameters['material']['sigma_D0'])
        tc = (parameters['material']['sigma_D0']/parameters['material']['E'])**(.5)
        ell = parameters['material']['ell']
        # import pdb; pdb.set_trace()
        fig1, ax1 =pp.plot_energy(parameters, data, tc)
        from math import tanh
        # L = parameters['geometry']['R']
        # F_L = L/2. - tanh(L/2.)
        # elast_en = [1./2.*parameters['material']['E']*t**2*F_L for t in data['load']]
        # visuals.setspines2()
        # print(data['elastic_energy'])
        mu = parameters['material']['E']/2.
        # elast_en = [1./2.*2.*mu*eps**2 for eps in data['load']]
        # Lx = 1.
        # Ly = .1
        # Omega = Lx*Ly
        # elast_en = [1./2.*parameters['material']['E']*eps**2 for eps in data['load']]
        # plt.plot(data['load'], elast_en, c='k', label='analytic')
        # plt.axhline(parameters['geometry']['Ly'], c='k')
        plt.legend()

        plt.ylim(0, 1.)
        plt.title('${}$'.format(lab))

        fig1.savefig(os.path.join(experiment, "energy.pdf"), bbox_inches='tight')

        (fig2, ax1, ax2) =pp.plot_spectrum(parameters, data, tc)
        plt.legend(loc='lower left')
        # ax2.set_ylim(-1e-7, 2e-4)
        fig2.savefig(os.path.join(experiment, "spectrum.pdf"), bbox_inches='tight')

    from dolfin import list_timings, TimingType, TimingClear

    list_timings(TimingClear.keep, [TimingType.wall, TimingType.system])
    t = list_timings(TimingClear.keep,
                [TimingType.wall, TimingType.user, TimingType.system])
    # Use different MPI reductions
    # t_sum = MPI.sum(MPI.comm_world, t[0])
    # t_min = MPI.min(MPI.comm_world, t[1])
    # t_max = MPI.max(MPI.comm_world, t[2])
    t_avg = MPI.avg(MPI.comm_world, t[3])

    # Print aggregate timings to screen
    # print('\n'+t_sum.str(True))
    # print('\n'+t_min.str(True))
    # print('\n'+t_max.str(True))
    print('\n'+t_avg.str(True))

    # Store to XML file on rank 0
    if MPI.rank(MPI.comm_world) == 0:
        f = File(MPI.comm_self, os.path.join(experiment, "timings_aggregate.xml"))
        # f << t_sum
        # f << t_min
        # f << t_max
        f << t_avg

    dump_timings_to_xml(os.path.join(experiment, "timings_avg_min_max.xml"), TimingClear.clear)

