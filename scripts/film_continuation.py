import sys
sys.path.append("../src/")
import numpy as np
import sympy
import pandas as pd
import os
import dolfin
import mshr
import ufl
import matplotlib.pyplot as plt
import solvers
from damage_elasticity_model import DamagePrestrainedElasticityModel
from utils import ColorPrint, get_versions
# from post_processing import make_figures, plot_global_data, plot_spectrum
# set_log_level(100)
dolfin.parameters["std_out_all_processes"] = False
from dolfin.cpp.log import log, LogLevel


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
                        "continuation": True,
                        "projection": 'none',
                        'maxmodes': 5,
                        }

petsc_options_alpha_tao = {"tao_type": "gpcg",
                           "tao_ls_type": "gpcg",
                           "tao_gpcg_maxpgits": 50,
                           "tao_max_it": 300,
                           "tao_steptol": 1e-7,
                           "tao_gatol": 1e-8,
                           "tao_grtol": 1e-8,
                           "tao_gttol": 1e-8,
                           "tao_catol": 0.,
                           "tao_crtol": 0.,
                           "tao_ls_ftol": 1e-6,
                           "tao_ls_gtol": 1e-6,
                           "tao_ls_rtol": 1e-6,
                           "ksp_rtol": 1e-6,
                           "tao_ls_stepmin": 1e-8,  #
                           "tao_ls_stepmax": 1e6,  #
                           "pc_type": "bjacobi",
                           "tao_monitor": True,  # "tao_ls_type": "more-thuente"
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
                      "solver_u": petsc_options_u,
                      # either
                      "solver_alpha": "snes",
                      "solver_alpha_snes": petsc_options_alpha_snes
                      # or
                      # "solver_alpha": "tao",
                     # "solver_alpha_tao": petsc_options_alpha_tao
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


    # import pdb; pdb.set_trace()

    # --------------------

    for par in parameters: parameters[par].update(cmd_parameters[par])

    if config:
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
    outdir = outdir+'-cont'
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
    

    # boundary_meshfunction = dolfin.MeshFunction("size_t", mesh, "meshes/%s-%s_facet_region.xml"%(fname, signature))
    # cells_meshfunction = dolfin.MeshFunction("size_t", mesh, "meshes/%s-%s_physical_region.xml"%(fname, signature))

    # ------------------
    geometry_parameters = parameters['geometry']

    geom_signature = hashlib.md5(str(geometry_parameters).encode('utf-8')).hexdigest()
    meshfile = "%s/meshes/%s-%s.xml"%(BASE_DIR, fname, geom_signature)
    # cmd_parameters['experiment']['signature']=signature

    if os.path.isfile(meshfile):
        print("Meshfile %s exists"%meshfile)
        mesh = dolfin.Mesh("meshes/%s-%s.xml"%(fname, geom_signature))
    else:
        print("Creating meshfile: %s"%meshfile)
        print(('DEBUG: (-Lx/2. ={} , -Ly/2.={})'.format(Lx/2., -Ly/2.)))
        geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))
        mesh = mshr.generate_mesh(geom, n * int(float(Lx / ell)))
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
    bcs_u = [DirichletBC(V_u, Constant((0., 0)), '(near(x[0], %f) or near(x[0], %f))'%(-Lx/2., Lx/2.))]

    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    bottom = dolfin.CompiledSubDomain("near(x[1],-Ly/2.)", Ly=Ly)
    top = dolfin.CompiledSubDomain("near(x[1],Ly/2.)", Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)
    bottom.mark(mf, 3)

    state = [u, alpha]

    Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)

    v, beta = dolfin.split(z)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)
    ds = dolfin.Measure("ds", subdomain_data=mf)

    # Files for output
    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_eig = dolfin.XDMFFile(os.path.join(outdir, "perturbations.xdmf"))
    file_con = dolfin.XDMFFile(os.path.join(outdir, "continuation.xdmf"))
    file_bif = dolfin.XDMFFile(os.path.join(outdir, "bifurcation_postproc.xdmf"))

    for f in [file_out, file_eig, file_con, file_bif]:
        f.parameters["functions_share_mesh"] = True
        f.parameters["flush_output"] = True

    # Problem definition

    foundation_density = 1./2.*1./ell_e**2.*dot(u, u)
    model = DamagePrestrainedElasticityModel(state, E, nu, ell, sigma_D0,
        user_functional=foundation_density, 
        eps0t=Expression([['t', 0.],[0.,0.]], t=0., degree=0))
    model.dx = dx
    model.ds = ds
    energy = model.total_energy_density(u, alpha)*dx
    # Alternate minimization solver
    # import pdb; pdb.set_trace()

    solver = solvers.AlternateMinimizationSolver(
        energy, [u, alpha], [bcs_u, bcs_alpha], parameters=parameters['alt_min'])

    rP =model.rP(u, alpha, v, beta)*dx + 1/ell_e**2.*dot(v, v)*dx
    rN =model.rN(u, alpha, beta)*dx

    stability = StabilitySolver(mesh, energy, [u, alpha], [bcs_u, bcs_alpha], z, parameters = parameters['stability'])
    # stability = StabilitySolver(mesh, energy, [u, alpha], [bcs_u, bcs_alpha], z, parameters = parameters['stability'], rayleigh=[rP, rN])

    # if isPeriodic:
    #     stability = StabilitySolver(mesh, energy, [u, alpha], [bcs_u, bcs_alpha], z,
    #         parameters = stability_parameters,
    #         constrained_domain = PeriodicBoundary(Lx))
    # else:
    #     stability = StabilitySolver(mesh, energy, [u, alpha], [bcs_u, bcs_alpha], z, parameters = parameters['stability'])

    load_steps = np.linspace(load_min, load_max, parameters['time_stepping']['nsteps'])
    if loads:
        load_steps = loads

    time_data = []

    linesearch = LineSearch(energy, [u, alpha])
    alpha_old = dolfin.Function(alpha.function_space())
    lmbda_min_prev = 0.000001
    bifurcated = False
    bifurcation_loads = []
    save_current_bifurcation = False
    bifurc_count = 0
    alpha_bif = dolfin.Function(V_alpha)
    alpha_bif_old = dolfin.Function(V_alpha)
    bifurcation_loads = []

    tot_energy = model.elastic_energy_density(model.eps(u), alpha)*dx + \
            1./2.*1/ell_e**2. * dot(u, u)*dx             + \
            model.damage_dissipation_density(alpha)*dx
    cont_atol = 1e-3

    for it, load in enumerate(load_steps):
        model.eps0t.t = load
        alpha_old.assign(alpha)
        # ColorPrint.print_warn('Solving load t = {:.2f}'.format(load))
        log(LogLevel.PROGRESS, 'Solving load t = {:.2f}'.format(load))

        # First order stability conditions
        (time_data_i, am_iter) = solver.solve()

        # Second order stability conditions
        (stable, negev) = stability.solve(solver.solver_alpha.problem.lb)
        # ColorPrint.print_pass('Current state is{}stable'.format(' ' if stable else ' un'))
        log(LogLevel.INFO, 'Current state is{}stable'.format(' ' if stable else ' un'))

        mineig = stability.mineig if hasattr(stability, 'mineig') else 0.0
        log(LogLevel.DEBUG, 'lmbda min {}'.format(lmbda_min_prev))
        log(LogLevel.DEBUG, 'mineig {}'.format(mineig))
        Deltav = (mineig-lmbda_min_prev) if hasattr(stability, 'eigs') else 0

        if (mineig + Deltav)*(lmbda_min_prev+dolfin.DOLFIN_EPS) < 0 and not bifurcated:
            bifurcated = True
            # save 3 bif modes
            log(LogLevel.DEBUG, 'About to bifurcate at load {} step {}'.format(load, it))
            bifurcation_loads.append(load)
            bifurc_count += 1

        lmbda_min_prev = mineig if hasattr(stability, 'mineig') else 0.
        if stable:
            solver.update()
        else:
            # Continuation
            iteration = 1
            energy_pre = dolfin.assemble(tot_energy)
            alpha_bif.assign(alpha)
            alpha_bif_old.assign(alpha_old)

            while stable == False and iteration < 30:
                # linesearch
                perturbation_v    = stability.perturbation_v
                perturbation_beta = stability.perturbation_beta

                h_opt, (hmin, hmax), energy_perturbations = linesearch.search(
                    [u, alpha, alpha_old],
                    perturbation_v, perturbation_beta)

                # import pdb; pdb.set_trace()
                # if h_opt != 0:
                if h_opt > cont_atol:

                    save_current_bifurcation = True

                    # admissible
                    uval = u.vector()[:]     + h_opt * perturbation_v.vector()[:]
                    aval = alpha.vector()[:] + h_opt * perturbation_beta.vector()[:]

                    u.vector()[:] = uval
                    alpha.vector()[:] = aval

                    u.vector().vec().ghostUpdate()
                    alpha.vector().vec().ghostUpdate()

                    (time_data_i, am_iter) = solver.solve()
                    (stable, negev) = stability.solve(alpha_old)
                    log(LogLevel.INFO, '    Continuation iteration #{}, current state is{}stable'.format(iteration, ' ' if stable else ' un'))
                    energy_post = dolfin.assemble(tot_energy)
                    ener_diff = energy_post - energy_pre
                    log(LogLevel.INFO, 'step {}, iteration {}, En_post - En_pre ={}'.format(it, iteration, energy_post - energy_pre))

                    iteration += 1
                    if ener_diff<0: bifurcated = False
                else:
                    # warn
                    log(LogLevel.WARNING, 'Found (almost) zero increment, we are stuck in the matrix')
                    log(LogLevel.WARNING, '  h_opt = {}'.format(h_opt))
                    log(LogLevel.WARNING, 'Continuing load program')
                    break

            solver.update()
            # stable == True    
            # modes = np.where(stability.eigs < 0)[0]
            # with file_bif as file:
            #     leneigs = len(modes)
            #     maxmodes = min(3, leneigs)
            #     for n in range(maxmodes):
            #         mode = dolfin.project(stability.linsearch[n]['beta_n'], V_alpha)
            #         modename = 'beta-%d'%n
            #         print(modename)
            #         file.write_checkpoint(mode, modename, 0, append=True)

            # bifurc_count += 1
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

        log(LogLevel.INFO,
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
                f.write_checkpoint(alpha, "alpha-{}".format(it), 0, append = True)
            # with file_bif as f:
                print('DEBUG: written step ', it)

        if save_current_bifurcation:
            # modes = np.where(stability.eigs < 0)[0]

            time_data_i['h_opt'] = h_opt
            time_data_i['max_h'] = hmax
            time_data_i['min_h'] = hmin

            with file_bif as file:
                beta0v = dolfin.project(stability.perturbation_beta, V_alpha)
                file.write_checkpoint(beta0v, 'beta0', bifurc_count-1, append = True)
                file.write_checkpoint(alpha_bif_old, 'alpha-old', bifurc_count-1, append=True)
                file.write_checkpoint(alpha_bif, 'alpha-bif', bifurc_count-1, append=True)
                file.write_checkpoint(alpha, 'alpha', bifurc_count-1, append=True)

                np.save(os.path.join(outdir, 'energy_perturbations'), energy_perturbations, allow_pickle=True, fix_imports=True)

            with file_eig as file:
                _v = dolfin.project(dolfin.Constant(h_opt)*perturbation_v, V_u)
                _beta = dolfin.project(dolfin.Constant(h_opt)*perturbation_beta, V_alpha)
                _v.rename('perturbation displacement', 'perturbation displacement')
                _beta.rename('perturbation damage', 'perturbation damage')
                # import pdb; pdb.set_trace()
                file.write(_v, load)
                file.write(_beta, load)
            with file_bif as file:
                file.write_checkpoint(_v, 'perturbation_v', bifurc_count-1, append=True)
                file.write_checkpoint(_beta, 'perturbation_beta', bifurc_count-1, append=True)

            save_current_bifurcation = False


        time_data_pd.to_json(os.path.join(outdir, "time_data.json"))
        dalpha = alpha - alpha_old
        # import pdb; pdb.set_trace()
        user_postprocess_timestep(dalpha, file_out, load, parameters, solver, stability)

    plt.figure()
    plt.semilogy()
    ax = plt.gca()
    ax.plot(time_data_pd["load"].values, time_data_pd["iterations"].values, label='iterations')
    ax2 = ax.twinx()
    ax2.plot(time_data_pd["load"].values, time_data_pd["alpha_error"].values, 'o', c='C1', label='alpha error') 
    plt.legend()
    plt.savefig(os.path.join(outdir, 'am.pdf'))
    plt.close()

    plt.figure()
    dolfin.plot(alpha)
    plt.savefig(os.path.join(outdir, "alpha.png"))
    plt.figure()
    dolfin.plot(u, mode="displacement")
    plt.savefig(os.path.join(outdir, "u.png"))
    _nu = parameters['material']['nu']
    _E = parameters['material']['E']
    _w1 = parameters['material']['sigma_D0']**2. / parameters['material']['E']

    # import pdb; pdb.set_trace()

    tc = np.sqrt(2*_w1/(_E*(1.-2.*_nu)*(1.+_nu)))
    if parameters['stability']['checkstability'] == True:
        pp.plot_spectrum(parameters, outdir, time_data_pd.sort_values('load'), tc)
    # plt.show()
    print(time_data_pd)
    print()
    print('Output in: '+outdir)

    return time_data_pd

def plot_trace_spectrum(eigendata, parameters, load, outdir):
    nmodes = len(eigendata)
    ell_e = parameters['material']['ell_e']
    ell = parameters['material']['ell']
    fig = plt.figure(figsize=(3*2, nmodes), dpi=80, facecolor='w', edgecolor='k')
    me = eigendata[0]['beta_n'].function_space().mesh()
    X = me.coordinates()
    nel = (max(X[:, 0])-min(X[:, 0]))/((me.hmin()+me.hmax())/2)
    xs = np.linspace(min(X[:, 0]), max(X[:, 0]), nel)
    dx = ((me.hmin()+me.hmax())/2)
    freq = np.fft.fftfreq(xs.shape[-1], d=dx)
    maxlen = 3
    for i,mode in enumerate(eigendata):
        sp = np.fft.fft([mode['beta_n'](x, 0) for x in xs])
        ax = plt.subplot(nmodes, 1, i+1)
        mask = np.where(freq > 0)
        power = np.abs(sp)
        plt.plot(freq[mask]*ell_e, power[mask], label='mode {}'.format(i), c='C1')
        peak_freq = freq[power[mask].argmax()]
        plt.plot(freq[mask]*ell_e, sp.real[mask], label='mode {}'.format(i), c='C3')
        plt.plot(freq[mask]*ell_e, sp.imag[mask], label='mode {}'.format(i), c='C2', lw=.5)
        plt.xlim(0, 3)
        plt.axvline(1/ell_e/2, c='k', lw=.5)
        plt.grid(b=True, which='major', linestyle='-', axis='x')
        plt.box(False)
        ax.axes.yaxis.set_ticks([])
        if i%3 == 0:
            plt.xlabel('$1/\\ell_e$')
        else: ax.axes.xaxis.set_ticks([])
        plt.ylabel('mode {}'.format(i))

        # ax = plt.subplot(nmodes, 2, 2*i+1)
        # if i==0: ax.set_title('Real')
        # plt.plot(freq/ell_e, sp.real, label='mode {}'.format(i), c='C1')
        # # import pdb; pdb.set_trace()
        # plt.xlim(0, maxlen)
        # plt.grid(b=True, which='major', linestyle='-', axis='x')
        # plt.legend()
        # ax.axes.get_yaxis().set_visible(False)
        # plt.box(False)
        # plt.axvline(ell, c='k', ls='dashed')

        # ax = plt.subplot(nmodes, 2, 2*i+2)
        # if i==0: ax.set_title('Imag')
        # plt.plot(freq/ell_e, sp.imag, label='mode {}'.format(i), c='C2')
        # plt.grid(b=True, which='major', linestyle='-', axis='x')
        # plt.legend()
        # plt.axvline(ell, c='k', ls='dashed')
        # ax.axes.get_yaxis().set_visible(False)
        # plt.box(False)
        # # plt.xlim(0, maxlen)

    plt.savefig(os.path.join(outdir, "trace_spectrum-{:3.4f}.pdf".format(load)))
    plt.close(fig)

def plot_energy_slices(eigendata, parameters, u, alpha, model, load, outdir):
    _u = Vector(u.vector())
    _alpha = Vector(alpha.vector())
    ell = parameters['material']['ell']
    w1 = parameters['material']['sigma_D0']**2. / parameters['material']['E']
    en0=assemble(model.total_energy_density(u, alpha)*dx)
    energy = model.total_energy_density(u, alpha)*dx
    energy_diss = model.damage_dissipation_density(alpha)*dx
    energy_elas = model.elastic_energy_density(model.eps(u), alpha)*dx + \
                    model.user_functional*dx
    nmodes = len(eigendata)
    # rows = int(nmodes/2+nmodes%2)
    rows = nmodes
    cols = 2
    fig = plt.figure(figsize=(cols*3,rows*4,), dpi=100, facecolor='w', edgecolor='k')
    for i,mode in enumerate(eigendata):
        plt.subplot(rows, cols, i%cols+1)
        ax = plt.gca()
        hstar = mode['hstar']
        (hmin,hmax) = mode['interval']
        envsu = []
        maxvar = max(abs(hmin), abs(hmax))
        htest = np.linspace(-maxvar, maxvar, 10)
        v_n = mode['v_n']
        beta_n = mode['beta_n']

        en = mode['en_diff']
        # z = np.polyfit(htest, en, mode['order'])
        # p = np.poly1d(z)

        # directional variations. En vs uh
        for h in htest:
            uval = _u[:]     + h*v_n.vector()
            u.vector().set_local(uval)
            envsu.append(assemble(energy)-en0)
        ax.plot(htest, envsu, label='E(u+h $v$, $\\alpha$)', lw=.5)

        ax.axvline(hmin, c='k')
        ax.axvline(hmax, c='k')

        u.vector().set_local(_u[:])
        alpha.vector().set_local(_alpha[:])

        envsa = []
        for h in htest:
            aval = _alpha[:] + h*beta_n.vector()
            alpha.vector().set_local(aval)
            envsa.append(assemble(energy)-en0)
        ax.plot(htest, envsa, label='E(u, $\\alpha$+h $\\beta$)', lw=.5)

        u.vector().set_local(_u[:])
        alpha.vector().set_local(_alpha[:])

        htest = np.linspace(hmin,hmax, mode['order']+1)
        envsh = []
        envsh_diss = []
        envsh_elas = []
        envsh_grad = []
        envsh_ltwo = []
        for h in htest:
            aval = _alpha[:] + h*beta_n.vector()
            uval = _u[:]     + h*v_n.vector()
            alpha.vector().set_local(aval)
            u.vector().set_local(uval)
            envsh.append(assemble(energy)-en0)
            envsh_diss.append(assemble(energy_diss))
            envsh_elas.append(assemble(energy_elas))
            envsh_grad.append(assemble( w1 * ell ** 2 * dot(grad(alpha), grad(alpha))*dx))
            envsh_ltwo.append(assemble(model.w(alpha)*dx))
        ax.plot(htest, envsh, label='$E_h$')
        ax.axvline(hstar)
        ax.axvline(0., c='k', lw=.5, ls=':')
        plt.subplot(rows, cols, i%cols+1+1)
        ax2 = plt.gca()
        # ax2 = ax.twinx()
        # import pdb; pdb.set_trace()
        ax2.plot(htest, np.array(envsh_elas)-min(np.array(envsh_elas)), label='$E_h$ ela', lw=1)
        ax2.plot(htest, np.array(envsh_diss)-min(np.array(envsh_diss)), label='$E_h$ diss', lw=1)
        ax2.plot(htest, np.array(envsh_diss)+np.array(envsh_elas)-min(np.array(envsh_diss)+np.array(envsh_elas)), label='$E_h$ tot')
        ax2.plot(htest, np.array(envsh_grad)-min(np.array(envsh_grad)), label='$grad$ diss', lw=1)
        ax2.plot(htest, np.array(envsh_ltwo)-min(np.array(envsh_ltwo)), label='$ltwo$ diss', lw=1)


        # ax.plot(np.linspace(hmin, hmax, 10), p(np.linspace(hmin, hmax, 10)),
            # label='interp h star = {:.5e}'.format(hstar))

        ax.legend()
        ax2.legend()
    plt.savefig(os.path.join(outdir, "en-{:3.4f}.pdf".format(load)))
    plt.close(fig)

def user_postprocess(self, load):
    # beta_n = self.stability.eigen
    from matplotlib.ticker import StrMethodFormatter
    outdir = self.parameters['outdir']
    alpha = self.solver.alpha

    adm_pert = np.where(np.array([e['en_diff'] for e in stability.eigendata]) < 0)[0]

    fig = plt.figure(figsize=(4, 1.5), dpi=180,)
    ax = plt.gca()
    X =alpha.function_space().tabulate_dof_coordinates()
    xs = np.linspace(min(X[:, 0]),max(X[:, 0]), 300)
    ax.plot(xs, [alpha(x, 0) for x in xs], label='$\\alpha$', lw=1, c='k')
    ax.axhline(0., lw=.5, c='k', ls='-')
    ax3 = ax.twinx()
    ax.legend(fontsize='small')
    # print(stability.eigendata)

    for mode in adm_pert:
        beta_n = stability.eigendata[mode]['beta_n']
        ax3.plot(xs, [beta_n(x, 0) for x in xs], label='$\\beta_{}$'.format(mode), ls=':')

    for axi in [ax, ax3]:
        axi.spines['top'].set_visible(False)
        axi.spines['bottom'].set_visible(False)

    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_yticks(np.linspace(0, 1, 3))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}')) # 2 decimal places
    plt.xlabel('$x$')
    ax.set_ylabel('$\\alpha$')
    ax3.set_ylabel('$\\beta$')
    ax.set_ylim(0., 1.)
    # ax.set_xlim(-.5, .5)
    ax3.legend(bbox_to_anchor=(0,-.45,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=len(adm_pert), frameon=False)

    fig.savefig(os.path.join(outdir, "profiles-{:.3f}.pdf".format(load)), bbox_inches="tight")

def user_postprocess_timestep(dalpha, file_out, load, parameters, solver, stability):
    # from matplotlib.ticker import FuncFormatter, MaxNLocator

    # alpha = self.solver.alpha
    # parameters = self.parameters
    # xresol = xresol
    # X =alpha.function_space().tabulate_dof_coordinates()
    # xs = np.linspace(min(X[:, 0]),max(X[:, 0]), xresol)

    inactive_set = stability.get_inactive_set()

    w = dolfin.Function(stability.Z)
    w.vector()[list(inactive_set)] = 1.
    wu, wa = w.split(deepcopy = True)
    wa.rename('beta support', 'beta support')
    file_out.write(wa, load)
    outdir = parameters['time_stepping']['outdir']
    if size == 1: 
        fig = plt.figure(figsize=(8, 6), dpi=180,)
        dolfin.plot(wa, vmin=0., vmax = 1.)
        fig.savefig(os.path.join(outdir, "beta_support.pdf".format(load)), bbox_inches="tight")
    # alpha0 = [alpha(x, 0) for x in xs]
    # spacetime[load] = alpha0
    # spacetime = spacetime.fillna(0)
    # mat = np.matrix(spacetime)
    # plt.imshow(mat, cmap = 'Greys', vmin = 0., vmax = 1., aspect=.1)
    # plt.colorbar()

    # def format_space(x, pos):
    #     return '$%1.1f$'%((-x+xresol/2)/xresol)

    # def format_time(t, pos):
    #     return '$%1.1f$'%((t-parameters['load_min'])/parameters['nsteps']*parameters['load_max'])

    # ax = plt.gca()

    # ax.yaxis.set_major_formatter(FuncFormatter(format_space))
    # ax.xaxis.set_major_formatter(FuncFormatter(format_time))

    # plt.xlabel('$t$')
    # plt.ylabel('$x$')
    # fig.savefig(os.path.join(outdir, "spacetime.pdf".format(load)), bbox_inches="tight")


    # spacetime.to_json(os.path.join(outdir + "/spacetime.json"))
    # dot(assemble(solver.problem_alpha.denergy), dalpha)
    # _res = 
    # import pdb; pdb.set_trace()

    res = Function(solver.problem_alpha.alpha.function_space())
    # res.vector()[:] = assemble(solver.problem_alpha.denergy/stability.cellarea.vector())[:]
    res.vector()[:] = assemble(solver.problem_alpha.denergy)[:]/stability.cellarea.vector()[:]
    res.rename('alpha residual', 'alpha residual')
    # import pdb; pdb.set_trace()
    file_out.write(res, load)
    if size == 1: 
        fig = plt.figure(figsize=(8, 6), dpi=180,)
        plt.colorbar(dolfin.plot(res))
        fig.savefig(os.path.join(outdir, "residual_alpha.pdf".format(load)), bbox_inches="tight")
    
    # res = computeResidual()
        # problem_alpha.denergy
        # pass
    # file_out.write(res, load)
    # fig.savefig(os.path.join(outdir, "residual.pdf".format(load)), bbox_inches="tight")

    pass
# 
    # def computeResidual():
    #     w = dolfin.Function(stability.Z)
    #     res = assemble(dot(de_alpha, alpha - alpha_old))
    #     return res

if __name__ == "__main__":

    import argparse
    from time import sleep
    from urllib.parse import unquote

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False,
                        help="JSON configuration string for this experiment")
    parser.add_argument("--ell", type=float, default=0.1)
    parser.add_argument("--ell_e", type=float, default=.3)
    parser.add_argument("--load_max", type=float, default=3.0)
    parser.add_argument("--load_min", type=float, default=0.)
    parser.add_argument("--Lx", type=float, default=1)
    parser.add_argument("--Ly", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--nsteps", type=int, default=30)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--postfix", type=str, default='')
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--savelag", type=int, default=1)
    parser.add_argument("--E", type=float, default=1)
    parser.add_argument("--parameters", type=str, default=None)
    parser.add_argument("--print", type=bool, default=False)
    parser.add_argument("--continuation", type=bool, default=False)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--periodic", action='store_true')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    if len(unknown):
        ColorPrint.print_warn('Unrecognised arguments:')
        print(unknown)
        ColorPrint.print_warn('continuing in 5s')
        sleep(5)
    # signature = md5().hexdigest()
    if args.outdir == None:
        args.postfix += '-cont' if args.continuation==True else ''
        outdir = "../output/{:s}".format('film')
    else:
        outdir = args.outdir

    if args.print and args.parameters is not None:
        cmd = ''
        with open(args.parameters, 'r') as params:
            parameters = json.load(params)
            for k,v in parameters.items():
                for c,u in v.items():
                    cmd = cmd + '--{} {} '.format(c, str(u))
        print(cmd)
        sys.exit()
    # import pdb; pdb.set_trace()

    config = '{}'
    if args.config:
        config = unquote(args.config).replace('\'', '"')

    if args.parameters is not None:
        experiment = ''
        with open(args.parameters, 'r') as params:
            config = str(json.load(params))
        config = unquote(config).replace('\'', '"')
        config = config.replace('"load"', '"time_stepping"')
        print(config)
        traction_test(configString=config)
    else:
        traction_test(
            ell=args.ell,
            ell_e=args.ell_e,
            load_min=args.load_min,
            load_max=args.load_max,
            nsteps=args.nsteps,
            n=args.n,
            Lx=args.Lx,
            Ly=args.Ly,
            outdir=outdir,
            postfix=args.postfix,
            savelag=args.savelag,
            continuation=args.continuation,
            periodic=args.periodic,
            configString=config,
            test=args.test
        )



