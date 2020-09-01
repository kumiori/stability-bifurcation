import sys
sys.path.append("../src/")
from utils import ColorPrint, get_versions, check_bool
from linsearch import LineSearch
from damage_elasticity_model import DamageElasticityModel1D
import solvers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import mshr
import dolfin
from dolfin import MPI
import os
import pandas as pd
import sympy
import numpy as np
import post_processing as pp
import petsc4py
from functools import reduce
import ufl
petsc4py.init(sys.argv)
from petsc4py import PETSc
from hashlib import md5
from post_processing import plot_global_data
from pathlib import Path
import json
import hashlib
from time_stepping import TimeStepping
from copy import deepcopy
import mpi4py

from slepc4py import SLEPc
from solver_stability import StabilitySolver


comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

form_compiler_parameters = {
    "representation": "uflacs",
    "quadrature_degree": 2,
    "optimize": True,
    "cpp_optimize": True,
}


timestepping_parameters = {"perturbation_choice": 1,
                            'savelag': 1}
                        # "perturbation_choice": 'steepest',               # admissible choices: steepest, first, #

stability_parameters = {"order": 4,
                        "projection": 'none',
                        'maxmodes': 5,
                        'checkstability': True,
                        'continuation': False,
                        'cont_rtol': 1e-5,
                        'inactiveset_atol': 1e-5
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
                           }

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
                     "solver_alpha_tao": petsc_options_alpha_tao
                     }

numerical_parameters = {"alt_min": alt_min_parameters,
                      "stability": stability_parameters,
                      "time_stepping": timestepping_parameters}

versions = get_versions()
versions.update({'filename': __file__})
parameters = {"alt_min": alt_min_parameters,
                "stability": stability_parameters,
                "time_stepping": timestepping_parameters,
                "material": {},
                "geometry": {},
                "experiment": {},
                "code": versions
                }


dolfin.parameters["std_out_all_processes"] = False
dolfin.parameters["form_compiler"].update(form_compiler_parameters)

def traction_test(
    ell=0.1,
    degree=1,
    n=3,
    E=1.,
    load_min=0,
    load_max=2,
    loads=None,
    nsteps=20,
    Lx=1,
    outdir="outdir",
    postfix='',
    savelag=1,
    sigma_D0=1.,
    continuation=False,
    checkstability=True,
    configString=''
):
    # constants
    ell = ell
    Lx = Lx
    load_min = load_min
    load_max = load_max
    nsteps = nsteps
    outdir = outdir
    loads=loads

    savelag = 1
    ell = dolfin.Constant(ell)
    E0 = dolfin.Constant(E)
    sigma_D0 = E0
    n = n
    continuation = continuation
    config = json.loads(configString) if configString != '' else ''

    cmd_parameters =  {
    'material': {
        "ell": ell.values()[0],
        "E": E0.values()[0],
        "sigma_D0": sigma_D0.values()[0]},
    'geometry': {
        'Lx': Lx,
        'n': n,
        },
    'experiment': {
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

    if config:
        for par in config: parameters[par].update(config[par])
    else:
        for par in parameters: parameters[par].update(cmd_parameters[par])
    print(parameters)

    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
    outdir += '-{}{}'.format(signature, cmd_parameters['time_stepping']['postfix'])
    # outdir += '-{}'.format(cmd_parameters['time_stepping']['postfix'])
    parameters['time_stepping']['outdir']=outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)
    print('Outdir is: '+outdir)

    with open(os.path.join(outdir, 'rerun.sh'), 'w') as f:
        configuration = deepcopy(parameters)
        configuration['time_stepping'].pop('outdir')
        str(configuration).replace("\'True\'", "True").replace("\'False\'", "False")
        rerun_cmd = 'python3 {} --config="{}"'.format(__file__, configuration)
        f.write(rerun_cmd)

    with open(os.path.join(outdir, 'parameters.pkl'), 'w') as f:
        json.dump(parameters, f)

    with open(os.path.join(outdir, 'signature.md5'), 'w') as f:
        f.write(signature)


    # geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))
    # mesh = mshr.generate_mesh(geom,  int(float(n * Lx / ell)))
    mesh = dolfin.IntervalMesh(int(float(n * Lx / ell)), -Lx/2., Lx/2.)
    meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
    meshf << mesh

    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    # bottom = dolfin.CompiledSubDomain("near(x[1],-Ly/2.)", Ly=Ly)
    # top = dolfin.CompiledSubDomain("near(x[1],Ly/2.)", Ly=Ly)
    # left_bottom_pt = dolfin.CompiledSubDomain("near(x[0],-Lx/2.) && near(x[1],-Ly/2.)", Lx=Lx, Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)
    # bottom.mark(mf, 3)
    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_u = dolfin.FunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = dolfin.Function(V_alpha, name="Damage")
    state = [u, alpha]

    Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)

    v, beta = dolfin.split(z)

    # BCs (homogenous version needed for residual evaluation)
    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u, dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u, ut, right)
             # dolfin.DirichletBC(V_u, (0, 0), left_bottom_pt, method="pointwise")
             ]

    bcs_alpha = []

    # Files for output
    ColorPrint.print_warn('Outdir = {}'.format(outdir))
    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_out.parameters["functions_share_mesh"] = True
    file_out.parameters["flush_output"] = True
    file_con = dolfin.XDMFFile(os.path.join(outdir, "cont.xdmf"))
    file_con.parameters["functions_share_mesh"] = True
    file_con.parameters["flush_output"] = True
    file_eig = dolfin.XDMFFile(os.path.join(outdir, "modes.xdmf"))
    file_eig.parameters["functions_share_mesh"] = True
    file_eig.parameters["flush_output"] = True

    import pdb; pdb.set_trace()
    # Problem definition
    model = DamageElasticityModel1D(state, E0, ell, sigma_D0)
    model.dx = dx
    model.ds = ds
    energy = model.total_energy_density(u, alpha)*model.dx

    # Alternate minimisation solver
    solver = solvers.AlternateMinimizationSolver(energy,
        [u, alpha], [bcs_u, bcs_alpha], parameters=parameters['alt_min'])

    # rP = model.rP(u, alpha, v, beta)*model.dx
    # rN = model.rN(u, alpha, beta)*model.dx

    # stability = StabilitySolver(mesh, energy,
        # [u, alpha], [bcs_u, bcs_alpha], z, rayleigh=[rP, rN], parameters = parameters['stability'])
    stability = StabilitySolver(mesh, energy, [u, alpha], [bcs_u, bcs_alpha], z, parameters = parameters['stability'])

    # Time iterations
    load_steps = np.linspace(load_min, load_max, parameters['time_stepping']['nsteps'])

    if loads:
        load_steps = loads

    stability.parameters['checkstability'] = True

    time_data = []

    linesearch = LineSearch(energy, [u, alpha])
    alpha_old = dolfin.Function(alpha.function_space())
    lmbda_min_prev = 0.000001
    bifurcated = False
    bifurcation_loads = []
    save_current_bifurcation = False
    bifurc_i = 0
    alpha_bif = dolfin.Function(V_alpha)
    alpha_bif_old = dolfin.Function(V_alpha)
    for it, load in enumerate(load_steps):
        ut.t = load
        alpha_old.assign(alpha)

        ColorPrint.print_warn('Solving load t = {:.2f}'.format(load))

        # First order stability conditions
        (time_data_i, am_iter) = solver.solve()

        # Second order stability conditions
        (stable, negev) = stability.solve(solver.problem_alpha.lb)
        ColorPrint.print_pass('Current state is{}stable'.format(' ' if stable else ' un'))


        mineig = stability.mineig if hasattr(stability, 'mineig') else 0.0
        print('lmbda min', lmbda_min_prev)
        print('mineig', mineig)
        Deltav = (mineig-lmbda_min_prev) if hasattr(stability, 'eigs') else 0

        if (mineig + Deltav)*(lmbda_min_prev+dolfin.DOLFIN_EPS) < 0 and not bifurcated:
            bifurcated = True

            # save 3 bif modes
            print('About to bifurcate load ', load, 'step', it)
            bifurcation_loads.append(load)
            print('DEBUG: decide what to do')
            # save_current_bifurcation = True
            bifurc_i += 1

        lmbda_min_prev = mineig if hasattr(stability, 'mineig') else 0.


        if stable:
            solver.update()
        else:
            # Continuation
            iteration = 1
            while stable == False:
                # linesearch
                perturbation_v    = stability.perturbation_v
                perturbation_beta = stability.perturbation_beta

                h_opt, (hmin, hmax), energy_perturbations = linesearch.search(
                    [u, alpha, alpha_old],
                    perturbation_v, perturbation_beta)

                if h_opt != 0:
                    save_current_bifurcation = True
                    alpha_bif.assign(alpha)
                    alpha_bif_old.assign(alpha_old)
                    # admissible
                    uval = u.vector()[:]     + h_opt * perturbation_v.vector()[:]
                    aval = alpha.vector()[:] + h_opt * perturbation_beta.vector()[:]

                    u.vector()[:] = uval
                    alpha.vector()[:] = aval

                    u.vector().vec().ghostUpdate()
                    alpha.vector().vec().ghostUpdate()

                    # import pdb; pdb.set_trace()
                    (time_data_i, am_iter) = solver.solve()
                    (stable, negev) = stability.solve(alpha_old)
                    ColorPrint.print_pass('    Continuation iteration {}, current state is{}stable'.format(iteration, ' ' if stable else ' un'))
                    iteration += 1

                else:
                    # warn
                    ColorPrint.print_warn('Found zero increment, we are stuck in the matrix')
                    ColorPrint.print_warn('Continuing load program')
                    break

            solver.update()
            # stable == True    

        time_data_i["load"] = load
        time_data_i["stable"] = stable
        time_data_i["elastic_energy"] = dolfin.assemble(
            model.elastic_energy_density(model.eps(u), alpha)*dx)
        time_data_i["dissipated_energy"] = dolfin.assemble(
            model.damage_dissipation_density(alpha)*dx)
        time_data_i["eigs"] = stability.eigs if hasattr(stability, 'eigs') else np.inf
        time_data_i["stable"] = stability.stable
        time_data_i["# neg ev"] = stability.negev
        # import pdb; pdb.set_trace()

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
            with dolfin.XDMFFile(os.path.join(outdir, "output_postproc.xdmf")) as f:
                f.write_checkpoint(alpha, "alpha-{}".format(it), 0, append = True)
                print('DEBUG: written step ', it)

        if save_current_bifurcation:
            # modes = np.where(stability.eigs < 0)[0]
            time_data_i['h_opt'] = h_opt
            time_data_i['max_h'] = hmax
            time_data_i['min_h'] = hmin
            with dolfin.XDMFFile(os.path.join(outdir, "bifurcation_postproc.xdmf")) as file:
                # leneigs = len(modes)
                # maxmodes = min(3, leneigs)
                beta0v = dolfin.project(stability.perturbation_beta, V_alpha)
                print('DEBUG: irrev ', alpha.vector()-alpha_old.vector())
                file.write_checkpoint(beta0v, 'beta0', 0, append = True)
                file.write_checkpoint(alpha_bif_old, 'alpha-old', 0, append=True)
                file.write_checkpoint(alpha_bif, 'alpha-bif', 0, append=True)
                file.write_checkpoint(alpha, 'alpha', 0, append=True)

                np.save(os.path.join(outdir, 'energy_perturbations'), energy_perturbations, allow_pickle=True, fix_imports=True)
                # import pdb; pdb.set_trace()

                # maxmodes = 2
                # for n in range(maxmodes):
                #     mode = dolfin.project(linsearch[n]['beta_n'], V_alpha)
                #     modename = 'beta-%d'%n
                #     print(modename)
                #     file.write_checkpoint(mode, modename, 0, append=True)
                #     import pdb; pdb.set_trace()

        # if np.mod(it, 10) == 0:
        #     alphatest = dolfin.Function(V_alpha)
        #     with dolfin.XDMFFile(os.path.join(outdir, "output_postproc.xdmf")) as f:
        #         f.read_checkpoint(alphatest, "alpha-{}".format(it), 0)
        #         print('DEBUG: read step ', it)

            time_data_pd.to_json(os.path.join(outdir, "time_data.json"))

    from post_processing import plot_global_data
    print(time_data_pd)
    print()
    print('Output in: '+outdir)


    # if size == 1:
    #     plt.figure()
    #     dolfin.plot(alpha)
    #     plt.savefig(os.path.join(outdir, "alpha.png"))
    #     plt.figure()
    #     dolfin.plot(u, mode="displacement")
    #     plt.savefig(os.path.join(outdir, "u.png"))
    #     plt.close('all')

    return time_data_pd

if __name__ == "__main__":

    import argparse
    from urllib.parse import unquote
    from time import sleep

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False,
                        help="JSON configuration string for this experiment")
    parser.add_argument("--ell", type=float, default=0.1)
    parser.add_argument("--load_max", type=float, default=2.0)
    parser.add_argument("--load_min", type=float, default=0.0)
    parser.add_argument("--E", type=float, default=1)
    parser.add_argument("--sigma_D0", type=float, default=1)
    parser.add_argument("--Lx", type=float, default=1)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--nsteps", type=int, default=30)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--postfix", type=str, default='')
    parser.add_argument("--savelag", type=int, default=1)
    parser.add_argument("--parameters", type=str, default=None)
    parser.add_argument("--print", type=bool, default=False)
    parser.add_argument("--continuation", type=bool, default=False)

    args, unknown = parser.parse_known_args()
    if len(unknown):
        ColorPrint.print_warn('Unrecognised arguments:')
        print(unknown)
        ColorPrint.print_warn('continuing in 3s')
        sleep(3)

    if args.outdir == None:
        args.postfix += '-cont' if args.continuation==True else ''
        outdir = "../output/{:s}{}".format('traction',args.postfix)
    else:
        outdir = args.outdir

    if args.print and args.parameters is not None:
        cmd = ''
        with open(args.parameters, 'r') as params:
            params = json.load(params)
            for k,v in params.items():
                for c,u in v.items():
                    cmd = cmd + '--{} {} '.format(c, str(u))
        print(cmd)

    config = '{}'
    if args.config:
        print(config)
        config = unquote(args.config).replace('\'', '"')
        config = config.replace('False', '"False"')
        config = config.replace('True', '"True"')
        print(config)

    if args.parameters is not None:
        experiment = ''
        with open(args.parameters, 'r') as params:
            config = str(json.load(params))
        config = unquote(config).replace('\'', '"')
        config = config.replace('False', '"False"')
        config = config.replace('True', '"True"')
        config = config.replace('"load"', '"time_stepping"')
        config = config.replace('"experiment"', '"stability"')
        print(config)
        traction_test(outdir=outdir, configString=config)
    else:
        traction_test(
            ell=args.ell,
            load_min=args.load_min,
            load_max=args.load_max,
            nsteps=args.nsteps,
            n=args.n,
            Lx=args.Lx,
            outdir=outdir,
            savelag=args.savelag,
            continuation=args.continuation,
            configString=config
        )
