import sys
sys.path.append("../src/")
from utils import get_versions, check_bool
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
import _post_processing as pp
import petsc4py
from functools import reduce
import ufl
petsc4py.init(sys.argv)
from petsc4py import PETSc
from hashlib import md5

from pathlib import Path
import json
import hashlib
# from loading import TimeStepping
from copy import deepcopy
import mpi4py

from slepc4py import SLEPc
from solver_stability import StabilitySolver
from dolfin.cpp.log import log, LogLevel, set_log_level
import yaml

set_log_level(LogLevel.INFO)

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

form_compiler_parameters = {
    "representation": "uflacs",
    "quadrature_degree": 2,
    "optimize": True,
    "cpp_optimize": True,
}


versions = get_versions()
versions.update({'filename': os.path.basename(__file__)})

dolfin.parameters["std_out_all_processes"] = False
dolfin.parameters["form_compiler"].update(form_compiler_parameters)

def traction_1d(
    outdir="outdir",
    configString='',
    parameters = '',
):

    parameters_file = parameters

    savelag = 1

    config = json.loads(configString) if configString != '' else ''

    with open('../parameters/form_compiler.yml') as f:
        form_compiler_parameters = yaml.load(f, Loader=yaml.FullLoader)

    with open('../parameters/solvers_default.yml') as f:
        solver_parameters = yaml.load(f, Loader=yaml.FullLoader)

    with open('../parameters/model1d.yaml') as f:
        material_parameters = yaml.load(f, Loader=yaml.FullLoader)['material']

    with open('../parameters/loading.yaml') as f:
        timestepping_parameters = yaml.load(f, Loader=yaml.FullLoader)['loading']

    with open('../parameters/stability.yaml') as f:
        stability_parameters = yaml.load(f, Loader=yaml.FullLoader)['stability']

    # with open('../parameters/geometry.yaml') as f:
    geometry_parameters = {'Lx': 1., 'n': 3}

    # default values
    parameters = {'solver':{**solver_parameters},
        'compiler': {**form_compiler_parameters},
        'loading': {**timestepping_parameters},
        'stability': {**stability_parameters},
        'material': {**material_parameters},
        'geometry': {**geometry_parameters}
        }

    # command line pointer parameter file

    if parameters_file:
        with open(parameters_file) as f:
            parameters = yaml.load(f, Loader=yaml.FullLoader)

    # command line config, highest priority

    for subpar in parameters.keys():
        if subpar in config.keys():
            parameters[subpar].update(config[subpar])    # parameters.update(config)
    # import pdb; pdb.set_trace()

    signature = hashlib.md5(str(parameters).encode('utf-8')).hexdigest()
    outdir += '-{}'.format(signature)

    print(parameters)

    # outdir += '-{}'.format(cmd_parameters['loading']['postfix'])
    parameters['loading']['outdir']=outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print('Outdir is: '+outdir)

    with open(os.path.join(outdir, 'parameters.yaml'), "w") as f:
        yaml.dump(parameters, f, default_flow_style=False)

    with open(os.path.join(outdir, 'rerun.sh'), 'w') as f:
        configuration = deepcopy(parameters)
        configuration['loading'].pop('outdir')
        # str(configuration).replace("\'True\'", "True").replace("\'False\'", "False")
        rerun_cmd = 'python3 {} --config="{}"'.format(os.path.basename(__file__), configuration)
        f.write(rerun_cmd)

    # with open(os.path.join(outdir, 'parameters.pkl'), 'w') as f:
    #     json.dump(parameters, f)

    with open(os.path.join(outdir, 'signature.md5'), 'w') as f:
        f.write(signature)

    Lx = parameters['geometry']['Lx']
    n = parameters['geometry']['n']
    ell = parameters['material']['ell']

    mesh = dolfin.IntervalMesh(int(float(n * Lx / ell)), -Lx/2., Lx/2.)
    meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
    meshf << mesh

    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)

    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_u = dolfin.FunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = dolfin.Function(V_alpha, name="Damage")
    state = {'u': u, 'alpha': alpha}

    Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)

    v, beta = dolfin.split(z)

    # BCs (homogenous version needed for residual evaluation)
    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u, dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u, ut, right)]

    bcs_alpha = [dolfin.DirichletBC(V_alpha, dolfin.Constant(0), left),
                dolfin.DirichletBC(V_alpha, dolfin.Constant(0), right)]

    bcs = {"damage": bcs_alpha, "elastic": bcs_u}
    # bcs_alpha = [dolfin.DirichletBC(V_alpha, dolfin.Constant(1.), right)]

    # Files for output
    log(LogLevel.WARNING, 'Outdir = {}'.format(outdir))

    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_out.parameters["functions_share_mesh"] = True
    file_out.parameters["flush_output"] = True
    file_con = dolfin.XDMFFile(os.path.join(outdir, "cont.xdmf"))
    file_con.parameters["functions_share_mesh"] = True
    file_con.parameters["flush_output"] = True
    file_eig = dolfin.XDMFFile(os.path.join(outdir, "modes.xdmf"))
    file_eig.parameters["functions_share_mesh"] = True
    file_eig.parameters["flush_output"] = True

    # Problem definition
    # model = DamageElasticityModel1D(state, E0, ell, sigma_D0)
    # 
    k_ell = 1e-8
    a = (1 - alpha) ** 2. + k_ell
    # import pdb; pdb.set_trace()
    w_1 = material_parameters['sigma_D0'] ** 2 / material_parameters['E']
    w = w_1 * alpha
    eps = u.dx(0)

    # sigma = material_parameters['E']*eps
    # energy = 1./2.* parameters['material']['E']*a*eps**2. * dx + (w + w_1 * parameters['material']['ell'] ** 2. * alpha.dx(0)**2.)*dx

    # # Rayleigh Ratio
    # rP = (dolfin.sqrt(a)*sigma + dolfin.diff(a, alpha)/dolfin.sqrt(a)*sigma*beta)*(dolfin.sqrt(a)*v.dx(0) + dolfin.diff(a, alpha)/dolfin.sqrt(a)*eps*beta)*dx + \
    #                 2*w_1*parameters['material']['ell'] ** 2 * beta.dx(0)**2*dx

    # da = dolfin.diff(a, alpha)
    # dda = dolfin.diff(dolfin.diff(a, alpha), alpha)
    # ddw = dolfin.diff(dolfin.diff(w, alpha), alpha)

    # rN = -(1./2.*(dda - da**2./a)*sigma*eps +1./2.*ddw)*beta**2.*dx
    # import pdb; pdb.set_trace()

    # ------------------------------

    # model = DamageElasticityModel1D(state, E0, ell, sigma_D0)
    model = DamageElasticityModel1D(state, material_parameters)
    model.dx = dx
    # energy = model.total_energy_density(u, alpha)*model.dx
    energy = 1./2.* material_parameters['E']*a*eps**2. * dx + w_1 *( alpha +  material_parameters['ell']** 2.*alpha.dx(0)**2.)*dx
    rP = model.rP(u, alpha, v, beta)*model.dx
    rN = model.rN(u, alpha, beta)*model.dx

    # Alternate minimisation solver
    # import pdb; pdb.set_trace()
    solver = solvers.EquilibriumSolver(energy, state, bcs, parameters=solver_parameters)

    # stability = StabilitySolver(mesh, energy,
    #     state, [bcs_u, bcs_alpha], z, rayleigh=[rP, rN], parameters = parameters['stability'])
    stability = StabilitySolver(mesh, energy, state, bcs, z, parameters = parameters['stability'])

    # Time iterations
    load_steps = np.linspace(parameters['loading']['load_min'], parameters['loading']['load_max'], parameters['loading']['n_steps'])
    # load_steps = np.logspace(np.log10(load_min), np.log10(load_max), parameters['loading']['nsteps'])

    time_data = []

    linesearch = LineSearch(energy, state)
    alpha_old = dolfin.Function(alpha.function_space())
    lmbda_min_prev = 0.000001
    bifurcated = False
    bifurcation_loads = []
    save_current_bifurcation = False
    bifurc_i = 0
    alpha_bif = dolfin.Function(V_alpha)
    alpha_bif_old = dolfin.Function(V_alpha)

    lmbda_min_prev = 0.000001
    bifurcated = False

    bifurcation_loads = []
    time_data_pd = []
    for it, load in enumerate(load_steps):
        ut.t = load
        # alpha_old.assign(alpha)
        # import pdb; pdb.set_trace()

        log(LogLevel.CRITICAL, 'CRITICAL: Solving load t = {:.2f}'.format(load))
        (time_data_i, am_iter) = solver.solve()
        # (stable, negev) = stability.solve(solver.damage_solver.problem.lb)

        # log(LogLevel.INFO, 'INFO: Current state is{}stable'.format(' ' if stable else ' un'))
        # import pdb; pdb.set_trace()
        solver.update()

        # mineig = stability.mineig if hasattr(stability, 'mineig') else 0.0
        # log(LogLevel.INFO, 'INFO: lmbda min {}'.format(lmbda_min_prev))
        # log(LogLevel.INFO, 'INFO: mineig {}'.format(mineig))
        # Deltav = (mineig-lmbda_min_prev) if hasattr(stability, 'eigs') else 0

        # if (mineig + Deltav)*(lmbda_min_prev+dolfin.DOLFIN_EPS) < 0 and not bifurcated:
        #     bifurcated = True

        #     # save 3 bif modes
        #     log(LogLevel.PROGRESS, 'About to bifurcate load {} step {}'.format(load, it))
        #     bifurcation_loads.append(load)
        #     modes = np.where(stability.eigs < 0)[0]

        #     with dolfin.XDMFFile(os.path.join(outdir, "postproc.xdmf")) as file:
        #         leneigs = len(modes)
        #         maxmodes = min(3, leneigs)
        #         for n in range(maxmodes):
        #             mode = dolfin.project(stability.linsearch[n]['beta_n'], V_alpha)
        #             modename = 'beta-%d'%n
        #             print(modename)
        #             file.write_checkpoint(mode, modename, 0, append=True)

        #     bifurc_i += 1

        # lmbda_min_prev = mineig if hasattr(stability, 'mineig') else 0.

        #     # stable == True    

        time_data_i["load"] = load
        # time_data_i["stable"] = stable

        time_data_i["elastic_energy"] = dolfin.assemble(
            1./2.* material_parameters['E']*a*eps**2. *dx)
        time_data_i["dissipated_energy"] = dolfin.assemble(
            (w + w_1 * material_parameters['ell'] ** 2. * alpha.dx(0)**2.)*dx)

        time_data_i["elastic_energy"] = dolfin.assemble(
            model.elastic_energy_density(u.dx(0), alpha)*dx)
        time_data_i["dissipated_energy"] = dolfin.assemble(
            model.damage_dissipation_density(alpha)*dx)

        # time_data_i["eigs"] = stability.eigs if hasattr(stability, 'eigs') else np.inf
        # time_data_i["stable"] = stability.stable
        # time_data_i["# neg ev"] = stability.negev
        # import pdb; pdb.set_trace()

        # time_data_i["S(alpha)"] = dolfin.assemble(1./(a)*dx)
        # time_data_i["a(alpha)"] = dolfin.assemble(a*dx)
        # time_data_i["avg_alpha"] = dolfin.assemble(alpha*dx)
        # import pdb; pdb.set_trace()
        log(LogLevel.CRITICAL,
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
            with dolfin.XDMFFile(os.path.join(outdir, "output_postproc.xdmf")) as f:
                f.write_checkpoint(alpha, "alpha-{}".format(it), 0, append = True)
                log(LogLevel.PROGRESS, 'PROGRESS: written step {}'.format(it))


        time_data_pd.to_json(os.path.join(outdir, "time_data.json"))

        # if size == 1:
        #     plt.figure()
        #     dolfin.plot(alpha, marker='o')
        #     plt.savefig(os.path.join(outdir, "alpha-{}.png".format(it)))
        #     plt.figure()
        #     dolfin.plot(u, marker='o')
        #     plt.savefig(os.path.join(outdir, "u-{}.png".format(it)))
        #     plt.clf()

        # if not stable and parameters['experiment']['break-if-unstable']: 
        #     print('Unstable state, breaking', parameters['experiment']['break-if-unstable'])
        #     break


    # print(time_data_pd)
    # print(time_data_pd['stable'])


    xs = np.linspace(-Lx/2., Lx/2., 100)
    profile = np.array([alpha(x) for x in xs])
    plt.figure()
    plt.plot(xs, profile, marker='o')
    # plt.plot(xs, np.array([u(x) for x in xs]))
    plt.savefig(os.path.join(outdir, 'profile.pdf'))
    # # solve optimal profile
    #   # Alternate minimisation solver
    # beta = dolfin.TestFunction(V_alpha)
    # dalpha = dolfin.TrialFunction(V_alpha)

    # F = (ell*alpha.dx(0)*alpha.dx(0) + w/ell*alpha)*dx
    # dF = dolfin.derivative(F,alpha,beta); ddF = dolfin.derivative(dF,alpha,dalpha)
    # # bcs_u = [dolfin.DirichletBC(V_u, dolfin.Constant(0.), 'on_boundary')]
    alpha = dolfin.Function(V_alpha)
    u = dolfin.Function(V_u)
    bcs_alpha = [dolfin.DirichletBC(V_alpha, dolfin.Constant(1.), right), dolfin.DirichletBC(V_alpha, dolfin.Constant(0.), left)]
    # solver = solvers.AlternateMinimizationSolver(energy, [u, alpha], [[], bcs_alpha], parameters=parameters['alt_min'])
    # solver.solve()
    # print('DEBUG: h1 norm alpha profile {}'.format(dolfin.norm(alpha, 'h1')))
    # ub = dolfin.interpolate(dolfin.Constant(1.), V_alpha); lb = dolfin.interpolate(dolfin.Constant(0.), V_alpha)
    # profile = dolfin.NonlinearVariationalProblem(dF, alpha, bcs_alpha, J = ddF)
    # profile.set_bounds(lb, ub)
    # solver_nl = dolfin.NonlinearVariationalSolver(profile)
    # snes_solver_parameters_bounds = {"nonlinear_solver": "snes",
    #                       "snes_solver": {"linear_solver": "cg",
    #                                       "maximum_iterations": 100,
    #                                       "report": True,
    #                                       "line_search": "basic",
    #                                       "method":"vinewtonrsls",
    #                                       "absolute_tolerance":1e-6,
    #                                       "relative_tolerance":1e-6,
    #                                       "solution_tolerance":1e-6}}
    # solver_nl.parameters.update(snes_solver_parameters_bounds)
    # solver_nl.solve()

    xs = np.linspace(-Lx/2., Lx/2., 100)
    profile = np.array([alpha(x) for x in xs])
    plt.figure()
    plt.plot(xs, profile, marker='o')
    # plt.plot(xs, np.array([u(x) for x in xs]))
    # plt.savefig(os.path.join(outdir, 'profile.pdf'))
    # import pdb; pdb.set_trace()

    return time_data_pd, outdir

if __name__ == "__main__":

    import argparse
    from urllib.parse import unquote
    from time import sleep

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False,
                        help="JSON configuration string for this experiment")
    parser.add_argument("--ell", type=float, default=0.1)
    parser.add_argument("--load_max", type=float, default=3.0)
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
    parser.add_argument("--parameters", type=str)
    parser.add_argument("--print", type=bool, default=False)
    parser.add_argument("--continuation", type=bool, default=False)
    parser.add_argument("--breakifunstable", type=bool, default=False)

    args, unknown = parser.parse_known_args()
    if len(unknown):
        log(LogLevel.WARNING, 'Unrecognised arguments:')
        print(unknown)
        log(LogLevel.WARNING, 'continuing in 3s')
        sleep(3)

    if args.outdir == None:
        args.postfix += '-cont' if args.continuation==True else ''
        outdir = "../output/{:s}{}".format('1d-traction',args.postfix)
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

    # config = '{}'
    # import pdb; pdb.set_trace()

    if args.config:
        print(args.config)
        config = unquote(args.config).replace('\'', '"')
        # config = config.replace('False', '"False"')
        # config = config.replace('True', '"True"')

    if args.parameters is not None:
        experiment = ''
        with open(os.path.join('../parameters', args.parameters)) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # config = unquote(config).replace('\'', '"')
        print(config)
        data, location = traction_1d(outdir=outdir, parameters=config)
    else:
        data, location = traction_1d(
            outdir=outdir,
            configString=config
        )



    print('Postprocess')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
    import numpy as np
    import sympy as sp
    import sys, os, sympy, shutil, math
    # import xmltodict
    # import pickle
    import json
    # import pandas
    import pylab
    from os import listdir
    import pandas as pd
    import visuals
    import postprocess as pp

    experiment = location

    params, data, signature = pp.load_data(experiment)
    # E0 = material_parameters['E']

    with open(os.path.join(location, 'parameters.yaml')) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    material_parameters = parameters['material']
    Lx = parameters['geometry']['Lx']
    Ly = 1.

    w1 = material_parameters['sigma_D0'] ** 2 / material_parameters['E']
    tc = np.sqrt(w1/material_parameters['E'])
    ell = material_parameters['ell']
    # import pdb; pdb.set_trace()
    lab = '\\ell={}, E={}, \\sigma_D = {}'.format(material_parameters['ell'], material_parameters['E'],material_parameters['sigma_D0'])
    # params['geometry']['Ly'] =1
    fig1, ax1 =pp.plot_energy(parameters, data, tc)
    # visuals.setspines2()
    mu = material_parameters['E']/2.
    En0 = w1 * Lx * Ly
    # elast_en = [1./2.*2.*mu*eps**2 for eps in data['load']]

    elast_en = [1./2.*material_parameters['E']*eps**2/En0 for eps in data['load']]

    plt.plot(data['load'], elast_en, c='k', lw=1, label='analytic')
    plt.legend()
    # ax1.axvline(pp.t_stab(ell), c='w', ls='-', lw=.5, label='$t^{cr}_s$')
    # ax1.axvline(pp.t_bif(ell), c='w', ls='-.', lw=.5, label=r'$t^{cr}_b$')
    plt.title('${}$'.format(lab))

    # (fig2, ax1, ax2) =pp.plot_spectrum(params, location, data, tc)
    # ax1.axvline(pp.t_stab(ell), c='k', ls='-', lw=2, label='$t^{cr}_s$')
    # ax1.axvline(pp.t_bif(ell), c='k', ls='-.', lw=2, label=r'$t^{cr}_b$')

    # plt.plot(np.linspace(1, params['loading']['load_max'], 30),
    #          [1- (t/material_parameters['sigma_D0']/material_parameters['E'])**(2/(1-2)) for t in np.linspace(1, params['loading']['load_max'], 30)],
    #         c='k', lw=.5)
    # plt.title('${}$'.format(lab))
    # # visuals.setspines2()
    # ax1.set_ylim(-1., .2)

    # fig1.savefig("/Users/kumiori/Documents/WIP/paper_stability/fig/energy-traction-{}.pdf".format(signature), bbox_inches='tight')
    # fig2.savefig("/Users/kumiori/Documents/WIP/paper_stability/fig/energy-spectrum-{}.pdf", bbox_inches='tight')
    fig1.savefig(os.path.join(location, "energy-traction-{}.pdf".format(signature)), bbox_inches='tight')
    # fig2.savefig(os.path.join(location, "energy-spectrum-{}.pdf".format(signature)), bbox_inches='tight')


    mesh = dolfin.Mesh(comm, os.path.join(experiment, 'mesh.xml'))
    fig = plt.figure()
    dolfin.plot(mesh)
    # visuals.setspines2()
    fig.savefig(os.path.join(location, "mesh.pdf"), bbox_inches='tight')
