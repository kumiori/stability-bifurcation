import sys
sys.path.append("../src/")
from utils import ColorPrint, get_versions
from damage_elasticity_model import DamageElasticityModel, UnilateralDamageElasticityModel
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
                           # "ksp_type": "preonly"  # "tao_ls_type": "more-thuente"
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
    nu=0.0,
    E=1.,
    load_min=0,
    load_max=2,
    loads=None,
    nsteps=20,
    Lx=1,
    Ly=0.1,
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
    Ly = Ly
    load_min = load_min
    load_max = load_max
    nsteps = nsteps
    outdir = outdir
    loads=loads

    savelag = 1
    nu = dolfin.Constant(nu)
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
        "nu": nu.values()[0],
        "sigma_D0": sigma_D0.values()[0]},
    'geometry': {
        'Lx': Lx,
        'Ly': Ly,
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
    parameters['time_stepping']['outdir']=outdir
    Path(outdir).mkdir(parents=True, exist_ok=True)
    print('Outdir is: '+outdir)

    with open(os.path.join(outdir, 'rerun.sh'), 'w') as f:
        configuration = deepcopy(parameters)
        configuration['time_stepping'].pop('outdir')
        str(configuration).replace("\'True\'", "True").replace("\'False\'", "False")
        rerun_cmd = 'python3 traction.py --config="{}"'.format(configuration)
        f.write(rerun_cmd)

    with open(os.path.join(outdir, 'parameters.pkl'), 'w') as f:
        json.dump(parameters, f)

    with open(os.path.join(outdir, 'signature.md5'), 'w') as f:
        f.write(signature)


    geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))
    mesh = mshr.generate_mesh(geom,  int(float(n * Lx / ell)))

    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    bottom = dolfin.CompiledSubDomain("near(x[1],-Ly/2.)", Ly=Ly)
    top = dolfin.CompiledSubDomain("near(x[1],Ly/2.)", Ly=Ly)
    left_bottom_pt = dolfin.CompiledSubDomain("near(x[0],-Lx/2.) && near(x[1],-Ly/2.)", Lx=Lx, Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)
    bottom.mark(mf, 3)
    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_u = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = dolfin.Function(V_alpha, name="Damage")
    state = [u, alpha]

    Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)

    v, beta = dolfin.split(z)

    # BCs (homogenous version needed for residual evaluation)
    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u.sub(0), dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u.sub(0), ut, right),
             dolfin.DirichletBC(V_u, (0, 0), left_bottom_pt, method="pointwise")]

    bcs_alpha = []

    # Files for output
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
    model = DamageElasticityModel(state, E0, nu, ell, sigma_D0)
    model.dx = dx
    model.ds = ds
    energy = model.total_energy_density(u, alpha)*model.dx

    # Alternate minimization solver
    solver = solvers.AlternateMinimizationSolver(energy,
        [u, alpha], [bcs_u, bcs_alpha], parameters=parameters['alt_min'])

    rP = model.rP(u, alpha, v, beta)*model.dx
    rN = model.rN(u, alpha, beta)*model.dx

    stability = StabilitySolver(mesh, energy,
        [u, alpha], [bcs_u, bcs_alpha], z, rayleigh=[rP, rN], parameters = parameters['stability'])
    # stability = StabilitySolver(mesh, energy, [u, alpha], [bcs_u, bcs_alpha], z, parameters = parameters['stability'])

    # Time iterations
    load_steps = np.linspace(load_min, load_max, nsteps)

    if loads:
        load_steps = loads

    stability.parameters['checkstability'] = True

    class TractionTS(TimeStepping):
        """docstring for Evolution"""
        def __init__(self,
                    model,
                    solver,
                    stability,
                    load_param,
                    outfiles,
                    parameters,
                    user_density=None):

            super(TractionTS, self).__init__(model,
                    solver,
                    stability,
                    load_param,
                    outfiles,
                    parameters,
                    user_density)
            self.spacetime = pd.DataFrame(columns = self.load_steps, )

        def user_postprocess_stability(self, load):
            from matplotlib.ticker import StrMethodFormatter
            outdir = self.parameters['outdir']
            alpha = self.solver.alpha
            if size > 1: return

            adm_pert = np.where(np.array([e['en_diff'] for e in stability.eigendata]) < 0)[0]

            fig = plt.figure(figsize=(4, 1.5), dpi=180,)
            ax = plt.gca()
            X =alpha.function_space().tabulate_dof_coordinates()
            xs = np.linspace(min(X[:, 0]),max(X[:, 0]), 300)
            ax.plot(xs, [alpha(x, 0) for x in xs], label='$\\alpha$', lw=1, c='k')
            ax.axhline(0., lw=.5, c='k', ls='-')
            ax3 = ax.twinx()
            ax.legend(fontsize='small')

            for mode in adm_pert:
                beta_n = stability.eigendata[adm_pert[mode]]['beta_n']
                ax3.plot(xs, [beta_n(x, 0) for x in xs], label='$\\beta_{}$'.format(mode), ls=':')

            for axi in [ax, ax3]:
                axi.spines['top'].set_visible(False)
                axi.spines['bottom'].set_visible(False)

            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 2 decimal places
            plt.xlabel('$x$')
            ax.set_ylabel('$\\alpha$')
            ax3.set_ylabel('$\\beta$')
            ax.set_ylim(0., 1.)
            ax.set_xlim(-.5, .5)
            ax3.legend(bbox_to_anchor=(0,-.45,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=len(adm_pert), frameon=False)

            fig.savefig(os.path.join(outdir, "profiles-{:.3f}.pdf".format(load)), bbox_inches="tight")

        def user_postprocess_timestep(self, load, xresol = 100):
            from matplotlib.ticker import FuncFormatter, MaxNLocator
            if size > 1: return
            alpha = self.solver.alpha
            alpha.set_allow_extrapolation(True)
            parameters = self.parameters
            xresol = xresol
            X =alpha.function_space().tabulate_dof_coordinates()
            xs = np.linspace(min(X[:, 0]),max(X[:, 0]), xresol)

            fig = plt.figure(figsize=(8, 6), dpi=180,)
            alpha0 = [alpha(x, 0) for x in xs]

            self.spacetime[load] = alpha0
            self.spacetime = self.spacetime.fillna(0)
            mat = np.matrix(self.spacetime)
            plt.imshow(mat, cmap = 'Greys', vmin = 0., vmax = 1., aspect=.1)
            plt.colorbar()

            def format_space(x, pos):
                return '$%1.1f$'%((-x+xresol/2)/xresol)

            def format_time(t, pos):
                return '$%1.1f$'%((t-parameters['load_min'])/parameters['nsteps']*parameters['load_max'])

            ax = plt.gca()

            ax.yaxis.set_major_formatter(FuncFormatter(format_space))
            ax.xaxis.set_major_formatter(FuncFormatter(format_time))

            plt.xlabel('$t$')
            plt.ylabel('$x$')
            fig.savefig(os.path.join(outdir, "spacetime.pdf".format(load)), bbox_inches="tight")

            self.spacetime.to_json(os.path.join(outdir + "/spacetime.json"))
            pass

    evo = TractionTS(model, solver, stability, ut, [file_out, file_con, file_eig], 
                parameters=parameters['time_stepping'])

    time_data_pd = evo.run()

    if size == 1:
        plt.figure()
        dolfin.plot(alpha)
        plt.savefig(os.path.join(outdir, "alpha.png"))
        plt.figure()
        dolfin.plot(u, mode="displacement")
        plt.savefig(os.path.join(outdir, "u.png"))
        plt.close('all')

    print(time_data_pd)

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
    parser.add_argument("--Ly", type=float, default=0.1)
    parser.add_argument("--nu", type=float, default=0.0)
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
            nu=args.nu,
            Lx=args.Lx,
            Ly=args.Ly,
            outdir=outdir,
            savelag=args.savelag,
            continuation=args.continuation,
            configString=config
        )
