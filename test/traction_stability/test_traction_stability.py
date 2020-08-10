import sys
sys.path.append("../../src/")
from utils import ColorPrint
from damage_elasticity_model import DamageElasticityModel
import solvers
import matplotlib.pyplot as plt
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
import petsc4py
import json
from dolfin import set_log_level
from dolfin.cpp.log import log, LogLevel
from dolfin import PETScOptions
from slepc4py import SLEPc
from solver_stability import StabilitySolver

set_log_level(LogLevel.ERROR)

form_compiler_parameters = {
    "representation": "uflacs",
    "quadrature_degree": 2,
    "optimize": True,
    "cpp_optimize": True,
}

PETScOptions.set('tao_converged_reason', True)

petsc_options_alpha_tao = {"tao_type": "gpcg",
                           "tao_ls_type": "gpcg",
                           "tao_gpcg_maxpgits": 50,
                           "tao_max_it": 300,
                           "tao_steptol": 1e-7,
                           "tao_gatol": 1e-8,
                           "tao_grtol": 0.,
                           "tao_gttol": 0.,
                           "tao_catol": 0.,
                           "tao_crtol": 0.,
                           "tao_ls_ftol": 1e-6,
                           "tao_ls_gtol": 1e-6,
                           "tao_ls_rtol": 1e-6,
                           "ksp_rtol": 1e-6,
                           "tao_ls_stepmin": 1e-8,  #
                           "tao_ls_stepmax": 1e6,  #
                           "pc_type": "bjacobi",
                           "tao_monitor": "True",  # "tao_ls_type": "more-thuente"
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



stability_parameters = {"order": 4,
                        "projection": 'none',
                        'maxmodes': 5,
                        'checkstability': True,
                        'continuation': False,
                        'cont_rtol': 1e-5,
                        'inactiveset_atol': 1e-5
                        }

dolfin.parameters["std_out_all_processes"] = False
dolfin.parameters["form_compiler"].update(form_compiler_parameters)



def traction_test(
    ell=0.1,
    degree=1,
    n=3,
    nu=0.0,
    load_min=0,
    load_max=2,
    loads=None,
    nsteps=20,
    Lx=1,
    Ly=0.1,
    outdir="outdir",
    savelag=1,
):
    # constants
    ell = ell
    Lx = Lx
    Ly = Ly
    load_min = load_min
    load_max = load_max
    nsteps = nsteps
    loads=loads

    savelag = 1
    nu = dolfin.Constant(nu)
    ell = dolfin.Constant(ell)
    E0 = dolfin.Constant(1.0)
    sigma_D0 = E0
    n = n


    params =  { 
    'material': {
        "ell":   ell.values()[0],
        "E": E0.values()[0],
        "nu": nu.values()[0],
        "sigma_D0": sigma_D0.values()[0]},
    'geometry': {
        'Lx': Lx,
        'Ly': Ly,
        'n': n,
        },
    'load':{
        'min': load_min,
        'max': load_max,
        'nsteps':  nsteps
    } }

    print(params)
    geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))

    nel = max(int(n * float(Lx / ell)), int(Ly/3.))
    mesh = mshr.generate_mesh(geom, nel)

    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    right_bottom_pt = dolfin.CompiledSubDomain("near(x[1], Lx/2.) && near(x[0],-Ly/2.)", Lx=Lx, Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)
    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    V_u = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = dolfin.Function(V_alpha, name="Damage")
    state = [u, alpha]

    Z = dolfin.FunctionSpace(mesh, dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)

    v, beta = dolfin.split(z)

    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u.sub(0), dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u.sub(0), ut, right),
             dolfin.DirichletBC(V_u, (0, 0), right_bottom_pt, method="pointwise")]

    bcs_alpha = []

    # Problem definition
    model = DamageElasticityModel(state, E0, nu, ell, sigma_D0)
    energy = model.total_energy_density(u, alpha)*dx

    # Alternate minimization solver
    solver = solvers.AlternateMinimizationSolver(
        energy, [u, alpha], [bcs_u, bcs_alpha], parameters = alt_min_parameters)

    rP = model.rP(u, alpha, v, beta)*dx
    rN = model.rN(u, alpha, beta)*dx

    stability = StabilitySolver(mesh, energy,
        [u, alpha], [bcs_u, bcs_alpha], z, rayleigh=[rP, rN], parameters = stability_parameters)

    # Time iterations
    time_data = []
    load_steps = np.linspace(load_min, load_max, nsteps)
    alpha_old = dolfin.Function(V_alpha)

    for it, load in enumerate(load_steps):
        stable = None; negev = 0; mineig = np.inf; iteration = 0
        ut.t = load
        ColorPrint.print_pass('load: {:4f} step {:d} ell {:f}'.format(load, it, ell.values()[0]))
        alpha_old.assign(alpha)
        time_data_i, am_iter = solver.solve()
        solver.update()
        (stable, negev) = stability.solve(alpha_old)

        time_data_i["load"] = load
        time_data_i["stable"] = stable
        time_data_i["# neg ev"] = negev
        time_data_i["elastic_energy"] = dolfin.assemble(
            model.elastic_energy_density(model.eps(u), alpha)*dx)
        time_data_i["dissipated_energy"] = dolfin.assemble(
            model.damage_dissipation_density(alpha)*dx)
        time_data_i["eigs"] = stability.eigs if hasattr(stability, 'eigs') else np.inf
        time_data_i["max alpha"] = np.max(alpha.vector()[:])
        time_data.append(time_data_i)
        time_data_pd = pd.DataFrame(time_data)

        if stable == False:
            break

    return time_data_pd

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ell", type=float, default=0.1)
    parser.add_argument("--load_max", type=float, default=2.0)
    parser.add_argument("--load_min", type=float, default=0.0)
    parser.add_argument("--Lx", type=float, default=1)
    parser.add_argument("--Ly", type=float, default=0.1)
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--nsteps", type=int, default=100)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--postfix", type=str, default='')
    parser.add_argument("--savelag", type=int, default=1)
    args = parser.parse_args()

    traction_test(
        ell=args.ell,
        load_min=args.load_min,
        load_max=args.load_max,
        nsteps=args.nsteps,
        n=args.n,
        nu=args.nu,
        Lx=args.Lx,
        Ly=args.Ly,
        savelag=args.savelag,
    )