# sys.path.append("../src/")
# from post_processing import compute_sig, local_project
import site
import sys
sys.path.append("../src/")


import pandas as pd

import sys
# from linsearch import LineSearch
# import solvers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mshr
# import mshr
import dolfin
from dolfin import MPI
import os
# import pandas as pd
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

from slepc4py import SLEPc
from dolfin.cpp.log import log, LogLevel, set_log_level
dolfin.parameters["std_out_all_processes"] = False

# from dolfin import NonlinearProblem, derivative, \
#         TrialFunction, TestFunction, inner, assemble, sqrt, \
#         Constant, interpolate, RectangleMesh, Point

from dolfin import *
import yaml
from utils import get_versions
code_parameters = get_versions()

set_log_level(LogLevel.INFO)

from solvers import EquilibriumAM
from solver_stability import StabilitySolver

def numerical_test(
    user_parameters,
    ell=0.05,
    nu=0.,
):

    # Create mesh and define function space
    # Lx = 1.; Ly = .1
    n = 3
    geometry_parameters = {'Lx': 1., 'Ly': .1, 'n': 3}
    Lx = geometry_parameters['Lx']; Ly = geometry_parameters['Ly']

    comm = MPI.comm_world
    # mesh = RectangleMesh(Point(-Lx/2, 0.), Point(Lx/2, Ly), 50, 10)
    geom = mshr.Rectangle(dolfin.Point(-Lx/2., -Ly/2.), dolfin.Point(Lx/2., Ly/2.))
    mesh = mshr.generate_mesh(geom,  int(float(n * Lx / ell)))

    # Define Dirichlet boundaries
    outdir = '../test/output/test_firstorderevo'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    meshf = dolfin.File(os.path.join(outdir, "mesh.xml"))
    meshf << mesh

    with open('../parameters/form_compiler.yml') as f:
        form_compiler_parameters = yaml.load(f, Loader=yaml.FullLoader)

    with open('../parameters/solvers_default.yml') as f:
        solver_parameters = yaml.load(f, Loader=yaml.FullLoader)

    with open('../parameters/tractionbar.yml') as f:
        material_parameters = yaml.load(f, Loader=yaml.FullLoader)['material']

    with open('../parameters/loading.yaml') as f:
        loading_parameters = yaml.load(f, Loader=yaml.FullLoader)['loading']

    Path(outdir).mkdir(parents=True, exist_ok=True)

    print('Outdir is: '+outdir)


    default_parameters = {
        'code': {**code_parameters},
        'compiler': {**form_compiler_parameters},
        'geometry': {**geometry_parameters},
        'loading': {**loading_parameters},
        'material': {**material_parameters},
        'solver':{**solver_parameters},
        }

    # import pdb; pdb.set_tracec()

    default_parameters.update(user_parameters)
    # FIXME: Not nice
    parameters = default_parameters

    with open(os.path.join(outdir, 'parameters.yaml'), "w") as f:
        yaml.dump(parameters, f, default_flow_style=False)

    savelag = 1
    left = dolfin.CompiledSubDomain("near(x[0], -Lx/2.)", Lx=Lx)
    right = dolfin.CompiledSubDomain("near(x[0], Lx/2.)", Lx=Lx)
    left_bottom_pt = dolfin.CompiledSubDomain("near(x[0],-Lx/2.) && near(x[1],-Ly/2.)", Lx=Lx, Ly=Ly)

    mf = dolfin.MeshFunction("size_t", mesh, 1, 0)
    right.mark(mf, 1)
    left.mark(mf, 2)

    ds = dolfin.Measure("ds", subdomain_data=mf)
    dx = dolfin.Measure("dx", metadata=form_compiler_parameters, domain=mesh)

    # Function Spaces
    V_u = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)
    u = dolfin.Function(V_u, name="Total displacement")
    alpha = Function(V_alpha)
    dalpha = TrialFunction(V_alpha)

    state = {'u': u, 'alpha': alpha}
    Z = dolfin.FunctionSpace(mesh, 
            dolfin.MixedElement([u.ufl_element(),alpha.ufl_element()]))
    z = dolfin.Function(Z)
    v, beta = dolfin.split(z)

    ut = dolfin.Expression("t", t=0.0, degree=0)
    bcs_u = [dolfin.DirichletBC(V_u.sub(0), dolfin.Constant(0), left),
             dolfin.DirichletBC(V_u.sub(0), ut, right),
             dolfin.DirichletBC(V_u, (0, 0), left_bottom_pt, method="pointwise")]

    bcs_alpha_l = DirichletBC(V_alpha,  Constant(0.0), left)
    bcs_alpha_r = DirichletBC(V_alpha, Constant(0.0), right)
    # bcs_alpha =[bcs_alpha_l, bcs_alpha_r]
    bcs_alpha = []

    bcs = {"damage": bcs_alpha, "elastic": bcs_u}

    # Problem definition
    k_res = parameters['material']['k_res']
    a = (1 - alpha) ** 2. + k_res
    w_1 = parameters['material']['sigma_D0'] ** 2 / parameters['material']['E']
    w = w_1 * alpha
    eps = sym(grad(u))
    lmbda0 = parameters['material']['E'] * parameters['material']['nu'] /(1. - parameters['material']['nu'])**2.
    mu0 = parameters['material']['E']/ 2. / (1.0 + parameters['material']['nu'])
    Wu = 1./2.* lmbda0 * tr(eps)**2. + mu0 * inner(eps, eps)

    energy = a * Wu * dx + w_1 *( alpha + \
            parameters['material']['ell']** 2.*inner(grad(alpha), grad(alpha)))*dx
    # import pdb; pdb.set_trace()

    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_out.parameters["functions_share_mesh"] = True
    file_out.parameters["flush_output"] = True

    with open('../parameters/solvers_default.yml') as f:
        solver_parameters = yaml.load(f, Loader=yaml.FullLoader)

    solver = EquilibriumSolver(energy, state, bcs, parameters=solver_parameters)

    load_steps = np.linspace(parameters['loading']['load_min'],
        parameters['loading']['load_max'],
        parameters['loading']['n_steps'])
    time_data = []
    time_data_pd = []
    spacetime = []


    # compute signature (incl bcs)
    # dump parameters?

    for it, load in enumerate(load_steps):
        log(LogLevel.CRITICAL, 'CRITICAL: Solving load t = {:.2f}'.format(load))
        ut.t = load
        (time_data_i, am_iter) = solver.solve()
        solver.update()

        time_data_i["load"] = load
        # time_data_i["stable"] = stable

        time_data_i["elastic_energy"] = dolfin.assemble(
            1./2.* material_parameters['E']*a*eps**2. *dx)
        time_data_i["dissipated_energy"] = dolfin.assemble(
            (w + w_1 * material_parameters['ell'] ** 2. * inner(grad(alpha), grad(alpha)))*dx)

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

        spacetime.append(get_trace(alpha))

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
    plt.clf()

    plot(mesh)
    plt.savefig(os.path.join(outdir, 'mesh.pdf'))
    plt.clf()

    from matplotlib.ticker import FuncFormatter, MaxNLocator
    plot(alpha)
    plt.savefig(os.path.join(outdir, 'alpha.pdf'))
    plt.clf()
    log(LogLevel.INFO, "Saved figure: {}".format(os.path.join(outdir, 'alpha.pdf')))


    xs = np.linspace(-Lx/2., Lx/2., 100)
    profile = np.array([alpha(x, 0) for x in xs])

    plt.figure()
    plt.plot(xs, profile, marker='o')
    plt.plot(xs, np.array([u(x, 0) for x in xs]))
    # plt.ylim(0., 1.)
    plt.savefig(os.path.join(outdir, 'profile.pdf'))



    return time_data_pd, outdir

def get_trace(alpha, xresol = 100):
    X =alpha.function_space().tabulate_dof_coordinates()
    xs = np.linspace(min(X[:, 0]),max(X[:, 0]), xresol)
    alpha0 = [alpha(x, 0) for x in xs]

    return alpha0

if __name__ == "__main__":

    # Parameters
    with open('../parameters/tractionbar.yml') as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    data, experiment = numerical_test(user_parameters = parameters)

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
    import pdb; pdb.set_trace()
    fig1, ax1 =pp.plot_energy(parameters, data, tc)
    # visuals.setspines2()
    print(data['elastic_energy'])
    mu = parameters['material']['E']/2.
    # elast_en = [1./2.*2.*mu*eps**2 for eps in data['load']]
    # Lx = 1.
    # Ly = .1
    # Omega = Lx*Ly
    elast_en = [1./2.*parameters['material']['E']*eps**2 for eps in data['load']]
    plt.plot(data['load'], elast_en, c='k', label='analytic')
    plt.legend()

    plt.ylim(0, 1.)
    plt.title('${}$'.format(lab))


    fig1.savefig(os.path.join(experiment, "energy.pdf"), bbox_inches='tight')

