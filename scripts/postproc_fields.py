import matplotlib.pyplot as plt
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
# import visuals
import dolfin

import os.path

print('postproc')
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_data(rootdir):
	with open(rootdir + '/parameters.pkl', 'r') as f:
		params = json.load(f)

	with open(rootdir + '/time_data.json', 'r') as f:
		data = json.load(f)
		dataf = pd.DataFrame(data).sort_values('load')

	if os.path.isfile(rootdir + '/signature.md5'):
#         print('sig file found')
		with open(rootdir + '/signature.md5', 'r') as f:
			signature = f.read()
	else:
		print('no sig file found')
		signature = hashlib.md5(str(params).encode('utf-8')).hexdigest()

	return params, dataf, signature 



# rootdir = '../output/traction-shortbar-dc94bca726c96b13a96912e21834b02a'
# rootdir = 'test-b73266c3fa7937d06a1b4b6ea4f01dca'
# outdir = 'test-'

# rootdir = '/Users/kumiori/Documents/WIP/paper_stability_code/output/traction-047c3f35cb3e998158cc243504ffd36a'
# rootdir = '/Users/kumiori/Documents/WIP/paper_stability_code/output/traction-9c1720ed3a8527baf2c8825d8a7c6ae6'
rootdir = '/Users/kumiori/Documents/WIP/paper_stability_code/output/traction-longbar-1f11d5cf5b3ae17a5062991bf39faeb5'

params, data, signature = load_data(rootdir)

mesh = dolfin.Mesh(comm, os.path.join(rootdir, 'mesh.xml'))
V_alpha = dolfin.FunctionSpace(mesh, "CG", 1)

load_min = params['time_stepping']['load_min']
load_max = params['time_stepping']['load_max']
nsteps = params['time_stepping']['nsteps']
print(nsteps)
load_steps = np.linspace(load_min, load_max, nsteps)
beta0 = dolfin.Function(V_alpha)
alpha = dolfin.Function(V_alpha)
alpha_old = dolfin.Function(V_alpha)
alpha_bif = dolfin.Function(V_alpha)

maxmodes = 2


perturbations = []

Lx = params['geometry']['Lx']
Ly = params['geometry']['Ly']
xs = np.linspace(-Lx/2, Lx/2, 100)
h0 = 0.

try:
	with dolfin.XDMFFile(os.path.join(rootdir, "bifurcation_postproc.xdmf")) as file:
		file.read_checkpoint(beta0, 'beta0')
		file.read_checkpoint(alpha, 'alpha')
		file.read_checkpoint(alpha_old, 'alpha-old')
		file.read_checkpoint(alpha_bif, 'alpha-bif')

	for field, name in zip(fields, ['beta0', 'alpha_old', 'alpha_bif', 'alpha']):
		fieldv = [field(x, h0) for x in xs]
		np.save(os.path.join(rootdir, name), fieldv,
			allow_pickle=True, fix_imports=True)

	fields = [beta0, alpha_old, alpha_bif, alpha]
		# for n in range(maxmodes):
		# 	modename = 'beta-%d'%n
		# 	print(modename)
		# 	file.read_checkpoint(betan, modename)

		# 	perturbations.append(betan)
		# 	betanv = [betan(x, h0) for x in xs]
		# 	np.save(os.path.join(rootdir, "beta-{}".format(n)), betanv,
		# 		allow_pickle=True, fix_imports=True)


except:
	print('no bifurcation data found')
# else:
# 	pass
# finally:
# 	pass



# nmodes = len(perturbations)


alpha = dolfin.Function(V_alpha)
alphas = []
for (step, load) in enumerate(load_steps):
	if not step % 10:
		with dolfin.XDMFFile(os.path.join(rootdir, "output_postproc.xdmf")) as file:
			print('reading step', step)
			file.read_checkpoint(alpha, 'alpha-{}'.format(step), 0)
			# dolfin.plot(alpha, vmin=0, vmax=1)
			# plt.savefig('_alpha_{}.pdf'.format(step))
			alphav = [alpha(x, h0) for x in xs]
			# print(alphav)
			alphas.append(alphav)

np.save(os.path.join(rootdir, "alpha"), alphas,
	allow_pickle=True, fix_imports=True)

data = np.load(os.path.join(rootdir, "alpha.npy".format(0)))

# ------------

import pdb; pdb.set_trace()

z = np.polyfit(htest, en, m)
p = np.poly1d(z)

3
np.linspace(hmin, hmax, 4)





