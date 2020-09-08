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
import visuals
import hashlib

import os.path

print('postproc')
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

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

def plot_spectrum(params, data, tc, ax=None, tol=1e-12):
	E0 = params['material']['E']
	w1 = params['material']['sigma_D0']**2/E0
	ell = params['material']['ell']
	fig = plt.figure()
	for i,d in enumerate(data['eigs']):
		if d is not (None and np.inf and np.nan):
			lend = len(d) if isinstance(d, list) else 1
			plt.scatter([(data['load'].values)[i]]*lend, d,
					   c=np.where(np.array(d)<tol, 'red', 'C2'))
					   # c=np.where(np.array(d)<tol, 'C1', 'C2'))

	plt.ylim(-6e-4, 3e-4)
	plt.axhline(0, c='k', lw=2.)
	plt.xlabel('$t$')
	plt.ylabel('Eigenvalues')
	#     plt.ylabel('$$\\lambda_m$$')
	plt.axvline(tc, lw=.5, c='k')
	ax1 = plt.gca()
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
	ax2 = plt.twinx()
	ax2.plot(data['load'].values, data['alpha_max'].values, label='$$max(\\alpha)$$')
	ax2.legend()
	tbif = t_bif(ell)
	tstab = t_stab(ell)
	ax2.set_ylabel('max $\\alpha$')
	ax2.set_ylim(0, 1)
	ax = plt.gca()
	ax.axvline(t_stab(ell), c='k', ls='-', lw=2, label='$t^{cr}_s$')
	ax.axvline(t_bif(ell), c='k', ls='-.', lw=2, label=r'$t^{cr}_b$')
	ax.set_xlim(params['time_stepping']['load_min'], params['time_stepping']['load_max'])

	stable = data['stable'].values

	ax2.scatter(data['load'].values[stable],  -1.+data['stable'].values[stable], c='k', marker='s', s=70, label='stable')
	ax2.scatter(data['load'].values[~stable],     data['stable'].values[~stable], c='red', marker='s', s=70, label='unstable')


	plt.legend(loc="upper left")


	# ax1.get_yaxis().set_major_formatter(ScalarFormatter())

	return fig, ax1, ax2


def plot_sigmaeps(params, dataf, tc):
	E0 = params['material']['E']
	w1 = params['material']['sigma_D0']**2/E0
	ell = params['material']['ell']
	L = params['geometry']['Lx']
	Ly = params['geometry']['Ly']
	
	fig = plt.figure()

	t = np.linspace(0., params['time_stepping']['load_max'], 100)
	fig = plt.figure()
	plt.ylabel('$$\sigma$$')
	plt.xlabel('$$t$$')

	# plt.plot(dataf['load'].values,
		# dataf['load'].values*pow(dataf['S(alpha)'].values, -1), marker='o', label='$$\sigma$$')

	plt.plot(dataf['load'].values,
		dataf['load'].values*dataf['A(alpha)'].values*E0/Ly, marker='o', label='$$\sigma$$')

	stable = dataf['stable'].values


	ax = plt.gca()
	ax.axvline(tc, c='k', lw=.5, label='$t^{cr}$')
	ax.axvline(t_stab(ell), c='k', ls='-', lw=2, label='$t^{cr}_s$')
	ax.axvline(t_bif(ell), c='k', ls='-.', lw=2, label=r'$t^{cr}_b$')
	ax.set_xlim(params['time_stepping']['load_min'], params['time_stepping']['load_max'])	
	plt.scatter(dataf['load'].values[stable], -1.+dataf['stable'].values[stable], c='k', marker='s', s=70, label='stable')
	plt.scatter(dataf['load'].values[~stable], dataf['stable'].values[~stable], c='red', marker='s', s=70, label='unstable')

	plt.legend(loc="upper left")

	return fig, ax

def plot_energy(params, dataf, tc):
	E0 = params['material']['E']
	w1 = params['material']['sigma_D0']**2/E0
	ell = params['material']['ell']
	fig = plt.figure()

	t = np.linspace(0., 3., 100)
	fig = plt.figure()
	plt.ylabel('Energy')
	plt.xlabel('$$t$$')

	plt.plot(dataf['load'].values,
		dataf['dissipated_energy'].values, marker='o', label='dissipated')

	plt.plot(dataf['load'].values,
		dataf['elastic_energy'].values, marker='o', lw=2, label='elastic')

	plt.plot(dataf['load'].values,
		(dataf['elastic_energy'].values+dataf['dissipated_energy'].values), marker='o', label='total')
	ax = plt.gca()
	ax.axvline(tc, c='k', lw=.5, label='$t^{cr}$')
	print(t_stab(ell))
	print(t_bif(ell))
	ax.axvline(t_stab(ell), c='k', ls='-', lw=2, label='$t^{cr}_s$')
	ax.axvline(t_bif(ell), c='k', ls='-.', lw=2, label=r'$t^{cr}_b$')
	ax.set_xlim(params['time_stepping']['load_min'], params['time_stepping']['load_max'])
	ax.set_ylim(0, w1*params['geometry']['Lx'])
	plt.legend()

	
	ax.get_yaxis().set_major_formatter(ScalarFormatter())
	ax.ticklabel_format(axis='both', style='plain', useOffset=True)

	return fig, ax


def plot_stability(prefix, tol=1e-5):
	import matplotlib.patches as patches

	# dirtree = os.path.join(dirroot, signature)
	fig = plt.figure()
	stab_diag = []
	global_dfs = []
	debug = False
	for subdir, dirs, files in os.walk(prefix):
		if not os.path.isfile(subdir + "/parameters.pkl"):
			print('file not found {}'.format(subdir + "/parameters.pkl"))
			continue
		with open(subdir + '/parameters.pkl', 'r') as f: 
			params = json.load(f)
			ell = params['material']['ell']
		if not os.path.isfile(subdir + "/time_data.json"):
			print('file not found {}'.format(subdir + "/time_data.json"))
			continue
		with open(subdir + "/time_data.json") as f:
			data = json.load(f)
			df = pd.DataFrame(data).sort_values('load')
			mineig = [min(eigs) if isinstance(eigs, (list,)) else 100 for eigs in df['eigs']]

			tol = tol
			# print(mineig)
			# print(np.array(mineig) < tol)
			loads = df['load'][np.where(np.array(mineig) < tol)[0]].values
			plt.plot(loads, [1/ell]*len(loads), c='C1', marker='+')
			# label='$\\lambda_{min}<\\eta_{tol}$')
			loads = df['load'][np.where(np.array(mineig) < 0)[0]].values
			plt.plot(loads, [1/ell]*len(loads), c='C1', marker='X')
			loads = df['load'][np.where(np.array(mineig) > tol)[0]].values
			plt.plot(loads, [1/ell]*len(loads), c='C2', marker='.')
			if debug:
				print('1/ell, mineog', 1/ell, mineig)
				print('nonunique loads')
				print(1/ell, np.where(np.array(mineig) < tol)[0])
				print('unstable')
				print(1/ell, np.where(np.array(mineig) < 0)[0])

	plt.plot((20, 20), (20, 20), ls='-', c='C1', marker='+', label='$\\lambda_0<{}$'.format(tol))
	plt.plot((20, 20), (20, 20), ls='', c='C1', marker='X', label='incr. unstable')
	plt.plot((20, 20), (20, 20), ls='', c='C2', marker='.', label='stable, unique')

	q=2
	coeff_sta = 2.*np.pi*q/(q+1)**(3./2.)*np.sqrt(2)
	coeff_bif = coeff_sta*(q+1)/(2.*q)
	loads = np.linspace(1., 10., 100)

	ax = plt.gca()
	# ax.plot(loads, [2.*2.*np.pi*q/(q+1)**(3./2.)*np.sqrt(2)/i for i in loads], lw=3, c='k's)
	ax.plot(loads, [coeff_sta/i for i in loads], '-', c='k', label='$$t_s^{cr}(\ell)$$')
	ax.plot(loads, [coeff_bif/i for i in loads], '-.', c='k', label='$$t^{cr}_b(\ell)$$')
	plt.axvline(1.0, c='k')

	ax.fill_betweenx([coeff_sta/i for i in loads], loads, 20., alpha=.3)
	ax.fill_betweenx([coeff_bif/i for i in loads], 0, loads, alpha=.3)
	ax.fill_betweenx([coeff_bif/i for i in loads], 1, loads, alpha=.3, facecolor='C0')

	ax.add_patch(patches.Rectangle((0, coeff_bif), 1, 10, facecolor = 'C1',fill=True, alpha=.3))
	ax.add_patch(patches.Rectangle((1, coeff_sta), 10, 10, facecolor = 'C0',fill=True, alpha=.3))
	ax.add_patch(patches.Rectangle((0, 0), 10, 1./loads[-2]*coeff_bif, facecolor = 'C1',fill=True, alpha=.3))
	ax.add_patch(patches.Rectangle((1, 0), loads[-1], coeff_bif/(loads[-1]), facecolor = 'C0',fill=True, alpha=.3))

	# ax.text(.55, .15, r'Elastic', fontsize=25)
	# ax.text(1.2, .15, r'Homogeneous (unique)', fontsize=25)
	# (?# ax.text(3.5, 2.5, r'Unstable', fontsize=25))
	plt.legend(loc='upper right')
	
	plt.xlabel('$t$')
	plt.ylabel('$$L/\ell$$')
	plt.ylim(0., 1.5*coeff_sta)
	plt.xlim(0., max(loads))

	ax.set_yticks([0, 1, 1/.5, 1/.25, coeff_sta, coeff_bif])
	ax.set_yticklabels(['0','1','2','4', '$$\\ell_s$$', '$$\\ell_b$$'])
	
	visuals.setspines()
	plt.loglog()
	plt.ylim(0.5, 3*coeff_sta)
	plt.xlim(0.5, max(loads))
	return fig

def load_cont(prefix):
	with open(prefix + '/continuation_data.json', 'r') as f:
		data = json.load(f)
		dataf = pd.DataFrame(data).sort_values('iteration')
	return dataf

def t_stab(ell, q=2):
	coeff = 2.*np.pi*q/(q+1)**(3./2.)*np.sqrt(2)
	if 1/ell > coeff:
#     print(1/ell, coeff)
		return 1.
	else:
		return coeff*ell

def t_bif(ell, q=2):
	coeff = t_stab(ell, q)*(q+1)/(2.*q)*np.sqrt(2)
	if 1/ell > coeff:
#     print(1/ell, coeff)
		return 1.
	else:
		return coeff*ell/1


def format_params(params):
	return ('\\ell ={:.2f},'.format(params['material']['ell']),             \
		'\\nu ={:.0f},'.format(params['material']['nu']),                  \
		'\\sigma_D0 ={:.0f},'.format(params['material']['sigma_D0']),     \
		'E ={:.0f}'.format(params['material']['E']))



