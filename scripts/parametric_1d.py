from traction_1d import *
import numpy as np
from utils import ColorPrint

# ell_list = np.linspace(.1, .5, 20)
# ell_min = 0.1
#ell_max = 2.
ell_list = np.logspace(np.log10(.15), np.log10(1.5), 20)

def t_stab(ell, q=2):
	coeff_stab = 2.*np.pi*q/(q+1)**(3./2.)*np.sqrt(2)
	if 1/ell > coeff_stab:
		return 1.
	else:
		return coeff_stab*ell/1.


def t_bif(ell, q=2):
	# coeff = t_stab(ell, q)*(q+1)/(2.*q)	
	coeff_bif = 2.*np.pi*q/(q+1)**(3./2.)*np.sqrt(2)*(q+1)/(2.*q)
	if 1/ell > coeff_bif:
		return 1.
	else:
		return coeff_bif*ell/1.

print([t_stab(ell) for ell in ell_list])
print([t_bif(ell) for ell in ell_list])
print([3./4.*t_stab(ell) for ell in ell_list])

# sys.exit()
for ell in ell_list:
	# tstab = 1./ell*4*np.pi/3.**(3./2.)
	eps = .3
	ell_r = ell*np.sqrt(2)
	# *np.sqrt(2)
	tstab = t_stab(ell_r, 2)
	tbif = t_bif(ell_r, 2)
	print('tstab, tbif', tstab, tbif)
	# sys.exit(//z)
	# tstab = 1.
	# lmin = tstab - 1.
	# load_min = load_min if lmin > 0. else 0.
	# load_min = tstab - 1. - tstab/10 
	load_min = .5 
	# load_max = tstab + 1. + tstab/5
	load_max = 5.
	# loads = [tstab-2*eps, tstab+2*eps]
	ColorPrint.print_info('Solving ell {}'.format(ell))
	ColorPrint.print_info('Load: [{} {}]'.format(load_min, load_max))
	ColorPrint.print_info('stab limit: {} '.format(tstab))
	ColorPrint.print_info('uniq limit: {} '.format(tbif))
	loads = np.logspace(load_min, load_max, 50)
	try:
		traction_1d(
			ell=ell,
			load_min=load_min,
			load_max=load_max,
			# loads = loads,
			nsteps=50,
			n=7,
			# Lx=Lx,
			# outdir='../output/parametric-traction-plane-stress/ell-{:2f}'.format(ell),
			outdir='../output/parametric-traction-1d-validation-paper/ell-{:2f}'.format(ell),
			# outdir='../output/parametric-traction-1d-validation-paper-auto/ell-{:2f}'.format(ell),
			# outdir='../output/parametric-traction-n-10/ell-{:2f}'.format(ell),
			breakifunstable = True
		)
	except:
		ColorPrint.print_warn("Something went somewhere, most likely an instability")

