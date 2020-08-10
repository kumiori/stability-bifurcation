from contextlib import contextmanager
import unittest
import sys
import os
sys.path.append("../src/")
sys.path.append("./")
import numpy as np
from test_traction_stability import traction_test as traction

class TestTraction(unittest.TestCase):
	def test_stability(self):
		ell_list = .5*np.logspace(-1, 0, 10)
		rtol = 0.1
		t_stab = []
		for ell in ell_list:
			eps = .1
			tstab = self.stab_limit(ell)
			load_min = tstab - 3*eps
			load_max = tstab + 1.
# ------------- silent
			# with self.suppress_stdout():
			# 	t = traction(
			# 			ell=ell,
			# 			load_min=load_min,
			# 			load_max=load_max,
			# 			nsteps=30, n=5,
			# 			nu=0., Lx=1., Ly=.1)
			# idx = np.where(t['stable'] == False)[0][0]
			# t_stab.append([t['load'][idx], ell])
		# import pdb; pdb.set_trace()
# ------------- silent
#
			hmesh = 0.01
			t = traction(
					ell=ell,
					load_min=load_min,
					load_max=load_max,
					nsteps=30, n=5,
					# nsteps=30, n=ell/hmesh	,
					nu=0., Lx=1., Ly=.1)
			idx = np.where(t['stable'] == False)[0][0]
			t_stab.append([t['load'][idx], ell])

		stab_analytic = np.array([self.stab_limit(ell) for ell in ell_list])
		stab_numeric  = np.array([t[0] for t in t_stab])
		np.linalg.norm(stab_analytic - stab_numeric, 2)
		print('')
		print('critical stability loads, numeric:  {}'.format(np.array(stab_numeric)))
		print('critical stability loads, analytic: {}'.format(np.array(stab_analytic)))
		diff = stab_analytic-stab_numeric
		print('difference:                         {}'.format(np.array(diff)))

		rel_error = np.linalg.norm(diff, 2)/np.linalg.norm(stab_analytic, 2)
		print('||ana-num||_2/||ana||_2 = {:e}'.format(rel_error))
		self.assertEqual(rel_error < rtol, True)

	def stab_limit(self, ell, q=2):
		coeff = 2.*2.*np.pi*q/(q+1)**(3./2.)
		if 1/ell > coeff:
			return 1.
		else:
			return coeff*ell/1.

	@contextmanager
	def suppress_stdout(self):
		with open(os.devnull, "w") as devnull:
			old_stdout = sys.stdout
			sys.stdout = devnull
			try:
				yield
			finally:
				sys.stdout = old_stdout

if __name__ == '__main__':
	unittest.main()
