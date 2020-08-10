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
		t_stab = []
		for ell in ell_list:
			eps = .05
			tstab = self.stab_limit(ell)
			load_min = tstab - eps
			load_max = tstab + 1.
			nsteps = 100
			# dt = (load_max - load_min)/nsteps
			# print('dt', dt)

			t = traction(
					ell=ell,
					load_min=load_min,
					load_max=load_max,
					nsteps=100, n=5,
					nu=0., Lx=1., Ly=.1)

			idx = np.where(t['stable'] == False)[0][0]
			t_stab.append([t['load'][idx], ell])

		rtol = 5.*(load_max - load_min)/nsteps
		print('rtol', rtol)
		stab_analytic = np.array([self.stab_limit(ell) for ell in ell_list])
		stab_numeric  = np.array([t[0] for t in t_stab])
		np.linalg.norm(stab_analytic - stab_numeric, 2)
		print('')
		print('Critical stability loads, numeric:  {}'.format(np.array(stab_numeric)))
		print('Critical stability loads, analytic: {}'.format(np.array(stab_analytic)))
		diff = stab_analytic-stab_numeric
		print('Difference:                         {}'.format(np.array(diff)))
# np.linalg.norm(stab_analytic, 2)
		rel_error = np.linalg.norm(diff, 2)/np.linalg.norm(stab_analytic, 2)
		print('Error = ||analytic-numeric||_2/||analytic|| = {:e}'.format(rel_error))
		print('Error vs target relative tolerance {:e}'.format(rtol))
		self.assertEqual(rel_error < rtol, True)

	def stab_limit(self, ell, q=2):
		coeff = 2.*2.*np.pi*q/(q+1)**(3./2.)
		if 1/ell > coeff:
			return 1.
		else:
			return coeff*ell/1.

if __name__ == '__main__':
	unittest.main()
