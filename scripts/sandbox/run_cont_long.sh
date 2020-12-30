python3 traction_continuation.py --config="{'alt_min': {'max_it': 300, 'tol':
1e-05, 'solver_alpha': 'tao', 'solver_u': {'u_snes_type': 'newtontr',
'u_snes_stol': 1e-06, 'u_snes_atol': 1e-06, 'u_snes_rtol': 1e-06,
'u_snes_max_it': 1000, 'u_snes_monitor': ''}, 'solver_alpha_tao':
{'tao_type': 'gpcg', 'tao_ls_type': 'gpcg', 'tao_gpcg_maxpgits': 50,
'tao_max_it': 300, 'tao_steptol': 1e-07, 'tao_gatol': 1e-05,
'tao_grtol': 1e-05, 'tao_gttol': 1e-05, 'tao_catol': 0.0, 'tao_crtol':
0.0, 'tao_ls_ftol': 1e-06, 'tao_ls_gtol': 1e-06, 'tao_ls_rtol': 1e-06,
'ksp_rtol': 1e-06, 'tao_ls_stepmin': 1e-08, 'tao_ls_stepmax':
1000000.0, 'pc_type': 'bjacobi', 'tao_monitor': ''}}, 'stability':
{'order': 4, 'projection': 'none', 'maxmodes': 5, 'checkstability':
True, 'continuation': True, 'cont_rtol': 1e-05, 'inactiveset_atol':
1e-05}, 'time_stepping': {'perturbation_choice': 1, 'savelag': 1,
'load_min': 0.0, 'load_max': 3.0, 'nsteps': 100, 'postfix': ''},
'material': {'ell': 0.2, 'E': 1.0, 'nu': 0.3,
'sigma_D0': 1.0}, 'geometry': {'Lx': 1, 'Ly': 0.1, 'n': 6},
'experiment': {'signature': ''}}"