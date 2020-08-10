alt_min_parameters = {"max_it": 300,
                      "tol": 1.e-5,
                      "solver_alpha": "tao"
                     }

petsc_options_alpha_tao = {
	"tao_type": "gpcg",
	"tao_ls_type": "gpcg",
	"tao_gpcg_maxpgits": 50,
	"tao_max_it": 300,
	"tao_steptol": 1e-7,
	"tao_gatol": 1e-4,
	"tao_grtol": 1e-4,
	"tao_gttol": 1e-4,
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

petsc_options_alpha_snes = {
    "alpha_snes_type": "vinewtonrsls",
    "alpha_snes_stol": 1e-5,
    "alpha_snes_atol": 1e-5,
    "alpha_snes_rtol": 1e-5,
    "alpha_snes_max_it": 500,
    "alpha_ksp_type": "preonly",
    "alpha_pc_type": "lu"}

petsc_options_u = {
    "u_snes_type": "newtontr",
    "u_snes_stol": 1e-6,
    "u_snes_atol": 1e-6,
    "u_snes_rtol": 1e-6,
    "u_snes_max_it": 1000,
    "u_snes_monitor": ''}

default_parameters = {
	"alt_min": alt_min_parameters,
	"solver_u": petsc_options_u,
	"solver_alpha_tao": petsc_options_alpha_tao,
	"solver_alpha_snes": petsc_options_alpha_snes}

