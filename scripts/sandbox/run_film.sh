python3 film_continuation.py --load_max=1.3 --nsteps=100 --config="{'material': {'ell': 0.1, 'ell_e': 0.3}, 'geometry': {'Lx': 5.0, 'Ly': 0.1, 'n': 10}}" --postfix='paper'

python3 postproc_fields.py --experiment='../output/film-paper-cont' --xres=1000 --stride=1

python3 film_stability.py --load_max=1.3 --nsteps=100 --config="{'material': {'ell': 0.1, 'ell_e': 0.3}, 'geometry': {'Lx': 5.0, 'Ly': 0.1, 'n': 10}}" --postfix='paper'

python3 postproc_fields.py --experiment='../output/film-paper' --xres=1000 --stride=1
