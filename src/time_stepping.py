import sys
sys.path.append("../src/")
from utils import ColorPrint
from dolfin import  Function, assemble, plot, norm
import numpy as np
import _post_processing as pp
import os
import pandas as pd
import json
from distutils.util import strtobool

class TimeStepping(object):
    """This class performs the incremental time stepping to solve the evolution problem"""
    def __init__(self,
                model,
                solver,
                stability,
                load_param,
                outfiles,
                parameters,
                user_density=None):

        super(TimeStepping, self).__init__()

        self.model = model
        self.solver = solver
        self.stability = stability
        self.load_param = load_param
        self.file_out = outfiles[0]
        self.file_con = outfiles[1]
        self.file_eig = outfiles[2]
        # if user_density:
        self.user_density = user_density
        self.parameters = parameters
        self.time_data, self.continuation_data = [], []
        self.load_steps = np.linspace(self.parameters['load_min'],
                                 self.parameters['load_max'],
                                 self.parameters['nsteps'])

    def user_postprocess_stability(self, load):
        pass

    def user_postprocess_timestep(self, load):
        pass

    def compile_continuation_data(self, load, iteration, perturbed):
        model = self.model
        u = self.solver.u
        alpha = self.solver.alpha

        if not perturbed:
            self.continuation_data_i = {}

            self.continuation_data_i["elastic_energy"] = assemble(model.elastic_energy_density(model.eps(u), alpha)*model.dx)
            if self.user_density is not None:
                self.continuation_data_i["elastic_energy"] += assemble(self.user_density*model.dx)
            self.continuation_data_i["dissipated_energy"] = assemble(model.damage_dissipation_density(alpha)*model.dx)

            self.continuation_data_i["total_energy"] = self.continuation_data_i["elastic_energy"] + self.continuation_data_i["dissipated_energy"]
            self.continuation_data_i["load"] = load
            self.continuation_data_i["iteration"] = iteration
            self.continuation_data_i["alpha_l2"] = alpha.vector().norm('l2')
            self.continuation_data_i["alpha_h1"] = norm(alpha, 'h1')
            self.continuation_data_i["alpha_max"] = np.max(alpha.vector()[:])
            self.continuation_data_i["eigs"] = self.stability.eigs

        else:
            elastic_post = assemble(model.elastic_energy_density(model.eps(u), alpha)*model.dx)
            if self.user_density is not None:
                elastic_post += assemble(self.user_density*model.dx)

            dissipated_post = assemble(model.damage_dissipation_density(alpha)*model.dx)

            self.continuation_data_i["elastic_energy_diff"] = elastic_post-self.continuation_data_i["elastic_energy"]
            self.continuation_data_i["dissipated_energy_diff"] = dissipated_post-self.continuation_data_i["dissipated_energy"]
            self.continuation_data_i["total_energy_diff"] = self.continuation_data_i["elastic_energy_diff"]\
                    +self.continuation_data_i["dissipated_energy_diff"]

            # ColorPrint.print_info('energy    {:4e}'.format(elastic_energy_post + dissipated_energy_post))
            # ColorPrint.print_info('estimate  {:4e}'.format(stability.en_estimate))
            # ColorPrint.print_info('en-est    {:4e}'.format(elastic_energy_post + dissipated_energy_post-stability.en_estimate))
        pass

    def compile_time_data(self, load):
        time_data_i = self.time_data_i
        model = self.model
        u = self.solver.u
        alpha = self.solver.alpha

        time_data_i["load"] = load
        time_data_i["stable"] = self.stability.stable
        time_data_i["# neg ev"] = self.stability.negev
        time_data_i["elastic_energy"] = assemble(
            model.elastic_energy_density(model.eps(u), alpha)*model.dx)
        if self.user_density is not None:
            time_data_i["elastic_energy"] += assemble(self.user_density*model.dx)
        time_data_i["dissipated_energy"] = assemble(
                model.damage_dissipation_density(alpha)*model.dx)
        # else:
        #     time_data_i["dissipated_energy"] = assemble(disspated_energy*model.dx)
        time_data_i["eigs"] = self.stability.eigs if hasattr(self.stability, 'eigs') else np.inf
        ColorPrint.print_pass(
            "Time step {:.4g}: it {:3d}, err_alpha={:.4g}".format(
                time_data_i["load"],
                time_data_i["iterations"],
                time_data_i["alpha_error"]))
        self.time_data.append(time_data_i)
        time_data_pd = pd.DataFrame(self.time_data)

        return time_data_pd

    def run(self):
        outdir = self.parameters['outdir']
        savelag = self.parameters['savelag']
        solver = self.solver
        stability = self.stability
        alpha = solver.alpha
        u = solver.u


        load_steps = self.load_steps 
        alpha_old = Function(alpha.function_space())
        self.time_data_i = []
        stable = None; negev = -1; mineig = np.inf; iteration = 0
        diff = alpha.copy(deepcopy=True)
        for it, load in enumerate(load_steps):
            self.load_param.t = load
            alpha_old.assign(alpha)
            print('')
            ColorPrint.print_warn('Solving load = {:.2f}'.format(load))
            self.time_data_i, am_iter = solver.solve()

            diff.vector()[:] = alpha.vector() - alpha_old.vector()
            try:
                assert all(alpha.vector()[:]>=alpha_old.vector()[:])
            except AssertionError:
                print('check alpha.vector()[:]>=alpha_old.vector()')

            try:
                assert all(solver.problem_alpha.lb.vector()[:]==alpha_old.vector()[:])
            except AssertionError:
                print('check all(solver.problem_alpha.lb.vector()[:]==alpha_old.vector()[:])')

            if bool(strtobool(str(stability.parameters['checkstability']))):
                (stable, negev) = stability.solve(solver.problem_alpha.lb)
                ColorPrint.print_pass('Current state is{}stable'.format(' ' if stable else ' not '))
                if hasattr(stability, 'eigs') and len(stability.eigs)>0 and min(stability.eigs)<0:
                    pp.plot_eigenmodes(stability.eigendata, alpha, load, outdir)
                    self.user_postprocess_stability(load)
            else:
                solver.update()
            alpha.copy(deepcopy = True)

            if stable:
                solver.update()
            elif not stable and not bool(strtobool(str(stability.parameters['continuation']))):
                solver.update()

            elif not stable and bool(strtobool(str(stability.parameters['continuation']))):
                while stable == False:
                    adm_pert = np.where(np.array([e['en_diff'] for e in stability.eigendata]) < 0)[0]
                    if len(adm_pert)==0:
                        ColorPrint.print_warn('No admissible perturbations found')
                        ColorPrint.print_pass('Continuing load program')
                        break
                    else:
                        continuation_data = []
                        steepest = np.argmin([e['en_diff']  for e in stability.eigendata])
                        if self.parameters['perturbation_choice']=='first':
                            mode = 0
                        elif self.parameters['perturbation_choice'] == 'steepest':
                            mode = steepest
                        elif isinstance(self.parameters['perturbation_choice'], int):
                            mode = self.parameters['perturbation_choice']

                        perturbation_v = stability.eigendata[mode]['v_n']
                        perturbation_beta = stability.eigendata[mode]['beta_n']
                        hstar = stability.eigendata[mode]['hstar']
                        perturbation_v.rename('step displacement perturbation', 'step displacement perturbation')
                        perturbation_beta.rename('step damage perturbation', 'step damage perturbation')
                        ColorPrint.print_pass('Perturbation choice {}'.format(self.parameters['perturbation_choice']))
                        ColorPrint.print_pass('Perturbing current state with mode {} Delta E={:.5%} (estimated)'.format(mode, stability.eigendata[mode]['en_diff']))
                        ColorPrint.print_pass('...... chosen mode {} vs. steepest {} Delta E={:.5%} (estimated)'.format(mode, steepest, stability.eigendata[mode]['en_diff']/stability.eigendata[steepest]['en_diff']))
                        ColorPrint.print_pass('...... steepest descent mode {} Delta E={:.5%} (estimated)'.format(steepest,stability.eigendata[steepest]['en_diff']))

                        # perturb current state
                        self.compile_continuation_data(load, iteration, perturbed=False)
                        solver.alpha.copy(deepcopy=True)

                        ColorPrint.print_pass('Perturbing current state, looking for stability, iteration {}'.format(iteration))
                        uval = u.vector()[:]     + hstar * perturbation_v.vector()[:]
                        aval = alpha.vector()[:] + hstar * perturbation_beta.vector()[:]

                        alpha.vector().vec().ghostUpdate()
                        u.vector().vec().ghostUpdate()

                        self.time_data_i, am_iter = solver.solve()

                        if self.file_con is not None:
                            with self.file_con as f:
                                f.write(alpha, iteration)
                                f.write(u, iteration)

                        self.compile_continuation_data(load, iteration, perturbed=True)

                        ColorPrint.print_pass('Energy diff {}, rel thhreshold {}'
                                        .format(self.continuation_data_i["total_energy_diff"]/self.continuation_data_i["total_energy"], 
                                                self.stability.parameters['cont_rtol']))
                        continuation_data.append(self.continuation_data_i)

                        if np.mod(it, self.parameters['savelag']) == 0:
                            continuation_data_pd=pd.DataFrame(continuation_data)
                            continuation_data_pd.to_json(os.path.join(outdir + "/continuation_data.json"))

                        if self.continuation_data_i["total_energy_diff"]/self.continuation_data_i["total_energy"] < - self.stability.parameters['cont_rtol']:
                            ColorPrint.print_pass('Updating irreversibility')
                            solver.update()
                        else:
                            ColorPrint.print_pass('Threshold not met, continuing load program')
                            break

                        (stable, negev) = stability.solve(alpha_old)

                        if self.file_eig is not None:
                            with self.file_eig as f:
                                f.write(perturbation_beta, iteration)
                                f.write(perturbation_v, iteration)

                        iteration += 1

            time_data_pd = self.compile_time_data(load)

            if np.mod(it, self.parameters['savelag']) == 0:
                time_data_pd.to_json(os.path.join(outdir + "/time_data.json"))
                ColorPrint.print_pass('written data to file {}'.format(str(os.path.join(outdir + "/time_data.json"))))

                if self.file_out is not None:
                    with self.file_out as f:
                        f.write(alpha, load)
                        f.write(u, load)

            pp.plot_global_data(time_data_pd,load,outdir)
            self.user_postprocess_timestep(load)
        return time_data_pd

