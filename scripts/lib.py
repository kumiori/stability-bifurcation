import yaml
import dolfin

def create_output(outdir):
    file_out = dolfin.XDMFFile(os.path.join(outdir, "output.xdmf"))
    file_out.parameters["functions_share_mesh"] = True
    file_out.parameters["flush_output"] = True
    file_postproc = dolfin.XDMFFile(os.path.join(outdir, "postprocess.xdmf"))
    file_postproc.parameters["functions_share_mesh"] = True
    file_postproc.parameters["flush_output"] = True
    file_eig = dolfin.XDMFFile(os.path.join(outdir, "perturbations.xdmf"))
    file_eig.parameters["functions_share_mesh"] = True
    file_eig.parameters["flush_output"] = True
    file_bif = dolfin.XDMFFile(os.path.join(outdir, "bifurcation.xdmf"))
    file_bif.parameters["functions_share_mesh"] = True
    file_bif.parameters["flush_output"] = True
    file_bif_postproc = dolfin.XDMFFile(os.path.join(outdir, "bifurcation_postproc.xdmf"))
    file_bif_postproc.parameters["functions_share_mesh"] = True
    file_bif_postproc.parameters["flush_output"] = True
    file_ealpha = dolfin.XDMFFile(os.path.join(outdir, "elapha.xdmf"))
    file_ealpha.parameters["functions_share_mesh"] = True
    file_ealpha.parameters["flush_output"] = True

    files = {'output': file_out, 
             'postproc': file_postproc,
             'eigen': file_eig,
             'bifurcation': file_bif,
             'ealpha': file_ealpha}

    return files

from dolfin import assemble
def compile_continuation_data(state, energy):
    continuation_data_i = {}
    continuation_data_i["energy"] = assemble(energy)
    return continuation_data_i


from utils import get_versions
code_parameters = get_versions()

def getDefaultParameters():
    with open('../../parameters/form_compiler.yml') as f:
        form_compiler_parameters = yaml.load(f, Loader=yaml.FullLoader)
    with open('../../parameters/solvers_default.yml') as f:
        equilibrium_parameters = yaml.load(f, Loader=yaml.FullLoader)['equilibrium']
    with open('../../parameters/solvers_default.yml') as f:
        damage_parameters = yaml.load(f, Loader=yaml.FullLoader)['damage']
    with open('../../parameters/solvers_default.yml') as f:
        elasticity_parameters = yaml.load(f, Loader=yaml.FullLoader)['elasticity']
    # with open('../../parameters/film.yaml') as f:
    #     material_parameters = yaml.load(f, Loader=yaml.FullLoader)['material']
    # with open('../../parameters/film.yaml') as f:
    #     newton_parameters = yaml.load(f, Loader=yaml.FullLoader)['newton']
    with open('../../parameters/loading.yaml') as f:
        loading_parameters = yaml.load(f, Loader=yaml.FullLoader)['loading']
    with open('../../parameters/stability.yaml') as f:
        stability_parameters = yaml.load(f, Loader=yaml.FullLoader)['stability']
    with open('../../parameters/stability.yaml') as f:
        inertia_parameters = yaml.load(f, Loader=yaml.FullLoader)['inertia']
    with open('../../parameters/stability.yaml') as f:
        eigen_parameters = yaml.load(f, Loader=yaml.FullLoader)['eigen']

    default_parameters = {
        'code': {**code_parameters},
        'compiler': {**form_compiler_parameters},
        'eigen': {**eigen_parameters},
        # 'geometry': {**geometry_parameters},
        'inertia': {**inertia_parameters},
        'loading': {**loading_parameters},
        # 'material': {**material_parameters},
        # 'newton': {**newton_parameters},
        'equilibrium':{**equilibrium_parameters},
        'damage':{**damage_parameters},
        'elasticity':{**elasticity_parameters},
        'stability': {**stability_parameters},
        }

    return default_parameters
