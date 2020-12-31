import os
import sys
from dolfin import MPI, timings, TimingClear, TimingType, File, dump_timings_to_xml
from subprocess import check_output, CalledProcessError
from distutils.util import strtobool
import json
import time
import mpi4py


def get_code_ver():
    version = check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    status = 'clean'
    try:
        check_output(['git','diff-index', '--quiet', '--exit-code', 'HEAD', '--'])
    except CalledProcessError as e:
        status = 'dirty'
        ColorPrint.print_warn('*** Warning: you are (or I am) working with a dirty repository ')
        ColorPrint.print_warn('***          with outstanding uncommitted changes.')
        ColorPrint.print_warn('***          Consider the importance of housekeeping.')
        ColorPrint.print_warn('***          Clean and commit.')
        ColorPrint.print_warn('***          And consider yourself warned.')
    return version+'-'+status

def get_petsc_ver():
    from petsc4py import __version__ as _ver
    return _ver

def get_slepc_ver():
    from slepc4py import __version__ as _ver
    return _ver

def get_dolfin_ver():
    from dolfin import __version__ as dolfin_ver
    return dolfin_ver

def get_versions():
    versions = {"petsc": get_petsc_ver(),
                         "slepc": get_slepc_ver(),
                         "dolfin": get_dolfin_ver(),
                         "mechanics": get_code_ver()}
    return versions

def check_bool(parameter, bool_val = True):
    return bool(strtobool(str(parameter))) == bool_val

class ColorPrint:
    @staticmethod
    def print_fail(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stderr.write("\x1b[1;31m" + message.strip() + "\x1b[0m" + end)
    
    @staticmethod
    def print_red(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stderr.write("\x1b[1;31m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_green(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stderr.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_pass(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stdout.write("\x1b[1;32m    " + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_warn(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stderr.write("\x1b[1;33m    " + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_info(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stdout.write("\x1b[1;34m    " + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_bold(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stdout.write("\x1b[1;37m" + message.strip() + "\x1b[0m" + end)

def collect_timings(outdir, tic):

    # list_timings(TimingClear.keep, [TimingType.wall, TimingType.system])

    # t = timings(TimingClear.keep, [TimingType.wall, TimingType.user, TimingType.system])
    t = timings(TimingClear.keep, [TimingType.wall])
    # Use different MPI reductions
    t_sum = MPI.sum(MPI.comm_world, t)
    # t_min = MPI.min(MPI.comm_world, t)
    # t_max = MPI.max(MPI.comm_world, t)
    t_avg = MPI.avg(MPI.comm_world, t)
    # Print aggregate timings to screen
    print('\n'+t_sum.str(True))
    # print('\n'+t_min.str(True))
    # print('\n'+t_max.str(True))
    print('\n'+t_avg.str(True))

    # Store to XML file on rank 0
    if MPI.rank(MPI.comm_world) == 0:
        f = File(MPI.comm_self, os.path.join(outdir, "timings_aggregate.xml"))
        f << t_sum
        # f << t_min
        # f << t_max
        f << t_avg

    dump_timings_to_xml(os.path.join(outdir, "timings_avg_min_max.xml"), TimingClear.clear)
    elapsed = time.time() - tic

    
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        with open(os.path.join(outdir, 'timings.pkl'), 'w') as f:
            json.dump({'elapsed': elapsed, 'size': size}, f)


    pass

