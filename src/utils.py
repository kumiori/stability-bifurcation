import os
import sys
from dolfin import MPI
from subprocess import check_output, CalledProcessError
# Colored printing functions for strings that use universal ANSI escape sequences.
# fail: bold red, pass: bold green, warn: bold yellow,
# info: bold blue, bold: bold white

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

class ColorPrint:
    @staticmethod
    def print_fail(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stderr.write("\x1b[1;31m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_pass(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_warn(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stderr.write("\x1b[1;33m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_info(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stdout.write("\x1b[1;34m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_bold(message, end="\n"):
        if MPI.comm_world.rank == 0:
            sys.stdout.write("\x1b[1;37m" + message.strip() + "\x1b[0m" + end)



