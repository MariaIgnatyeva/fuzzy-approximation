import sys
from cx_Freeze import setup, Executable

base = None
if sys.platform == 'win32':
    base = 'Win32GUI'

options = {
    'build_exe': {'packages': ['pkg_resources._vendor', 'numpy', 'scipy']}
}

executables = [
    Executable('interface.py', base=base, targetName='application.exe')
]

setup(name='application',
      version='1.0',
      description='The program for 2D-function approximation using fuzzy rule base',
      options=options,
      executables=executables
      )