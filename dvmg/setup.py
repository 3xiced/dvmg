from setuptools import setup
from os.path import join, dirname

from dvmg import __version__  # type: ignore

setup(
    name='dvmg',
    version=__version__,
    packages=['dvmg', 'dvmg.patterns',
              'dvmg.processors', 'dvmg.worker', 'dvmg.utils'],
    long_description=open(join(dirname(__file__), 'README.rst')).read(),
)
