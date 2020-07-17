from setuptools import setup

# Disco_theque: library of useful scripts, functions and classes for this project
setup(
    name='disco_theque',
    description='DIStributed semi-COnstrained microphone arrays',
    author='Nicolas Furnon',
    author_email='nicolas.furnon@loria.fr',
    license='MIT',
    install_requires=['numpy',
                      'soundfile',
                      'pyroomacoustics',
                      'acoustics',
                      'matplotlib',
                      'scipy']
)
