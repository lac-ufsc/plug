from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE.md", "r") as fh:
    license = fh.read()
    
setup(name='plug',
      version='0.1',
      description='Plug flow reactor w/ surface reactions',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Thiago P. de Carvalho',
      author_email='tpcarvalho2@gmail.com',
      license=license,
      packages=['plug','plug.reactor','plug.kinetics','plug.utils'],
      install_requires=['numpy','scipy','sklearn','Assimulo'],
      zip_safe=False,    
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL ",
        "Operating System :: OS Independent",
    ]) 
