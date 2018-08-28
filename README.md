# Plug

Plug is a Python implementation of a plug flow reactor (PFR) with support for homogeneous and heterogeneous reactions, 
using Cantera (https://cantera.org/documentation/docs-2.3/sphinx/html/index.html) as the chemical interpreter.

In addition to the PFR model, Plug also contains a modified implementation of the surface kinetics model from Cantera, 
which adds the option of having coverage-dependent enthalpy for surface species. Lastly, it comes with some utility functions 
for mechanism reduction, such as PCA for example.

This package relies on either Assimulo (https://jmodelica.org/assimulo/) or scikits.odes (https://github.com/bmcage/odes) solver packages. Please, refer to the links for more information.

# Installation

The current version of Plug is still preliminary. It can be installed locally on your machine by the command 'pip install -e .' within the folder where setup.py is located. There are some packages dependencies which are listed below. Numpy, Scipy, Sklearn and Assimulo are handled automatically if setup.py is run. Cantera and scikits.odes must be installed manually by the user. Additionally, the user must have installed the Sundials 2.7.0 library (https://computation.llnl.gov/projects/sundials).

Package requirements:

* numpy
* scipy
* cantera
* sklearn
* assimulo
* scikits.odes

# Usage
A simple example solving a PFR problem of hydrogen-oxygen combustion. This example was taken from Cantera's documentation (https://cantera.org/documentation/docs-2.3/sphinx/html/cython/examples/reactors_pfr.html).

    import cantera as ct
    import numpy as np
    import plug as pfr

    #### Input mechanism data ####:                  
    input_file = 'h2o2.xml'

    #### Current thermo state ####:
    X_in = 'H2:2, O2:1, AR:0.1'         
    T_in = 1500.0
    P_in = 101325.0

    #Load Cantera solution objects
    gas = ct.Solution(input_file)

    #Reactor parameters:
    carea = 1e-04
    tube_d = np.sqrt(4*carea/np.pi)

    #Initialize reactor                                          
    r = pfr.PlugFlowReactor(gas,
                            energy = 1,
                            momentum = 0,
                            z_out = 2.3e-07, 
                            diam = tube_d)

    #Reactor current thermo state
    r.TPX = T_in, P_in, X_in

    #Set reactor flow velocity [m/s]
    r.u = 0.006

    #Create a ReactorSolver object
    sim = pfr.ReactorSolver(r,
                            atol = 1e-08,
                            rtol = 1e-06,
                            grid = 300,
                            max_steps = 1000)
                            
    #Solve the reactor
    sim.solve()

    #Axial coordinates
    zcoord = sim.z
    
    #Gas molar fractions, temperature and pressure along the reactor
    mdot = sim.mdot
    Pg = sim.P
    Tg = sim.T
    Xg = sim.X
    uz = sim.u
