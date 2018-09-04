# Plug

Plug is a Python implementation of a plug flow reactor (PFR) with support for homogeneous and heterogeneous reactions, 
using Cantera (https://cantera.org/documentation/docs-2.3/sphinx/html/index.html) as the chemical interpreter. The code
was designed to take as input thermodynamic, chemical kinetics and transport data in Cantera's .cti format, which derives
from CHEMKIN's own format. 

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

## PFR with homogeneous reactions

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
    
    #Gas molar fractions and temperature along the reactor
    Xg = sim.X
    Tg = sim.T

    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()

    #Molar fractions  along the reactor
    ax = fig.add_subplot(121)     
    ax.plot(zcoord, Xg[:, gas.species_index('H2')], '-g', label='H2')
    ax.plot(zcoord, Xg[:, gas.species_index('H2O')], '-r',  label='H2O')
    ax.plot(zcoord, Xg[:, gas.species_index('O2')], '-b',  label='O2')
    ax.plot(zcoord, Xg[:, gas.species_index('O')], '-c',  label='O')
    ax.plot(zcoord, Xg[:, gas.species_index('OH')], '-m',  label='OH')
    ax.plot(zcoord, Xg[:, gas.species_index('H')], '-k',  label='H') 
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('$X_k$ [-]')
    ax.legend(loc='best')
    
    #Temperature along the reactor
    ax1 = fig.add_subplot(122)  
    ax1.plot(zcoord, Tg, '-b')
    ax1.set_xlabel('$z$ [mm]')
    ax1.set_ylabel('$T$ [K]') 
    
    plt.show()
    
## PFR with heterogeneous reactions

This example solves a plug flow reactor problem, where the chemistry is surface chemistry. The specific problem simulated is the partial oxidation of methane over a platinum catalyst in a packed bed reactor. This example was taken from Cantera's documentation (https://cantera.org/documentation/docs-2.3/sphinx/html/cython/examples/reactors_surf_pfr.html).

    import cantera as ct
    import numpy as np
    import plug as pfr 

    #### Input mechanism data ####:      
    input_file = 'methane_pox_on_pt.cti' 
    surf_name = 'Pt_surf'

    #Load phases solution objects
    gas = ct.Solution(input_file)
    surf = ct.Interface(input_file,surf_name,[gas])

    #### Inlet thermo state ####:  
    T_in = 773.15                #Inlet temperature [K]
    P_in = 101325.0
    #Inlet molar composition: 
    X_in = {'CH4':1,'O2':1.5, 'AR':5} 

    #Reactor diameter [m]:
    tube_d = np.sqrt(4*1e-04/np.pi)        
    carea = np.pi*tube_d**2/4

    #Initialize reactor                                          
    r = pfr.PlugFlowReactor(gas,
                            energy = 1,
                            z_out = 0.001,
                            diam = tube_d)

    #Initialize the reacting wall
    rs = pfr.ReactingSurface(r,surf,
                             cat_area_pvol = 355)

    #Set inlet thermo state
    r.TPX = T_in, P_in, X_in

    #Set inlet velocity [m/s]
    r.u = 0.005

    #Create a ReactorSolver object
    sim = pfr.ReactorSolver(r,
                            atol = 1e-15,
                            rtol = 1e-05,
                            grid = 1000,
                            max_steps = 1000,
                            solver_type = 'dae')

    #Solve the PFR
    sim.solve()

    #Axial coordinates [mm]
    zcoord = sim.z*1e03
    
    #Gas molar fractions, temperature and surface coverages the reactor        
    Xg = sim.X
    Tg = sim.T
    covs = sim.coverages
    
    #Check that the mass is conserved
    mdot = sim.mdot    #Mass flow rate along reactor [kg/s]

    #Get inlet mass flow rate [kg/s]
    mdot_in = r.ac*rho[0]*uz[0]

    #Get outlet mass flow rate [kg/s]
    mdot_out = r.ac*rho[-1]*uz[-1]

    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()

    #Molar fractions  along the reactor
    ax = fig.add_subplot(121)     
    ax.plot(zcoord, Xg[:, gas.species_index('H2')], '-g', label='H2')
    ax.plot(zcoord, Xg[:, gas.species_index('H2O')], '-r',  label='H2O')
    ax.plot(zcoord, Xg[:, gas.species_index('CO')], '-c',  label='CO')
    ax.plot(zcoord, Xg[:, gas.species_index('CH4')], '-k',  label='CH4') 
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('$X_k$ [-]')
    ax.axis((0.0,zcoord[-1],0.0,0.25))
    ax.legend(loc='best')
    
    #Temperature along the reactor
    ax1 = fig.add_subplot(122)  
    ax1.plot(zcoord, Tg, '-b')
    ax1.set_xlabel('$z$ [mm]')
    ax1.set_ylabel('$T$ [K]') 
    ax1.axis((0.0,zcoord[-1],800,3200))
    
    plt.show()
