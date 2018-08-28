import cantera as ct
import numpy as np
import plug as pfr
import time
start = time.time()  

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

#Gas molar fractions, temperature and pressure along the reactor
zcoord = sim.z*1e03
covs = sim.coverages
Xg = sim.X
Tg = sim.T
uz = sim.u
rho = sim.rho
mdot = sim.mdot

#Get inlet mass flow rate [kg/s]
mdot_in = r.ac*rho[0]*uz[0]

#Get outlet mass flow rate [kg/s]
mdot_out = r.ac*rho[-1]*uz[-1]

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

vis = 1

if vis==1:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(16, 7)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 14})  

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