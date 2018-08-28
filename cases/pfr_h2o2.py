import cantera as ct
import numpy as np
import plug as pfr
import time
start = time.time()  

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

#Gas molar fractions, temperature and pressure along the reactor
mdot = sim.mdot
Pg = sim.P
Tg = sim.T
Xg = sim.X
uz = sim.u

#Axial coordinates
zcoord = sim.z

#Print time elapsed
print('It took {0:0.8f} seconds'.format(time.time() - start)) 

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