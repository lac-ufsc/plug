import cantera as ct
#import numpy as np
import plug as pfr
import time
start = time.time()     

#### Input mechanism data ####:          
input_file = 'prf-curran.xml'
#input_file = 'c8h18_gaseous.xml'

#Load Cantera solution objects
gas = ct.Solution(input_file)

#### Inlet thermo state ####:
X_in = 'IC8H18:8e-03, O2:0.2099, N2:0.7892'         
T_in = 1200 
P_in = 101325.0

#Reactor parameters:
tube_d = 0.019

#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        energy = 1,
                        momentum = 0,
                        z_out = 0.01, 
                        diam = tube_d)

#Reactor current thermo state
r.TPX = T_in, P_in, X_in

#Set reactor flow velocity [m/s]
r.u = 2   #0.235

#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        atol = 1e-14,
                        rtol = 1e-06,
                        grid = 500,
                        max_steps = 10000)

#Solve the reactor
sim.solve()

#Gas molar fractions, coverages and 
res_t = sim.rtime
Tg = sim.T
Xg = sim.X

#Axial coordinates [mm]
zcoord = sim.z*1e03

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
#    ax.plot(res_t, Xg[:, gas.species_index('co')], '-c',  label='CO')
    ax.plot(zcoord, Xg[:, gas.species_index('c2h2')], '-y',  label='C2H2')
    ax.plot(zcoord, Xg[:, gas.species_index('ch4')], '-m',  label='CH4')
    ax.plot(zcoord, Xg[:, gas.species_index('ic4h8')], '-g',  label='iC4H8')  
    ax.plot(zcoord, Xg[:, gas.species_index('c3h6')], '-r',  label='C3H6')  
    ax.plot(zcoord, Xg[:, gas.species_index('ic8h18')], '-k',  label='iC8H18')  
#    ax.axis((0.0,zcoord[-1],2e-04,1.2e-03))
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('$X_k$ [-]')
    ax.legend(loc='best')
    
    ax1 = fig.add_subplot(122)     
    ax1.plot(zcoord, Tg, '-b',  label='T[K]')
    ax1.set_xlabel('z [mm]')
    ax1.set_ylabel('$T$ [K]')
    
plt.show()
    