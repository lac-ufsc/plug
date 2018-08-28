import cantera as ct
import numpy as np
import plug as pfr
import time
start = time.time()  
   
#### Input mechanism data ####:                  
input_file = 'diesel_surrogate_reduced_mech.cti'

#### Current thermo state ####:
X_in = {'C12H24-1':1.0,'O2':18/1.576}         
T_in = 680
P_in = 9.718e06

#Load Cantera solution objects
gas = ct.Solution(input_file)

#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        energy = 1,
                        momentum = 0,
                        z_out = 2, 
                        diam = 1.25)

#Reactor current thermo state
r.TPX = T_in, P_in, X_in

#Set reactor mass flow rate [kg/s]
r.mdot = 271.4

#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        atol = 1e-08,
                        rtol = 1e-06,
                        grid = 80000,
                        max_steps = 1000)
                       
#Solve the reactor
sim.solve()

#Gas molar fractions, temperature and pressure along the reactor
mdot = sim.mdot
Pg = sim.P
Tg = sim.T
Xg = sim.X
uz = sim.u
rho = sim.rho

#Axial coordinates
zcoord = sim.z

#Temperature derivative
dTdz = np.gradient(Tg, 2)/np.gradient(zcoord, 2)

#Find the index of peak values
id_peak = np.argmax(dTdz)

#Print time elapsed
print('It took {0:0.8f} seconds'.format(time.time() - start)) 

#%%
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
    ax.plot(zcoord, Xg[:, gas.species_index('C12H24-1')], '-g', label='C12H24')
    ax.plot(zcoord, Xg[:, gas.species_index('CO')], '-r',  label='CO')
    ax.plot(zcoord, Xg[:, gas.species_index('CO2')], '-c',  label='CO2')
    ax.plot(zcoord, Xg[:, gas.species_index('OH')], '-m',  label='OH')
    ax.plot(zcoord, Xg[:, gas.species_index('H')], '-k',  label='H') 
    ax.plot(zcoord, Xg[:, gas.species_index('H2')], '-y',  label='H2') 
#    ax.plot(zcoord, Xg[:, gas.species_index('O2')], '-b',  label='O2')
    ax.set_xlabel('z [m]')
    ax.set_ylabel('$X_k$ [-]')
    ax.legend(loc='best')

    #2nd subplot, zoomed in scale,
    #Set the plot window for second plot:
    tw_i = int(np.around(id_peak*0.999))
    tw_f = int(np.around(id_peak*1.001))
    
    ax1 = fig.add_subplot(122)  
    ax1.plot(zcoord[tw_i:tw_f],Xg[tw_i:tw_f, gas.species_index('C12H24-1')],'g-')
    ax1.plot(zcoord[tw_i:tw_f],Xg[tw_i:tw_f, gas.species_index('CO')],'r-')
    ax1.plot(zcoord[tw_i:tw_f],Xg[tw_i:tw_f, gas.species_index('CO2')],'c-')
    ax1.plot(zcoord[tw_i:tw_f],Xg[tw_i:tw_f, gas.species_index('OH')], '-m',  label='OH')
    ax1.plot(zcoord[tw_i:tw_f],Xg[tw_i:tw_f, gas.species_index('H')], '-k',  label='H') 
    ax1.plot(zcoord[tw_i:tw_f],Xg[tw_i:tw_f, gas.species_index('H2')], '-y',  label='H2') 
#    ax1.plot(zcoord[tw_i:tw_f],Xg[tw_i:tw_f, gas.species_index('O2')], '-b',  label='O2')
    ax1.plot(zcoord[tw_i:tw_f],Tg[tw_i:tw_f]/1e04,'k-')
    ax1.set_xlabel('$z$ [m]')
    ax1.set_ylabel('$X_k$ [-]') 