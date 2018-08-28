import cantera as ct
import numpy as np
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:      
input_file = 'sif_gas.cti'
surf_name = 'SI3N4'
bulk_name = ['N(D)','SI(D)']

#Load phases solution objects
gas = ct.Solution(input_file)
bulk_n = ct.Solution(input_file,bulk_name[0])
bulk_si = ct.Solution(input_file,bulk_name[1])
surf = ct.Interface(input_file,surf_name,[gas,bulk_n,bulk_si])

#### Current thermo state ####:
X_in = {'SIF4':0.1427, 'NH3':0.8573}    #Inlet molar composition       
T_in = 1713.0                           #Inlet temperature [K]
P_in = 133.322*2                        #Inlet pressure (2 torr) [Pa]

#Reactor parameters (SI units):
length = 0.6                    #Axial length [m]
tube_d = 0.0508                 #Reactor diameter [m]
carea = np.pi/4*tube_d**2       #Reactor cross-sectional area [m**2]

#Wall area [m**2]
wallarea = np.pi*tube_d*length

#Catalytic area per volume [m**-]
cat_area_pvol = wallarea/(length*carea)

#Volumetric flow [m**3/s]
vdot_in = 0.02337
u_in = vdot_in/carea



#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        momentum = 1,
                        z_out = length,
                        diam = tube_d)

#Initialize the reacting wall
rs = pfr.ReactingSurface(r,surf,
                         bulk = [bulk_n,bulk_si],                   
                         cat_area_pvol = cat_area_pvol)

#Set inlet thermo state
r.TPX = T_in, P_in, X_in

#Set inlet mass flow rate [kg/s]
r.u = u_in

#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        atol = 1e-08,
                        rtol = 1e-06,
                        grid = 7,
                        max_steps = 1000,
                        solver_type = 'dae')

#Solve the PFR
sim.solve()

#Gas molar fractions, temperature and pressure along the reactor
zcoord = sim.z*1e03
covs = sim.coverages
Xg = sim.X
Pg = sim.P
uz = sim.u
rho = sim.rho
rt = sim.rtime
mdot = sim.mdot

#Deposition rates [kmol/m**2 s] at outlet
id_N = surf.kinetics_species_index('N(D)')
id_SI = surf.kinetics_species_index('SI(D)')

N_dot = surf.net_production_rates[id_N]
Si_dot = surf.net_production_rates[id_SI]

#Pressure drop [Torr]
dP = (P_in - gas.P)/133.322

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

#Open PLUG CHEMKIN manual results
filename='/home/tpcarvalho/carva/python_data/SiF_PLUG_results.csv'
import csv
ref_data=[]
with open(filename, 'rt') as f:
    reader = csv.reader(f)
    next(reader, None)      #Skip one line
    for row in reader:
        ref_data.append(np.array(row,dtype=float))
f.close()
#Convert list to Numpy array  
ref_data=np.array(ref_data)
zcoord_ref = np.linspace(0.0,600,len(ref_data[:,0]))

vis = 1

if vis==1:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(16, 7)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 14})  
    
    #Temperature along the reactor
    ax = fig.add_subplot(121)  
    ax.plot(zcoord, Xg[:, gas.species_index('SIF4')], '-g', label='SIF4')
    ax.plot(zcoord, Xg[:, gas.species_index('NH3')], '-b',  label='NH3')
    ax.plot(zcoord, Xg[:, gas.species_index('HF')], '-r',  label='HF')

    ax.plot(zcoord_ref, ref_data[:, 0], 'ok', label='SIF4 ref')
    ax.plot(zcoord_ref, ref_data[:, 1], 'ok',  label='NH3 ref')
    ax.plot(zcoord_ref, ref_data[:, 2], 'ok',  label='HF ref')

    ax.set_xlabel('z [mm]')
    ax.set_ylabel('$X_k$ [-]')
    ax.legend(loc='best')
    
    ax1 = fig.add_subplot(122)  
    ax1.semilogy(zcoord, covs[:, surf.species_index('HN_NH2(S)')], '-k', label='HN_NH2(S)')
    ax1.semilogy(zcoord, covs[:, surf.species_index('HN_SIF(S)')], '-r',  label='HN_SIF(S)')
    ax1.semilogy(zcoord, covs[:, surf.species_index('F2SINH(S)')], '-g',  label='F2SINH(S)')
    
    ax1.semilogy(zcoord_ref, ref_data[:, 3], 'sk', label='HN_SIF(S) ref')
    ax1.semilogy(zcoord_ref, ref_data[:, 4], 'sk', label='HN_NH2(S) ref')
    ax1.semilogy(zcoord_ref, ref_data[:, 5], 'sk', label='F2SINH(S) ref')
        
    ax1.axis((0.0,zcoord[-1],1e-03,1))
    ax1.set_xlabel('z [mm]')
    ax1.set_ylabel('$\Phi_i$ [-]')
    ax1.legend(loc='best')
    plt.show()