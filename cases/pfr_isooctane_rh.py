import numpy as np
import cantera as ct
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:      
input_file = 'c8h18_rh_surf.cti'
surf_name = 'Rh_surface'

#Load phases solution objects
gas = ct.Solution(input_file)
surf = ct.Interface(input_file,surf_name,[gas]) 

#### Current thermo state ####:  
#T_in = 1070                #Inlet temperature [K]
T_in = 1359                #Inlet temperature [K]
P_in = 101325.0            #Inlet pressure [Pa]

#C/O ratio
co_ratio = 1.0

#Compute reactants molar composition
o2_mols = 20.0/(1.0+0.25*co_ratio)
c8h18_mols = 20.0 - o2_mols

#Inlet molar composition: 
X_in = {'IC8H18':c8h18_mols,'O2':o2_mols,'AR':80} 

#Monolith channel diameter [m]:
d_channel = 0.019

#Monolith cross-sectional area [m**2]
Ac = np.pi/4.0*0.019**2
    
#Total volumetric flow rate is 4 SLPM
vdot_in = 4.0*1.66667e-5    #[m**3/s]
u_in = vdot_in/Ac

#Catalytic active area per reactor volume [m**-1]
cat_area_pvol = 3.5e02

#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        z_out = 0.001,
                        diam = d_channel,
                        support_type = 'foam',
                        sp_area = 3e03,
                        dc = 1e-04,
                        porosity = 0.8,
                        ext_mt = 1)

#Initialize the reacting wall
rs = pfr.ReactingSurface(r,surf,                   
                         cat_area_pvol = cat_area_pvol)

#Set inlet thermo state
r.TPX = T_in, P_in, X_in

#Set inlet velocity [m/s]
r.u = u_in

#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        atol = 1e-14,
                        rtol = 1e-06,
                        grid = 300,
                        tcovs = 1e04,
                        solver_type = 'dae')
                       
#Solve the PFR
sim.solve()

#Gas results
zcoord = sim.z
covs = sim.coverages
Xg = sim.X

#Print time elapsed
print('It took {0:0.8f} seconds'.format(time.time() - start)) 

vis = 1

if vis==1:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(17, 7)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 14})  

    #Molar fractions  along the reactor
    ax = fig.add_subplot(121)  
    ax.plot(zcoord, Xg[:, gas.species_index('IC8H18')], '-k',  label='iC8H18') 
    ax.plot(zcoord, Xg[:, gas.species_index('O2')], '-y',  label='O2')
    ax.plot(zcoord, Xg[:, gas.species_index('H2')], '-c',  label='H2')
    ax.plot(zcoord, Xg[:, gas.species_index('H2O')], '-b',  label='H2O')
    ax.plot(zcoord, Xg[:, gas.species_index('CO')], '-r',  label='CO')
    ax.plot(zcoord, Xg[:, gas.species_index('CO2')], '-g',  label='CO2')  
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('$X_k$ [-]')
    ax.legend(loc='best')

    ax = fig.add_subplot(122)     
    ax.semilogy(zcoord, covs[:, surf.species_index('_Rh_')], '-k',  label='Rh(S)')
    ax.semilogy(zcoord, covs[:, surf.species_index('CO_Rh')], '-r',  label='CO(S)')
    ax.semilogy(zcoord, covs[:, surf.species_index('C_Rh')], '-g',  label='C(S)')  
    ax.semilogy(zcoord, covs[:, surf.species_index('H_Rh')], '-c',  label='H(S)')  
    ax.semilogy(zcoord, covs[:, surf.species_index('O_Rh')], '-y',  label='O(S)') 
    ax.semilogy(zcoord, covs[:, surf.species_index('OH_Rh')], '-m',  label='OH(S)') 
    ax.semilogy(zcoord, covs[:, surf.species_index('H2O_Rh')], '-b',  label='H2O(S)')
    ax.axis((0.0,zcoord[-1],1e-08,1.0))
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('$\Theta_k$ [-]')
    ax.legend(loc='best')
    
    plt.show()