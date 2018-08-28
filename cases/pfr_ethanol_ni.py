import numpy as np
import cantera as ct
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'ethanol_nib.cti'      
surf_name = 'Ni_surface'
bulk_name = 'Ni_bulk'

#### Coverage dependency matrix file ####: 
cov_file = ('/home/tpcarvalho/carva/python_data/kinetic_mechanisms/input_files'
            '/cov_matrix/covmatrix_et_ni_2.inp')

#### Experimental data filename ####:
expfile = ('/home/tpcarvalho/carva/python_data/ethanol/nickel/et_conv_923.csv') 

#### Save results into file? ####
savefile = 0
savepath = '/home/tpcarvalho/carva/python_data/ethanol/'
savename = 'ethanol_ni.pickle'

#Load phases solution objects
gas = ct.Solution(input_file)
bulk = ct.Solution(input_file,bulk_name)
surfCT = ct.Interface(input_file,surf_name,[gas,bulk])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT, 
                        bulk = bulk, 
                        cov_file = cov_file)

#### Current thermo state ####:
X_in = 'CH3CH2OH:0.016, H2O:0.09, AR:0.894'        #Ethanol
T_in = 923
P_in = 101325.0

#Inlet gas solution object
gas_in = ct.Solution(input_file) 
gas_in.TPX = 273.15, 1e05, X_in

#Volumetric flow range [liters/min]
vdot_in = np.geomspace(20,1.0,15)

#Inlet mass flow rate [kg/s] at variable SLPM
mdot_in = list(vdot_in*1.66667e-5*gas_in.density)

#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        energy = 0,
                        z_out = 0.004,
                        diam =  0.01)

#Initialize the reacting wall
rs = pfr.ReactingSurface(r,surf,
                         bulk = [bulk],
                         cat_area_pvol = 30)
                     
#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        tcovs = 1e02,
                        atol = 1e-14,
                        rtol = 1e-07,
                        grid = 300,
                        max_steps = 5000,
                        solver_type = 'dae')

#CO index
et_idx = r.gas.species_index('CH3CH2OH')

#Create lists to store results at each temperature
X = []
C = []
covs = []
et_conv = []

#Loop over temperature range
for m in mdot_in:   

    #Current inlet temperature
    r.TPX = T_in, P_in, X_in 
    
    #Set inlet flow velocity [m/s]
    r.mdot = m
     
    #Solve the PFR
    sim.solve()
    
    #Gas molar fractions and coverages along the reactor
    covs.append(sim.coverages)
    X.append(sim.X)
    #Gas species concentrations at the outlet
    C.append(gas.concentrations)
    
    #CO conversion    
    et_conv.append(((sim.X[0,et_idx] - sim.X[-1,et_idx])/sim.X[0,et_idx])*1e02)
    
#Axial coordinates [mm]
zcoord = sim.z*1e03

#Convert lists to array
X = np.squeeze(np.array(X))
covs = np.squeeze(np.array(covs))
C = np.array(C)

#Molar at the reactor outlet 
et_conv = np.array(et_conv)

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

#%%

#Estimate amount of catalyst
dispersion = 0.20
weight_perc = 0.003
cat_mw = surf.molecular_weights[surf.species_index('NI(S)')]
cat_loading = rs.cat_area*rs.surface.site_density*cat_mw/(dispersion*weight_perc)
#Weight of catalyst in grams
cat_loading *= 1e03  

#Space time [mg min/mL]
space_time = (cat_loading*1e03/(vdot_in*1e03))*1e05

vis = 1

if vis==1:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(17, 8)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 18})  
    if len(mdot_in) == 1:

        #temperature along the reactor
        ax = fig.add_subplot(121)  
        ax.plot(zcoord, X[:, gas.species_index('CH3CH2OH')], '-k', label='Ethanol')
        ax.plot(zcoord, X[:, gas.species_index('H2O')], '-b',  label='H2O')
        ax.plot(zcoord, X[:, gas.species_index('H2')], '-g',  label='H2')
        ax.plot(zcoord, X[:, gas.species_index('CO')], '-r',  label='CO')
        ax.plot(zcoord, X[:, gas.species_index('CO2')], '-m',  label='CO2')   
        ax.set_xlabel('z [mm]')
        ax.set_ylabel('$X_k$ [-]')
        ax.legend(loc='lower right')
        
        ax1 = fig.add_subplot(122)  
        ax1.semilogy(zcoord, covs[:, surf.species_index('NI(S)')], '-k',  label='NI(S)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('H(S)')], '-g',  label='H(S)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('C(S)')], '-y',  label='C(S)')  
        ax1.semilogy(zcoord, covs[:, surf.species_index('CO(S)')], '-r',  label='CO(S)') 
        ax1.semilogy(zcoord, covs[:, surf.species_index('CH(S)')], '-b',  label='CH(S)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CHCH(S)')], '-c',  label='CHCH(S)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CCOOH(S)')], '-m',  label='CCOOH(S)')
        ax1.axis((0.0,zcoord[-1],1e-04,1))
        ax1.set_xlabel('z [mm]')
        ax1.set_ylabel('$\Phi_i$ [-]')
        ax1.legend(loc='lower right')
        plt.show()
        
    else:
        #Import paper experimental data from CSV file
        import csv
        exp_data=[]
        with open(expfile, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                exp_data.append(np.array(row,dtype=float))
        f.close()
        #Convert list to Numpy array  
        exp_data = np.array(exp_data)

        #Ethanol conversion variation with space time
        
        ax = fig.add_subplot(121)    
        ax.plot(space_time, et_conv, '-b', label='Numerical')
        ax.plot(exp_data[:,0], exp_data[:,1]*1e02, 'ks',
                markersize=10, label='Exp. data')
        ax.set_xlabel('Space time [mg min/mL] $10^5$')
        ax.set_ylabel('Conversion [%]')
        ax.legend(loc='lower right',fontsize=18)
        
        ax1 = fig.add_subplot(122) 
        covs_o = covs[:,-1,:]
        
        #Find 5 largest coverage values
        cov_idx = np.argsort(-covs_o[0,:])[:5]
        
        for i in cov_idx:
            ax1.semilogy(space_time, covs_o[:, i], label=surf.species_names[i])

        ax1.set_xlabel('Space time [mg min/mL] $10^5$')
        ax1.set_ylabel('$\Phi_i$ [-]')
        ax1.legend(loc='best')
        plt.xlim(space_time[0],space_time[-1])
        plt.show() 
