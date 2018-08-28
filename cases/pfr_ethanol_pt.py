import numpy as np
import cantera as ct
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'ethanol_ptb.cti'     
surf_name = 'Pt_surface'
bulk_name = 'Pt_bulk'

#### Coverage dependency matrix file ####: 
cov_file = ('/home/tpcarvalho/carva/python_data/kinetic_mechanisms/input_files'
            '/cov_matrix/covmatrix_et_pt.inp')

#### Save results into file? ####
savefile = 0
savepath = '/home/tpcarvalho/carva/python_data/ethanol/'
savename = 'ethanol_pt.pickle'

#Load phases solution objects
gas = ct.Solution(input_file)
bulk = ct.Solution(input_file,bulk_name)
surfCT = ct.Interface(input_file,surf_name,[gas,bulk])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT, 
                        bulk = bulk, 
                        cov_file = cov_file)

#### Current thermo state ####:
X_in = 'CH3CH2OH:0.125, H2O:0.375, HE:0.50'        #Ethanol
T_in = 668
P_in = 101325.0

#Inlet gas solution object
gas_in = ct.Solution(input_file) 
gas_in.TPX = 273.15, 1e05, X_in

#Inlet mass flow rate [kg/s] at 0.12 SLPM
mdot0 = 0.12*1.66667e-5*gas_in.density

#temperature range to simulate
Trange = [T_in]
#Trange = list(np.linspace(273.15+300,273.15+400,20))

#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        energy = 0,
                        z_out = 0.01,
                        diam =  0.012)

#Initialize the reacting wall
rs = pfr.ReactingSurface(r,surf,
                            bulk = [bulk],
                            cat_area_pvol = 7e04)

#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        tcovs = 1e03,
                        atol = 1e-14,
                        rtol = 1e-06,
                        grid = 300,
                        max_steps = 5000,
                        solver_type = 'dae')

#Ethanol index
et_idx = r.gas.species_index('CH3CH2OH')

#Create lists to store results at each temperature
X = []
C = []
covs = []
et_conv = []

#Loop over temperature range
for temp in Trange:   

    #Current inlet temperature
    r.TPX = temp, P_in, X_in 
    
    #Set inlet flow velocity [m/s]
    r.mdot = mdot0
     
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

#Index of main products containing H2
idx_h2 = [1,5]
m_factor_h2 = np.array([2,1])

#Index of main products containing C
idx_c = [1,3,4]
m_factor_c = np.array([1,1,1])

#Compute selectivities
sl_h2 = C[:,gas.species_index('H2')]/np.sum(m_factor_h2*C[:,idx_h2],axis=1)
sl_ch4 = C[:,gas.species_index('CH4')]/np.sum(m_factor_c*C[:,idx_c],axis=1)
sl_co = C[:,gas.species_index('CO')]/np.sum(m_factor_c*C[:,idx_c],axis=1)
sl_co2 = C[:,gas.species_index('CO2')]/np.sum(m_factor_c*C[:,idx_c],axis=1)

#Save data to dict 
out = dict()
out['T'] = np.array(Trange)
out['X'] = X
out['covs'] = covs
out['conv'] = et_conv
out['sl_h2'] = sl_h2
out['sl_ch4'] = sl_ch4
out['sl_co'] = sl_co
out['sl_co2'] = sl_co2

#Store data (serialize)
if savefile != 0:
    import pickle
    #Save file using pickle
    with open((savepath+savename), 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Estimate amount of catalyst
dispersion = 0.99
weight_perc = 0.005
cat_mw = surf.molecular_weights[surf.species_index('PT(S)')]
cat_loading = rs.cat_area*rs.surface.site_density*cat_mw/(dispersion*weight_perc)
#Weight of catalyst in grams
cat_loading *= 1e03  
       
#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

#%%
vis = 1

if vis==1:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(17, 8)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 18})  
    if len(Trange) == 1:

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
        ax1.semilogy(zcoord, covs[:, surf.species_index('PT(S)')], '-k',  label='PT(S)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('H(S)')], '-g',  label='H(S)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CHCO(S)')], '-y',  label='CHCO(S)')  
        ax1.semilogy(zcoord, covs[:, surf.species_index('CO(S)')], '-r',  label='CO(S)') 
        ax1.semilogy(zcoord, covs[:, surf.species_index('CH(S)')], '-b',  label='CH(S)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CH3CO(S)')], '-m',  label='CH3CO(S)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CH3(S)')], '--r',  label='CH3(S)') 
        ax1.semilogy(zcoord, covs[:, surf.species_index('CH2(S)')], '--b',  label='CH2(S)') 
        ax1.axis((0.0,zcoord[-1],1e-04,1))
        ax1.set_xlabel('z [mm]')
        ax1.set_ylabel('$\Phi_i$ [-]')
        ax1.legend(loc='lower right')
        plt.show()
        
    else:
        #Import paper experimental data from CSV file
        import csv
        exp_data=[]
        filename = ('/home/tpcarvalho/carva/python_data/ethanol/et_conv_pt.csv') 
        with open(filename, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                exp_data.append(np.array(row,dtype=float))
        f.close()
        #Convert list to Numpy array  
        exp_data = np.array(exp_data)

        #Ethanol conversion variation with temperature
        Tplot = np.array(Trange)
        Tplot -= 273.15
        
        ax = fig.add_subplot(121)    
        ax.plot(Tplot, et_conv, '-b', label='Numerical')
        ax.plot(exp_data[:,0], exp_data[:,1], 'ks',
                markersize=10, label='Exp. data')
        ax.set_xlabel('$T$ [$^\circ$C]')
        ax.set_ylabel('Conversion [%]')
        ax.legend(loc='lower right',fontsize=18)
        
        ax1 = fig.add_subplot(122) 
        covs_o = covs[:,-1,:]
        
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('PT(S)')], '-k',  label='PT(S)')
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('H(S)')], '-g',  label='H(S)')
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('CHCO(S)')], '-y',  label='CHCO(S)')  
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('CO(S)')], '-r',  label='CO(S)') 
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('CH(S)')], '-b',  label='CH(S)')
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('CH3CO(S)')], '-m',  label='CH3CO(S)')
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('CH3(S)')], '--r',  label='CH3(S)') 
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('CH2(S)')], '--b',  label='CH2(S)') 
        ax1.axis((Tplot[0],Tplot[-1],1e-08,1))
        ax1.set_xlabel('$T$ [$^\circ$C]')
        ax1.set_ylabel('$\Phi_i$ [-]')
        ax1.legend(loc='lower right')
        plt.show() 