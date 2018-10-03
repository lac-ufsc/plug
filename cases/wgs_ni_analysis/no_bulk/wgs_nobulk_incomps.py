import numpy as np
import cantera as ct
import plug as pfr
import os
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'wgs_ni_redux_nobulk.cti'        
surf_name = 'Ni_surface'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'../../..','data')

#### Coverage dependency matrix file ####: 
cov_file = os.path.join(filepath,'cov_matrix/covmatrix_wgs_ni.inp')

#### Save results into file? ####
savefile = 0
savepath = '/home/tpcarvalho/carva/python_data/wgs_ni/'
savename = 'wgs_nib_redux_incomps.pickle'

#Load phases solution objects
gas = ct.Solution(input_file)
surfCT = ct.Interface(input_file,surf_name,[gas])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT,  
                        cov_file = cov_file)

#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Vary the inlet CO:H2O ratio 
ratio = np.linspace(1,10,10)
flipratio = np.flip(ratio,axis=0)

#CO and H2O inlet molar fractions
co_x = 0.55*(ratio/(ratio+flipratio))
h2o_x = 0.55*(flipratio/(ratio+flipratio))

#Temperature range to simulate
#Trange = [T_in]
Trange = list(np.linspace(273.15+300,273.15+800,50))

#Inlet pressure [Pa]
P_in = 1e05
                                     
#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        z_out = 0.01,
                        diam =  0.01520)

#Initialize reacting wall
rs = pfr.ReactingSurface(r,surf,             
                         cat_area_pvol = 1.75e04)
 
#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        tcovs = 1e02,
                        atol = 1e-14,
                        rtol = 1e-07,
                        grid = 250,
                        max_steps = 5000,
                        solver_type = 'dae')

#CO index
CO_idx = r.gas.species_index('CO')
ratio_act = []
out = {'Xg': [], 'covs': [], 'CO_conv': [], 'CO_conv_eq': []}

for j in range(len(ratio)):
    
    #Create dict to store results at each temperature
    temp = {'Xg': [], 'covs': [], 'CO_conv': [], 'CO_conv_eq': []}
    
    #### Inlet thermo state ####:
    X_in = {'H2O':h2o_x[j], 'CO':co_x[j], 'H2':0.25, 'N2':0.2}          #WGS Ni
    
    #Load gas phase at inlet conditions
    gas_in = ct.Solution(input_file)
    gas_in.TPX = 273.15, P_in, X_in
    
    #Inlet mass flow rate [kg/s] at 3 SLPM
    mdot0 = 3*1.66667e-5*gas_in.density

    #Compute CO/H2O ratio
    ratio_act.append( gas_in.X[gas.species_index('CO')]
                     /gas_in.X[gas.species_index('H2O')] )
    
    #Loop over temperature range
    for i in range(len(Trange)):   
        
        #Current inlet temperature
        r.TPX = Trange[i], P_in, X_in 
        
        #Set inlet flow velocity [m/s]
        r.mdot = mdot0
    
        #Solve the PFR
        sim.solve()
        
        #Gas molar fractions and coverages along the reactor
        covs = sim.coverages
        Xg = sim.X
        uz = sim.u
        
        #Compute equilibrium
        gas_eq.TPX = Trange[i], P_in, X_in
        gas_eq.equilibrate('TP',solver='auto')
        Xg_eq = gas_eq.X
        
        #CO conversion    
        CO_conv = ( (Xg[0,CO_idx] - Xg[-1,CO_idx])/Xg[0,CO_idx] )*1e02
        CO_conv_eq = ( (Xg[0,CO_idx] - Xg_eq[CO_idx])/Xg[0,CO_idx] )*1e02
    
        #Append results to list
        temp['Xg'].append(Xg)
        temp['covs'].append(covs)        
        temp['CO_conv'].append(CO_conv)
        temp['CO_conv_eq'].append(CO_conv_eq)
 
    #Convert lists to array
    Xg = np.squeeze(np.array(temp['Xg']))
    covs = np.squeeze(np.array(temp['covs']))
    
    #Molar at the reactor outlet 
    CO_conv = np.array(temp['CO_conv'])
    CO_conv_eq = np.array(temp['CO_conv_eq'])

    #Append results to list
    out['Xg'].append(Xg)
    out['covs'].append(covs)        
    out['CO_conv'].append(CO_conv)
    out['CO_conv_eq'].append(CO_conv_eq)
        
#Axial coordinates [mm]
zcoord = sim.z*1e03

#Convert to lists
Xg_all = np.array(out['Xg'])
covs_all = np.array(out['covs'])

co_conv_all = np.array(out['CO_conv'])
eq_conv_all = np.array(out['CO_conv_eq'])

ratio_act = np.array(ratio_act)

#Coverages at reactor inlet and mean values along reactor
covs_in = covs_all[:,:,0,:]
covs_mean = np.mean(covs_all,axis=2) 

wgs_data = {'Xg': Xg_all, 'covs': covs_all, 'co_conv_all': co_conv_all, 
            'eq_conv_all': eq_conv_all, 'Tout': np.array(Trange) - 273.15,
            'ratio': ratio_act}

#Coverage look-up table
temp = np.tile(Trange,(len(ratio),1))
table_covs = covs_mean[:,:,[2,3]]
table_covs = np.dstack((table_covs,temp))

#Store data (serialize)
if savefile != 0:
    import pickle
    #Save file using pickle
    with open((savepath+savename), 'wb') as handle:
        pickle.dump(wgs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
              
#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

vis = 1

if vis != 0:
    
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(17, 8)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 18})  
    
    if vis==1:
  
        if len(Trange) == 1:
            #Temperature along the reactor
            ax = fig.add_subplot(121)  
            ax.plot(zcoord, Xg[:, gas.species_index('H2')], '-g', label='H2')
            ax.plot(zcoord, Xg[:, gas.species_index('H2O')], '-r',  label='H2O')
            ax.plot(zcoord, Xg[:, gas.species_index('CO2')], '-b',  label='CO2')
            ax.plot(zcoord, Xg[:, gas.species_index('CO')], '-c',  label='CO')  
            ax.set_xlabel('z [mm]')
            ax.set_ylabel('$X_k$ [-]')
            ax.legend(loc='best')
            
            ax1 = fig.add_subplot(122)  
            ax1.semilogy(zcoord, covs[:, surf.species_index('NI(S)')], '-k', label='NI(S)')
            ax1.semilogy(zcoord, covs[:, surf.species_index('CO(S)')], '-r',  label='CO(S)')
            ax1.semilogy(zcoord, covs[:, surf.species_index('H(S)')], '-g',  label='H(S)')
            ax1.semilogy(zcoord, covs[:, surf.species_index('H2O(S)')], '-b',  label='H2O(S)')
            ax1.semilogy(zcoord, covs[:, surf.species_index('CO2(S)')], '-c',  label='CO2(S)')
            ax1.semilogy(zcoord, covs[:, surf.species_index('OH(S)')], '-m',  label='OH(S)')
            ax1.axis((0.0,zcoord[-1],1e-10,1))
            ax1.set_xlabel('z [mm]')
            ax1.set_ylabel('$\Phi_i$ [-]')
            ax1.legend(loc='best')

        else:
                        
            #CO conversion variation with temperature
            Tout = np.array(Trange) - 273.15
    
            ax = fig.add_subplot(121)  
            
            for k in range(ratio.size):
                ax.plot(Tout, co_conv_all[k,:], '-b',)
                ax.plot(Tout, eq_conv_all[k,:], ':k',)
                 
            ax.axis((300,800,0.0,100.0))
            ax.set_xlabel('$T$ [$^\circ$C]')
            ax.set_ylabel('CO conversion [%]')
            
            #Coverages (mean) at reactor exit
            covs_out = np.mean(covs_all[:,:,-1,:],axis=0)
            
            ax1 = fig.add_subplot(122)
            ax1.semilogy(Tout, covs_out[:, surf.species_index('NI(S)')], '-k', label='NI(S)')
            ax1.semilogy(Tout, covs_out[:, surf.species_index('H(S)')], '-g',  label='H(S)')
            ax1.semilogy(Tout, covs_out[:, surf.species_index('H2O(S)')], '-b',  label='H2O(S)')
            ax1.semilogy(Tout, covs_out[:, surf.species_index('CO(S)')], '-r', label='CO(S)')
            ax1.semilogy(Tout, covs_out[:, surf.species_index('O(S)')], '-m',  label='O(S)')
            ax1.semilogy(Tout, covs_out[:, surf.species_index('OH(S)')], '-c',  label='OH(S)')
            ax1.axis((300,800,1e-06,1))
            ax1.set_xlabel('$T$ [$^\circ$C]')
            ax1.set_ylabel('$\Theta_k$ [-]')
            ax1.legend(loc='best')
            
        plt.show()
            