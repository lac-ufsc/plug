import cantera as ct
import numpy as np
import plug as pfr
from scikits.odes import ode
from scipy.interpolate import interp2d 
from wgs_pfr_class import PFR_simple
import os
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'wgs_nib_redux.cti'        
surf_name = 'Ni_surface'
bulk_name = 'Ni_bulk'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'../../..','data')

#### Coverage dependency matrix file ####: 
cov_file = os.path.join(filepath,'cov_matrix/covmatrix_wgs_ni.inp')

#### Save results into file? ####
savefile = 0
savepath = '/home/tpcarvalho/carva/python_data/wgs_ni/'
savename = 'wgs_nib_simple_mdots.pickle'

#Import WGS data at various inlet mass flow rates
filename = ('/home/tpcarvalho/carva/python_data/wgs_ni/wgs_nib_redux_mdots.pickle') 

import pickle
with open(filename, 'rb') as handle:
    wgs_data_mdots = pickle.load(handle)
    
#Temperature of WGS data (convert to K)
Tdata = wgs_data_mdots['Tout'] + 273.15

#Coverage data from micro-kinetic simulation
cov_data = np.array(wgs_data_mdots['covs'])

#Inlet conditions
T_in = 750
P_in = 1e05
X_in = {'H2O':0.457, 'CO':0.114, 'H2':0.229, 'N2':0.2}

#Load phases solution objects
gas = ct.Solution(input_file)
bulk = ct.Solution(input_file,bulk_name)
surfCT = ct.Interface(input_file,surf_name,[gas,bulk])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT, 
                        bulk = bulk, 
                        cov_file = cov_file)

#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Load gas phase at inlet conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, P_in, X_in

#Inlet mass flow rate [kg/s] at 0.3-30 SLPM
vdot0 = np.array([0.03,0.3,3,30,300,3000])
mdot0 = vdot0*1.66667e-5*gas_in.density

'''Test PFR with simplified kinetics expression'''

#Initial conditions vector
y0 = gas_in.Y  

#Initialize simple PFR
r = PFR_simple(gas = gas,
                 surf = surf,
                 bulk = bulk,
                 diam = 0.01520,
                 rate_eff = 0.75e-05)

#Instantiate the solver
solver = ode('cvode', 
             r.eval_reactor_wgs,
             atol = 1e-09,
             rtol = 1e-06,
             max_steps = 2000,
             old_api = False) 
    
#Temperature range
#Trange = [T_in]
Trange = list(np.linspace(273.15+300,273.15+700,60))

#Reactor axial coordinates [m]
zcoord = np.linspace(0.0, 0.01, 250)

#Coverage values at reactor inlet  
#cov_data_mean = cov_data[:,:,0,:]
cov_data_mean = np.mean(cov_data,axis=2)

#Mean coverage values under the kinetic regime [375-475 C]
cov_mean = np.mean(cov_data_mean[:,15:35,:],axis=1)

#Set fixed values for NI(S), H(S), CO(S) 
cov_fixed = np.zeros(surf.n_species)

#Species index
idx_ni = surf.species_index('NI(S)')
idx_h = surf.species_index('H(S)')
idx_co = surf.species_index('CO(S)')
 
#Create interpolands for the coverages of H(S) and CO(S)
covs_interp = np.zeros(surf.n_species)

covs_h_f = interp2d(mdot0,Tdata,cov_data_mean[:,:,idx_h].T, kind='cubic')
covs_co_f = interp2d(mdot0,Tdata,cov_data_mean[:,:,idx_co].T, kind='cubic')

#Dict to store results
out = {'co_conv': []}

#CO index
co_idx = gas.species_index('CO')

for j in range(mdot0.size):
    
    #Empty list to store results         
    Yg = []
    Yg_eq = []
    covs = []
    
    #Set fized coverage values for entire temperature range
    cov_fixed[idx_ni] = 1.0 - (cov_mean[j,idx_h] + cov_mean[j,idx_co])
    cov_fixed[idx_h] = cov_mean[j,idx_h]
    cov_fixed[idx_co] = cov_mean[j,idx_co] 

    #Loop over temperature range
    for i in range(len(Trange)):   
                
        #Update coverage values
        covs_h = covs_h_f(mdot0[j],Trange[i])
        covs_co = covs_co_f(mdot0[j],Trange[i])
        covs_ni = 1.0 - covs_h - covs_co
        
        #Update within coverages array
        covs_interp[idx_ni] = covs_ni
        covs_interp[idx_h] = covs_h
        covs_interp[idx_co] = covs_co
        
        #Update surface initial coverages
#        surf.set_unnormalized_coverages(cov_fixed)
        surf.set_unnormalized_coverages(covs_interp)
    
        #Update thermodynamic states:  
        r.TPX = Trange[i], P_in, X_in 
        
        #Set inlet mass flow [kg/s]
        r.mdot = mdot0[j]
        
        #Compute rates for current inlet state
        r.compute_k()
        
        #Compute solution and return it along supplied axial coords.
        sol = solver.solve(zcoord, y0)
        
        #If an error occurs
        if sol.errors.t != None:
            raise ValueError(sol.message,sol.errors.t)
        
        #Get solution vector
        values = sol.values.y
        
        #Get molar fractions
        Yg.append(values) 
    
        #Get coverages at exit
        covs.append(surf.kinetics.surfCT.coverages)
    
        #Compute equilibrium
        gas_eq.TPX = Trange[i], P_in, X_in 
        gas_eq.equilibrate('TP',solver='auto')
        
        #Append concentrations vector to list 
        Yg_eq.append(gas_eq.Y)
        
    #Convert list to array    
    Yg = np.array(Yg)
    covs = np.array(covs)

    #Mass fractions at the inlet, outlet and equilibrium
    Yg_in = Yg[:,0,:]
    Yg_out = Yg[:,-1,:]
    
    Yg_eq = np.array(Yg_eq)
    
    #CO conversion [%]
    co_conv =( (Yg_in[:,co_idx] - Yg_out[:,co_idx])/Yg_in[:,co_idx] )*1e02
    co_conv_eq =( (Yg_in[:,co_idx] - Yg_eq[:,co_idx])/Yg_in[:,co_idx] )*1e02

    #Append to dict
    out['co_conv'].append(co_conv)

co_conv_all = np.array(out['co_conv'])

#Save data to file
wgs_simple = {'co_conv': co_conv_all, 'Tout': np.array(Trange) - 273.15}

#Store data (serialize)
if savefile != 0:
    import pickle
    #Save file using pickle
    with open((savepath+savename), 'wb') as handle:
        pickle.dump(wgs_simple, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Print time elapsed
print('It took {0:0.8f} seconds'.format(time.time() - start)) 

vis = 1

if vis == 1:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(11, 8)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 24})  
            
    #CO conversion
    Twgs = wgs_data_mdots['Tout'] #- 273.15
    co_conv_redux = wgs_data_mdots['co_conv']
    
    #CO conversion variation with temperature
    Tout = np.array(Trange) - 273.15
    
    ax = fig.add_subplot(111)   
    
    for k in range(mdot0.size):
        ax.plot(Tout, co_conv_all[k,:], '--ko',)
        ax.plot(Twgs, co_conv_redux[k,:], '-k',)

    ax.plot(Tout, co_conv_eq, ':k', label='Equilibrium')

    ax.axis((300,700,0.0,100.0))
    ax.set_xlabel('$T$ [$^\circ$C]')
    ax.set_ylabel('CO conversion [%]')
    ax.legend(loc='best',fontsize=20,frameon=False)    
