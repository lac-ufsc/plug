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
savefile = 1
savepath = '/home/tpcarvalho/carva/python_data/wgs_ni/'
savename = 'wgs_nib_simple_incomps.pickle'

#Import WGS data at various inlet mass flow rates
filename = ('/home/tpcarvalho/carva/python_data/wgs_ni/'
            'wgs_nib_redux_incomps.pickle') 
    
import pickle
with open(filename, 'rb') as handle:
    wgs_data_incomps = pickle.load(handle)

#Temperature of WGS data (convert to K)
Tdata = wgs_data_incomps['Tout'] + 273.15

#CO/H2O ratio at the inlet 
Xg_in = wgs_data_incomps['Xg'][:,:,0,:]
ratio_data = wgs_data_incomps['ratio']
    
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

#Vary the inlet CO:H2O ratio 
ratio = np.linspace(1,10,10)
flipratio = np.flip(ratio,axis=0)

#CO and H2O inlet molar fractions
co_x = 0.55*(ratio/(ratio+flipratio))
h2o_x = 0.55*(flipratio/(ratio+flipratio))

'''Test PFR with simplified kinetics expression'''

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

#Inlet pressure [Pa]
P_in = 1e05
  
#Temperature range
#Trange = [T_in]
Trange = list(np.linspace(273.15+300,273.15+800,100))

#Reactor axial coordinates [m]
zcoord = np.linspace(0.0, 0.01, 250)
       
#Coverage data from micro-kinetic simulation
cov_data = np.array(wgs_data_incomps['covs'])

#Coverage values at reactor inlet  
cov_data_mean = cov_data[:,:,0,:]
#cov_data_mean = np.mean(cov_data,axis=2)

#Mean coverage values under the kinetic regime [375-475 C]
cov_mean = np.mean(cov_data_mean[:,15:35,:],axis=1)

#Set fixed values for NI(S), H(S), CO(S) 
cov_fixed = np.zeros(surf.n_species)

#Create array to fill with interpolated values
covs_interp = np.zeros(surf.n_species)

#Species index
idx_ni = surf.species_index('NI(S)')
idx_h = surf.species_index('H(S)')
idx_co = surf.species_index('CO(S)')

#Create interpolands for the coverages of H(S) and CO(S)
covs_h_f = interp2d(ratio_data,Tdata, cov_data_mean[:,:,idx_h].T, kind='cubic')
covs_co_f = interp2d(ratio_data,Tdata, cov_data_mean[:,:,idx_co].T, kind='cubic')

co_idx = gas.species_index('CO')
    
#Dict to store results
out = {'Yg': [], 'covs': [], 'co_conv': [], 'co_conv_eq': [], 'Cg': [],
       'K2': [], 'K4': []}

for j in range(ratio.size):
    
    #### Inlet composition ####:
    X_in = {'H2O':h2o_x[j], 'CO':co_x[j], 'H2':0.25, 'N2':0.2}          #WGS Ni
    
    #Load gas phase at inlet conditions
    gas_in = ct.Solution(input_file)
    gas_in.TPX = 273.15, P_in, X_in

    #Inlet mass flow rate [kg/s] at 3 SLPM
    mdot0 = 3*1.66667e-5*gas_in.density
    
    #Initial mass fractions
    y0 = gas_in.Y
    
    #Compute CO/H2O ratio
    ratio_act = ( gas_in.X[gas.species_index('CO')]
                 /gas_in.X[gas.species_index('H2O')] )

    #Update fixed coverages for temperature range
    cov_fixed[idx_ni] = 1.0 - (cov_mean[j,idx_h] + cov_mean[j,idx_co])
    cov_fixed[idx_h] = cov_mean[j,idx_h]
    cov_fixed[idx_co] = cov_mean[j,idx_co] 

    #Empty list to store results         
    Yg = []
    Yg_eq = []
    covs = []
    K2 = []
    K4 = []
    Cg = []

    #Loop over temperature range
    for i in range(len(Trange)):   
    
        #Update coverage values
        covs_h = covs_h_f(ratio_act,Trange[i])
        covs_co = covs_co_f(ratio_act,Trange[i])
        covs_ni = 1.0 - covs_h - covs_co
        
        #Update within coverages array
        covs_interp[idx_ni] = covs_ni
        covs_interp[idx_h] = covs_h
        covs_interp[idx_co] = covs_co

        #Update surface phase state
#        surf.set_unnormalized_coverages(cov_fixed)
        surf.set_unnormalized_coverages(covs_interp)
        
        #Update thermodynamic states:  
        r.TPX = Trange[i], P_in, X_in 
        
        #Set inlet mass flow [kg/s]
        r.mdot = mdot0
        
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
        covs.append(r.theta)
    
        #Compute equilibrium
        gas_eq.TPX = Trange[i], P_in, X_in 
        gas_eq.equilibrate('TP',solver='auto')
        
        #Append mass fractions vector to list 
        Yg_eq.append(gas_eq.Y)
    
        K2.append(r.K2)
        K4.append(r.K4)
        Cg.append(gas.concentrations)
        
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
    out['Yg'].append(Yg)
    out['covs'].append(covs)
    out['co_conv'].append(co_conv)
    out['co_conv_eq'].append(co_conv_eq)
    out['Cg'].append(np.array(Cg))
    out['K2'].append(np.array(K2))
    out['K4'].append(np.array(K4))

co_conv_all = np.array(out['co_conv'])
co_conv_eq = np.array(out['co_conv_eq'])

#Save data to file
wgs_simple = {'co_conv': co_conv_all, 'co_conv_eq': co_conv_eq, 'ratio': ratio, 
              'Tout': np.array(Trange) - 273.15}

#Store data (serialize)
if savefile != 0:
    import pickle
    #Save file using pickle
    with open((savepath+savename), 'wb') as handle:
        pickle.dump(wgs_simple, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Print time elapsed
print('It took {0:0.8f} seconds'.format(time.time() - start)) 

#%%

vis = 1

if vis != 0:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(11, 8)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 24})  
    
    if vis == 1:      
        #CO conversion
        co_conv_redux = wgs_data_incomps['co_conv_all']
        
        #CO conversion variation with temperature
        Tplot = np.array(Trange) - 273.15
        
        ax = fig.add_subplot(111)   
        
        for k in range(ratio.size):
            ax.plot(Tplot, co_conv_all[k,:], 'ko',)
            ax.plot(Tplot, co_conv_eq[k,:], ':k',)
            ax.plot(wgs_data_incomps['Tout'], co_conv_redux[k,:], '-k',)
            
        ax.axis((300,700,0.0,100.0))
        ax.set_xlabel('$T$ [$^\circ$C]')
        ax.set_ylabel('CO conversion [%]')
        
    elif vis == 2:        
        #Gas concentration
        conc = np.array(out['Cg'])
        
        #H2 term
        K4 = np.array(out['K4'])
        h2_term = np.sqrt(conc[:,:,gas.species_index('H2')]*K4)
        
        #CO term
        K2 = np.array(out['K2'])
        co_term = conc[:,:,gas.species_index('CO')]*K2
        
        #Ni term
        ni_term = (1.0 + h2_term + co_term)
        
        #Temperature
        Tplot = np.array(Trange) - 273.15
        
        ax = fig.add_subplot(111)   
        
        for k in range(ratio.size):
            if k == 0:
                ax.plot(Tplot, h2_term[k,:], '-b', label='$\sqrt{K_4 C_{H2}}$',linewidth=3)
                ax.plot(Tplot, co_term[k,:], '-r', label='$K_2 C_{CO}$',linewidth=3)
                ax.plot(Tplot, ni_term[k,:], '-k', label='$1 + \sqrt{K_4 C_{H2}} + K_2 C_{CO}$',linewidth=3)
            else:
                ax.plot(Tplot, h2_term[k,:], '-b',linewidth=3)
                ax.plot(Tplot, co_term[k,:], '-r',linewidth=3)
                ax.plot(Tplot, ni_term[k,:], '-k',linewidth=3)

        ax.set_xlabel('$T$ [$^\circ$C]')
        ax.set_ylabel('Eq. terms [-]') 
        ax.legend(loc='best',frameon=False)
    
    
