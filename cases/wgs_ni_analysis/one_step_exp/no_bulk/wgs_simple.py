import cantera as ct
import numpy as np
import csv
import plug as pfr
from scikits.odes import ode
from scipy.interpolate import interp1d 
from wgs_pfr_class import PFR_simple
import os
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'wgs_ni_redux_nobulk.cti'        
surf_name = 'Ni_surface'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'../../../..','data')

#### Coverage dependency matrix file ####: 
cov_file = os.path.join(filepath,'cov_matrix/covmatrix_wgs_ni.inp')

#### Save results into file? ####
savefile = 0
savepath = '/home/tpcarvalho/carva/python_data/wgs_ni/no_bulk/'
savename = 'wgs_ni_simple.pickle'

#Import surface coverages
filename = ('/home/tpcarvalho/carva/python_data/wgs_ni/no_bulk/wgs_ni_redux.pickle') 

import pickle
with open(filename, 'rb') as handle:
    wgs_data = pickle.load(handle)

#Temperature of WGS data (convert to K)
Tdata = wgs_data['Tout'] + 273.15

#Coverage data from micro-kinetic simulation
cov_data = np.array(wgs_data['covs'])

#Inlet conditions
T_in = 750
P_in = 1e05
X_in = {'H2O':0.457, 'CO':0.114, 'H2':0.229, 'N2':0.2}

#X_in = {'H2O':0.005, 'CO':0.495, 'H2':0.3, 'N2':0.2}

#Load phases solution objects
gas = ct.Solution(input_file)
surfCT = ct.Interface(input_file,surf_name,[gas])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT, 
                        cov_file = cov_file)

#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Load gas phase at inlet conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, P_in, X_in

#Inlet mass flow rate [kg/s] at 3 SLPM
mdot0 = 3*1.66667e-5*gas_in.density

'''Test PFR with simplified kinetics expression'''

#Initial conditions vector
y0 = gas_in.Y  

#Initialize simple PFR
r = PFR_simple(gas = gas,
                 surf = surf,
                 diam = 0.01520,
                 rate_eff = 1)

#Instantiate the solver
solver = ode('cvode', 
             r.eval_reactor_wgs,
             atol = 1e-09,
             rtol = 1e-06,
             max_steps = 2000,
             old_api = False) 
    
#Temperature range
#Trange = [T_in]
Trange = list(np.linspace(273.15+300,273.15+800,100))

#Reactor axial coordinates [m]
zcoord = np.linspace(0.0, 0.01, 250)

#Coverage values at reactor inlet  
cov_data_mean = cov_data[:,0,:] 

#Mean coverage values under the kinetic regime [375-475 C]
cov_mean = np.mean(cov_data_mean[15:35,:],axis=0)

#Set fixed values for NI(S), H(S), CO(S) 
cov_fixed = np.zeros(surf.n_species)

#Species index
idx_ni = surf.species_index('NI(S)')
idx_h = surf.species_index('H(S)')
idx_co = surf.species_index('CO(S)')

cov_fixed[idx_ni] = 1.0 - (cov_mean[idx_h] + cov_mean[idx_co])
cov_fixed[idx_h] = cov_mean[idx_h]
cov_fixed[idx_co] = cov_mean[idx_co] 
 
#Create interpolands for the coverages of H(S) and CO(S)
covs_interp = np.zeros(surf.n_species)

covs_h_f = interp1d(Tdata, cov_data_mean[:,idx_h], kind='cubic', 
                    fill_value ='extrapolate')
covs_co_f = interp1d(Tdata, cov_data_mean[:,idx_co], kind='cubic', 
                     fill_value ='extrapolate')
 
#Empty list to store results         
Yg = []
Yg_eq = []
covs = []
K2 = []
K4 = []
C = []

#Loop over temperature range
for i in range(len(Trange)):   
            
    #Update coverage values
    covs_h = covs_h_f(Trange[i])
    covs_co = covs_co_f(Trange[i])
    covs_ni = 1.0 - covs_h - covs_co
    
    #Update within coverages array
    covs_interp[idx_ni] = covs_ni
    covs_interp[idx_h] = covs_h
    covs_interp[idx_co] = covs_co
    
    #Update surface phase state
    surf.set_unnormalized_coverages(covs_interp)
    
    #Update thermodynamic states:  
    r.TPX = Trange[i], P_in, X_in 
    
    #Set inlet mass flow [kg/s]
    r.mdot = mdot0
 
    #Compute rates for current inlet state
    r.compute_k()
    
    #Conc
    C.append(gas.concentrations)
    
    #K2 K4
    K2.append(r.K2)
    K4.append(r.K4)
    
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
    
    #Append concentrations vector to list 
    Yg_eq.append(gas_eq.Y)
    
#Convert list to array    
Yg = np.array(Yg)
covs = np.array(covs)

#Mass fractions at the inlet, outlet and equilibrium
Yg_in = Yg[:,0,:]
Yg_out = Yg[:,-1,:]

Yg_eq = np.array(Yg_eq)

C = np.array(C)

#CO conversion [%]
co_idx = gas.species_index('CO')
co_conv =( (Yg_in[:,co_idx] - Yg_out[:,co_idx])/Yg_in[:,co_idx] )*1e02
co_conv_eq =( (Yg_in[:,co_idx] - Yg_eq[:,co_idx])/Yg_in[:,co_idx] )*1e02

#Save data to file
wgs_simple = {'co_conv': co_conv, 'Yg': Yg, 'Tout': np.array(Trange) - 273.15}

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
    
    
    if len(Trange) == 1:
        #Remove singleton dimensions
        Yg = np.squeeze(Yg)
        
        #Temperature along the reactor
        ax = fig.add_subplot(111)  
        ax.plot(zcoord, Yg[:, gas.species_index('H2')], '-g', label='H2')
        ax.plot(zcoord, Yg[:, gas.species_index('H2O')], '-r',  label='H2O')
        ax.plot(zcoord, Yg[:, gas.species_index('CO2')], '-b',  label='CO2')
        ax.plot(zcoord, Yg[:, gas.species_index('CO')], '-c',  label='CO')  
        ax.set_xlabel('z [m]')
        ax.set_ylabel('$X_k$ [-]')
        ax.legend(loc='best')

        plt.show()
    
    else:
        
        #Import paper experimental data from CSV file
        exp_data=[]
        filename = os.path.join(filepath,'exp_data/wheeler_co_conv_ni.csv') 
        with open(filename, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                exp_data.append(np.array(row,dtype=float))
        f.close()
        #Convert list to Numpy array  
        exp_data = np.array(exp_data)
        
        #WGS redux data
        Twgs = wgs_data['Tout']
        co_conv_redux = wgs_data['co_conv']
        
        #CO conversion variation with temperature
        Tout = np.array(Trange) - 273.15
        
        ax = fig.add_subplot(111)    
        ax.plot(exp_data[:,0], exp_data[:,1], 'ks',
                markersize=10, label='Exp. data')
        ax.plot(Tout, co_conv, '--k', label='Simplified Exp.')
        ax.plot(Twgs, co_conv_redux, '-k', label='10-step mech.')
        ax.plot(Tout, co_conv_eq, ':k', label='Equilibrium')    

        ax.axis((300,800,0.0,100.0))
        ax.set_xlabel('$T$ [$^\circ$C]')
        ax.set_ylabel('CO conversion [%]')
        ax.legend(loc='best',fontsize=20)      
