import cantera as ct
import numpy as np
import plug as pfr
from plug.utils.pca_reduction import pca_reduction
import os
import time
start = time.time()   

#### Input mechanism data ####:    
input_file = 'wgs_nib.cti'        
surf_name = 'Ni_surface'
bulk_name = 'Ni_bulk'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'../..','data')

#### Coverage dependency matrix file ####: 
cov_file = os.path.join(filepath,'cov_matrix/covmatrix_wgs_ni.inp')

#Load phases solution objects
gas = ct.Solution(input_file)
bulk = ct.Solution(input_file,bulk_name)
surfCT = ct.Interface(input_file,surf_name,[gas,bulk])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT, 
                        bulk = bulk, 
                        cov_file = cov_file)

#### Current thermo state ####:
X_in = 'H2O:0.457, CO:0.114, H2:0.229, N2:0.2'          
T_in = 750.0
P_in = 101325.0

#Load gas phase at inlet conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, 1e05, X_in

#Inlet mass flow rate [kg/s] at 3 SLPM
mdot0 = 3*1.66667e-5*gas_in.density

#Temperature range to simulate (~623 - 773 K)
n_temps = 10
Trange = list(np.linspace(273.15+375,273.15+475,n_temps))

#Trange = [T_in]
u_in = np.geomspace(1,1e03,len(Trange))*1.0

#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        z_out = 0.01,
                        diam =  0.01520)

#Initialize reacting wall
rs = pfr.ReactingSurface(r,surf,
                         bulk = [bulk],              
                         cat_area_pvol = 7e03)

#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        tcovs = 1e03,
                        atol = 1e-10,
                        rtol = 1e-06,
                        grid = 250,
                        max_steps = 5000,
                        solver_type = 'dae')

#CO index
co_idx = r.gas.species_index('CO')

#Create lists
Sm = []
Sm_all = []
co_conv = []
delta_yco = []

#Reaction rate perturbation factor
factor = 0.95

#Loop over temperature range
for i,temp in enumerate(Trange):   

    #Current thermodynamic state
    r.TPX = temp, P_in, X_in 
    
    #Set inlet flow velocity [m/s]
#    r.mdot = mdot0
    r.u = u_in[i]
    
    #Set reaction rate multiplier to one
    surf.set_multiplier(1.0)
    
    #Solve the PFR
    sim.solve()
    
    #Gas molar fractions along the reactor
    X = sim.X
    
    #Compute reference CO conversion
    co_conv_ref = ( 1.0 - X[-1,co_idx]/X[0,co_idx] )*1e02
    
    #Create lists to store values
    Sm = []
    co_conv_pt = []
    
    #Deactivate each reaction sequentially
    for j in range(surf.n_reactions):
        
        #Set reaction rate multiplier to one
        surf.set_multiplier(1.0)
    
        #Perturb each reaction rate by 5 %
        surf.set_multiplier(factor,j)
        
        #Current thermodynamic state
        r.TPX = temp, P_in, X_in 
        
        #Set inlet flow velocity [m/s]
#        r.mdot = mdot0
        r.u = u_in[i]
        
        #Solve the PFR
        sim.solve()
        
        #Perturbed gas molar fractions along the reactor
        Xpt = sim.X

        #Compute reference CO conversion
        co_conv_pt.append(( 1.0 - Xpt[-1,co_idx]/Xpt[0,co_idx] )*1e02)
        
        #Compute deviation of perturbed values from reference values
        delta = np.abs(co_conv_ref - co_conv_pt[j])

        #Normalized sensitivity coefficient for CO conversion
        Sm.append( (factor*delta)/(co_conv_ref*(1.0-factor)) )
    
    #Convert list to array
    Sm = np.array(Sm)
    co_conv_pt = np.array(co_conv_pt)
    
    #Append sensitivity data to list
    Sm_all.append(Sm)

#Convert list to array
Sm_all = np.array(Sm_all)

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 
    
#%%
#Perform PCA-based mechanism reduction
res = pca_reduction(Sm_all,1)

#PCA eigenvalues
eigenvalues = res[0]

#PCA eigenvectors
eigenvectors = res[1]

#Cut-off value 
cut_off = 2.7e-05

#Index of largest eigenvector components (relevant reactions)
if len(eigenvalues) == 1:
    
    idx_redux = np.abs(eigenvectors)>cut_off
else:
    idx_redux = np.any(np.abs(eigenvectors)>cut_off,axis=1).astype(int)

#Index of reduced reactions
idx_redux = np.nonzero(idx_redux)[0]
    
#Print number of reaction in reduced mechanism
print('Reduced number of reactions: ',len(idx_redux))

#Test reduced mech class
rmech = pfr.ReduceMechanism(gas,surf,bulk)

#Test reduced mech class
rmech.write_to_cti(idx_redux)


