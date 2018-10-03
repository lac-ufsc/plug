import numpy as np
import cantera as ct
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'wgs_ni_nobulk.cti'        
surf_name = 'Ni_surface'

#### Coverage dependency matrix file ####: 
cov_file = ('/home/tpcarvalho/carva/python_data/kinetic_mechanisms/'
            'input_files/cov_matrix/covmatrix_wgs_ni.inp')

#Load phases solution objects
gas = ct.Solution(input_file)
surfCT = ct.Interface(input_file,surf_name,[gas])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT,cov_file = cov_file)

#### Current thermo state ####:
P_in = 1e05 #Pa
X_in = {'CO':0.25,'H2O':0.25,'CO2':0.25,'H2':0.25}   

#temperature range to simulate
Trange = np.linspace(273.15+375,273.15+500,3)

#%%
#Create instance of reduced mech class
rmech = pfr.ReduceMechanism(gas,surf)

#Compute reduced mechanism
err = rmech.solve_reduction_problem(Trange,(P_in,X_in))

#Get reduced mechanism index at each condition  
idx_redux = rmech.get_reduced_index(err,1e-2) 
   
#Save mechanism folder
folder = '/home/tpcarvalho/carva/python_data/kinetic_mechanisms/input_files/'
mech_file = folder+'wgs_ni_redux.cti'

#Get CTI version of reduced mechanism
rmech.write_to_cti(idx_redux)
#rmech.write_to_cti(idx_redux,mech_file)

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 