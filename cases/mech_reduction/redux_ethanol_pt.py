import numpy as np
import cantera as ct
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'ethanol_pt.cti'     
surf_name = 'Pt_surface'

#### Coverage dependency matrix file ####: 
cov_file = ('/home/tpcarvalho/carva/python_scripts/cantera/pfr/plug'
            '/data/cov_matrix/covmatrix_et_pt.inp')

#Load phases solution objects
gas = ct.Solution(input_file)
surfCT = ct.Interface(input_file,surf_name,[gas])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT,cov_file = cov_file)

#### Current thermo state ####:
P_in = 1e05 #Pa
X_sr = {'CH3CH2OH':0.21,'H2O':0.61,'CO2':0.045,'H2':0.135}   #Steam reforming

#temperature range to simulate
Trange = np.linspace(273.15+250,273.15+450,3)

#%%
#Create instance of reduced mech class
rmech = pfr.ReduceMechanism(gas,surf)

#Compute reduced mechanism
err_sr = rmech.solve_reduction_problem(Trange,(P_in,X_sr))

#Get reduced mechanism index at each condition  
idx_redux = rmech.get_reduced_index(err_sr,1e-02) 
   
#Save mechanism folder
folder = '/home/tpcarvalho/carva/python_scripts/cantera/pfr/plug/data/'
mech_file = folder+'ethanol_pt_redux.cti'

#Get CTI version of reduced mechanism
rmech.write_to_cti(idx_redux)
#rmech.write_to_cti(idx_redux,mech_file)

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 