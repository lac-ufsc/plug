import numpy as np
import cantera as ct
import plug as pfr
import os
import time
start = time.time()  

#### Input mechanism data ####:   
input_file0= 'ethanol_pt.cti'     
input_file = 'ethanol_pt_redux.cti'    
surf_name = 'Pt_surface'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'../..','data')

#### Coverage dependency matrix file ####: 
cov_file = os.path.join(filepath,'cov_matrix/covmatrix_et_pt.inp')

#Load phases solution objects
gas0 = ct.Solution(input_file0)
surfCT0 = ct.Interface(input_file0,surf_name,[gas0])

#Load in-house surface phase object
surf0 = pfr.SurfacePhase(gas0,surfCT0,cov_file = cov_file)

#Load phases solution objects for reduced mechanism
gas = ct.Solution(input_file)
surfCT = ct.Interface(input_file,surf_name,[gas])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT,cov_file = cov_file)

#### Current thermo state ####:
X_in = 'CH3CH2OH:0.125, H2O:0.375, HE:0.50'        
T_in = 668
P_in = 101325.0

#Inlet gas solution object
gas_in = ct.Solution(input_file) 
gas_in.TPX = 273.15, 1e05, X_in

#Inlet mass flow rate [kg/s] at 0.12 SLPM
mdot0 = 0.12*1.66667e-5*gas_in.density

#temperature range to simulate
Trange = [T_in]
#Trange = np.linspace(273.15+250,273.15+450,10)

#Catalytic area per volume [m**-1]
cat_area = 8.5e04

#Initialize reactors                                          
r0 = pfr.PlugFlowReactor(gas0,
                         energy = 0,
                         z_out = 0.01,
                         diam =  0.012)

r = pfr.PlugFlowReactor(gas,
                        energy = 0,
                        z_out = 0.01,
                        diam =  0.012)

#Initialize the reacting wall
rs0 = pfr.ReactingSurface(r0,surf0,
                          cat_area_pvol = cat_area)

rs = pfr.ReactingSurface(r,surf,
                         cat_area_pvol = cat_area)


#Create a ReactorSolver object
sim0 = pfr.ReactorSolver(r0,
                         tcovs = 1e03,
                         atol = 1e-14,
                         rtol = 1e-07,
                         grid = 100,
                         max_steps = 5000,
                         solver_type = 'dae')

sim = pfr.ReactorSolver(r,
                        tcovs = 1e03,
                        atol = 1e-14,
                        rtol = 1e-07,
                        grid = 100,
                        max_steps = 5000,
                        solver_type = 'dae')

#Ethanol index
et_idx = r.gas.species_index('CH3CH2OH')

#Simulate over temperature range
sim0.solve_trange(Trange,(P_in,X_in),mdot_in=mdot0)
sim.solve_trange(Trange,(P_in,X_in),mdot_in=mdot0)

#Get output data
output0 = sim0.output_all
output = sim.output_all

#Axial coordinates [mm]
zcoord = sim0.z*1e03

#Get reactor variables from list
X0 = []
covs0 = []
for o in output0:
    X0.append(o['X'])
    covs0.append(o['coverages'])
    
X = []
covs = []
for o in output:
    X.append(o['X'])
    covs.append(o['coverages'])

X0 = np.array(X0)
covs0 = np.array(covs0)
X = np.array(X)
covs = np.array(covs)

#Ethanol conversion [%]
et_conv0 = (1 - X0[:,-1,et_idx]/X0[:,0,et_idx])*1e02
et_conv = (1 - X[:,-1,et_idx]/X[:,0,et_idx])*1e02

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
    if len(Trange) != 1:

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
        ax.plot(exp_data[:,0], exp_data[:,1], 'ks',
                markersize=10, label='Exp. data')
        ax.plot(Tplot, et_conv0, '-b', label='Full')
        ax.plot(Tplot, et_conv, ':b', label='Reduced')
        ax.set_xlabel('$T$ [$^\circ$C]')
        ax.set_ylabel('Conversion [%]')
        ax.legend(loc='lower right',fontsize=18)
        
        ax1 = fig.add_subplot(122) 
        covs0_o = covs0[:,-1,:]
        covs_o = covs[:,-1,:]
        
        ax1.semilogy(Tplot, covs0_o[:, surf0.species_index('PT(S)')], '-k',  label='PT(S)')
        ax1.semilogy(Tplot, covs0_o[:, surf0.species_index('H(S)')], '-g',  label='H(S)')
        ax1.semilogy(Tplot, covs0_o[:, surf0.species_index('CH(S)')], '-b',  label='CH(S)')
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('PT(S)')], ':k',  label='PT(S)')
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('H(S)')], ':g',  label='H(S)')
        ax1.semilogy(Tplot, covs_o[:, surf.species_index('CH(S)')], ':b',  label='CH(S)')
        ax1.axis((Tplot[0],Tplot[-1],1e-04,1))
        ax1.set_xlabel('$T$ [$^\circ$C]')
        ax1.set_ylabel('$\Phi_i$ [-]')
        ax1.legend(loc='lower right')
    plt.show()