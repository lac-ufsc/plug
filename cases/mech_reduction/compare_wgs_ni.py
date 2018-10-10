import numpy as np
import cantera as ct
import plug as pfr
import os
import time
start = time.time()  

#### Input mechanism data ####:   
input_file0= 'wgs_ni_nobulk.cti'     
input_file = 'wgs_ni_redux.cti'    
surf_name = 'Ni_surface'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'../..','data')

#### Coverage dependency matrix file ####: 
cov_file = os.path.join(filepath,'cov_matrix/covmatrix_wgs_ni.inp')

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
X_in = 'H2O:0.457, CO:0.114, H2:0.229, N2:0.2'          
T_in = 750.0
P_in = 101325.0

#Inlet gas solution object
gas_in = ct.Solution(input_file) 
gas_in.TPX = 273.15, 1e05, X_in

#Inlet mass flow rate [kg/s] at 3 SLPM
mdot0 = 3*1.66667e-5*gas_in.density

#Temperature range to simulate
Trange = [T_in]
Trange = list(np.linspace(273.15+300,273.15+700,30))

#Catalytic area per volume [m**-1]
cat_area =  1.75e04

#Initialize reactors                                          
r0 = pfr.PlugFlowReactor(gas0,
                         energy = 0,
                         z_out = 0.01,
                         diam =  0.01520)

r = pfr.PlugFlowReactor(gas,
                        energy = 0,
                        z_out = 0.01,
                        diam =  0.01520)

#Initialize the reacting wall
rs0 = pfr.ReactingSurface(r0,surf0,
                          cat_area_pvol = cat_area)

rs = pfr.ReactingSurface(r,surf,
                         cat_area_pvol = cat_area)


#Create a ReactorSolver object
sim0 = pfr.ReactorSolver(r0,
                         tcovs = 1e03,
                         atol = 1e-12,
                         rtol = 1e-07,
                         grid = 100,
                         max_steps = 5000,
                         solver_type = 'dae')

sim = pfr.ReactorSolver(r,
                        tcovs = 1e03,
                        atol = 1e-12,
                        rtol = 1e-07,
                        grid = 100,
                        max_steps = 5000,
                        solver_type = 'dae')

#Ethanol index
co_idx = r.gas.species_index('CO')

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
X = []
covs = []

for i in range(len(output0)):
    X0.append(output0[i]['X'])
    covs0.append(output0[i]['coverages'])
    X.append(output[i]['X'])
    covs.append(output[i]['coverages'])

X0 = np.array(X0)
covs0 = np.array(covs0)
X = np.array(X)
covs = np.array(covs)

#Ethanol conversion [%]
co_conv0 = (1 - X0[:,-1,co_idx]/X0[:,0,co_idx])*1e02
co_conv = (1 - X[:,-1,co_idx]/X[:,0,co_idx])*1e02

#Simulate the equilibrium conversion
gas_eq = ct.Solution(input_file)
co_conv_eq = []
for temp in Trange:
    #Set gas to current thermo state
    gas_eq.TPX = temp,P_in,X_in 
    #Solve the equilibrium
    gas_eq.equilibrate('TP',solver='auto')
    X_eq = gas_eq.X
    
    #CO conversion  [%]   
    co_conv_eq.append(( 1 - X_eq[co_idx]/X[0,0,co_idx] )*1e02)
    
#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

#### Save results into file? ####
savefile = 0
savepath = '/home/tpcarvalho/carva/python_data/wgs_ni/'
savename = 'wgs_ni_redux.pickle'

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
        #Squeeze arrays
        X = np.squeeze(X)
        covs = np.squeeze(covs)
        
        #Temperature along the reactor
        ax = fig.add_subplot(121)  
        ax.plot(zcoord, X[:, gas.species_index('H2')], '-g', label='H2')
        ax.plot(zcoord, X[:, gas.species_index('H2O')], '-r',  label='H2O')
        ax.plot(zcoord, X[:, gas.species_index('CO2')], '-b',  label='CO2')
        ax.plot(zcoord, X[:, gas.species_index('CO')], '-c',  label='CO')  
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
        #Import paper experimental data from CSV file
        import csv
        exp_data=[]
        filename = os.path.join(filepath,'exp_data/wheeler_co_conv_ni.csv')
        with open(filename, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                exp_data.append(np.array(row,dtype=float))
        f.close()
        #Convert list to Numpy array  
        exp_data = np.array(exp_data)
        
        #CO conversion variation with temperature
        Tout = np.array(Trange) - 273.15
        ax = fig.add_subplot(121)    
        ax.plot(Tout, co_conv0, '-b', label='Full')
        ax.plot(Tout, co_conv, ':b', label='Reduced')
        ax.plot(Tout, co_conv_eq, ':k', label='Equilibrium')
        
        ax.plot(exp_data[:,0], exp_data[:,1], 'ks',
                markersize=10, label='Exp. data')
        ax.axis((300,700,0.0,100.0))
        ax.set_xlabel('$T$ [$^\circ$C]')
        ax.set_ylabel('CO conversion [%]')
        ax.legend(loc='best',fontsize=20)
        
        #Coverages at the reactor outlet
        cov_out0 = covs0[:,-1,:]
        cov_out = covs[:,-1,:]
        
        ax1 = fig.add_subplot(122)
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('NI(S)')], '-k', label='NI(S)')
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('H(S)')], '-g',  label='H(S)')
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('H2O(S)')], '-b',  label='H2O(S)')
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('CO(S)')], '-r', label='CO(S)')
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('O(S)')], '-m',  label='O(S)')
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('OH(S)')], '-c',  label='OH(S)')
        
        ax1.semilogy(Tout, cov_out[:, surf.species_index('NI(S)')], ':k', label='NI(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('H(S)')], ':g',  label='H(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('H2O(S)')], ':b',  label='H2O(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('CO(S)')], ':r', label='CO(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('O(S)')], ':m',  label='O(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('OH(S)')], ':c',  label='OH(S)')
        ax1.axis((300,700,1e-06,1))
        ax1.set_xlabel('$T$ [$^\circ$C]')
        ax1.set_ylabel('$\Theta_k$ [-]')
        ax1.legend(loc='best')
    plt.show()