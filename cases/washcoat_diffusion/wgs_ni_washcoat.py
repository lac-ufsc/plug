import numpy as np
import cantera as ct
import plug as pfr
import os
import time
start = time.time()  

'''############################################################################
 Example that tests the washcoat models for internal mass transfer resistence .
 Two models are tested: the effectiveness factor approach and the detailed 
 washcoat reaction-diffusion model. Washcoat and monolith parameters are 
 arbitrary.
############################################################################'''

#### Input mechanism data ####:   
input_file= 'wgs_ni_nobulk.cti'       
surf_name = 'Ni_surface'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'../..','data')

#### Coverage dependency matrix file ####: 
cov_file = os.path.join(filepath,'cov_matrix/covmatrix_wgs_ni.inp')

#Load phases solution objects
gas = ct.Solution(input_file)
surfCT = ct.Interface(input_file,surf_name,[gas])

gas_wc = ct.Solution(input_file)
surfCT_wc = ct.Interface(input_file,surf_name,[gas_wc])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT, 
                    cov_file = cov_file)

surf_wc = pfr.SurfacePhase(gas_wc,surfCT_wc, 
                       cov_file = cov_file)

#Reactor geometric parameters
diam = 0.017                   #Diameter [m]
length = 0.01                  #Length [m]
vol = np.pi*diam**2/4*length   #Volume [m**3]

#Foam monolith properties
porosity = 0.8         #Foam porosity [-]
dp_foam = 1e-3       #Foam mean pore size [m]
ds_foam = dp_foam*np.sqrt(4/(3*np.pi)*(1-porosity))  #Foam mean strut size [m]

#Foam specific surface [m**-1]
sv = 4/ds_foam*(1-porosity)

#Create instance of washcoat area object
wc = pfr.WashcoatArea(thickness = 3e-05,
                      pore_diam = 1e-08,
                      epsilon = 0.35,
                      tortuosity = 5)

#Compute washcoat surface area assuming packing of uniform spheres [m**2]
Aw = wc.washcoat_area_spheres(sv*vol)

#Compute metallic area [m**2]
Aws = wc.compute_metallic_area(support_density = 3950,
                               metal_fraction = 0.005,
                               mean_cristal_size = 20,
                               cat_mw = surf.molecular_weights[surf.species_index('NI(S)')],
                               site_density = surf.site_density)

#Catalytic area per volume [m**-1]
cat_area_pvol = Aws/(vol*porosity)

#### Current thermo state ####:
X_in = 'H2O:0.457, CO:0.114, H2:0.229, N2:0.2'          
T_in = 800.0
P_in = 101325.0

#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Load gas phase at inlet conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, 1e05, X_in

#Inlet mass flow rate [kg/s] at 3 SLPM
mdot0 = 3*1.66667e-5*gas_in.density

#Temperature range to simulate
Trange = [T_in]
#Trange = list(np.linspace(273.15+300,273.15+700,30))

#Initialize reactor                                          
r0 = pfr.PlugFlowReactor(gas,
                         z_out = length,
                         diam =  diam,
                         sp_area = sv,
                         support_type = 'foam',
                         porosity = porosity,
                         dc = ds_foam,
                         ext_mt = 0)

r = pfr.PlugFlowReactor(gas,
                        z_out = length,
                        diam =  diam,
                        sp_area = sv,
                        support_type = 'foam',
                        porosity = porosity,
                        dc = ds_foam,
                        ext_mt = 0)

#Initialize reacting wall
rs0 = pfr.ReactingSurface(r0,surf,             
                          cat_area_pvol = cat_area_pvol,
                          int_mt = 1,
                          lim_species = 'CO',
                          thickness = wc.thickness,
                          pore_diam = wc.pore_diam,
                          epsilon = wc.epsilon,
                          tortuosity = wc.tortuosity)

rs = pfr.ReactingSurface(r,surf,             
                         cat_area_pvol = cat_area_pvol,
                         int_mt = 2,
                         thickness = wc.thickness,
                         pore_diam = wc.pore_diam,
                         epsilon = wc.epsilon,
                         tortuosity = wc.tortuosity,
                         gas_wc = gas_wc,
                         surf_wc = surf_wc,
                         ngrid = 20,
                         stch = 1.1)

#Create a ReactorSolver object
sim0 = pfr.ReactorSolver(r0,
                         tcovs = 1e02,
                         atol = 1e-14,
                         rtol = 1e-08,
                         grid = 200,
                         solver_type = 'dae')

sim = pfr.ReactorSolver(r,
                        tcovs = 1e02,
                        atol = 1e-10,
                        rtol = 1e-05,
                        grid = 200,
                        solver_type = 'dae')

#CO index
co_idx = gas.species_index('CO')

#Create lists to store results at each temperature
output0 = []
output = []
Xeq = []

#Loop over temperature range
for i,temp in enumerate(Trange):   

    #Inlet thermo state for r0
    r0.TPX = temp, P_in, X_in 
    
    #Set inlet flow velocity [m/s]
    r0.mdot = mdot0

    #Solve the PFR
    sim0.solve()

    #Append results to list
    output0.append(sim0.output)

    #Inlet thermo state 
    r.TPX = temp, P_in, X_in
    
    #Set inlet flow velocity [m/s]
    r.mdot = mdot0

    #Solve the PFR
    sim.solve()  
    
    #Append results to list
    output.append(sim.output)
    
    #Compute equilibrium
    gas_eq.TPX = temp, P_in, X_in
    gas_eq.equilibrate('TP')
    Xeq.append(gas_eq.X)

#Axial coordinates [mm]       
zcoord = sim0.z*1e03

#Get reactor variables from list
X0 = []
covs0 = []
rtime0 = []
X = []
Xwc = []
covs = []

for i in range(len(output0)):
    X0.append(output0[i]['X'])
    covs0.append(output0[i]['coverages'])
    rtime0.append(output0[i]['rtime'])
    X.append(output[i]['X'])
    Xwc.append(output[i]['Xwc'])
    covs.append(output[i]['coverages'])

X0 = np.array(X0)
covs0 = np.array(covs0)
rtime0 = np.array(rtime0)
X = np.array(X)
Xwc = np.array(Xwc)
covs = np.array(covs)
Xeq = np.array(Xeq)

#Ethanol conversion [%]
co_conv0 = (1 - X0[:,-1,co_idx]/X0[:,0,co_idx])*1e02
co_conv = (1 - X[:,-1,co_idx]/X[:,0,co_idx])*1e02
co_conv_eq = (1 - Xeq[:,co_idx]/X0[:,0,co_idx])*1e02

#Final residence time [ms]
rtime0_f = rtime0[:,-1]*1e03

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
        #Squeeze arrays
        X0 = np.squeeze(X0)
        X = np.squeeze(X)
        covs0 = np.squeeze(covs0)
        covs = np.squeeze(covs)
        
        #Temperature along the reactor
        ax = fig.add_subplot(121)  
        ax.plot(zcoord, X0[:, gas.species_index('H2')], '-g', label='H2 - Eff. factor')
        ax.plot(zcoord, X0[:, gas.species_index('H2O')], '-r',  label='H2O - Eff. factor')
        ax.plot(zcoord, X0[:, gas.species_index('CO2')], '-b',  label='CO2 - Eff. factor')
        ax.plot(zcoord, X0[:, gas.species_index('CO')], '-c',  label='CO - Eff. factor')  
        
        ax.plot(zcoord, X[:, gas.species_index('H2')], ':g', label='H2 - RD model')
        ax.plot(zcoord, X[:, gas.species_index('H2O')], ':r',  label='H2O - RD model')
        ax.plot(zcoord, X[:, gas.species_index('CO2')], ':b',  label='CO2 - RD model')
        ax.plot(zcoord, X[:, gas.species_index('CO')], ':c',  label='CO - RD model')  
        ax.set_xlabel('z [mm]')
        ax.set_ylabel('$X_k$ [-]')
        ax.legend(loc='best',fontsize=14)
        
        ax1 = fig.add_subplot(122)  
        ax1.semilogy(zcoord, covs0[:, surf.species_index('NI(S)')], '-k', label='NI(S) - Eff. factor')
        ax1.semilogy(zcoord, covs0[:, surf.species_index('CO(S)')], '-r',  label='CO(S) - Eff. factor')
        ax1.semilogy(zcoord, covs0[:, surf.species_index('H(S)')], '-g',  label='H(S) - Eff. factor')

        ax1.semilogy(zcoord, covs[:, surf.species_index('NI(S)')], ':k', label='NI(S) - RD model')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CO(S)')], ':r',  label='CO(S) - RD model')
        ax1.semilogy(zcoord, covs[:, surf.species_index('H(S)')], ':g',  label='H(S) - RD model')
        ax1.axis((0.0,zcoord[-1],1e-4,1))
        ax1.set_xlabel('z [mm]')
        ax1.set_ylabel('$\Phi_i$ [-]')
        ax1.legend(loc='best',fontsize=14)
           
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
        ax.plot(Tout, co_conv0, '-b', label='Eff. factor')
        ax.plot(Tout, co_conv, ':b', label='RD model')
        ax.plot(Tout, co_conv_eq, ':k', label='Equilibrium')
        
        ax.plot(exp_data[:,0], exp_data[:,1], 'ks',
                markersize=8, label='Exp. data')
        ax.axis((300,700,0.0,100.0))
        ax.set_xlabel('$T$ [$^\circ$C]')
        ax.set_ylabel('CO conversion [%]')
        ax.legend(loc='best',fontsize=20)
        
        #Coverages at the reactor outlet
        cov_out0 = covs0[:,-1,:]
        cov_out = covs[:,-1,:]
        
        ax1 = fig.add_subplot(122)
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('NI(S)')], '-k', label='NI(S) - Eff. factor')
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('H(S)')], '-g',  label='H(S) - Eff. factor')
        ax1.semilogy(Tout, cov_out0[:, surf.species_index('CO(S)')], '-r', label='CO(S) - Eff. factor')

        ax1.semilogy(Tout, cov_out[:, surf.species_index('NI(S)')], ':k', label='NI(S) - RD model')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('H(S)')], ':g',  label='H(S) - RD model')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('CO(S)')], ':r', label='CO(S) - RD model')
        
        ax1.axis((300,700,1e-06,1))
        ax1.set_xlabel('$T$ [$^\circ$C]')
        ax1.set_ylabel('$\Theta_k$ [-]')
        ax1.legend(loc='best')
    plt.show()