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
input_file = 'wgs_rh_gas.cti' 
surf_name = 'Rh_surface'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'../..','data')
filename = os.path.join(filepath,'exp_data/wgs_rh_case10.csv')  

#Load phases solution objects
gas = ct.Solution(input_file)
surf = ct.Interface(input_file,surf_name,[gas])
gas_wc = ct.Solution(input_file)
surf_wc = ct.Interface(input_file,surf_name,[gas_wc])

#### Current thermo state ####:  
T_in = 850                     #Inlet temperature [K]
P_in = 101325.0                #Inlet pressure [Pa]
#X_in = {'H2O':0.113, 'CO':0.112}   
X_in = {'H2O':0.102, 'CO':0.1008, 'CO2':0.02}   

#Balance inlet composition dict with N2
X_in['N2'] = 1.0-np.sum(np.fromiter(X_in.values(),dtype=float))

#Set the temperature range [K]
Trange = [T_in] 
#Trange = list(np.linspace(273.15+330,273.15+900,20))

#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Compute inlet mass flow rate and set to standard conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, P_in, X_in

#Monolith properties:
dcell = 1e-03                 #Monolith cell size [m]
porosity = 0.7                #Monolith porosity [-]
sv = 4*porosity/dcell         #Monolith specific surface [m**-1]
length = 0.01                 #Length [m]
vol = np.pi*dcell*2/4*length  #Volume [m**3]

#Create instance of washcoat area object
wc = pfr.WashcoatArea(thickness = 4e-05,
                      pore_diam = 2e-08,
                      epsilon = 0.6,
                      tortuosity = 5)

#Compute washcoat surface area assuming packing of uniform spheres [m**2]
Aw = wc.washcoat_area_spheres(sv*vol)

#Compute metallic area [m**2]
Aws = wc.compute_metallic_area(support_density = 3950,
                               metal_fraction = 0.0025,
                               mean_cristal_size = 20,
                               cat_mw = surf.molecular_weights[surf.species_index('Rh(s)')],
                               site_density = surf.site_density)

#Catalytic area per volume [m**-1]
cat_area_pvol = Aws/(vol*porosity)

#Tube cross-sectional area [m**2]
carea = np.pi*0.0195**2/4

#Channel cross-sectional area [m**2]
carea_c = np.pi*dcell**2/4

#Total inlet mass flow rate [kg/s] at 5 SLPM
mdot_tot = 5*1.66667e-5*gas_in.density

#Single channel inlet mass flow rate [kg/s]
mdot0 = mdot_tot*carea_c/carea

#Initialize reactor                                          
r0 = pfr.PlugFlowReactor(gas,
                         energy = 1,
                         z_out = length,
                         diam = dcell,
                         support_type = 'honeycomb',
                         porosity = 1.0,             #Simulates a clear channel
                         dc = dcell,
                         ext_mt = 0)

r = pfr.PlugFlowReactor(gas,
                        energy = 1,
                        z_out = length,
                        diam = dcell,
                        support_type = 'honeycomb',
                        porosity = 1.0,             #Simulates a clear channel
                        dc = dcell,
                        ext_mt = 0)

#Initialize the reacting wall
rs0 = pfr.ReactingSurface(r0,surf,
                          cat_area_pvol = cat_area_pvol,
                          int_mt = 1,
                          lim_species = 'CO',
                          thickness = wc.thickness,
                          pore_diam = wc.pore_diam,
                          epsilon = wc.epsilon,
                          tortuosity = wc.tortuosity)

#Initialize the reacting wall
rs = pfr.ReactingSurface(r,surf,
                         cat_area_pvol = cat_area_pvol,
                         int_mt = 2,
                         thickness = wc.thickness,
                         pore_diam = wc.pore_diam,
                         epsilon = wc.epsilon,
                         tortuosity = wc.tortuosity,
                         gas_wc = gas_wc,
                         surf_wc = surf_wc,
                         ngrid = 15,
                         stch = 1.1)

#Create a ReactorSolver object
sim0 = pfr.ReactorSolver(r0,
                         tcovs = 1e05,
                         atol = 1e-10,
                         rtol = 1e-07,
                         grid = 200,
                         solver_type = 'dae')

sim = pfr.ReactorSolver(r,
                        tcovs = 1e05,
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
    gas_eq.equilibrate('HP')
    Xeq.append(gas_eq.X)
    
#Axial coordinates [mm]
zcoord = sim0.z*1e03

#Get reactor variables from list
X0 = []
covs0 = []
X = []
Xwc = []
covs = []

for i in range(len(output0)):
    X0.append(output0[i]['X'])
    covs0.append(output0[i]['coverages'])
    X.append(output[i]['X'])
    Xwc.append(output[i]['Xwc'])
    covs.append(output[i]['coverages'])

X0 = np.array(X0)
covs0 = np.array(covs0)    
X = np.array(X)
Xwc = np.array(Xwc)
covs = np.array(covs)
Xeq = np.array(Xeq)

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

#%%
plot = 1

if plot != 0:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.set_size_inches(17, 9)
    fig.canvas.manager.window.move(100,100)  
    plt.rcParams.update({'font.size': 14})  
    
    if len(Trange) == 1:
        #Squeeze arrays
        X0 = np.squeeze(X0)
        X = np.squeeze(X)
        covs0 = np.squeeze(covs0)
        covs = np.squeeze(covs)
        
        #Temperature along the reactor
        ax = fig.add_subplot(121)  
        ax.plot(zcoord, X0[:, gas.species_index('H2O')], '-g', label='H2 - Eff. factor')
        ax.plot(zcoord, X0[:, gas.species_index('CO2')], '-r',  label='H2O - Eff. factor')
        
        ax.plot(zcoord, X[:, gas.species_index('H2O')], ':g',  label='H2O - RD model')
        ax.plot(zcoord, X[:, gas.species_index('CO2')], ':r',  label='CO2 - RD model')
        ax.set_xlabel('z [mm]')
        ax.set_ylabel('$X_k$ [-]')
        ax.legend(loc='best',fontsize=14)

        #Surface coverages along reactor 
        ax1 = fig.add_subplot(122)
        ax1.semilogy(zcoord, covs0[:, surf.species_index('Rh(s)')], '-k',  label='Rh(s) - Eff. factor')
        ax1.semilogy(zcoord, covs0[:, surf.species_index('CO(s)')], '-g',  label='CO(s) - Eff. factor')
        ax1.semilogy(zcoord, covs0[:, surf.species_index('H(s)')], '-r',  label='H(s) - Eff. factor')
        ax1.semilogy(zcoord, covs0[:, surf.species_index('C(s)')], '-m',  label='C(s) - Eff. factor')

        ax1.semilogy(zcoord, covs[:, surf.species_index('Rh(s)')], ':k',  label='Rh(s) - RD model')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CO(s)')], ':g',  label='CO(s) - RD model')
        ax1.semilogy(zcoord, covs[:, surf.species_index('H(s)')], ':r',  label='H(s) - RD model')
        ax1.semilogy(zcoord, covs[:, surf.species_index('C(s)')], ':m',  label='C(s) - RD model')
        ax1.axis((0.0,zcoord[-1],1e-04,1))
        ax1.set_xlabel('$z$ [mm]')
        ax1.set_ylabel('$\Theta_k$ [-]')
        ax1.legend(loc='center right',fontsize=14)
        
    else:
        fig.set_size_inches(12, 8)
        fig.canvas.manager.window.move(400,100) 
        #Import paper experimental data from CSV file
        import csv
        exp_data=[]
        with open(filename, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                exp_data.append(np.array(row,dtype=float))
        f.close()
        
        #Convert list to Numpy array  
        exp_data = np.array(exp_data) 
        
        #Reaction temperature (inlet)
        Tplot = np.array(Trange) - 273.15
        
        #Molar at the reactor outlet 
        X0out = X0[:,-1,:]
        Xout = X[:,-1,:]

        ax = fig.add_subplot(111)
        ax.plot(Tplot, X0out[:, gas.species_index('H2O')], '-g',  label='H2O - Eff. factor')
        ax.plot(Tplot, X0out[:, gas.species_index('CO2')], '-r',  label='CO2 - Eff. factor')      

        ax.plot(Tplot, Xout[:, gas.species_index('H2O')], ':g',  label='H2O - RD model')
        ax.plot(Tplot, Xout[:, gas.species_index('CO2')], ':r',  label='CO2 - RD model')     
        
        ax.plot(Tplot, Xeq[:,gas.species_index('CO2')], '--r',  label= 'CO2 equil.')
        
        ax.plot(exp_data[:,2], exp_data[:,3], 'sg',  
                markersize=8, label='%s Exp.' %'H2O')
        ax.plot(exp_data[:,0], exp_data[:,1], 'or',  
                markersize=8, label='%s Exp.' %'CO2') 
        plt.xlim(Tplot[0],Tplot[-1])
        ax.set_xlabel('T [$^\circ$C]')
        ax.set_ylabel('$X_{k}$ [-]')
        ax.legend(loc='upper right',fontsize=14)
        
    plt.show()