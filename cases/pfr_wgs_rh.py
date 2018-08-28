import cantera as ct
import numpy as np
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:      
input_file = 'wgs_rh_gas.cti' 
surf_name = 'Rh_surface'

#Load phases solution objects
gas = ct.Solution(input_file)
surf = ct.Interface(input_file,surf_name,[gas])

#### Current thermo state ####:  
T_in = 600                     #Inlet temperature [K]
P_in = 101325.0                #Inlet pressure [Pa]

#Set the temperature range [K]
Trange = [T_in] 
Trange = list(np.linspace(273.15+320,273.15+900,30))

#Select wich case to run
case = 2

#Data filepath
filepath = '/home/tpcarvalho/carva/python_data/wgs_rh/'
   
#Inlet molar composition and filename of experimental data
if case==1:
    ### Case09 - WGS ###
    X_in = {'H2O':0.113, 'CO':0.112}         
    filename = filepath+'case9.csv' 
    #Washcoat parameters:
    wc_spc = 'CO'
    
elif case==2:
    ### Case10 - WGS ###            
    X_in = {'H2O':0.102, 'CO':0.1008, 'CO2':0.02}   
    filename = filepath+'case10.csv' 
    #Washcoat parameters:
    wc_spc = 'CO'
    
elif case==3:
    ### Case11 - RWGS ###    
    X_in = {'H2':0.1040, 'CO2':0.1088}
    filename = filepath+'case11.csv' 
    #Washcoat parameters:
    wc_spc = 'CO2'
    
elif case==4:
    ### Case12 - RWGS ###   
    X_in = {'H2':0.1003, 'CO2':0.0952, 'CO':0.0204}
    filename = filepath+'case12.csv' 
    #Washcoat parameters:
    wc_spc = 'CO2'

#Balance inlet composition dict with N2
X_in['N2'] = 1.0-np.sum(np.fromiter(X_in.values(),dtype=float))

#Compute inlet mass flow rate and set to standard conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, P_in, X_in

#Tube cross-sectional area [m**2]
carea = np.pi*0.0195**2/4

#Total inlet mass flow rate [kg/s] at 5 SLPM
mdot_tot = 5*1.66667e-5*gas_in.density

#Monolith properties:
dcell = 1e-03         #Monolith cell size [m]
porosity = 0.7        #Monolith porosity [-]
sv = 4*porosity/dcell #Monolith specific surface [m**-1]

#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        energy = 1,
                        momentum = 0,
                        z_out = 0.01,
                        diam = 0.0195,
                        support_type = 'honeycomb',
                        sp_area = sv,
                        porosity = 0.7,
                        dc = 1e-03,
                        ext_mt = 0)

#Initialize the reacting wall
rs = pfr.ReactingSurface(r,surf,
                         cat_area_pvol = sv*8.5,
                         int_mt = 1,
                         lim_species = wc_spc,
                         thickness = 4e-05,
                         pore_diam = 2.5e-08,
                         epsilon = 0.3,
                         tortuosity = 5)

#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        tcovs = 1e5,
                        atol = 1e-09,
                        rtol = 1e-07,
                        grid = 300,
                        solver_type = 'dae')

#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Create lists to store results at each temperature
output = []
neff = []
X_eq = []

#Loop over temperature range
for temp in Trange:   

    #Current inlet temperature
    r.TPX = temp, P_in, X_in 

    #Set inlet mass flow rate [kg/s]
    r.mdot = mdot_tot

    #Change solver relative tolerance depending on temperature
    if temp < 650.0:
        sim.atol = 1e-08
        sim.rtol = 1e-05
    else:
        sim.atol = 1e-14
        sim.rtol = 1e-08

    #Solve PFR
    sim.solve()

    #Append results to list
    output.append(sim.output)

    #Append effectiveness factor
    if r.rs.flag_int_mt == 1:
        neff.append(rs.wc.neff)
    
    #Compute equilibrium
    gas_eq.TPX = temp, P_in, X_in 
    gas_eq.equilibrate('HP',solver='auto')
    X_eq.append(gas_eq.X)
        
#Axial coordinates [mm]
zcoord = sim.z*1e03

#Get reactor variables from list
X = []
Xs = []
covs = []

for o in output:
    X.append(o['X'])
    Xs.append(o['Xs'])
    covs.append(o['coverages'])

X = np.array(X)
Xs= np.array(Xs)
X_eq = np.array(X_eq)
covs = np.array(covs)
neff = np.array(neff)

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

#%%
vis = 1

if vis==1:
    #Plot some results
    import matplotlib.pyplot as plt
    #Species to plot depending on type of reaction: WGS or RWGS
    if case==1 or case==2:
        spc1 = 'H2O'
        spc2 = 'CO2'
    elif case==3 or case==4:
        spc1 = 'H2'
        spc2 = 'CO'
        
    #Set figure size and position   
    plt.rcParams.update({'font.size': 20})  
    fig = plt.figure()
    if len(Trange) == 1:
        fig.set_size_inches(18, 8)
        fig.canvas.manager.window.move(100,100)  
        #Remove singleton dimensions
        X = np.squeeze(X)
        Xs = np.squeeze(Xs)
        covs = np.squeeze(covs)
        
        #Molar fractions along reactor 
        ax = fig.add_subplot(121)  
        ax.plot(zcoord, X[:, gas.species_index(spc1)], '-g', label=spc1)
        ax.plot(zcoord, X[:, gas.species_index(spc2)], '-r',  label=spc2)
        ax.plot(zcoord, Xs[:, gas.species_index(spc1)], ':g', label=spc1+' surf')
        ax.plot(zcoord, Xs[:, gas.species_index(spc2)], ':r',  label=spc2+' surf')
    
        ax.set_xlabel('$z$ [mm]')
        ax.set_ylabel('$X_k$ [-]')
        ax.legend(loc='best',fontsize=18)
        
        #Surface coverages along reactor 
        ax1 = fig.add_subplot(122)
        ax1.semilogy(zcoord, covs[:, surf.species_index('Rh(s)')], '-k',  label='Rh(s)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CO(s)')], '-g',  label='CO(s)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('H(s)')], '-r',  label='H(s)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('H2O(s)')], '-b',  label='H2O(s)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('C(s)')], '-m',  label='C(s)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('OH(s)')], '--m',  label='OH(s)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('O(s)')], '--k',  label='O(s)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('CO2(s)')], '--b',  label='CO2(s)')
        ax1.semilogy(zcoord, covs[:, surf.species_index('COOH(s)')], '--r',  label='COOH(s)')
        ax1.axis((0.0,zcoord[-1],1e-09,1))
        ax1.set_xlabel('$z$ [m]')
        ax1.set_ylabel('$\Theta_k$ [-]')
        ax1.legend(loc='center right',fontsize=16)
        
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
        exp_data=np.array(exp_data) 
        
        #Reaction temperature (inlet)
        Tplot = np.array(Trange) - 273.15
        
        #Molar at the reactor outlet 
        Xout = X[:,-1,:]

        ax = fig.add_subplot(111)
        ax.plot(Tplot, Xout[:, gas.species_index(spc1)], '-g',  label=spc1+' Model')
        ax.plot(Tplot, Xout[:, gas.species_index(spc2)], '-r',  label=spc2+' Model')      
        ax.plot(Tplot, X_eq[:,gas.species_index(spc2)], ':r',  label=spc2+' equil.')

        ax.plot(exp_data[:,2], exp_data[:,3], 'sg',  
                markersize=8, label='%s Exp.' %spc1)
        ax.plot(exp_data[:,0], exp_data[:,1], 'or',  
                markersize=8, label='%s Exp.' %spc2) 
        plt.xlim(Tplot[0],Tplot[-1])
        ax.set_xlabel('T [$^\circ$C]')
        ax.set_ylabel('$X_{k}$ [-]')
        ax.legend(loc='upper right',fontsize=18)
        
    plt.show()

