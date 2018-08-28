import cantera as ct
import numpy as np
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'wgs_nib.cti'        
surf_name = 'Ni_surface'
bulk_name = 'Ni_bulk'

#### Coverage dependency matrix file ####: 
cov_file = ('/home/tpcarvalho/carva/python_data/kinetic_mechanisms/'
            'input_files/cov_matrix/covmatrix_wgs_ni.inp')

#### Save results into file? ####
savefile = 0
savepath = '/home/tpcarvalho/carva/python_data/wgs_ni/'
savename = 'wgs_nib_redux.pickle'

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

#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Load gas phase at inlet conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, 1e05, X_in

#Inlet mass flow rate [kg/s] at 3 SLPM
mdot0 = 3*1.66667e-5*gas_in.density

#Temperature range to simulate
Trange = [T_in]
#Trange = list(np.linspace(273.15+300,273.15+700,50))
#u_in = np.geomspace(1,1e05,len(Trange))*1.0

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

#Create dict to store results at each temperature
temp = {'Xg': [], 'covs': [], 'co_conv': [], 'co_conv_eq': [], 
        'tau': [], 'qf': [], 'qb': []}

#CO index
co_idx = r.gas.species_index('CO')

#Loop over temperature range
for i in range(len(Trange)):   
    
    #Current inlet temperature
    r.TPX = Trange[i], P_in, X_in 
    
    #Set inlet flow velocity [m/s]
    r.mdot = mdot0
#    r.u0 = u_in[i]

    #Solve the PFR
    sim.solve()
    
    #Gas molar fractions and coverages along the reactor
    covs = sim.coverages
    Xg = sim.X
    
    #Compute equilibrium
    gas_eq.set_unnormalized_mole_fractions(Xg[0,:])
    gas_eq.TP = Trange[i], 101325.0 
    gas_eq.equilibrate('TP',solver='auto')
    Xg_eq = gas_eq.X
    
    #CO conversion    
    co_conv = ( (Xg[0,co_idx] - Xg[-1,co_idx])/Xg[0,co_idx] )*1e02
    co_conv_eq = ( (Xg[0,co_idx] - Xg_eq[co_idx])/Xg[0,co_idx] )*1e02

    #Append results to list
    temp['Xg'].append(Xg)
    temp['covs'].append(covs)        
    temp['co_conv'].append(co_conv)
    temp['co_conv_eq'].append(co_conv_eq)
    temp['tau'].append(sim.rtime[-1])
    temp['qf'].append(surf.forward_rates_of_progress)
    temp['qb'].append(surf.reverse_rates_of_progress)
    
#Axial coordinates [mm]
zcoord = sim.z*1e03

#Convert lists to array
Xg = np.squeeze(np.array(temp['Xg']))
covs = np.squeeze(np.array(temp['covs']))

#Molar at the reactor outlet 
co_conv = np.array(temp['co_conv'])
co_conv_eq = np.array(temp['co_conv_eq'])

#Residence time [s]
tau = np.array(temp['tau'])

#Append variables to dict
Tout = np.array(Trange) - 273.15   #Reactor exit temperature [C]
temp['zcoord'] = sim.z
temp['Tout'] = Tout

if len(Trange)>1:
    
    #Reactant/product stoichiometric coefficients
    st_r = surf.reactant_stoich_coeffs()
    st_p = surf.product_stoich_coeffs()
    
    #Get the stoichiometric coefficients for CO(S)
    st_r = st_r[surf.kinetics_species_index('CO(S)'),:]
    st_p = st_p[surf.kinetics_species_index('CO(S)'),:]
    
    #Rates of progress [kmol/m**2 s]
    qf = np.squeeze(np.array(temp['qf']))
    qb = np.squeeze(np.array(temp['qf']))
     
    #CO consumption rate
    qf_co = qf[:,st_r>0]
    qb_co = qb[:,st_p>0]
    
    qall = np.hstack((qf_co,qb_co))
    qall = (qall.T/np.sum(qall ,axis=1)).T*1e02
    
    qf_co = (qf_co.T/np.sum(qf_co,axis=1)).T*1e02
    qb_co = (qb_co.T/np.sum(qb_co,axis=1)).T*1e02
    
    #Append variables to dict
    temp['qf_co'] = qf_co
    temp['qb_co'] = qb_co   

#Store data (serialize)
if savefile != 0:
    import pickle
    #Save file using pickle
    with open((savepath+savename), 'wb') as handle:
        pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

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
        #Temperature along the reactor
        ax = fig.add_subplot(121)  
        ax.plot(zcoord, Xg[:, gas.species_index('H2')], '-g', label='H2')
        ax.plot(zcoord, Xg[:, gas.species_index('H2O')], '-r',  label='H2O')
        ax.plot(zcoord, Xg[:, gas.species_index('CO2')], '-b',  label='CO2')
        ax.plot(zcoord, Xg[:, gas.species_index('CO')], '-c',  label='CO')  
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
        plt.show()
           
    else:
        #Import paper experimental data from CSV file
        import csv
        exp_data=[]
        filename = (savepath+'wheeler_co_conv_ni.csv') 
        with open(filename, 'rt') as f:
            reader = csv.reader(f)
            for row in reader:
                exp_data.append(np.array(row,dtype=float))
        f.close()
        #Convert list to Numpy array  
        exp_data = np.array(exp_data)
        
        #CO conversion variation with temperature

        ax = fig.add_subplot(121)    
        ax.plot(Tout, co_conv, '-b', label='Numerical')
        ax.plot(Tout, co_conv_eq, ':k', label='Equilibrium')
        
        ax.plot(exp_data[:,0], exp_data[:,1], 'ks',
                markersize=10, label='Exp. data')
        ax.axis((300,700,0.0,100.0))
        ax.set_xlabel('$T$ [$^\circ$C]')
        ax.set_ylabel('CO conversion [%]')
        ax.legend(loc='best',fontsize=20)
        
        #Coverages at the reactor outlet
        cov_out = covs[:,-1,:]
        
        ax1 = fig.add_subplot(122)
        ax1.semilogy(Tout, cov_out[:, surf.species_index('NI(S)')], '-k', label='NI(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('H(S)')], '-g',  label='H(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('H2O(S)')], '-b',  label='H2O(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('CO(S)')], '-r', label='CO(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('O(S)')], '-m',  label='O(S)')
        ax1.semilogy(Tout, cov_out[:, surf.species_index('OH(S)')], '-c',  label='OH(S)')
        ax1.axis((300,700,1e-06,1))
        ax1.set_xlabel('$T$ [$^\circ$C]')
        ax1.set_ylabel('$\Theta_k$ [-]')
        ax1.legend(loc='best')
        plt.show()