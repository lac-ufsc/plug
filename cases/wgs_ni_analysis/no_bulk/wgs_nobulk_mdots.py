import numpy as np
import cantera as ct
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'wgs_ni_redux_nobulk.cti'        
surf_name = 'Ni_surface'

#### Coverage dependency matrix file ####: 
cov_file = ('/home/tpcarvalho/carva/python_data/kinetic_mechanisms/'
            'input_files/cov_matrix/covmatrix_wgs_ni.inp')

#### Save results into file? ####
savefile = 0
savepath = '/home/tpcarvalho/carva/python_data/wgs_ni/'
savename = 'wgs_nib_redux_mdots.pickle'

#Load phases solution objects
gas = ct.Solution(input_file)
surfCT = ct.Interface(input_file,surf_name,[gas])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT,  
                        cov_file = cov_file)

#### Current thermo state ####:
X_in = 'H2O:0.457, CO:0.114, H2:0.229, N2:0.2'          #WGS Ni
T_in = 750.0
P_in = 1e05

#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Load gas phase at inlet conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, P_in, X_in

#Inlet mass flow rate [kg/s] at 0.3-30 SLPM
vdot0 = np.array([0.03,0.3,3,30,300,3000])
mdot0 = vdot0*1.66667e-5*gas_in.density

#Temperature range to simulate
#Trange = [T_in]
Trange = list(np.linspace(273.15+300,273.15+700,60))
#u_in = np.geomspace(1,1e05,len(Trange))*1.0

#Initialize reactor                                          
r = pfr.PlugFlowReactor(gas,
                        z_out = 0.01,
                        diam =  0.01520)

#Initialize reacting wall
rs = pfr.ReactingSurface(r,surf,             
                         cat_area_pvol = 1.75e04)

#Create a ReactorSolver object
sim = pfr.ReactorSolver(r,
                        tcovs = 1e04,
                        atol = 1e-12,
                        rtol = 1e-06,
                        grid = 250,
                        max_steps = 5000,
                        solver_type = 'dae')

#Create dict to store results at each temperature
out = {'Xg': [], 'covs': [], 'co_conv': [], 'co_conv_eq': [], 
       'tau': []}

#CO index
co_idx = r.gas.species_index('CO')

for j in range(mdot0.size):
    
    #Create empty lists 
    Xg = []
    covs = []
    co_conv = []
    co_conv_eq = []
    tau = []
        
    #Loop over temperature range
    for i in range(len(Trange)):   
        
        #Current inlet temperature
        r.TPX = Trange[i], P_in, X_in 
        
        #Set inlet flow velocity [m/s]
        r.mdot = mdot0[j]

        #Solve the PFR
        sim.solve()
        
        #Gas molar fractions and coverages along the reactor
        covs.append(sim.coverages)
        Xg.append(sim.X)
        
        #Compute equilibrium
        gas_eq.TPX = Trange[i], P_in, X_in 
        gas_eq.equilibrate('TP',solver='auto')
        xg_eq = gas_eq.X
        
        #CO conversion    
        co_conv.append( ( (sim.X[0,co_idx] - sim.X[-1,co_idx])
                          /sim.X[0,co_idx] )*1e02 )
        co_conv_eq.append( ( (sim.X[0,co_idx] - xg_eq[co_idx])
                          /sim.X[0,co_idx] )*1e02 )
        
        #Residence time [s]
        tau.append(sim.rtime[-1])
    
    #Append data to dict
    out['Xg'].append(Xg)
    out['covs'].append(covs)
    out['co_conv'].append(co_conv)
    out['co_conv_eq'].append(co_conv_eq)
    out['tau'].append(tau)

#Axial coordinates [mm]
zcoord = sim.z*1e03

#Append variables to dict
out['Xg'] = np.array(out['Xg'])
out['covs'] = np.array(out['covs'])
out['co_conv'] = np.array(out['co_conv'])
out['co_conv_eq'] = np.array(out['co_conv_eq'])
out['tau'] = np.array(out['tau'])
out['zcoord'] = sim.z
out['Tout'] = np.array(Trange) - 273.15   #Reactor exit temperature [C]

#Store data (serialize)
if savefile != 0:
    import pickle
    #Save file using pickle
    with open((savepath+savename), 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

vis = 1

if vis == 1:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(11, 8)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams.update({'font.size': 24})  
            
    #CO conversion variation with temperature
    Tout = np.array(Trange) - 273.15
    co_conv_all = out['co_conv']
    
    ax = fig.add_subplot(111)   
    
    for k in range(mdot0.size):
        ax.plot(Tout, co_conv_all[k,:], '-k',)

    ax.plot(Tout, np.array(co_conv_eq), ':k', label='Equilibrium')

    ax.axis((300,800,0.0,100.0))
    ax.set_xlabel('$T$ [$^\circ$C]')
    ax.set_ylabel('CO conversion [%]')
    ax.legend(loc='best',fontsize=20,frameon=False)  

plt.show()