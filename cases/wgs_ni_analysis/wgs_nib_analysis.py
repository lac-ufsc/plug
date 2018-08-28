import numpy as np
import cantera as ct
import plug as pfr
import time
start = time.time()  

#### Input mechanism data ####:   
input_file = 'wgs_nib_redux.cti'        
surf_name = 'Ni_surface'
bulk_name = 'Ni_bulk'

#### Coverage dependency matrix file ####: 
cov_file = ('/home/tpcarvalho/carva/python_data/kinetic_mechanisms/'
            'input_files/cov_matrix/covmatrix_wgs_ni.inp')

#Export figure? 
exp_fig = 0

#Load phases solution objects
gas = ct.Solution(input_file)
bulk = ct.Solution(input_file,bulk_name)
surfCT = ct.Interface(input_file,surf_name,[gas,bulk])

#Load in-house surface phase object
surf = pfr.SurfacePhase(gas,surfCT, 
                        bulk = bulk, 
                        cov_file = cov_file)

#Export figure? 
exp_fig = 0

#Reactions object
R = surf.reactions()

#Get mechanism parameters
surf_names = surf.species_names    #Species names
stm_n = surf.net_stoich_coeffs()   #Stoichiometric coefficients
stm_r = surf.reactant_stoich_coeffs()  #Stoichiometric reactant coefficients
stm_p = surf.product_stoich_coeffs()   #Stoichiometric product coefficients

#Get stoichiometric coefficients for surface species (exclude vacancies)
stm_ns = stm_n[gas.n_species:,:]
stm_rs = stm_r[gas.n_species:,:]
stm_ps = stm_p[gas.n_species:,:]

#Create binary matrix for coefficients
stm_bin = stm_ns.copy()
stm_bin[np.nonzero(stm_ns)] = 1   

#Create dict with the site balances and coefficients
site_balances = dict().fromkeys(surf_names)
sb_coeffs = dict().fromkeys(surf_names)

#Write the site balance for each species
for key in surf_names:
    idx = surf.species_index(key)
    site_balances[key] = np.nonzero(stm_ns[idx,:])[0]
    #Site balance stoichiometric coefficients
    sb_coeffs[key] = stm_ns[idx,site_balances[key]]

#### Current thermo state ####:
#X_in = {'CO': 0.5, 'H2': 0.25, 'H2O': 0.05, 'N2': 0.2}   #CO:H2O 10:1
X_in = {'CO': 0.114, 'H2': 0.229, 'H2O': 0.457, 'N2': 0.2}
T_in = 750.0
P_in = 101325.0
    
#Set up equilibrium gas phase
gas_eq = ct.Solution(input_file)

#Load gas phase at inlet conditions
gas_in = ct.Solution(input_file)
gas_in.TPX = 273.15, 1e05, X_in

#Inlet mass flow rate [kg/s] at 3 SLPM
mdot0 = 3*1.66667e-5*gas_in.density

#Temperature range to simulate (~623 - 773 K)
n_temps = 3
#Trange = [T_in]
Trange = list(np.linspace(273.15+375,273.15+475,n_temps))
#u_in = np.geomspace(.1,100,n_temps)*2.0

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
                        grid = 100,
                        max_steps = 5000,
                        solver_type = 'dae')

#Create dict to store results at each temperature
temp = {'Xg': [], 'covs': [], 'CO_conv': [], 'CO_conv_eq': [], 'qf': [],
        'qr': [],'qf_mag': [], 'qr_mag': [], 'qn_mag': []}

#CO index
CO_idx = r.gas.species_index('CO')

#Rate cut-off 
rate_cut_off = 0.1

peq = []
delta_g = []

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
    
    #Net species production rate
    sdot = surf.net_production_rates
    
    #Delta Gibbs
    delta_g.append(surf.kinetics.delta_gibbs)
    
    #Forward and reverse rates of progress 
    qf = surf.kinetics.qf
    qr = surf.kinetics.qr
        
    #Multiply rates by stoichiometric matrix
    qf_all = qf*stm_bin
    qr_all = qr*stm_bin

    #Total forward and reverse rates (excluding ads-des steps)
    qf_tot = np.sum(qf_all,axis=1)
    qr_tot = np.sum(qr_all,axis=1)
#    qf_tot = np.sum(qf_all[:,4:],axis=1)
#    qr_tot = np.sum(qr_all[:,4:],axis=1)
    qn_tot = qf_tot + qr_tot
    
    #Rate for each species balances (both forward and reverse)
    qn_all = np.hstack((qf_all,qr_all))
       
    #Get the relative rates of progress magnitudes for each species
    qf_mag = np.divide(qf_all.T, qn_tot, 
                       out=np.zeros_like(qf_all.T), 
                       where=qn_tot!=0).T

    qr_mag = np.divide(qr_all.T, qn_tot, 
                       out=np.zeros_like(qr_all.T), 
                       where=qn_tot!=0).T
                           
    #Eliminate rates smaller than x% of balance for each species
    qf_mag[qf_mag<rate_cut_off] = 0.0
    qr_mag[qr_mag<rate_cut_off] = 0.0
    
    #Partial equilibrium
    peq.append(qf/(qf + qr))
    
    #Compute equilibrium
    gas_eq.set_unnormalized_mole_fractions(Xg[0,:])
    gas_eq.TP = Trange[i], 101325.0 
    gas_eq.equilibrate('TP',solver='auto')
    Xg_eq = gas_eq.X
    
    #CO conversion    
    CO_conv = ( (Xg[0,CO_idx] - Xg[-1,CO_idx])/Xg[0,CO_idx] )*1e02
    CO_conv_eq = ( (Xg[0,CO_idx] - Xg_eq[CO_idx])/Xg[0,CO_idx] )*1e02

    #Append results to list
    temp['Xg'].append(Xg)
    temp['covs'].append(covs)        
    temp['CO_conv'].append(CO_conv)
    temp['CO_conv_eq'].append(CO_conv_eq)
    temp['qf'].append(qf)
    temp['qr'].append(qr)
    temp['qf_mag'].append(qf_mag)
    temp['qr_mag'].append(qr_mag)
    
#Axial coordinates [mm]
zcoord = sim.z*1e03

#Convert lists to array
Xg = np.squeeze(np.array(temp['Xg']))
covs = np.squeeze(np.array(temp['covs']))
peq = np.array(peq).T
delta_g = np.array(delta_g).T

#CO conversion 
CO_conv = np.array(temp['CO_conv'])
CO_conv_eq = np.array(temp['CO_conv_eq'])

#Temperature range [C]
Tout = np.array(Trange) - 273.15

#Conver lists to arrays
qf_mag = np.array(temp['qf_mag'])
qr_mag = np.array(temp['qr_mag'])

#Get the relevant rates (>1% of max) for each surface species at all temps.
rf_mag = np.max(np.array(temp['qf_mag']),axis=0)
rr_mag = np.max(np.array(temp['qr_mag']),axis=0)

#Store relevant rates into dict for species 
rf_dict = dict().fromkeys(surf_names)
rr_dict = dict().fromkeys(surf_names)

for key in surf_names:
    idx = surf.species_index(key)
    #Store reaction index (start at 1)
    rf_dict[key] = np.nonzero(rf_mag[idx+1,:])[0]
    rr_dict[key] = np.nonzero(rr_mag[idx+1,:])[0]
               
#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 

vis = 1

if vis!=0:
    #Plot some results
    import matplotlib.pyplot as plt
    #Set figure size and position   
    fig = plt.figure()
    fig.set_size_inches(11, 8)
    fig.canvas.manager.window.move(200,100)  
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
    plt.rcParams.update({'font.size': 25})  

    if vis==1:

        #Barplot of PE
        ind = np.arange(len(peq[:,0])) + 1 # Location of the bars
        width = 0.27                   # Width of the bars
        
        #Tick labels (reaction numbers)
        ylabels = ['R1','R2','R3','R4','R5','R8','R9','R10','R11','R12']
#        ylabels = ['R'+str(ind[i]) for i in range(len(ind))]
        
        ax = fig.add_subplot(111)
        
        #Lower temp PE
        pe_low = ax.barh(ind, peq[:,0], width, color=[0.2,0.2,0.2])  
        
        pe_mid = ax.barh(ind+width, peq[:,1], width, color=[0.8,0.8,0.8]) 
        
        pe_high = ax.barh(ind+2*width, peq[:,2], width, color=[0.5,0.5,0.5]) 
        
        ax.axvline(x=0.5,color='k',linestyle='--')
        ax.set_xlabel('$\phi$ [-]')
        ax.set_ylabel('Reaction number')
        ax.set_yticks(ind+width)
        ax.set_yticklabels(ylabels)
        ax.set_xlim(0.0,1.0)
        
        ax.legend((pe_low[0], pe_mid[0], pe_high[0]), 
                  (str(int(Tout[0]))+' [$^\circ$C]', 
                  str(int(Tout[1]))+' [$^\circ$C]', 
                  str(int(Tout[2]))+' [$^\circ$C]'),frameon=False)

        plt.gca().invert_yaxis()

        figname = ('/home/tpcarvalho/carva/python_data/wgs_ni/wgs_peq.eps')
        
    if vis==2:
        
        #Net reaction rates [kmol/m**2 s]
        qf = np.array(temp['qf']).T
        
        #Barplot of qn
        ind = np.arange(len(qf[:,0])) + 1 # Location of the bars
        width = 0.27                   # Width of the bars
        
        #Tick labels (reaction numbers)
        ylabels = ['R1','R2','R3','R4','R5','R8','R9','R10','R11','R12']
#        ylabels = ['R'+str(ind[i]) for i in range(len(ind))]
        
        ax = fig.add_subplot(111)
        
        #Lower temp PE
        qf_low = ax.barh(ind, qf[:,0], width, color=[0.2,0.2,0.2],log=True)  
        
        qf_mid = ax.barh(ind+width, qf[:,1], width, color=[0.8,0.8,0.8],log=True) 
        
        qf_high = ax.barh(ind+2*width, qf[:,2], width, color=[0.5,0.5,0.5],log=True) 
        
        ax.set_xlabel('$\dot{r}_{j,f}$ [kmol/m$^2$ s]')
        ax.set_ylabel('Reaction number')
        ax.set_yticks(ind+width)
        ax.set_yticklabels(ylabels)
        ax.set_xlim(1e-17,1e00)
        
        ax.legend((qf_low[0], qf_mid[0], qf_high[0]),                   
                  (str(int(Tout[0]))+' [$^\circ$C]', 
                  str(int(Tout[1]))+' [$^\circ$C]', 
                  str(int(Tout[2]))+' [$^\circ$C]'),frameon=False)

        plt.gca().invert_yaxis()
        
        figname = ('/home/tpcarvalho/carva/python_data/wgs_ni/wgs_qf.eps')
    
    plt.tight_layout()     
    plt.show()

if exp_fig == 1:
    #Export figure as EPS
    plt.savefig(figname,format='eps',dpi=600)