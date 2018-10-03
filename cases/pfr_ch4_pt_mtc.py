import cantera as ct
import numpy as np
import plug as pfr
import os
import time
start = time.time()  

#### Input mechanism data ####:      
input_file = 'ch4_pt.cti' 
surf_name = 'PT_SURFACE'

#### Data files path ####:
basepath = os.path.dirname(__file__)
filepath = os.path.join(basepath,'..','data/exp_data/') 

#Load phases solution objects
gas = ct.Solution(input_file)
surf = ct.Interface(input_file,surf_name,[gas])

#Select which case to run
case = 1

#Re = 200
if case == 1:
    vis = 1
    #Inlet pressure [Pa]
    P_in = 101325.0                   
    #Import reference data
    filename = (filepath+'ch4_pt_Re200_Yk.csv')
#Re = 2000
elif case == 2:
    vis = 1
    #Inlet pressure [Pa]
    P_in = 101325.0*10                   
    #Import reference data
    filename = (filepath+'ch4_pt_Re2000_Yk.csv') 

#Re = 2000 (show coverages)
elif case == 3:
    vis = 2
    #Inlet pressure [Pa]
    P_in = 101325*10                   
    #Import reference data
    filename = (filepath+'ch4_pt_Re2000_covs.csv') 
    
#### Current thermo state ####:  
T_in = 1290                #Inlet temperature [K]
#Inlet molar composition: 
X_in = {'CH4':2.91,'O2':20.3889, 'N2':76.7011} 

#Reactor diameter and length [m]:
tube_d = 0.002        
length = 0.09

#Cross-sectional area [m**2]              
carea = np.pi*tube_d**2/4          

#Wall area [m**2]
wallarea = np.pi*tube_d*length 

#Catalytic area per volume [m**-]
cat_area_pvol = wallarea/(length*carea)

#Inlet velocity [m/s]
u_in_600 = 5

#Inlet gas properties @600 K
gas_600 = ct.Solution(input_file)
gas_600.TPX = 600, P_in, X_in
gas_600.transport_model = 'Mix'

#Compute inlet mass flow rate @600 K
vdot_in = carea*u_in_600             #Volumetric flow rate [m**3/s]
mdot_in = vdot_in*gas_600.density #Mass flow rate [kg/s]

#Compute Reynolds number (at inlet conditions)
Re = gas_600.density*u_in_600*tube_d/gas_600.viscosity

#Initialize reactor                                          
r1 = pfr.PlugFlowReactor(gas,
                         z_in = 0.01,
                         z_out = 0.1,
                         diam = tube_d)

#Initialize the reacting wall
rwall1 = pfr.ReactingSurface(r1,surf,
                             cat_area_pvol = cat_area_pvol)

#Set inlet thermo state
r1.TPX = T_in, P_in, X_in

#Set inlet mass flow rate [kg/s]
r1.mdot = mdot_in

#Create a ReactorSolver object
sim1 = pfr.ReactorSolver(r1,
                         atol = 1e-10,
                         rtol = 1e-08,
                         grid = 300,
                         max_steps = 1000,
                         solver_type = 'dae')
                       
#Solve the PFR
sim1.solve()

#Gas molar fractions, temperature and pressure along the reactor
covs1 = sim1.coverages
Yg1 = sim1.Y
uz1 = sim1.u
rho1 = sim1.rho

#Axial coordinates [cm]
zcoord = sim1.z*1e02

#Get inlet mass flow rate [kg/s]
mdot_in = r1.ac*rho1[0]*uz1[0]

#Get outlet mass flow rate [kg/s]
mdot_out = r1.ac*rho1[-1]*uz1[-1]

#Solve another PFR

#Initialize reactor                                          
r2 = pfr.PlugFlowReactor(gas,
                         z_in = 0.01,
                         z_out = 0.1,
                         diam = tube_d,
                         support_type = 'honeycomb',
                         porosity = 1.0,
                         dc = tube_d,
                         ext_mt = 1)

#Initialize the reacting wall
rwall2 = pfr.ReactingSurface(r2,surf,
                             cat_area_pvol = cat_area_pvol)

#Set inlet thermo state
r2.TPX = T_in, P_in, X_in

#Set inlet mass flow rate [kg/s]
r2.mdot = mdot_in

sim2 = pfr.ReactorSolver(r2,
                         atol = 1e-08,
                         rtol = 1e-07,
                         grid = 300,
                         max_steps = 1000,
                         solver_type = 'dae')

#Solve the PFR
sim2.solve()

#Gas molar fractions, temperature and pressure along the reactor
covs2 = sim2.coverages
Yg2 = sim2.Y
uz2 = sim2.u
rho2 = sim2.rho
       
#Print time elapsed
print('Time elapsed: {0:0.8f} seconds'.format(time.time() - start)) 
print('Reynolds number: {0:0.8f}'.format(Re)) 

#Import CSV data
import csv
ref_data=[]
with open(filename, 'rt') as f:
    reader = csv.reader(f)
    next(f)
    for row in reader:
        #Substitute empty entries for NaNs
        row = np.array(row)
        row[row==''] = 'nan'
        ref_data.append(row.astype(np.float))
f.close()
ref_data = np.array(ref_data)

#Plot results
import matplotlib.pyplot as plt
#Set figure size and position   
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-pastel')
plt.rcParams.update({'font.size': 24})  
plt.rcParams.update({'lines.linewidth': 2})  
    
if vis==1:
    fig = plt.figure()
    fig.set_size_inches(11, 8)
    fig.canvas.manager.window.move(200,100)  

    #Molar fractions along reactor
    ax = fig.add_subplot(111) 
    
    ax.plot(ref_data[:,16], ref_data[:, 17], '--k', label='CH$_4$ N-S')
    ax.plot(ref_data[:,18], ref_data[:, 19], '--r', label='CO$_2$ N-S')
    ax.plot(ref_data[:,20], ref_data[:, 21], '--b', label='H$_2$O N-S')
    
    ax.plot(zcoord, Yg1[:, gas.species_index('CH4')], ':k', label='CH$_4$ PFR')
    ax.plot(zcoord, Yg1[:, gas.species_index('CO2')], ':r',  label='CO$_2$ PFR')
    ax.plot(zcoord, Yg1[:, gas.species_index('H2O')], ':b',  label='H$_2$O PFR')

    ax.plot(zcoord, Yg2[:, gas.species_index('CH4')], '-k', label='CH$_4$ PFR-MTC')
    ax.plot(zcoord, Yg2[:, gas.species_index('CO2')], '-r',  label='CO$_2$ PFR-MTC')
    ax.plot(zcoord, Yg2[:, gas.species_index('H2O')], '-b',  label='H$_2$O PFR-MTC')
    
#    ax.plot(ref_data[:,0], ref_data[:, 1], ':k', label='CH4 - PLUG')
#    ax.plot(ref_data[:,2], ref_data[:, 3], ':g', label='CO2 - PLUG')
#    ax.plot(ref_data[:,4], ref_data[:, 5], ':b', label='H2O - PLUG')

#    ax.plot(ref_data[:,6], ref_data[:, 7], '--k', label='CH4 - BL')
#    ax.plot(ref_data[:,8], ref_data[:, 9], '--r', label='CO2 - BL')
#    ax.plot(ref_data[:,10], ref_data[:, 11], '--b', label='H2O - BL')
  
    ax.set_xlabel('$z$ [cm]')
    ax.set_ylabel('$Y_k$ [-]')
    ax.legend(loc='best',fontsize=18)
    
#    ax1 = fig.add_subplot(111)  
#    ax1.semilogy(ref_data[:,22], ref_data[:, 23], '--k', label='CO N-S')
#    ax1.semilogy(zcoord, Yg1[:, gas.species_index('CO')], ':k', label='CO PFR')
#    ax1.semilogy(zcoord, Yg2[:, gas.species_index('CO')], '-k', label='CO PFR-MTC')
#    
##    ax1.semilogy(ref_data[:,6], ref_data[:, 7], ':k', label='CO - PLUG')
#
#    ax1.axis((1.0,zcoord[-1],1e-08,1e-04))
#    ax1.set_xlabel('$z$ [cm]')
#    ax1.set_ylabel('$Y_k$ [-]')
#    ax1.legend(loc='best',fontsize=20)
    plt.show()
    
elif vis==2:
    
    fig = plt.figure()
    fig.set_size_inches(17, 8)
    fig.canvas.manager.window.move(200,100)  
    
    #Coverages along reactor
    ax = fig.add_subplot(121) 
    ax.semilogy(zcoord, covs1[:, surf.species_index('O_Pt')], ':y',  label='O(S) PFR')   
    ax.semilogy(zcoord, covs1[:, surf.species_index('_Pt_')], ':k',  label='Pt(S) PFR')
    ax.semilogy(zcoord, covs1[:, surf.species_index('OH_Pt')], ':r',  label='OH(S) PFR')

    ax.semilogy(zcoord, covs2[:, surf.species_index('O_Pt')], '-y',  label='O(S) PFR-MTC')   
    ax.semilogy(zcoord, covs2[:, surf.species_index('_Pt_')], '-k',  label='Pt(S) PFR-MTC')
    ax.semilogy(zcoord, covs2[:, surf.species_index('OH_Pt')], '-r',  label='OH(S) PFR-MTC')
    
#    ax.semilogy(ref_data[:,0], ref_data[:, 1], ':y',  label='O(S)  - PLUG')
#    ax.semilogy(ref_data[:,2], ref_data[:, 3], ':k',  label='Pt(S) - PLUG')
#    ax.semilogy(ref_data[:,4], ref_data[:, 5], ':c',  label='OH(S) - PLUG')  

    ax.semilogy(ref_data[:,14], ref_data[:, 15], '--y',  label='O(S) N-S')
    ax.semilogy(ref_data[:,16], ref_data[:, 17], '--k',  label='Pt(S) N-S')
    ax.semilogy(ref_data[:,18], ref_data[:, 19], '--r',  label='OH(S) N-S')  
    
    ax.axis((1.0,zcoord[-1],1e-03,1.0))
    ax.set_xlabel('$z$ [cm]')
    ax.set_ylabel('$\Theta_k$ [-]')
    ax.legend(loc='right',fontsize=13)
    
    #Plot coverages
    ax1 = fig.add_subplot(122)  
    ax1.semilogy(zcoord, covs1[:, surf.species_index('C_Pt')], ':r',  label='C(S) PFR')
    ax1.semilogy(zcoord, covs1[:, surf.species_index('CO_Pt')], ':m',  label='CO(S) PFR') 
    ax1.semilogy(zcoord, covs1[:, surf.species_index('H_Pt')], ':b',  label='H(S) PFR')
    ax1.semilogy(zcoord, covs1[:, surf.species_index('H2O_Pt')], ':g', label='H2O(S) PFR')

    ax1.semilogy(zcoord, covs2[:, surf.species_index('C_Pt')], '-r',  label='C(S) PFR-MTC')
    ax1.semilogy(zcoord, covs2[:, surf.species_index('CO_Pt')], '-m',  label='CO(S) PFR-MTC') 
    ax1.semilogy(zcoord, covs2[:, surf.species_index('H_Pt')], '-b',  label='H(S) PFR-MTC')
    ax1.semilogy(zcoord, covs2[:, surf.species_index('H2O_Pt')], '-g', label='H2O(S) PFR-MTC')
    
#    ax1.semilogy(ref_data[:,6], ref_data[:, 7], ':r',  label='C(S)  - PLUG')
#    ax1.semilogy(ref_data[:,8], ref_data[:, 9], ':m',  label='CO(S) - PLUG')
#    ax1.semilogy(ref_data[:,10], ref_data[:, 11], ':b',  label='H(S) - PLUG') 
#    ax1.semilogy(ref_data[:,12], ref_data[:, 13], ':g',  label='H2O(S) - PLUG')

    ax1.semilogy(ref_data[:,20], ref_data[:, 21], '--r',  label='C(S) N-S')
    ax1.semilogy(ref_data[:,22], ref_data[:, 23], '--m',  label='CO(S) N-S')
    ax1.semilogy(ref_data[:,24], ref_data[:, 25], '--b',  label='H(S) N-S') 
    ax1.semilogy(ref_data[:,26], ref_data[:, 26], '--g',  label='H2O(S) N-S')
    
    ax1.axis((1.0,zcoord[-1],1e-07,1e-04))
    ax1.set_xlabel('$z$ [cm]')
    ax1.set_ylabel('$\Theta_k$ [-]')
    ax1.legend(loc='best',fontsize=13)
    plt.show()
