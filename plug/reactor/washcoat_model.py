import numpy as np

class Washcoat(object):
    '''########################################################################
    Class that defines washcoat model to account for internal mass transfer
    limitations. Two methods are available, the effectiveness factor and the
    detailed reaction-diffusion method. Input parameters are described in the
    ReactingSurface class.    
    ########################################################################'''  
    
    def __init__(self,reactor,params):
 
        #Make reference to reactor and reacting wall objects
        self._r = reactor

        #Washcoat properties
        self.thickness = params.get('thickness',None)    #Washcoat thickness [m]
        self.epsilon = params.get('epsilon',None)        #Washcoat porosity [-]
        self.tortuosity = params.get('tortuosity',None)  #Washcoat tortuosity [-]
        self.pore_diam = params.get('pore_diam',None)    #Washcoat pore diameter [m]
       
        #Group washcoat properties into a single dict
        self.parameters = {'thickness': self.thickness, 
                           'epsilon': self.epsilon,
                           'tortuosity': self.tortuosity, 
                           'pore_diam': self.pore_diam}
              
        #Limiting gas species (for effectiveness factor calculation) 
        self.lim_species = params.get('lim_species',None)
        
        if self.lim_species != None:
            #Limiting gas species index
            self.lim_species_idx = self._r.gas.species_index(self.lim_species)
            
            #Append to dict
            self.parameters['lim_species'] = self.lim_species
            
        #Ratio of tortuosity to porosity
        self.tp_ratio = self.tortuosity/self.epsilon
        
        #Universal gas constant [J/kmol K]
        self.rg = 8314.4621
        
        #Precompute Knudsen diffusion constant terms
        self.knudsen = self.pore_diam/3.0*np.sqrt(8.0*self.rg/(np.pi*self._r.wg))
        
        #Ratio of active catalytic area to washcoat volume
        self.gamma = self._r.rs.fcatgeo/self.thickness
        
    def _init_reaction_diffusion_params(self,params):
        
        #Initialize variables defined below if reaction-diffusion problem
        #is to be solved
        
        #Cantera gas and surface phase objects within the washcoat
        self._gas = params.get('gas_wc',None)
        self._surface = params.get('surf_wc',None)
    
        if self._gas == None or self._surface == None:
            raise ValueError('Cantera phase objects were not provided')
            
        #Number of grid points
        self.ngrid = params.get('ngrid',15)
        
        #Create non-uniform washcoat coordinates
        self.stch = params.get('stch',1.0)       #Mesh stretching factor        
        w0 = 1
        self.wcoord = [w0]   
        for i in range(self.ngrid-1):
            w0 *= self.stch
            self.wcoord.append(w0)
            
        self.wcoord = np.array(self.wcoord) - 1.0
        self.wcoord = self.thickness*self.wcoord/np.max(self.wcoord)
        
        #Pre-alloc input arrays for mass fractions and coverages
        self.y = np.zeros((self.ngrid,self._r.ng+1))
        self.covs = np.zeros((self.ngrid,self._r.rs.ns+1))
        
        #Create ODE solver object to solve transient problem
        from scikits.odes import ode
        self.solver = ode('cvode',self.eval_transient, 
                          atol=1e-10,
                          rtol=1e-05,
                          old_api=False,
                          max_steps=5000)
      
    '''Internal mass transport effectiveness factor computation''' 
    def get_effectiveness_factor(self,gas,gdot): 
        
        #Compute molecular diffusion coefficients [m**2/s]
        self.diff_mol = gas.mix_diff_coeffs_mass
                
        #Molecular diffusion coefficient[m**2/s]
        self.diff_knud = self.knudsen*np.sqrt(gas.T) 

        #Effective diffusivity [m^2/s]
        self.diff_eff = 1.0/(self.tp_ratio*(1.0/self.diff_mol + 1.0/self.diff_knud) )
        
        #Net rate of production for species within the washcoat [kmol/m**2 s]
        self.gdot_int = np.abs(gdot)*self.gamma
        
        #Get surface concentration species in the gas-washcoat interface [kmol/m**3]
        self.c_int = gas.concentrations
        
        #Prevent zero or negative values in concentration array
        self.c_int[self.c_int <= 0.0] = 1e-20
        
        #Forward rate constant 
        self.kf = self.gdot_int/self.c_int
        
        #Compute the Thiele modulus  
        self.thiele_mod = self.thickness*np.sqrt(self.kf/self.diff_eff)

        #Prevent zero values in Thiele modulus array
        self.thiele_mod[self.thiele_mod == 0.0] = 1e-20
        
        #Effectiveness factor
        self.neff = np.tanh(self.thiele_mod)/self.thiele_mod
        
        return self.neff
    
    def eval_washcoat(self,y):
        
        #Reshape input vector of concentrations
        self.y[:,:-1] = np.reshape(y[:self._r.ng*self.ngrid],(self.ngrid,self._r.ng))
        
        #Append last gas mass fraction value to keep a constant sum   
        self.y[:,-1] = 1.0 - np.sum(self.y[:,:-1],axis=1)
        
        #Reshape input vector of coverages
        self.covs[:,self._r.rs._cov_index] = np.reshape(y[self._r.ng*self.ngrid:],
                                                    (self.ngrid,self._r.rs.ns))
        
        #Append first mass fraction column to keep a constant sum
        self.covs[:,self._r.rs._covmax_index] = ( 1.0 - 
                                np.sum(self.covs[:,self._r.rs._cov_index],axis=1) )
                                         
        #Create lists to store values 
        self.gdot = []
        self.sdot = []
        self.diff_mol_wc = []
        
        for i in range(self.ngrid):
            
            #Set gas mass fractions
            self._gas.set_unnormalized_mass_fractions(self.y[i,:])
            
            #Update gas phase to current temperature and pressure
            self._gas.TP = self._r.gas.T, self._r.gas.P
                        
            #Molecular diffusion coefficients [m**2/s]
            self.diff_mol_wc.append(self._gas.mix_diff_coeffs_mass)
            
            #Set surface coverages
            self._surface.set_unnormalized_coverages(self.covs[i,:])

            #Update surface phase to current temperature and pressure
            self._surface.TP = self._r.gas.T, self._r.gas.P
        
            #Get net production rates from all heterogeneous reactions [kmol/m**2 s] 
            self.hdot = self._surface.net_production_rates
        
            #Get heterogeneous production rates [kmol/m**2 s]
            self.gdot.append(self.hdot[self._r.rs.gas_idx])
            self.sdot.append(self.hdot[self._r.rs.surface_idx])
            
        #Convert lists to arrays
        self.diff_mol_wc = np.array(self.diff_mol_wc)
        self.gdot = np.array(self.gdot)
        self.sdot = np.array(self.sdot)

        #Gas density [kg/m**3]
        self.rho = self._gas.density
        
        #Molecular diffusion coefficient of limiting species [m**2/s]
        self.diff_knud_wc = self.knudsen*np.sqrt(self._gas.T)
        
        #Effective diffusivity [m^2/s]
        self.diff_eff_wc = 1.0/(self.tp_ratio*(1.0/self.diff_mol_wc 
                                            + 1.0/self.diff_knud_wc) )
        
        #Reaction rate term [kmol/m**3 s]
        self.rdot = self.gamma*self.gdot*self._r.wg
      
        #Compute the FD approximations 
        self.yx = np.gradient(self.y,self.wcoord,axis=0,edge_order=2)
        
        #Apply zero-flux BC
        self.yx[-1,:] = 0.0
        
        #Diffusive flux [m**2/s]
        self.jdot = self.diff_eff_wc*self.yx
        
        #Compute 2nd derivative
        self.yxx = np.gradient(self.yx,self.wcoord,axis=0,edge_order=2)
        
    def eval_residuals(self,y):
        
        #Evaluate washcoat equations
        self.eval_washcoat(y)
        
        #Compute residual for washcoat gas species concentrations
        self.resy = self.rho*self.diff_eff_wc*self.yxx + self.rdot
        
        #Compute mass flux at the gas-washcoat interface
        self.gdot0 = self.jdot[0,:]*self.rho/self._r.wg
        
        #Compute coverages residuals
        self.rescovs = self.sdot
        
        #Gas species mass fractions at the interface:        
        #If there are external mass transfer effects
        if self._r.flag_ext_mt == 1:
            #Gas-surface mass fractions [-]
            self.ygas = self._r.rs.Ys
        else:
            #Gas-bulk mass fractions [-]
            self.ygas = self._r.Y
        
        #Apply BCs
        self.resy[0,:] = self.ygas - self.y[0,:]
        self.rescovs[0,:] = self._r.rs.theta - self.covs[0,:]
        
        #Reshape both output arrays
        self.resy_flat = self.resy[:,:-1].flatten()
        self.rescovs_flat = self.rescovs[:,self._r.rs._cov_index].flatten()
        
    def eval_transient(self,t,y,ydot):
        
        #Evaluate washcoat equations
        self.eval_washcoat(y)

        #Compute time-like derivative for gas-species mass fractions
        self.dydt = (self.diff_eff_wc*self.yxx + self.rdot/self.rho)
        
        #Compute coverages derivative
        self.dthetadt = self.sdot*self._r.rs.rsden
        
        #Apply ICs
        self.dydt[0,:] = 0.0
        self.dthetadt[0,:] = 0.0
        
        #Reshape both output arrays
        self.dydt_flat = self.dydt[:,:-1].flatten()
        self.dthetadt_flat = self.dthetadt[:,self._r.rs._cov_index].flatten()

        ydot[:] = np.hstack((self.dydt_flat,self.dthetadt_flat))

    def solve_initial_guess(self,tf,yg,covs):
        
        #Create initial gas concentrations and coverages matrices
        self.y0_tile = np.tile(yg[:-1].T,(self.ngrid,1))
        self.cov0_tile = np.tile(covs[self._r.rs._cov_index],(self.ngrid,1))
        
        self.y0flat = self.y0_tile.flatten()
        self.cov0flat = self.cov0_tile.flatten()
        
        #Initial conditions vector
        self.y0 = np.hstack((self.y0flat,self.cov0flat))
        
        #Create time array
        self.t = np.linspace(0,tf,2)
        
        #Solve the equations
        self.solution = self.solver.solve(self.t,self.y0)
        
        #If errors take place during integration
        if self.solution.errors.t != None:
            print ('Error: ', self.solution.message, 
                   'Error at time', self.solution.errors.t)
        
        #Solution variables at final time step  
        self.guess = self.solution.values.y[-1,:] 
        
        #Pre-alloc output arrays
        self.y0_guess = np.zeros((self.ngrid,self._r.ng+1))
        self.covs0_guess = np.zeros((self.ngrid,self._r.rs.ns+1))
        
        #Mass fractions for gas species within the washcoat
        self.y0_guess[:,:-1] = np.reshape(self.guess[:self._r.ng*self.ngrid],
                                          (self.ngrid,self._r.ng))
       
        #Append last gas mass fraction value to keep a constant sum   
        self.y0_guess[:,-1] = 1.0 - np.sum(self.y0_guess[:,:-1],axis=1)
        
        #Coverages within the washcoat
        self.covs0_guess[:,self._r.rs._cov_index] = np.reshape(
                                            self.guess[self._r.ng*self.ngrid:],
                                            (self.ngrid,self._r.rs.ns))
        
        #Compute coverage of first species 
        self.covs0_guess[:,self._r.rs._covmax_index] = ( 1.0
                    - np.sum(self.covs0_guess[:,self._r.rs._cov_index],axis=1) )