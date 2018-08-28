import numpy as np        
from plug.reactor.washcoat_model import Washcoat
                   
class ReactingSurface(object):
    '''########################################################################
    Class that defines a reacting surface to be used in conjunction with the
    FlowReactor class.
    -----------------
    Input parameters:
    -----------------
        reactor: FlowReactor object
        surface: surface-phase object in Cantera Interface format
        bulk: solid bulk-phase object in Cantera Solution format 
        cat_area_pvol: catalytic area per volume (m**-1)
        int_mt: flag, turns support for internal mass transfer limitations 
                on[1]/off[0] (default is off [0]). Extra parameters needed:   
                ---
                thickness: washcoat thickness (m)
                epsilon: washcoat porosity (-)
                tortuosity: washcoat tortuosity (-)
                pore_diam: washcoat mean pore diameter (m) 
                ---
                ### Parameters for effectiveness factor method:
                lim_species: reactant limitant species
                ---
                ### Parameters for reaction-diffusion model:
                gas_wc: gas-phase object in Cantera Solution format 
                surf_wc: surface-phase object in Cantera Interface format
                ngrid: number of radial grid points
                stch: stretching factor for radial mesh. If > 1, mesh is 
                refined near the reacting surface
                ---
    ########################################################################'''
    
    def __init__(self,reactor,surface,**params):
        
        #### MAKE REFERENCE TO REACTOR OBJECT ####: 
        self._r = reactor
        
        #### LOAD SURFACE PHASE OBJECTS ####:    
        
        #One gas phase and one surface phase must be supplied by default.        
        self.surface = surface   #Cantera surface solution object 
        
        #Cantera bulk solution object (must be supplied as a list)
        self.bulk = params.get('bulk',None)
        
        #If a solid bulk phase was supplied
        if self.bulk != None:
            
            #Number of bulk phases
            self.n_bulk = len(self.bulk)
            
            self.bulk_idx = []          #Kinetics bulk species index
            self.wb = []                #Bulk species molar masses [kg/kmol]
            
            #Get kinetics index of bulk species
            for i in range(self.n_bulk):
                
                #Kinetics index for solid bulk species
                self.bulk_idx.append( self.surface.kinetics_species_index(
                                      self.bulk[i].species_names[0]) )
                
                #Solid bulk species molecular weight [kg/kmol]
                self.wb.append(self.bulk[i].molecular_weights)
                
            #Convert list to Numpy array   
            self.wb = np.array(self.wb)
      
        #### CHEMICAL SPECIES DATA ####:
        
        self.ns = self.surface.n_species - 1    #Number of surface species

        #Get surface site occupancy number
        self.n_sites = np.array([self.surface.species(n).size 
                                 for n in range(self.surface.n_species)])

        #Site density [kmol/m**2]
        self.sden = self.surface.site_density
        
        #Site occupancy and density term (only used w/ pseudo surface vel.)
        self.rsden = self.n_sites/self.sden
        
        #Gas species kinetics index
        self.gas_idx = np.array([self.surface.kinetics_species_index(
                                 self._r.gas.species_names[n]) 
                                 for n in range(self._r.gas.n_species)])  
    
        #Surface species kinetics index
        self.surface_idx = np.array([self.surface.kinetics_species_index(
                                     self.surface.species_names[n]) 
                                     for n in range(self.surface.n_species)]) 

        #### CATALYTIC AND GEOMETRIC DATA ####:      
            
        #Catalytic area per volume [m**-1]
        self.cat_area_pvol = params.get('cat_area_pvol',0.0)  

        #Catalytic area [m**2]       
        self.cat_area = self.cat_area_pvol*self._r.vol_eff
        
        #Catalytic active area per unit length [m]  
        self.ai = self.cat_area/self._r.length
        
        #Compute ratio of catalytic to geometric surface area [-]
        self.fcatgeo = self.cat_area/self._r.geo_area
        
        #### INSTALL REACTING SURFACE IN REACTOR OBJECT ####: 
        
        #Install wall in the reactor (make reference to this class)
        self._r._add_reacting_surface(self)
        
        #Pre-allocate coverages array
        self.theta = np.zeros(self.ns+1)

        #Array slicing pointers
        self.aux2 = self._r.aux1 + self.ns
        
        #### MASS TRANSFER PARAMETERS ####: 
 
        #Internal mass transfer effects flag (off by default)
        self.flag_int_mt = params.get('int_mt',0)
        
        #Internal mass transfer effects
        if self.flag_int_mt != 0: 
            
            #Initialize Washcoat model
            self.wc = Washcoat(self._r,params)

            #If the effectiveness factor approach is used
            if self.flag_int_mt == 1:
                
                #Check if limiting species has been supplied
                if self.wc.lim_species == None:
                    #Raise error if not given
                    raise ValueError('Limiting species was not provided')
                    
            #If detailed reaction-diffusion model is used      
            elif self.flag_int_mt == 2:
                
                #Initialize reaction-diffusion parameters
                self.wc._init_reaction_diffusion_params(params)
                
        #External mass transfer parameters
        if self._r.flag_ext_mt == 1:

            #Pre-allocate mass fraction array
            self.Ys = np.zeros(self._r.ng+1)
        
            #Array slicing pointer
            self.aux3 = self.aux2 + self._r.ng
                 
        else:    
            #Array slicing pointer
            self.aux3 = self._r.aux1 + self.ns
            
        #Initialize pseudo-velocity for surface species (only use if there 
        #are convergence issues)
        self._pseudo_velocity = 1e-06 #[m/s]
                        
    #Get/set pseudo-velocity (only if system is solved as ODE)
    @property
    def pseudo_velocity(self):       
        return self._pseudo_velocity

    @pseudo_velocity.setter
    def pseudo_velocity(self, value):
        #Set mass flow rate
        self._pseudo_velocity = value
        
    def get_max_coverage(self):
        
        #Get maximum coverage index
        self._covmax_index = np.argmax(self.surface.coverages)
        
        #Get boolean array with all coverages except the largest one
        self._cov_index = (self.surface.coverages != 
                           self.surface.coverages[self._covmax_index])
        
    '''Sync Cantera object thermo states with reactor's'''
    def _sync_state(self):

        #Set surface to same pressure and temperature as gas phase
        self.surface.TP = self._r.gas.T, self._r.gas.P
        
        #If solid bulk phase(s) are present
        if self.bulk != None:
            
            #Set bulk phase(s) to same pressure and temperature as gas phase
            for i in range(self.n_bulk):
                self.bulk[i].TP = self._r.gas.T, self._r.gas.P
                                     
    def update_state(self,y):
        
        #Get state variables:
        self.theta[self._cov_index] = y[self._r.aux1:self.aux2]
        
        #Substitute largest coverage value by constraint
        self.theta[self._covmax_index] = 1.0 - np.sum(self.theta[self._cov_index])
        
        #If there are external mass transfer effects
        if self._r.flag_ext_mt == 1:
            
            #Update thermodynamic state of surface-adjacent gas phase
            self.update_gas_state(y)
                
        #If a bulk phase if also present:
        if self.bulk != None:   
            
            #Update bulk phase(s) to current temperature and pressure
            for i in range(self.n_bulk):
                self.bulk[i].TP = self._r.T, self._r.P
        
        #Update surface coverages
        self.surface.set_unnormalized_coverages(self.theta)
        
        #Update surface phase to current temperature and pressure   
        self.surface.TP = self._r.T, self._r.P
    
    def update_gas_state(self,y):
        
        #Mass fractions of gas species next to surface
        self.Ys[:-1] = y[self.aux2:self.aux3]
        
        #Last mass fraction  value ensures the sum stays constant
        self.Ys[-1] = 1.0 - np.sum(self.Ys[:-1])
        
        #Update gas phase mass fractions
        self._r.gas.set_unnormalized_mass_fractions(self.Ys)

        #Update gas phase to current temperature and pressure
        self._r.gas.TP = self._r.T, self._r.P
        
        #Current gas density [kg/m**3]
        self.rho_s = self._r.gas.density

    def update_rates(self,y):
              
        #### Net species production rates ###: 
         
        #Get net production rates from all heterogeneous reactions [kmol/m**2 s] 
        self.hdot = self.surface.net_production_rates
        
        #Gas species from heterogeneous reactions [kmol/m**2 s]  
        self.gdot = self.hdot[self.gas_idx]  
        
        #Surface species from heterogeneous reactions [kmol/m**2 s]  
        self.sdot = self.hdot[self.surface_idx]
        
        #If internal diffusion effects are considered
        if self.flag_int_mt == 1:

            #Compute effectiveness factor 
            self.neff = self.wc.get_effectiveness_factor(self._r.gas,self.gdot)
            
            #Multiply gas species production rates by effectiveness factor
            self.gdot *= self.neff[self.wc.lim_species_idx]
            self.sdot *= self.neff[self.wc.lim_species_idx]
            
        elif self.flag_int_mt == 2: 
            
            #Solve washcoat diffusion-reaction equations 
            self.wc.eval_residuals(y[self.aux3:])
            
            #Gas species from heterogeneous reactions [kmol/m**2 s]  
            self.gdot = self.wc.gdot0/self.fcatgeo
                           
    def eval_theta(self):
        
        #Evaluate surface coverages as ODEs (using a pseudo-velocity)
        self.dthetadz = (self.sdot*self.rsden)[self._cov_index] 
        self.dthetadz /= self.pseudo_velocity
    
    def eval_rtheta(self):
        
        #Evaluate coverages balance as algebraic equation in residual form
        self.restheta = self.sdot[self._cov_index]
        
    def eval_ys_residuals(self,z):
        
        #Evaluate external mass transfer coefficients [m/s]
        self.emt_coefficients(z)
        
        #Residuals for gas mass fractions at the surface 
        self.resYs = ( self.km*(self.rho_s*self.Ys - self._r.rho*self._r.Y) 
                       - self.gdot*self._r.wg*self.fcatgeo )
        
    '''External mass transport coefficient correlations'''
    def emt_coefficients(self,z):

        #Compute transport properties
        self._r.eval_transport_props()
        
        if self._r.support_flag == 0:
            
            #Compute Sherwood numbers 
            if self._r.Re < 2320.0:
                #Laminar flow through circular pipe
                self.Sh = 1.62*(self._r.dc**2*self._r.us/
                            (self._r.length*self._r.diff_mol))**(1/3)
            else:
                #Turbulent flow through circular pipe
                self.Sh = 0.026*self._r.Re**0.8*self._r.Sc**0.333
             
        elif self._r.support_flag == 1:

            #Avoid z being equal to zero
            z = np.maximum(z,1e-10)
            
            #Compute Z^star
            self.zstar = z/(self._r.dc*self._r.Re*self._r.Sc)
            
            #Compute Sherwood numbers 
            self.Sh = 3.675 + (8.827*(1e03*self.zstar)**(-0.545)
                               *np.exp(-48.2*self.zstar))
            
        elif self._r.support_flag == 2:

            #Compute Sherwood numbers 
            self.Sh = 1.1*self._r.Re**0.43*self._r.Sc**0.333
            
        #Compute mass transfer coefficients [m/s]
        self.km = self.Sh*self._r.diff_mol/self._r.dc 