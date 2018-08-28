import numpy as np

#Create a PFR class for the reduced expression:
class PFR_simple(object):
    def __init__(self,**p):
        #Make reference to gas phase object
        self.gas = p.get('gas',None)

        #Make reference to surface phase object
        self.surf = p.get('surf',None)

        #Make reference to bulk phase object
        self.bulk = p.get('bulk',None)
        
        #Reactor parameters:
        self.diam = p.get('diam',0.0)         #Reactor diameter [m]  
        self.carea = np.pi*self.diam**2/4     #Reactor cross-sectional area [m**2]

        #### CHEMICAL SPECIES DATA ####:
        self.mw_g = self.gas.molecular_weights      #Molar masses [kg/kmol]
        self.n_species = self.gas.n_species         #Number of gas species
        
        #Universal gas constant [J/kmol K]
        self.Rg = 8314.4621 
        
        #Site density [kmol/m**2]
        self.sden = self.surf.site_density
        
        #Stoichiometric coefficients for CO + H2O = CO2 + H2
        self.stq_coeffs = np.array([-1.0,1.0,-1.0,1.0,0.0])

        #Rate effectiveness factor
        self.rate_eff = p.get('rate_eff',1.0)
        
        #Initialize reactor inlet properties:
        self._u0 = 0.0
        self._mdot = 0.0        
        self.theta = np.zeros(3)
    
    #### INLET PROPERTIES ####:   
    
    #Get/set current flow velocity [m/s]
    @property
    def u0(self):
        return self._u0

    @u0.setter
    def u0(self, value):
        #Set flow velocity
        self._u0 = value
        #Compute mass flow rate
        self._mdot = self._u0*self.carea*self.gas.density        
    
    #Get/set the mass flow rate [kg/s]    
    @property
    def mdot(self):       
        return self._mdot

    @mdot.setter
    def mdot(self, value):
        #Set mass flow rate
        self._mdot = value
        #Compute flow velocity
        self._u0 = self._mdot/(self.carea*self.gas.density) 
    
    #Get/set current thermodynamic state (temperature,pressure,molar fractions)
    @property
    def TPX(self):
        return self.gas.T, self.gas.P, self.gas.X

    @TPX.setter
    def TPX(self, values):
        assert len(values) == 3, 'Incorrect number of input values'
        #Set gas phase thermodynamic state
        self.gas.TPX = values[0], values[1], values[2] 
        
        if self.bulk != None:
            #Set bulk phase thermodynamic state
            self.bulk.TP = values[0], values[1]
        
        #Set surface thermodynamic state
        self.surf.TP = values[0], values[1]
   
    #Get/set current thermodynamic state (temperature,pressure,molar fractions)
    @property
    def TPY(self):
        return self.gas.T, self.gas.P, self.gas.Y

    @TPY.setter
    def TPY(self, values):
        assert len(values) == 3, 'Incorrect number of input values'
        #Set gas phase thermodynamic state
        self.gas.TPY = values[0], values[1], values[2] 
    
        if self.bulk != None:
            #Set bulk phase thermodynamic state
            self.bulk.TP = values[0], values[1]
        
        #Set surface thermodynamic state
        self.surf.TP = values[0], values[1]
       
    def compute_k(self):
        '''Compute rate constants at current state'''

        #P_atm/RT term
        self.p_RT = 101325.0/(self.Rg*self.gas.T) 
     
        #Compute WGS equilibrium constant
        self.compute_kequil()
           
        #Forward and reverse rate constants based on inlet conditions                
        self.kf = self.surf.forward_rate_constants
        self.kr = self.surf.reverse_rate_constants
 
        #Equilibrium constants (concentration units)
        self.K = self.surf.equilibrium_constants
        
        #Rate constants used in reduced kinetic expression     
        self.kf9 = self.kf[6]*self.sden    
        
        self.K1 = self.K[0]*self.p_RT
        self.K2 = self.K[1]*self.p_RT
        self.K3 = self.K[2]*self.p_RT
        self.K4 = self.K[3]*self.p_RT
        self.K5 = self.K[4]

    def compute_kequil(self):

        #Get non-dimensional enthalpies
        self.h_gas = self.gas.standard_enthalpies_RT
        
        #Get non-dimensional entropies
        self.s_gas = self.gas.standard_entropies_R
        
        #Delta Gibbs
        self.delta_gibbs = np.inner(self.stq_coeffs,(self.s_gas-self.h_gas))
        
        #WGS equilibrium constant (pressure units)
        self.Kp = np.exp(self.delta_gibbs)

    def compute_state(self,y):
        
        '''Update gas state'''
        #Get state variables:
        self.Y = y
        
        #Update gas phase mass fractions
        self.gas.set_unnormalized_mass_fractions(self.Y)

        #Gas species concentrations [kmol/m**3]
        self.C = self.gas.concentrations
        self.Cco = self.C[self.gas.species_index('CO')] 
        self.Cco2 = self.C[self.gas.species_index('CO2')] 
        self.Ch2 = self.C[self.gas.species_index('H2')] 
        self.Ch2o = self.C[self.gas.species_index('H2O')] 
        
        #Current gas density [kg/m**3]
        self.rho = self.gas.density
        
        #Current flow velocity [m/s]
        self.u = self.mdot/(self.carea*self.rho)
        
        '''Compute coverages expressions'''
        #Vacant sites expression
        self.theta[0] = 1.0/(1.0 + self.K2*self.Cco + np.sqrt(self.K4*self.Ch2))
        
        #H coverage
        self.theta[1] = np.sqrt(self.K4*self.Ch2)*self.theta[0]   
        
        #CO coverage
        self.theta[2] = self.K2*self.Cco*self.theta[0] 
             
    def compute_rates(self): 
        
        #Current coverages
        self.cov_h = self.surf.coverages[self.surf.species_index('H(S)')]
        self.cov_co = self.surf.coverages[self.surf.species_index('CO(S)')]
        self.cov_ni = 1.0 - self.cov_h - self.cov_co
     
        #Compute forward rate constant for CO consumption
        self.kf9_co = ( self.kf9*self.K1*self.K2*self.K5
                        /(self.cov_h*self.cov_ni**(-3)) )
        
        #Effective forward constant
        self.kf_eff = self.kf9_co 
        
        #Forward WGS reaction rate [kmol/m**3 s]
        self.r_fwgs = ( self.kf_eff*self.Cco*self.Ch2o )
        
        #Reverse WGS reaction rate [kmol/m**3 s]
        self.r_rwgs = ( (self.kf_eff/self.Kp)*self.Cco2*self.Ch2 )
        
        #Net WGS reaction rate [kmol/m**3 s]
        self.r_net = self.r_fwgs - self.r_rwgs

        #Reaction rates [kmol/m**3 s]
        self.wdot = self.stq_coeffs*self.r_net
        self.wdot *= self.rate_eff
         
    def eval_reactor_wgs(self,z,y,ydot): 
        
        #Update gas state and compute coverages
        self.compute_state(y)
               
        #Compute reaction rates
        self.compute_rates()
          
        #Evaluate species mass balance equation    
        self.dYdz = self.mw_g*self.wdot
        self.dYdz /= self.rho*self.u
        
        #Vector with derivatives
        ydot[0:] = self.dYdz
        
