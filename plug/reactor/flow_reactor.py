import numpy as np
from assimulo.problem import Explicit_Problem, Implicit_Problem

class FlowReactorBase(object):
    '''########################################################################
    Base class that defines a tubular flow reactor with homogeneous and/or
    heterogeneous reactions. This class defines methods that are used by
    FlowReactor class.
    -----------------
    Input parameters:
    -----------------
        gas: gas-phase object in Cantera Solution format
        energy: flag, turns energy equation on[1]/off[0] (default is off [0])
        momentum: flag, turns momentum equation on[1]/off[0] (default is off [0])
        ext_mt: flag, turns support for external mass transfer limitations 
                on[1]/off[0] (default is off [0])   
        ext_ht: flag, turns support for external heat transfer (default is off 
                [0]) but requires two extra parameters:
                ---
                Tinf: ambient temperature (default is 298 K)
                uht: overall heat-transfer coefficient (W/m**2 K)
                ---
        z_in: reactor inlet axial coordinate (m)
        z_out: reactor outlet axial coordinate (m)
        diam: reactor diameter (m)
        support_type: type of porous support present in the reactor. Default
                is 'none'. Options include 'honeycomb' and 'foam'. Additional
                parameters are necessary if 'honeycomb' or 'foam' are used:
                ---
                porosity: ratio of void to total volume (default is 1)
                dc: characteristic support diameter (m)
                sp_area: support specific surface area (m**-1)
                ---
    ########################################################################'''
    
    def __init__(self,gas,**params):
               
        #### PROBLEM FLAGS (1=on, 0=off) ####:   
        
        #Energy equation flag (default is off)
        self.flag_energy = params.get('energy',0)
        
        #Momentum equation flag (default is off)
        self.flag_momentum = params.get('momentum',0)

        #External mass transfer effects flag (off by default)
        self.flag_ext_mt = params.get('ext_mt',0)  
        
        #External heat transfer from reactor walls flag
        self.flag_ext_ht = params.get('ext_ht',0)
        
        #External heat transfer input paramters:
        if self.flag_ext_ht == 1:

            #Temperature of external surface [K]
            self.Tinf = params.get('Tinf',298.0)     
            
            #Overall heat-transfer coefficient [W/m**2 K]
            self.uht = params.get('uht',None)
            
            #Raise error if overall heat-transfer coefficient is not provided
            if self.uht == None:
                raise ValueError('Overall heat-transfer coefficient was not supplied')

        #### LOAD GAS PHASE OBJECT ####:    
        
        #The plug flow reactor must always have a Cantera gas phase object
        self.gas = gas                  
             
        #### CHEMICAL SPECIES DATA ####:
        self.wg = self.gas.molecular_weights     #Molar masses [kg/kmol]
        self.ng = self.gas.n_species - 1         #Number of gas species

        #### REACTOR GEOMETRIC PARAMETERS ####:
        
        #Initialize inputs:
        self._init_reactor_properties(**params)
        
        #### AUXILIARY PARAMETERS ####:
        
        #Initialize reactor inlet properties:
        self._u = 0.0
        self._mdot = 0.0
        
        #Initialize reacting surface
        self.rs = None
        
        #Pre-allocate mass fractions array
        self.Y = np.zeros(self.ng+1)
        
        #Array slicing position
        self.aux1 = 4+self.ng

    def _init_reactor_properties(self,**params):
        
        #Parse inputs:
        self.z_in = params.get('z_in',0.0)       #Reactor inlet axial position [m]
        self.z_out = params.get('z_out',0.01)    #Reactor outlet axial position [m]
        self.d = params.get('diam',0.0)          #Reactor diameter [m]
        self.length = (self.z_out - self.z_in)   #Reactor length [m]

        #Compute derived parameters:      
        self.ac = np.pi*self.d**2/4              #Reactor cross-sectional area [m**2]
        self.perim = np.pi*self.d                #Reactor perimeter [m]
        self.vol = self.length*self.ac           #Reactor total volume [m**3]
        self.ae = self.perim*self.length         #Reactor total external area [m**2]
        
        #Get support type (default is 'none', which means there is no porous
        #support and only the pipe walls are considered for calculation of mass
        #transfer coefficients and friction factor)
        self.support_type = params.get('support_type','none')
        
        #Support porosity (default value is 1.0) [-]
        self.porosity = params.get('porosity', 1.0)  
                        
        #Depending on the support type assign an integer flag
        if self.support_type == 'none':
            #If there is no support (only the pipe itself)        
            self.support_flag = 0
            
            #Characteristic support length scale [m]
            self.dc = params.get('diam', 0.0) 
            
            #Compute the geometric surface area [m**2]
            self.geo_area = self.perim*self.length
            
        elif self.support_type == 'honeycomb':         
            #If support is a honeycomb, flag = 1           
            self.support_flag = 1
      
        elif self.support_type == 'foam':
            #If support is a foam, flag = 2
            self.support_flag = 2
        
        #Check input data is correct if 'honeycomb' or 'foam' supports are used
        if self.support_flag == 1 or self.support_flag == 2:

            #Characteristic support length scale [m]
            self.dc = params.get('dc', None) 
    
            #Need to supply characteristic support diameter [m]         
            if self.dc  == None:                
                raise ValueError('Characteristic support diameter was not supplied')
            
            #Get the surface area per unit volume [m**-1]
            self.sp_area = params.get('sp_area', None) 
            
            #If surface area per unit volume is not provided
            if self.sp_area == None:
                #Assume reactor tube external area 
                self.sp_area = np.pi*self.dc*self.length/self.vol
            
            #Compute the geometric surface area [m**2]
            self.geo_area = self.sp_area*self.vol
            
        #Compute effective flow parameters
        self.ac_eff = self.ac*self.porosity      #Effective flow cross-sectional area [m**2]
        self.vol_eff = self.vol*self.porosity    #Effective flow volume [m**3]
        self.aei_eff = self.geo_area/self.length #Geometric area per unit length [m]
            
    def _add_reacting_surface(self,reacting_surface):
        
        #Add reacting surface object to reactor
        self.rs = reacting_surface
        
    def _sync_reacting_surface(self):
        
        #Sync thermodynamic state of the wall with the reactor
        if self.rs != None:
            #Sync all phases
            self.rs._sync_state()

    def eval_transport_props(self):
        
        #Compute molecular diffusion coefficients [m**2/s]
        self.diff_mol = self.gas.mix_diff_coeffs_mass
        
        #Compute superficial velocity [m/s]
        self.us = self.porosity*self.u
        
        #Compute Schimidt number
        self.Sc = self.gas.viscosity/(self.rho*self.diff_mol)

        #Compute Reynolds number (using char. length)
        self.Re = self.dc*self.rho*self.us/self.gas.viscosity

    def eval_friction_factor(self):

        if self.support_flag == 0:
            #Friction factor correlations for pipe flow
            if self.Re < 2320.0:
                #Laminar flow
                self.fric = 16.0/self.Re
            else:
                #Turbulent flow
                self.fric = 0.0791*self.Re**(-0.25)
                
        elif self.support_flag == 1: 
             #Friction factor correlations for honeycomb monolith
             self.fric = 14.3/self.Re
        
        elif self.support_flag == 2: 
             #Friction factor correlations for foam monolith
             self.fric = (0.87 + 13.65/self.Re)
             
    def update_state(self,y):
        
        #Get state variables:
        self.mdot = y[1]
        self.T = y[2]
        self.P = y[3]
        self.Y[:-1] = y[4:self.aux1]
        
        #Compute inert species mass fraction
        self.Y[-1] = 1.0 - np.sum(self.Y[:-1])
        
        #Update gas phase mass fractions
        self.gas.set_unnormalized_mass_fractions(self.Y)
        
        #Update gas phase to current temperature and pressure
        self.gas.TP = self.T, self.P
        
        #Current gas density [kg/m**3]
        self.rho = self.gas.density
        
        #Current flow velocity [m/s]
        self.u = self.mdot/(self.ac_eff*self.rho)
        
        #If there is a reacting wall, update its state
        if self.rs != None:
            
            #Update reacting wall coverages and thermo state
            self.rs.update_state(y)
        
    def update_rates(self,y):

        #### Net species production rates ###:            
        #Gas species from homogeneous reactions [kmol/m**3 s] 
        self.wdot = self.gas.net_production_rates
        
        #Rates in mass basis [kg/m**3 s]
        self.wdot_mass = self.wdot*self.wg
        
        #If there is a reacting wall, update its state
        if self.rs != None:
            
            #Update reacting wall production rates [kmol/m**2 s]
            self.rs.update_rates(y)
            
            #Rates in mass basis [kg/m**3 s]
            self.gdot_mass = self.rs.gdot*self.wg

    def update_derivatives(self,yd):
        
        #Update state derivatives:
        self.dt = yd[0]
        self.dm = yd[1]
        self.dT = yd[2]
        self.dP = yd[3]
        self.dY = yd[4:self.aux1]

    def eval_equations(self,z,y):
        
        #### UPDATE GAS PHASE THERMODYNAMIC STATE ####:
        self.update_state(y)

        #### UPDATE HOMOGENEOUS NET PRODUCTION RATES ####:
        self.update_rates(y)
        
        #### COMPUTE REACTOR BALANCE EQUATIONS ####:
                
        #Residence time:
        self.eval_rtime()

        #Continuity and species conservation:
        self.eval_mass_balances(z)
              
        #Evaluate energy equation
        self.eval_energy()     

        #Evaluate momentum equation
        self.eval_mom()
        
    def eval_as_dae(self,z,y,yd):
        
        #Evaluate reactor conservation equations
        self.eval_equations(z,y)
        
        #Update derivatives
        self.update_derivatives(yd)
                
        #Evaluate coverages balance residuals 
        self.rs.eval_rtheta()

        #Residual for residence time
        self.rest = self.dt - self.dtdz              

        #Residual for mass conservation
        self.resm = self.dm - self.dmdz 
        
        #Residuals for species conservation equations
        self.resY = self.dY -self.dYdz[:-1]  
        
        #Residual for energy balance
        self.resT = self.dT - self.dTdz

        #Residual for momentum balance
        self.resP = self.dP - self.dPdz

    #### REACTOR PROPERTIES ####:   
    
    #Get/set current flow velocity [m/s]
    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        #Set flow velocity
        self._u = value
        #Compute mass flow rate
        self._mdot = self._u*self.ac_eff*self.gas.density        
    
    #Get/set the mass flow rate [kg/s]    
    @property
    def mdot(self):       
        return self._mdot

    @mdot.setter
    def mdot(self, value):
        #Set mass flow rate
        self._mdot = value
        #Compute flow velocity
        self._u = self._mdot/(self.ac_eff*self.gas.density) 
    
    #Get/set current thermodynamic state (temperature,pressure,molar fractions)
    @property
    def TPX(self):
        return self.gas.T, self.gas.P, self.gas.X

    @TPX.setter
    def TPX(self, values):
        assert len(values) == 3, 'Incorrect number of input values'
        #Set gas phase thermodynamic state
        self.gas.TPX = values[0], values[1], values[2] 
        
        #Set surface and bulk phases (if there are any) to same temperature 
        #and pressure of the gas phase
        self._sync_reacting_surface()       
        
    #Get/set current thermodynamic state (temperature,pressure,mass fractions)
    @property
    def TPY(self):
        return self.gas.T, self.gas.P, self.gas.Y

    @TPY.setter
    def TPY(self, values):
        assert len(values) == 3, 'Incorrect number of input values'
        #Set gas phase thermodynamic state
        self.gas.TPY = values[0], values[1], values[2]  
        
        #Set surface and bulk phases (if there are any) to same temperature 
        #and pressure of the gas phase
        self._sync_reacting_surface()

    #Get/set current thermodynamic state (temperature,density,molar fractions)
    @property
    def TDX(self):
        return self.gas.T, self.gas.density, self.gas.X

    @TDX.setter
    def TDX(self, values):
        assert len(values) == 3, 'Incorrect number of input values'
        #Set gas phase thermodynamic state
        self.gas.TDX = values[0], values[1], values[2] 
        
        #Set surface and bulk phases (if there are any) to same temperature 
        #and pressure of the gas phase
        self._sync_reacting_surface()       

    #Get/set current thermodynamic state (temperature,density,mass fractions)
    @property
    def TDY(self):
        return self.gas.T, self.gas.density, self.gas.X

    @TDY.setter
    def TDY(self, values):
        assert len(values) == 3, 'Incorrect number of input values'
        #Set gas phase thermodynamic state
        self.gas.TDY = values[0], values[1], values[2] 
        
        #Set surface and bulk phases (if there are any) to same temperature 
        #and pressure of the gas phase
        self._sync_reacting_surface() 
        
class PlugFlowReactor(FlowReactorBase):
    '''########################################################################
    Class that defines a standard plug flow reactor (PFR) with homogeneous 
    and/or heterogeneous reactions.
    ########################################################################'''
    
    def __init__(self,gas,**params):
        #Call base reactor class constructor 
        super().__init__(gas,**params)
        
    def eval_rtime(self):  
        
        #Evaluate residence time
        self.dtdz = 1.0/self.u

    def eval_mass_balances(self,z):  
        
        #Evaluate continuity equation
        self.dmdz = 0.0

        #Gas-species conservation equation:
        self.dYdz = self.ac_eff*self.wdot_mass
        
        if self.rs != None:
            
            #Mass ablation/depletion term due to heterogeneous reactions
            self.m_source = self.rs.ai*np.sum(self.gdot_mass)
            
            #Evaluate continuity equation
            self.dmdz += self.m_source

            #Terms due to heterogeneous reactions
            self.dYdz += self.rs.ai*self.gdot_mass
            self.dYdz -= self.Y*self.m_source
            
            #If external mass transfer is consided
            if self.flag_ext_mt == 1:
            
                #Evaluate residuals for gas mass fractions 
                #adjacent to the surface 
                self.rs.eval_ys_residuals(z)
            
        self.dYdz /= self.mdot
        
    def eval_energy(self):   
        
        #Evaluate energy equation
        self.dTdz = 0.0
        
        #If energy equation is enabled
        if self.flag_energy == 1:
            
            #Gas-phase mixture specific heat [J/kg K]
            self.cp = self.gas.cp_mass             
            
            #Gas-phase species specific enthalpies [J/kg]
            self.hk = self.gas.partial_molar_enthalpies/self.wg
                            
            #Contributions from external heat flux
            if self.flag_ext_ht == 1:
                
                #External heat flux [W/m**2]
                self.qe = self.uht*(self.Tinf - self.T)
                
                #Contributions due to external heat-flux
                self.dTdz += self.aei_eff*self.qe

            #Enthalpic and kinetic contributions
            self.dTdz -= (self.mdot*np.dot(self.hk,self.dYdz) 
                          + self.u**2*self.dmdz)
                
            #If there is a reacting wall
            if self.rs != None:
                
                #Contributions due to heterogeneous reactions
                self.dTdz -= ( (np.dot(self.hk,self.Y) + 0.5*self.u**2)
                                *self.rs.ai*np.sum(self.gdot_mass) )
                
                #Contributions from solid bulk phase(s)                 
                if self.rs.bulk != None:
                
                    #Solid bulk pecies specific enthalpies [J/kg]
                    self.hk_b = np.array([o.enthalpy_mass 
                                          for o in self.rs.bulk])
                    
                    #Bulk species net production rates from 
                    #heterogeneous reactions [kmol/m**2 s]  
                    self.bdot = self.rs.surface.net_production_rates[self.rs.bulk_idx]
                    
                    #Contributions accumulation of enthalpy in the bulk solid
                    self.dTdz -= self.rs.ai*np.sum(self.bdot*self.hk_b*self.rs.wb)   
                
            self.dTdz /= self.cp*self.mdot
        
    def eval_mom(self):

        #Evaluate momentum equation
        self.dPdz = 0.0

        #If momentum equation is enabled
        if self.flag_momentum == 1:

            #Compute transport properties
            self.eval_transport_props()
    
            #Compute friction factor
            self.eval_friction_factor()
            
            #Wall friction
            self.tw = 0.5*self.aei_eff*self.rho*self.u**2*self.fric
                
            #Compute term due to heterogeneous reactions
            if self.rs != None:
                self.dPdz -= self.u*self.m_source
  
            #Compute wall friction term
            self.dPdz -= self.tw + self.u*self.dmdz
            
            self.dPdz /= self.ac_eff

class ScikitsODE(object):
    '''Class that defindes the PFR as a ODE system'''     
    def __init__(self,reactor):
        
        #Make reference to reactor object
        self._r = reactor

        #Array slicing pointers
        self.p1 = self._r.aux1
        
        #Only if there is a reacting wall
        if self._r.rs != None:
            self.p2 = self._r.rs.aux2
        
    def eval_ode(self,z,y,ydot):
        
        #Evaluate reactor conservation equations
        self._r.eval_equations(z,y)
        
        #Check if reactor has reacting walls
        if self._r.rs != None:
                       
            #Evaluate coverages w/ pseudo-velocity term
            self._r.rs.eval_theta()
            
            #Add coverages to output vector
            ydot[self.p1:self.p2] = self._r.rs.dthetadz

        #Output vector
        ydot[0] = self._r.dtdz
        ydot[1] = self._r.dmdz
        ydot[2] = self._r.dTdz
        ydot[3] = self._r.dPdz
        ydot[4:self.p1] = self._r.dYdz[:-1]

class ScikitsDAE(object):
    '''Class that defindes the PFR as a DAE system''' 
    def __init__(self,reactor):
        
        #Make reference to reactor object
        self._r = reactor    
        
        #Array slicing pointers
        self.p1 = self._r.aux1
        self.p2 = self._r.rs.aux2
        self.p3 = self._r.rs.aux3
                
    def eval_dae(self,z,y,yd,res):
        
        #Evaluate reactor conservation equations
        self._r.eval_as_dae(z,y,yd)
                         
        #Vector with residuals
        res[0] = self._r.rest
        res[1] = self._r.resm
        res[2] = self._r.resT
        res[3] = self._r.resP
        res[4:self.p1] = self._r.resY
        res[self.p1:self.p2] = self._r.rs.restheta

        #If external mass transfer is consided
        if self._r.flag_ext_mt == 1:
     
            #Add to output residuals
            res[self.p2:self.p3] = self._r.rs.resYs[:-1]
            
        #If detailed washcoat model is employed
        if self._r.rs.flag_int_mt == 2:  

            #Add to output residuals
            res[self.p3:] = np.hstack((self._r.rs.wc.resy_flat,
                                       self._r.rs.wc.rescovs_flat))            
                       
class AssimuloODE(Explicit_Problem):
    '''Class that defindes the PFR as a ODE system for ASSIMULO package'''     
    def __init__(self,reactor,t0, y0):
        
        #Make reference to reactor object
        self._r = reactor
    
        #Initial conditions
        self.t0 = t0
        self.y0 = y0
        
        #Create output array
        self.yout = np.zeros_like(self.y0)
        
        #Array slicing pointers
        self.p1 = self._r.aux1
        
        #Only if there is a reacting wall
        if self._r.rs != None:
            self.p2 = self._r.rs.aux2
        
    def rhs(self,z,y):
        
        #Evaluate reactor conservation equations
        self._r.eval_equations(z,y)
       
        #Check if reactor has reacting walls
        if self._r.rs != None:

            #Evaluate coverages w/ pseudo-velocity term
            self._r.rs.eval_theta()
            
            #Add coverages to output vector
            self.yout[self.p1:self.p2] = self._r.rs.dthetadz
                      
        #Output vector
        self.yout[0] = self._r.dtdz
        self.yout[1] = self._r.dmdz
        self.yout[2] = self._r.dTdz
        self.yout[3] = self._r.dPdz
        self.yout[4:self.p1] = self._r.dYdz[:-1]
            
        return self.yout
    
class AssimuloDAE(Implicit_Problem):
    '''Class that defindes the PFR as a ODE system for ASSIMULO package'''     
    def __init__(self,reactor,t0,y0,yd0):
        
        #Make reference to reactor object
        self._r = reactor
    
        #Initial conditions
        self.t0 = t0
        self.y0 = y0
        self.yd0 = yd0
    
        #Create output array
        self.rout = np.zeros_like(self.y0)
        
        #Array slicing pointers
        self.p1 = self._r.aux1
        self.p2 = self._r.rs.aux2
        self.p3 = self._r.rs.aux3

    def res(self, z, y, yd):
        
        #Evaluate reactor conservation equations
        self._r.eval_as_dae(z,y,yd)
                
        #Vector with residuals
        self.rout[0] = self._r.rest
        self.rout[1] = self._r.resm
        self.rout[2] = self._r.resT
        self.rout[3] = self._r.resP
        self.rout[4:self.p1] = self._r.resY
        self.rout[self.p1:self.p2] = self._r.rs.restheta

        #If external mass transfer is consided
        if self._r.flag_ext_mt == 1:

            #Add to output residuals
            self.rout[self.p2:self.p3] = self._r.rs.resYs[:-1]
            
        #If detailed washcoat model is employed
        if self._r.rs.flag_int_mt == 2:   

            #Add to output residuals
            self.rout[self.p3:] = np.hstack((self._r.rs.wc.resy_flat,
                                             self._r.rs.wc.rescovs_flat))                 
        return self.rout