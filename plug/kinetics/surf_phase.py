import numpy as np
from scikits.odes import ode

class SurfacePhase(object):
    def __init__(self,gas,surf,**p): 
        
        #Load Cantera solution objects
        self.gasCT = gas   
        self.surfCT = surf 
        self.bulkCT = p.get('bulk',None)  
                    
        #If a bulk phase exists
        if self.bulkCT != None:           
            #Turn bulk flag on
            self.flag_bulk = 1 
        else:
            #Turn bulk flag off
            self.flag_bulk = 0       
        
        #Check if a coverage enthalpy depedency file was given
        self.cov_file = p.get('cov_file',None)
        
        ##Initialize SurfaceKinetics class
        if self.flag_bulk == 1:
            #With a bulk phase
            self.kinetics = SurfaceKinetics(self.gasCT,self.surfCT,
                                            bulk = self.bulkCT, 
                                            cov_file = self.cov_file)           
        else:
            #Without a bulk phase
            self.kinetics = SurfaceKinetics(self.gasCT,self.surfCT,
                                            cov_file = self.cov_file)
        
        ### PHASE NAMES ###
        self.gas_name = self.gasCT.name
        self.surf_name = self.surfCT.name
                
        ### CHEMICAL SPECIES DATA ###: 
                
        #Occupancy number [-]   
        self._n_sites = np.array([self.surfCT.species(n).size 
                                 for n in range(self.surfCT.n_species)])
    
    #### GET/SET PROPERTIES ####
    
    #Get phase name
    @property
    def name(self): 
        return self.surf_name

    #Get number of surface species 
    @property
    def n_species(self): 
        return self.surfCT.n_species
    
    #Get number of reactions
    @property
    def n_reactions(self): 
        return self.surfCT.n_reactions
    
    #Get species names
    @property
    def species_names(self): 
        return self.surfCT.species_names
         
    #Get surface site sensity [kmol/m**2]
    @property
    def site_density(self): 
        return self.surfCT.site_density

    #Get the site occupancy numbers
    @property
    def n_sites(self): 
        return self._n_sites
    
    @property
    def molecular_weights(self): 
        return self.surfCT.molecular_weights

    #Get coverages values from Cantera object
    @property
    def coverages(self):    
        return self.surfCT.coverages

    @coverages.setter
    def coverages(self, values):        
        #Set surface coverages in Cantera object
        self.surfCT.coverages = values     

    @property    
    def equilibrium_constants(self):
  
        #Compute equilibrium constants
        self.kinetics.get_equilibrium_constants()

        return self.kinetics.kc

    @property    
    def forward_rate_constants(self):

        #Compute forward rate constants
        self.kinetics.get_forward_rate_constants()

        return self.kinetics.kf

    @property    
    def reverse_rate_constants(self):

        #Compute reverse rate constants
        self.kinetics.get_reverse_rate_constants()

        return self.kinetics.kr

    @property    
    def forward_rates_of_progress(self):

        #Compute net production rates
        self.kinetics.get_net_rates()

        return self.kinetics.qf

    @property    
    def reverse_rates_of_progress(self):

        #Compute net production rates
        self.kinetics.get_net_rates()

        return self.kinetics.qr

    @property    
    def net_rates_of_progress(self):

        #Compute net production rates
        self.kinetics.get_net_rates()

        return self.kinetics.qnet
    
    @property    
    def net_production_rates(self):

        #Compute net production rates
        self.kinetics.get_net_rates()

        return self.kinetics.sdot
    
    #Get current temperature
    @property
    def T(self): 
        return self.kinetics.T

    #Get current pressure
    @property
    def P(self): 
        return self.kinetics.P
    
    #Get/set current thermodynamic state (temperature,pressure)
    @property
    def TP(self):
        return self.kinetics.T, self.kinetics.P

    @TP.setter
    def TP(self, values):
        assert len(values) == 2, 'Incorrect number of input values'

        #Set thermodynamic state
        self.kinetics.update_state(values[0], values[1])
        
    #### CLASS METHODS ####
    
    def get_net_production_rates(self,phase):       
        #Check phase names
        if phase.name == self.gas_name:
            #If supplied phase is the gas phase
            phase_idx = self.kinetics.gas_idx
            
        elif phase.name == self.surf_name:
            #If supplied phase is the surface phase
            phase_idx = self.kinetics.surf_idx

        #Compute net production rates
        self.kinetics.get_net_rates()
        
        return self.kinetics.sdot[phase_idx]
    
    def set_unnormalized_coverages(self,values):        
        #Set unnormalized coverages values within Cantera object
        self.surfCT.set_unnormalized_coverages(values)
        
    def set_multiplier(self,*args):        
        #Set reaction rate multiplier
        self.kinetics.set_multiplier(*args)
    
    def species(self,*value):       
        #If no argument is given
        if len(value) == 0:
            #Species object list
            return self.surfCT.species()            
        else:
            #Species object specified by index
            return self.surfCT.species(value[0])

    def species_index(self,value):        
        #Get species index position within this phase
        return self.surfCT.species_index(value)  
    
    def reaction(self,value):
        #Return reaction object
        return self.surfCT.reaction(value)

    def reactions(self):
        #Return all reactions object
        return self.surfCT.reactions()
    
    def kinetics_species_index(self,value):        
        #Get species index position within this phase
        return self.surfCT.kinetics_species_index(value)  

    def reactant_stoich_coeffs(self):
        #The array of reactant stoichiometric coefficients
        return self.surfCT.reactant_stoich_coeffs()

    def product_stoich_coeffs(self):
        #The array of reactant stoichiometric coefficients
        return self.surfCT.product_stoich_coeffs()

    def net_stoich_coeffs(self):
        #The array of reactant stoichiometric coefficients
        nstc = (self.surfCT.product_stoich_coeffs()
                -self.surfCT.reactant_stoich_coeffs())
        
        return nstc
     
    def advance_coverages(self,value):        
        #Advance coverages for specified amount of time
        self.kinetics.advance_coverages(value)        
                     
#Surface kinetics data class
class SurfaceKinetics(object):
    def __init__(self,gas,surf,**p): 

        #Load Cantera solution objects
        self.gasCT = gas 
        self.surfCT = surf  
        self.bulkCT = p.get('bulk',None)  
        
        #If a Cantera solution object is not given
        if self.gasCT == None:      
            raise ValueError('Cantera gas solution object was not provided!')  
        else:

            #Get gas species index
            self.gas_idx = np.array(
                           [self.surfCT.kinetics_species_index(
                                   self.gasCT.species_names[n]) 
                            for n in range(self.gasCT.n_species)])  
    
            #Total number of species in kinetic mechanism
            self.n_species = self.gasCT.n_species
    
        #If a Cantera solution object is not given
        if self.surfCT == None:      
            raise ValueError('Cantera surface solution object was not provided!')  
        else:

            #Get surface species index
            self.surf_idx = np.array(
                           [self.surfCT.kinetics_species_index(
                                   self.surfCT.species_names[n]) 
                            for n in range(self.surfCT.n_species)])
            
            #Total number of species in kinetic mechanism
            self.n_species += self.surfCT.n_species
        
        #If a bulk phase exists
        if self.bulkCT != None:
            
            #Turn bulk flag on
            self.flag_bulk = 1   
            
            #Get bulk species index
            self.bulk_idx = np.array(
                           [self.surfCT.kinetics_species_index(
                                   self.bulkCT.species_names[n]) 
                            for n in range(self.bulkCT.n_species)]) 
    
            #Total number of species in kinetic mechanism
            self.n_species += self.bulkCT.n_species
            
        else:
            
            #Turn bulk flag off
            self.flag_bulk = 0     
                
        #Check if a coverage enthalpy depedency file was given
        self.cov_file = p.get('cov_file',None)
        
        if self.cov_file != None:
            
            #Set the enthanlpy-dependency flag on
            self.flag_enthalpy_covs = 1
            
            #Build coverage dependency matrix
            self.build_cov_matrix()
            
        else:
            #Set the enthanlpy-dependency flag off
            self.flag_enthalpy_covs = 0

        #Gas constant [J/kmol K]
        self.Rg = 8314.4621   
        
        #Reaction rate multiplier
        self.rm = 1.0
        
        #Get kinetics data from Cantera solution objects
        self.get_kinetics_data()
        
        #Initialize auxiliary terms for net rates calculation
        self.init_aux_terms()

        #Instantiate the solver to guess the initial coverages values
        self.solver = ode('cvode', 
                          self.dt_coverages,
                          atol = 1e-20,
                          rtol = 1e-05,
                          max_steps = 2000,
                          old_api = False) 
        
        #Set initial thermodynamic state (300 K, 1 atm)
        
        #Gas phase 
        self.gasCT.TP = 300.0, 101325.0
        
        if self.flag_bulk == 1:
            
            #Bulk phase
            self.bulkCT.TP = 300.0, 101325.0
        
        #Surface kinetics
        self.set_initial_state(300.0,101325.0)
               
    def get_kinetics_data(self):
        
        ### HETEROGENEOUS KINETICS DATA ###
        
        #Get reactions object
        self.R = self.surfCT.reactions() 
        
        #Get number of heterogeneous reactions        
        self.n_reactions = len(self.R)
        
        #Pre-allocate lists 
        self.rate_A = []
        self.rate_b = []
        self.Ea = []
        self.motz_wise = []
        self.reversible = []
        self.stick = []
        self.stick_mw = []
        self.rxn_cov_id = []
        self.rxn_cov_spc_idx = []
        self.rxn_cov_deps = []
        
        #Loop over reactions
        for i in range(self.n_reactions):
            #Get Arrhenius parameters:
            self.rate_A.append(self.R[i].rate.pre_exponential_factor)
            self.rate_b.append(self.R[i].rate.temperature_exponent)
            self.Ea.append(self.R[i].rate.activation_energy)
    
            #Get a boolean indicating whether to use the correction factor 
            #developed by Motz & Wise
            self.motz_wise.append(self.R[i].use_motz_wise_correction)
            
            #Get if reaction is reversible flags
            self.reversible.append(self.R[i].reversible == False)

            #Check for adsorption reactions (w/ sticking coefficient)
            if self.R[i].is_sticking_coefficient == True:
                
               #Get index for sticking reaction
               self.stick.append(i)
               
               #Molecular weight of sticking gas species
               self.stick_mw.append(
                       self.gasCT.molecular_weights[self.gasCT.species_index(
                                                    self.R[i].sticking_species)])
    
            #Check for reactions with coverage dependencies
            if np.any(self.R[i].coverage_deps):
                
               #Reaction index 
               self.rxn_cov_id.append(i)
               
               #Coverage-dependent species                  
               cov_spc = list(self.R[i].coverage_deps.keys())
               
               #Coverage-dependent species indexes
               self.rxn_cov_spc_idx.append(np.array([self.surfCT.species_index(cov_spc[n]) 
                                            for n in range(len(cov_spc))]))
               
               #Coverage-dependent values (ak,mk,Eak)                  
               self.rxn_cov_deps.append(np.array(list(self.R[i].coverage_deps.values())))

        #Convert lists to arrays
        self.rate_A = np.array(self.rate_A)
        self.rate_logA = np.log(self.rate_A)
        self.rate_b = np.array(self.rate_b)
        self.Ea = np.array(self.Ea)
        self.reversible = np.array(self.reversible)
        self.stick = np.array(self.stick) 
        self.stick_mw = np.array(self.stick_mw) 
        self.rxn_cov_id = np.array(self.rxn_cov_id) 
    
        #Get stoichiometric matrices: 
        self.stm_r = self.surfCT.reactant_stoich_coeffs() #Reagents matrix
        self.stm_p = self.surfCT.product_stoich_coeffs()  #Products matrix
        self.stm_n = self.stm_p - self.stm_r              #Net coeff. matrix
     
        #Stoichiometric matrix for surface species only:
        self.stm_r_s = self.stm_r[self.surf_idx,:]      #Surface reagents matrix
        self.stm_p_s = self.stm_p[self.surf_idx,:]      #Surface products matrix
        self.stm_n_s = self.stm_n[self.surf_idx,:]      #Surface net coeff. matrix
        
        #Occupancy number [-]   
        self.n_sites = np.array([self.surfCT.species(n).size 
                                 for n in range(self.surfCT.n_species)])
        
        #Occupancy number and site density term
        self.rsden = self.n_sites/self.surfCT.site_density
        
        #Expand occupancy number term
        nsites_expand = np.expand_dims(self.n_sites,axis=0).T 
        
        #If there are reactions with sticking coefficients
        if np.any(self.stick): 
            #Flag for reactions with sticking coefficient
            self.flag_stick_rxn = 1
            
            #Compute constant terms in rate expression 
            #with sticking coefficients
            prod_term = self.rate_A[self.stick]*np.prod(
                        nsites_expand**self.stm_r_s[:,self.stick],axis=0)
            
            exp_term = ( self.surfCT.site_density
                         **np.sum(self.stm_r_s[:,self.stick],axis=0) )
    
            #Forward rate constants with sticking coeff.   
            self.kf_stick = (prod_term/exp_term)
 
        else:
            #Flag for reactions with sticking coefficient
            self.flag_stick_rxn = 0
        
        #Compute constant terms in equilibrium rate expression 
        self.kc_sites_term = ( self.surfCT.site_density**np.sum(self.stm_n_s,axis=0)   
                               *np.prod(nsites_expand**-self.stm_n_s,axis=0) )   
            
        #Equilibrium constant term expoents:   
        self.kc_sites_term_exp = np.sum(self.stm_n[self.gas_idx,:],axis=0)
        
        #If there is a solid bulk phase as well
        if self.flag_bulk == 1:
            #Compute bulk phase term
            self.kc_bulk_term = ( (self.bulkCT.density/self.bulkCT.molecular_weights)
                                 **np.sum(self.stm_n[self.bulk_idx,:],axis=0) ) 
        else:
            self.kc_bulk_term = 1.0
        
        #If there are reactions with rate coverage dependencies
        if np.any(self.rxn_cov_id):
            
            #Flag for reactions with coverage dependency
            self.flag_covs_rxn = 1

            #Pre-allocate matrices
            self.rxn_cov_ak = np.zeros((len(self.rxn_cov_id),
                                        self.surfCT.n_species),dtype=float)
            self.rxn_cov_mk = np.zeros_like(self.rxn_cov_ak,dtype=float)
            self.rxn_cov_Ea = np.zeros_like(self.rxn_cov_ak,dtype=float)
            
            #Loop over coverage dependent reactions
            for i in range(len(self.rxn_cov_id)):

                #Current dependency parameters
                deps = self.rxn_cov_deps[i]

                #Populate matrices with respective values
                self.rxn_cov_ak[i,self.rxn_cov_spc_idx[i]] = deps[:,0]
                self.rxn_cov_mk[i,self.rxn_cov_spc_idx[i]] = deps[:,1]
                self.rxn_cov_Ea[i,self.rxn_cov_spc_idx[i]] = -deps[:,2]
                    
        else:
            #Flag for reactions with coverage dependency
            self.flag_covs_rxn = 0
           
    def build_cov_matrix(self):
        ''' ###################################################################
        Function that reads the contents of coverage-dependent enthalpies of 
        surface species and builds a coverage dependency matrix.
        An example of the file format is given below:
            
            CO(S)/SURFACE/2
            CO(S)/SURFACE/-36.35
            H(S)/SURFACE/-4.63
            --
            COOH(S)/SURFACE/2
            CO(S)/SURFACE/-15
            H(S)/SURFACE/-10
            --
        The first line of each block defines which species has a coverage
        -dependent enthalpy, and on how many species it is dependent on. The 
        following lines in each block define the species on which it is 
        dependent,  along with the alpha coefficients in units of KCAL/MOL, 
        which are converted to J/KMOL.
        ####################################################################'''    
        import re
        #Read text file line by line, skipping the first 5 lines (header),
        #and store content inside list
        with open(self.cov_file) as f:
            content = f.readlines()[5:]
            
        #Close the file
        f.close()
        
        #Current line to be processed
        current_line = 0
        
        #Surface species names
        species_names = self.surfCT.species_names

        #Create a matrix with enthalpy coverage dependencies coefficients 
        self.cov_matrix0 = np.zeros((self.surfCT.n_species,self.surfCT.n_species))
            
        #Loop until end of content
        while current_line < len(content):
            
            #Split current line
            line = re.split('/',content[current_line]) 
            
            #Number of coverages dependencies that affect current species
            n_spc = int(line[-1])
            
            #If species exists within the mechanism 
            if line[0] in species_names:
        
                #Affected species index
                rxn_cov_spc_idx = self.surfCT.species_index(line[0])

                #Buffers to store intermediate values
                cov_alpha = []
                cov_deps_idx = []
                
                #Loop over dependencies
                for i in range(current_line+1,current_line+1+n_spc):
                    
                    #Split current line
                    line_deps = re.split('/',content[i]) 
                    
                    #If current species is present within mechanism
                    if line_deps[0] in species_names:
                
                        #Append species names and coefficient values
                        cov_alpha.append(float(line_deps[-1]))
                        cov_deps_idx.append(self.surfCT.species_index(line_deps[0]))
                
                #Add values to coverage matrix
                self.cov_matrix0[rxn_cov_spc_idx,cov_deps_idx] = cov_alpha
            
            #Update current line number
            current_line += (n_spc + 2)
        
        #Convert from [kcal/mol] to [J/kmol]
        self.cov_matrix0 *= 4.186798e06
                     
    def init_aux_terms(self):
        
        #Get the contiguous 1-D arrays of the stoichiometric matrices for both
        #reactants and products, length will be: n_species*n_reactions
        self.stm_r1d = np.ravel(self.stm_r)
        self.stm_p1d = np.ravel(self.stm_p)
        
        #Find the index of non-zeros coefficients in both arrays
        self.idx_nonzero_r = np.where(self.stm_r1d>0.0)[0]
        self.idx_nonzero_p = np.where(self.stm_p1d>0.0)[0]
        
        #Get the species index (line positions) in the original stoichiometric
        #matrices
        self.idx_lines_r = np.where(self.stm_r>0.0)[0]
        self.idx_lines_p = np.where(self.stm_p>0.0)[0]
        
        #Find the index of coefficients larger than 1 for reactants and products
        self.idx_exps_r = np.where(self.stm_r1d>1.0)[0]
        self.idx_exps_p = np.where(self.stm_p1d>1.0)[0]
        
        #Create arrays of expoents larger than 1 for reactants and products
        self.exps_r = self.stm_r1d[self.idx_exps_r]
        self.exps_p = self.stm_p1d[self.idx_exps_p]
        
        #Create 1-D arrays for reactants and products concentration
        self.Ca_1d_r = np.ones(len(self.stm_r1d))
        self.Ca_1d_p = np.ones(len(self.stm_p1d))

    def set_multiplier(self,*args):
        
        #If extra arguments were given
        if len(args) == 2:
            #Create array the same size to number of reactions
            self.rm = np.ones(self.n_reactions)
            
            #Reaction index
            ridx = args[1]
            
            #Set multiplier for reaction specified by index
            self.rm[ridx] = args[0]
            
        elif len(args) == 1:
            
            #Set all reaction rates of progress multiplier factor
            self.rm = args[0] 
            
    def get_thermo_data(self):

        #Get standard-state enthalpies for the gas phase 
        self.hg = self.gasCT.standard_enthalpies_RT
        self.hs = self.surfCT.standard_enthalpies_RT

        #Get standard-state entropies for both gas and surface phases
        self.sg = self.gasCT.standard_entropies_R
        self.ss = self.surfCT.standard_entropies_R
        
        #Get standard-state enthalpies for surface phase.
        #Check if enthalpies are affected by coverages:
        if self.flag_enthalpy_covs == 1:    

            #Modify surface species enthalpy according to surface coverages 
            self.hs -= np.sum(self.cov_matrix0*self.Ta
                              *self.surfCT.coverages,axis=1)            
        
        #If there is a bulk phase
        if self.flag_bulk == 1:
            
            #Get bulk phase standard-state enthalpies and entropies
            self.hb = self.bulkCT.standard_enthalpies_RT
            self.sb = self.bulkCT.standard_entropies_R
            
            #Concatenate values
            self.h_all = np.hstack((self.hg, self.hb, self.hs))
            self.s_all = np.hstack((self.sg, self.sb, self.ss))
            
        else:
            #Concatenate values
            self.h_all = np.hstack((self.hg, self.hs))
            self.s_all = np.hstack((self.sg, self.ss))

    def get_concentrations(self):
        
        #Update concentrations
        if self.flag_bulk == 1:
            
            self.Ca = np.hstack((self.gasCT.concentrations, 
                                 self.bulkCT.concentrations,
                                 self.surfCT.concentrations))
        else:
            self.Ca = np.hstack((self.gasCT.concentrations, 
                                 self.surfCT.concentrations))  
            
    def get_forward_rate_constants(self):     

        #Pre-compute term within exponential
        arrhenius = self.rate_logA + self.rate_b*np.log(self.T) - self.Ea*self.Ta
        
        #Forward rate constants from modified Arrhenius: 
        self.kf = np.exp(arrhenius)
        
        #Check if there are surface reactions with sticking coefficients
        if self.flag_stick_rxn == 1: 
            
            #Forward rate constants with sticking coeff.   
            self.kf[self.stick] = ( self.kf_stick
                                  *np.sqrt((self.Rg*self.T/(2*np.pi*self.stick_mw)))
                                  *np.exp(-self.Ea[self.stick]*self.Ta) )   
            
        #Compute activation energy coverage effects
        self.get_rate_cov_effects()
            
    def get_equilibrium_constants(self):
     
        #Compute delta Gibbs
        self.delta_gibbs = np.inner(self.stm_n.T,(self.s_all-self.h_all))

        #Compute equilibrium constant Kp
        self.kp = np.exp(self.delta_gibbs)
        
        #Compute equilibrium constant Kc   
        self.kc =( self.kp*(self.P*self.Ta)**self.kc_sites_term_exp
                  *self.kc_sites_term*self.kc_bulk_term )
                      
    def get_reverse_rate_constants(self):
        
        #Reverse rate constant from thermodynamics
        self.kr = self.kf/self.kc
        
        #Set reverse rate constant of reactions identified 
        #by irreversibility flag to zero
        self.kr[self.reversible] = 0.0

    def get_rate_cov_effects(self):
        
        #Compute terms only if flag is on
        if self.flag_covs_rxn == 1:
            
            #Forward rate constants from modified Arrhenius:
            self.kf_cov = ( self.rate_A[self.rxn_cov_id]
                        *(self.T**self.rate_b[self.rxn_cov_id])
                        *np.exp(-self.Ea[self.rxn_cov_id]*self.Ta) ) 
            
            #Evaluate reactions with rates dependent on surface coverages
            self.cov_term = ( np.prod(self.surfCT.coverages**self.rxn_cov_mk,axis=1)
                       *10.0**(np.inner(self.rxn_cov_ak,self.surfCT.coverages))
                       *np.exp(np.inner(self.rxn_cov_Ea*self.Ta,self.surfCT.coverages)) )
            
            #Modified forward rate constant
            self.kf_cov *= self.cov_term
    
            #Recompute forward and rate constants
            self.kf[self.rxn_cov_id] = self.kf_cov 

    def get_net_rates(self): 
        
        #Broadcast concentration values to the appropriate positions in the
        #contiguous 1D reactants and products arrays
        self.Ca_1d_r[self.idx_nonzero_r] = self.Ca[self.idx_lines_r]
        self.Ca_1d_p[self.idx_nonzero_p] = self.Ca[self.idx_lines_p]
        
        #Get the power of required values for both arrays
        self.Ca_1d_r[self.idx_exps_r] **= self.exps_r
        self.Ca_1d_p[self.idx_exps_p] **= self.exps_p

        #Forward rates of progress [kmol/m**2 s]
        self.qf = self.rm*self.kf*np.reshape(self.Ca_1d_r,(self.n_species,
                                             self.n_reactions)).prod(axis=0)
        
        #Reverse rates of progress [kmol/m**2 s]
        self.qr = self.rm*self.kr*np.reshape(self.Ca_1d_p,(self.n_species,
                                             self.n_reactions)).prod(axis=0)
        
        #Net rates of progress [kmol/m**2 s]
        self.qnet = self.qf - self.qr
        
        #Net production rates dut to heterogeneous reactions [kmol/m**2 s]
        self.sdot = np.inner(self.stm_n,self.qnet)
        
    def set_initial_state(self,T,P):
        
        #Set all Cantera objects to current thermostate
        self.surfCT.TP = T, P
                        
        #Current temperature and pressure
        self.T, self.P = T, P
                 
        #Inverse temperature and gas constant term
        self.Ta = 1.0/(self.Rg*self.T)
        
        #Get thermo data
        self.get_thermo_data()
        
        #Compute forward rate constants kf
        self.get_forward_rate_constants()
           
        #Compute equilibrium constants kc
        self.get_equilibrium_constants()
        
        #Compute reverse rate constants kr
        self.get_reverse_rate_constants()
        
        #Compute species concentrations
        self.get_concentrations()
        
        #Create temperature and pressure cache
        self.Tcache, self.Pcache = self.T, self.P
        
    def update_state(self,T,P):
        
        #If temperature is same as cached
        if (T != self.Tcache or P != self.Pcache):
            
            #Set new initial state at updated temperature
            self.set_initial_state(T,P)   

        else:
            
            #Compute coverage terms 
            self.get_rate_cov_effects()
            
            #If surface enthalpies are affected by coverages
            if self.flag_enthalpy_covs == 1:
                
                #Re-compute thermo data
                self.get_thermo_data()
                
                #Re-compute equilibrium constants
                self.get_equilibrium_constants()
                    
                #Compute reverse rate constants
                self.get_reverse_rate_constants()
            
            #Compute concentrations
            self.get_concentrations()
    
    def update_cov_rates(self):
        
        #Compute coverage-dependent forward rate constant terms
        self.get_rate_cov_effects()
                
        #If surface enthalpies are affected by coverages
        if self.flag_enthalpy_covs == 1:
            
            #Re-compute thermo data
            self.get_thermo_data()
            
            #Re-compute equilibrium constants
            self.get_equilibrium_constants()
                
            #Compute reverse rate constants
            self.get_reverse_rate_constants()
            
        else:
            
            #Compute reverse rate constants
            self.get_reverse_rate_constants()
            
        #Compute concentrations
        self.get_concentrations()

        #Compute net species production rates        
        self.get_net_rates() 
     
    def dt_coverages(self,t,y,ydot):
        '''Solve for Ns - 1 surface species. The equation corresponding for the
        largest initial coverage value is substituted by constraint to ensure
        the sum of all coverages remain equal to one. '''
        
        #Substitute largest coverage value by constraint
        coverages = np.insert(y,self.covmax_idx,1.0-np.sum(y))
        
        #Update coverages values
        self.surfCT.set_unnormalized_coverages(coverages)
        
        #Update reaction rates
        self.update_cov_rates()
        
        #Surface reaction rates
        self.srates = self.sdot[self.surf_idx][self.cov_idx]
        
        #Compute the coverage variation with time
        ydot[:] = self.srates*self.rsden[self.cov_idx]

    def advance_coverages(self,tf):      
        
        #Get index of largest coverage value
        self.covmax_idx = np.argmax(self.surfCT.coverages)
        
        #Get boolean array with all coverages except the largest one
        self.cov_idx = self.surfCT.coverages!=self.surfCT.coverages[self.covmax_idx]
         
        #Integrate the transient coverage ODEs. 
        self.sol_covs = self.solver.solve([0.0,tf], self.surfCT.coverages[self.cov_idx])
        
        #If an error occurs during integration
        if self.sol_covs.errors.t != None:
            raise ValueError(self.sol_covs.message,self.sol_covs.errors.t)
    
        #Get solution vector
        y = self.sol_covs.values.y[-1,:]
        
        #Insert sum constraint into the array
        y = np.insert(y,self.covmax_idx,1.0-np.sum(y))
            
        #Update coverages within the Cantera surface object
        self.surfCT.set_unnormalized_coverages(y)
        
