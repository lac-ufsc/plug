import numpy as np
from plug.reactor.flow_reactor import ScikitsODE, ScikitsDAE, AssimuloODE, AssimuloDAE
        
class ReactorSolverBase(object):
    '''########################################################################
    Base class for the PFR solver. Two implementations can be used, the one by
    Assimulo (https://jmodelica.org/assimulo/) and the Scikits.ODES 
    (https://github.com/bmcage/odes) . Both packages offer Python wrappers 
    for the solvers defined in the Sundials library 
    (https://computation.llnl.gov/projects/sundials). Input parameters defined
    in the base class are common for both implementations. Note that CVODE is 
    used for ODE problems and IDA is used for DAE problems.
    -----------------
    Input parameters:
    -----------------
        reactor: reference to FlowReactor object
        solver_type: type of numerical solver employed, options are 'ode' if
                     problem involves only ordinary differencial equations or
                     'dae' if problem involves differential algebraic equations
        tcovs:
        psv:
        implementation: which package is to be employed, 'assimulo' or 'scikits'
        
        #Solver parameters common for both 'assimulo' and 'scikits'. Values in
        #parentheses denote default values.
        
        #Parameters valid for both CVODE and IDA:
        grid: number of solution output grid points. (100)
        atol: absolute solver tolerance (1e-08)
        rtol: relative solver tolerance (1e-06)
        max_steps: maximum number of integration steps (5000)
        max_step_size: maximum integration step size (0)
        first_step_size: first integration step size (0)
        order: maximal order that is be used by the solver (5)
        
        #Parameters valid for CVODE only:
        min_step_size: minimum integration step size (0)
        max_conv_fails: maximum number of convergence failures (10)
        max_nonlin_iters: maximum number of nonlinear iterations (5)
        bdf_stability: stability limit detection for BDF (False) 
        
        #Parameters valid for IDA only:
        exclude_algvar_from_error: exclude algebraic variables from local error 
                                   calculation (False)           
    ########################################################################'''
    
    def __init__(self,reactor,**params):
        
        #Make reference to reactor object class
        self._r = reactor
        
        #Type of solver used ['ode' or 'dae']
        self.solver_type = params.get('solver_type','ode')
           
        #### SOLVER PARAMETERS ####:
        #Parameters valid for both CVODE and IDA:
        self.grid = params.get('grid',100)     #Number of solution output grid points   
        self.atol = params.get('atol',1e-08)   #Absolute tolerance
        self.rtol = params.get('rtol',1e-06)   #Relative tolerance
        self.order = params.get('order',5)            #BDF order (1 to 5)
        self.max_steps = params.get('max_steps',5000) #Maximum number of integration steps
        self.first_step_size = params.get('first_step_size',0.0) #Sets the first step size
        self.max_step_size = params.get('max_step_size',0.0)      #Sets the maximum step size
        
        #Parameters valid for CVODE only:
        self.min_step_size = params.get('min_step_size',0.0)      #Sets the minimum step size 
        self.max_conv_fails = params.get('max_conv_fails',10)     #Maximum number of convergence failures
        self.max_nonlin_iters = params.get('max_nonlin_iters',5)  #Maximum number of nonlinear iterations
        self.bdf_stability = params.get('bdf_stability',False)    #Stability limit detection for BDF
        
        #Parameters valid for IDA only:
        #Exclude algebraic variables from local error calculation
        self.exclude_algvar_from_error = params.get('exclude_algvar_from_error',False)
        
        #Only used when solving PFR with heterogeneous reactions
        self.tcovs = params.get('tcovs',1e02)  #Final integration time for coverages [s]
        self.psv = params.get('psv',None)      #Pseudo-velocity for surface species [m/s]
        
        #Set pseudo-velocity if supplied
        if self.psv != None:
            self._r.pseudo_velocity = self.psv
            
        #Universal gas constant [J/kmol K]
        self._rg = 8314.4621
        
    def init_conditions(self):
        
        #Reactor axial coordinates [m]
        self._z = np.linspace(self._r.z_in, self._r.z_out, self.grid)
           
        #Initial conditions vector
        self.y0 = np.hstack((0.0, self._r.mdot, self._r.gas.T, 
                             self._r.gas.P, self._r.gas.Y[:-1])) 
        
        #If reactor has surface reactions
        if self._r.rs != None:

            #Advance coverages in time to pseudo-steady-state
            if self.tcovs != 0.0:
                self._r.rs.surface.advance_coverages(self.tcovs)
            
            #Get maximum coverage index
            self._r.rs.get_max_coverage()
            
            #Get coverage array pointers
            self._cov_idx = self._r.rs._cov_index
            self._covmax_idx = self._r.rs._covmax_index
            
            #Initial conditions vector
            self.y0 = np.append(self.y0,
                                self._r.rs.surface.coverages[self._cov_idx])   
            
            #If external mass transfer effects are considered
            if self._r.flag_ext_mt == 1:
                
                #Append initial surface-adjacent gas mass fractions
                self.y0 = np.append(self.y0,self._r.gas.Y[:-1])
            
            #If internal mass transfer effects are considered
            if self._r.rs.flag_int_mt == 2:

                #Solve transient reaction-diffusion problem to get an initial
                #guess for the washcoat composition
                self._r.rs.wc.solve_initial_guess(1e-03,self._r.gas.Y,
                                                  self._r.rs.surface.coverages)
                
                #Append values to initial guess vector
                self.y0 = np.append(self.y0,self._r.rs.wc.guess)

    def get_slice_pointers(self):
        
        #Get array slicing pointers
        if self.solver_type == 'ode':
            #Array slicing pointers for ODE problem
            self.p1 = self._rode.p1
            
            if self._r.rs != None:                
                self.p2 = self._rode.p2
                
        elif self.solver_type == 'dae':
            #Array slicing pointers for DAE problem
            self.p1 = self._rdae.p1
            self.p2 = self._rdae.p2
            self.p3 = self._rdae.p3
             
    def sort_solution_values(self,vals):
        
        #Get array slicing pointers
        self.get_slice_pointers()
                  
        #Get solution variables 
        self._rtime = vals[:,0]      #Residence time [s]
        self._mdot = vals[:,1]       #Mass flow rate [kg/s]
        self._T = vals[:,2]          #Temperature [K]
        self._P = vals[:,3]          #Pressure [Pa]
            
        #Bulk gas mass fractions [-]:  
        self._Y = np.zeros((self.grid,self._r.ng+1))      
        self._Y[:,:-1] = vals[:,4:self.p1]        
        self._Y[:,-1] = 1.0 - np.sum(self._Y[:,:-1],axis=1) 

        #Get gas mixture mean molecular weight [kg/kmol]
        self._wg_mix = 1.0/np.sum(self._Y/self._r.wg,axis=1)

        #Gas gas mixture density [kg/m**3]
        self._rho = self._P*self._wg_mix/(self._rg*self._T)
        
        #Get gas molar fractions [-]
        self._X = (self._Y.T*self._wg_mix).T/self._r.wg
        
        #Axial velocity [m/s]
        self._u = self._mdot/(self._r.ac*self._rho)
            
        #Get surface solution variables if they exist
        if self._r.rs != None:
            
            #Surface coverages [-]
            self._covs = np.zeros((self.grid,self._r.rs.ns+1))
            self._covs[:,self._cov_idx] = vals[:,self.p1:self.p2]
            self._covs[:,self._covmax_idx] = ( 1.0 - np.sum(self._covs
                                                  [:,self._cov_idx],axis=1) ) 

            #If external mass transfer effects are considered
            if self._r.flag_ext_mt == 1:
                
                #Surface gas mass fractions [-]:  
                self._Ys = np.zeros((self.grid,self._r.ng+1))                             
                self._Ys[:,:-1] = vals[:,self.p2:self.p3]     
                self._Ys[:,-1] = 1.0 - np.sum(self._Ys[:,:-1],axis=1) 
            
            else:  
                #Surface gas mass fractions equal to bulk [-]:  
                self._Ys = self._Y
                   
            #Get gas mixture mean molecular weight [kg/kmol]
            self._wg_mix_s = 1.0/np.sum(self._Ys/self._r.wg,axis=1)
    
            #Gas gas mixture density [kg/m**3]
            self._rho_s = self._P*self._wg_mix_s/(self._rg*self._T)
            
            #Get gas molar fractions [-]
            self._Xs = (self._Ys.T*self._wg_mix_s).T/self._r.wg
                
            #If internal mass transfer effects are considered
            if self._r.rs.flag_int_mt == 2:

                #Washcoat solution values
                wc_values = vals[:,self.p3:] 
                
                #Washcoat gas species mass fractions [-]
                self._Ywc = np.zeros((self.grid,self._r.rs.wc.ngrid,self._r.ng+1))
                self._Ywc[:,:,:-1] = np.reshape(
                                     wc_values[:,:self._r.ng*self._r.rs.wc.ngrid],
                                     (self.grid,self._r.rs.wc.ngrid,self._r.ng))    
                self._Ywc[:,:,-1] = 1.0 - np.sum(self._Ywc[:,:,:-1],axis=2)

                #Get mixture mean molecular weight [kg/kmol]
                self._wg_mix_wc = 1.0/np.sum(self._Ywc/self._r.wg,axis=2)
                 
                #Get molar fractions [-]
                self._Xwc = (self._Ywc.T*self._wg_mix_wc.T).T/self._r.wg
                
                #Washcoat surface coverages [-]
                self._covs_wc = np.zeros((self.grid,self._r.rs.wc.ngrid,self._r.rs.ns+1))
                self._covs_wc[:,:,self._cov_idx] = np.reshape(
                                           wc_values[:,self._r.ng*self._r.rs.wc.ngrid:],
                                          (self.grid,self._r.rs.wc.ngrid,self._r.rs.ns))    
                self._covs_wc[:,:,self._covmax_idx] = ( 1.0 - 
                                       np.sum(self._covs_wc[:,:,self._cov_idx],axis=2) )
                
            else:
                #Set values washcoat variables to None
                self._Ywc = None
                self._wg_mix_wc  = None
                self._Xwc = None
                self._covs_wc = None
                
                
                
        #Store results into dict
        self._output = dict()
        
        #Gas-phase data
        self._output['z'] = self._z
        self._output['rtime'] = self._rtime
        self._output['mdot'] = self._mdot
        self._output['u'] = self._u
        self._output['rho'] = self._rho
        self._output['wg_mix'] = self._wg_mix
        self._output['T'] = self._T
        self._output['P'] = self._P
        self._output['X'] = self._X
        self._output['Y'] = self._Y
        
        #Surface-phase data
        if self._r.rs != None:
            self._output['coverages'] = self._covs
            self._output['rho_s'] = self._rho_s
            self._output['wg_mix_s'] = self._wg_mix_s
            self._output['Xs'] = self._Xs
            self._output['Ys'] = self._Ys
            #Washcoat data (None if detailed model is not employed)
            self._output['coverages_wc'] = self._covs_wc
            self._output['wg_mix_wc'] = self._wg_mix_wc
            self._output['Xwc'] = self._Xwc
            self._output['Ywc'] = self._Ywc            
            #Internal mass transfer data: Thiele modulus and effectiveness
            #factor.
            if self._r.rs.flag_int_mt != 0:
                #If detailed washcoat model is employed, compute internal mass 
                #transfer parameters using gas-solid interface properties 
                if self._r.rs.flag_int_mt == 2:
                    #Set gas phase object composition to that of gas-solid interface
                    self._r.rs.wc._gas.set_unnormalized_mass_fractions(
                                                         self._r.rs.wc.y[0,:])
                    #Compute the Thiele modulus and effectiveness factor
                    self._r.rs.wc.get_effectiveness_factor(self._r.rs.wc._gas,
                                                           self._r.rs.wc.gdot[0,:])  
                    
                #Save internal mass transfe variables into dict
                self._output['thiele_mod'] = self._r.rs.wc.thiele_mod
                self._output['neff'] = self._r.rs.wc.neff
                
'''Reactor solver class using scikits.odes wrapper for SUNDIALS'''
import scikits.odes.sundials.ida as ida
import scikits.odes.sundials.cvode as cvode

#Make it a subclass of ReactorSolver  
class ReactorSolverScikits(ReactorSolverBase):
    
    def __init__(self,reactor,**params):
        
        #Initialize base solver class
        ReactorSolverBase.__init__(self,reactor,**params)
               
        #### SOLVER PARAMETERS ####:  
        self.iter_type = params.get('iter_type','NEWTON')  #Iteration method (NEWTON or 'FUNCTIONAL')
        self.lmm_type = params.get('lmm_type','BDF')       #Discretization method (BDF or ADAMS)
        
        #Safety factor in the nonlinear convergence test
        self.nonlin_conv_coef = params.get('nonlin_conv_coef',0.33)
        
        #Set up the linear solver from the following list:
        #  ['dense' (= default), 'lapackdense', 'band', 'lapackband', 
        #   'spgmr', 'spbcg', 'sptfqmr']
        self.linsolver = params.get('linsolver','dense')
        
    def solve_ode(self):
        
        #Get initial conditions vector
        self.init_conditions()
        
        #Instantiate reactor as ODE class
        self._rode = ScikitsODE(self._r)
        
        #Instantiate the solver
        self.solver = cvode.CVODE(self._rode.eval_ode,
                                  atol = self.atol,
                                  rtol = self.rtol,
                                  order = self.order,
                                  max_steps = self.max_steps,
                                  lmm_type = self.lmm_type,
                                  iter_type = self.iter_type,
                                  linsolver = self.linsolver,
                                  first_step_size = self.first_step_size,
                                  min_step_size = self.min_step_size,
                                  max_step_size = self.max_step_size,
                                  max_conv_fails = self.max_conv_fails,
                                  max_nonlin_iters = self.max_nonlin_iters,
                                  bdf_stability_detection = self.bdf_stability,
                                  old_api = False) 
        
        #Compute solution and return it along supplied axial coords.
        self.sol = self.solver.solve(self._z, self.y0)
        
        #If an error occurs
        if self.sol.errors.t != None:
            raise ValueError(self.sol.message,self.sol.errors.t)
    
        #Get solution vector
        self.values = self.sol.values.y

    def solve_dae(self):
        
        #Get initial conditions vector
        self.init_conditions()

        #Instantiate reactor as DAE class
        self._rdae = ScikitsDAE(self._r)
                
        #Initial derivatives vector
        self.yd0 = np.zeros(len(self.y0))
        
        #Get list of differential and algebraic variables
        varlist = np.ones(len(self.y0))
           
        #Set algebraic variables
        varlist[self._rdae.p1:] = 0.0  
        
        #Instantiate the solver
        self.solver = ida.IDA(self._rdae.eval_dae, 
                              user_data = self._r,
                              atol = self.atol,
                              rtol = self.rtol,
                              order = self.order,
                              max_steps = self.max_steps,
                              linsolver = self.linsolver,
                              first_step_size = self.first_step_size,
                              max_step_size = self.max_step_size,
                              compute_initcond='yp0',
                              compute_initcond_t0 = 1e-06,
                              algebraic_vars_idx = np.where(varlist==0.0)[0],
                              exclude_algvar_from_error = self.exclude_algvar_from_error,
                              old_api=False)
        
        #Compute solution and return it along supplied axial coords.
        self.sol = self.solver.solve(self._z, self.y0, self.yd0)
        
        #If an error occurs
        if self.sol.errors.t != None:
            raise ValueError(self.sol.message,self.sol.errors.t)
    
        #Get solution vector
        self.values = self.sol.values.y
                      
'''Reactor solver class using ASSIMULO wrapper for SUNDIALS'''
from assimulo.solvers import CVode, IDA

#Make it a subclass of ReactorSolver  
class ReactorSolverAssimulo(ReactorSolverBase):

    def __init__(self,reactor,**params):
        
        #Initialize base solver class
        ReactorSolverBase.__init__(self,reactor,**params)
        
        #### SOLVER PARAMETERS ####:
        self.iter = params.get('iter','Newton') #Iteration method (Newton or FixedPoint)
        self.discr = params.get('discr','BDF')  #Discretization method (BDF or Adams)
        
        #Set up the linear solver from the following list:
        #  ['DENSE'(= default) or 'SPGMR']
        self.linear_solver  = params.get('linear_solver','DENSE')
        
        #Boolean value to turn OFF Sundials LineSearch when calculating 
        #initial conditions (IDA only)
        self.lsoff = params.get('lsoff',False)
        
    def solve_ode(self):
        
        #Get initial conditions vector
        self.init_conditions()

        #Instantiate reactor as ODE class
        self._rode = AssimuloODE(self._r,self._r.z_in,self.y0)  
     
        #Define the CVode solver
        self.solver = CVode(self._rode) 
        
        #Solver parameters
        self.solver.atol = self.atol         
        self.solver.rtol = self.rtol         
        self.solver.iter = self.iter
        self.solver.discr = self.discr
        self.solver.linear_solver = self.linear_solver
        self.solver.maxord = self.order
        self.solver.maxsteps = self.max_steps
        self.solver.inith = self.first_step_size
        self.solver.minh = self.min_step_size
        self.solver.maxh = self.max_step_size
        self.solver.maxncf = self.max_conv_fails
        self.solver.maxnef = self.max_nonlin_iters
        self.solver.stablimdet = self.bdf_stability
        self.solver.verbosity = 50

        #Solve the equations
        z, self.values = self.solver.simulate(self._r.z_out,self.grid-1) 
        
        #Convert axial coordinates list to array
        self._z = np.array(z)

    def solve_dae(self):
        
        #Get initial conditions vector
        self.init_conditions()
        
        #Initial derivatives vector
        self.yd0 = np.zeros(len(self.y0))

        #Instantiate reactor as DAE class
        self._rdae = AssimuloDAE(self._r,self._r.z_in,
                                       self.y0,self.yd0)
        
        #Get list of differential and algebraic variables
        varlist = np.ones(len(self.y0))

        #Set algebraic variables
        varlist[self._rdae.p1:] = 0.0   
               
        #Set up the solver and its parameters
        self.solver = IDA(self._rdae)
        self.solver.atol = self.atol         
        self.solver.rtol = self.rtol         
        self.solver.linear_solver = self.linear_solver
        self.solver.maxord = self.order
        self.solver.maxsteps = self.max_steps
        self.solver.inith = self.first_step_size
        self.solver.maxh = self.max_step_size
        self.solver.algvar = varlist
        self.solver.make_consistent('IDA_YA_YDP_INIT')
        self.solver.tout1 = 1e-06
        self.solver.suppress_alg = self.exclude_algvar_from_error
        self.solver.lsoff = self.lsoff
        self.solver.verbosity = 50

        #Solve the DAE equation system
        z, self.values, self.valuesd = self.solver.simulate(self._r.z_out,
                                                            self.grid-1)    
        
        #Convert axial coordinates list to array
        self._z = np.array(z)
        
#ReactorSolver class to be instantiated in the simulations
class ReactorSolver(object):
    
    def __init__(self,reactor,**params):

        #Type of solver implementation used ['scikits' or 'assimulo']
        self.implementation = params.get('implementation','scikits')
        
        #Check the implementation and instantiate subclass
        if self.implementation == 'scikits':
            
            #Scikits.odes implementation
            self.solver = ReactorSolverScikits(reactor,**params)
            
        elif self.implementation == 'assimulo':
            #Assimulo implementation
            self.solver = ReactorSolverAssimulo(reactor,**params)
            
    def solve(self):
        
        #Call the solver function depending on type of solver
        if self.solver.solver_type == 'ode':
            
            #Call the CVODE solver
            self.solver.solve_ode()
            
        elif self.solver.solver_type == 'dae':
            
            #Call the IDA solver
            self.solver.solve_dae()
            
        else:
            raise ValueError('Invalid solver type')

        #Sort solution values accordingly
        self.solver.sort_solution_values(self.solver.values)
        
        #Collect solution variables into class
        self.collect_results()
                              
    def solve_trange(self,trange,state,**kwargs):
        
        #Parse keywords arguments
        self.mdot_in = kwargs.get('mdot_in',[])
        self.u_in = kwargs.get('u_in',[])
        self.sa = kwargs.get('sa',None)

        #Temperature range (convert to list)
        self.trange = trange
        if not isinstance(self.trange,list):
            self.trange = list(self.trange)

        #Other state variables (pressure,composition)
        self.state_p = state[0]
        self.state_comp = state[1]
        
        #Initialize flags
        self.flag_mdot_in = False
        self.flag_u_in = False
        self.flag_sa = False

        #Test whether a mass flow rate or a velocity was specified
        if np.asarray(self.mdot_in).size != 0:
            
            #Turn flag single for single input value
            self.flag_mdot_in = True
            
            #Check if input is a number
            if isinstance(self.mdot_in,(int,float)):
                #Convert number to list
                self.mdot_in = list(np.repeat(self.mdot_in,len(self.trange)))    
            else:
                #Convert array to list
                self.mdot_in = list(self.mdot_in)

        if np.asarray(self.u_in).size != 0:      

            #Turn flag single for single input value
            self.flag_u_in = True
                
            #Check if input is a number
            if isinstance(self.u_in,(int,float)):
                #Convert number to list
                self.u_in = list(np.repeat(self.u_in,len(self.trange)))
            else:
                #Convert array to list
                self.u_in = list(self.u_in)

        #Perform brute force sensitivity analysis. Perturb each surface 
        #reaction by same factor and compute the normalized sensitivity 
        #coefficient for a user-defined gas-phase reactant species.
        if self.sa != None:
                
            #Unpack input. Must be a tuple in the form: (factor,species name)
            self.factor, name = self.sa
            
            #Get major reactant species index
            self.sp_idx = self.solver._r.gas.species_index(name)
            
            #Store sensitivity coefficients within list
            self.sa_coeffs = []
            self.overall_sa_coeffs = []
        
            #Turn flag for SA
            self.flag_sa = True
        
        #Store results within list
        self.output_all = []

        for i,self.state_t in enumerate(self.trange):
            
            #Current reactor thermodynamic state
            self.cstate = (self.state_t,self.state_p,self.state_comp)
            
            #Current inlet thermodynamic state
            self.solver._r.TPX = self.cstate
            
            #Set reactor inlet flow conditions:
            if self.flag_mdot_in == True:
                #Set inlet mass flow rate [kg/s]
                self.solver._r.mdot = self.mdot_in[i]
                
            if self.flag_u_in == True:
                #Set inlet flow velocity [m/s]
                self.solver._r.u = self.u_in[i]
            
            #Solve the PFR problem at current temperature
            self.solve() 
            
            #Append dictionary to list
            self.output_all.append(self.output)
            
            #Perform brute force SA
            if self.flag_sa == True:
                
                #Molar fractions at the reactor outlet
                self._Xout_ref = self.X[-1,:]
                    
                #Compute reactant conversion [-]
                self.conv_ref = (1.0 
                                 - self.X[-1,self.sp_idx]/self.X[0,self.sp_idx])
        
                #Normalized sensitivity coefficients list
                coeffs = []
                overall_coeffs = []
                
                #Loop over all surface reactions
                for j in range(self.solver._r.rs.surface.n_reactions):
                    
                    #Perturb each reaction sequentially
                    self.solver._r.rs.surface.set_multiplier(self.factor,j)
                    
                    #Current inlet thermodynamic state
                    self.solver._r.TPX = self.cstate
                    
                    #Set reactor inlet flow conditions:
                    if self.flag_mdot_in == True:
                        #Set inlet mass flow rate [kg/s]
                        self.solver._r.mdot = self.mdot_in[i]
                        
                    if self.flag_u_in == True:
                        #Set inlet flow velocity [m/s]
                        self.solver._r.u = self.u_in[i]
                     
                    #Solve the PFR
                    self.solve()

                    #Molar fractions at the reactor outlet
                    self._Xout_pt = self.X[-1,:]
                
                    #Compute perturbed reactant conversion [-]
                    self.conv_pt = (1.0  
                                    - self.X[-1,self.sp_idx]/self.X[0,self.sp_idx])
                    
                    #Compute difference between reference and perturbed values
                    self._delta = np.abs(self.conv_ref-self.conv_pt)
                    self._delta_x = np.abs(self._Xout_ref-self._Xout_pt)
                    
                    #Normalized sensitivity coefficients 
                    coeffs.append((self.factor*self._delta)/
                                            (self.conv_ref*(1.0-self.factor)) )
                    
                    #Auxiliary variables
                    _a = self.factor*self._delta_x
                    _b = self._Xout_ref*(1.0-self.factor) 
                    
                    #Overall sensitivity coefficients 
                    overall_coeffs.append(np.sum(np.divide(_a,_b,
                                        out=np.zeros_like(_a),where=_b!=0)**2))
                    
                    #Reset reaction rate multiplier to 1
                    self.solver._r.rs.surface.set_multiplier(1.0)
                    
                #Append SA coefficients to outer list
                self.sa_coeffs.append(np.array(coeffs))
                self.overall_sa_coeffs.append(np.array(overall_coeffs))
                
    def collect_results(self):
        
        #Make reference to output dict
        self.output = self.solver._output
        
        #Individual variables:
        
        #Gas-phase data
        self.z = self.solver._z                      #Axial grid coordinates [m]
        self.rtime = self.solver._rtime              #Residence time [s]
        self.mdot = self.solver._mdot                #Mass flow rate [kg/s]
        self.u = self.solver._u                      #Flow velocity [m/s]
        self.rho = self.solver._rho                  #Gas-phase density [kg/m**3]
        self.wg_mix = self.solver._wg_mix            #Gas-phase average molar mass [kg/kmol]
        self.T = self.solver._T                      #Temperature [K]
        self.P = self.solver._P                      #Pressure [Pa]
        self.X = self.solver._X                      #Gas-phase molar fractions [-]
        self.Y = self.solver._Y                      #Gas-phase mass fractions [-]
        
        #Surface-phase data
        if self.solver._r.rs != None: 
            self.coverages = self.solver._covs       #Surface coverages [-]
            self.rho_s = self.solver._rho_s          #Gas-phase (surface) density [kg/m**3]
            self.wg_mix_s = self.solver._wg_mix_s    #Gas-phase (surface) average molar mass [kg/kmol]
            self.Xs = self.solver._Xs                #Gas-phase (surface) molar fractions [-]
            self.Ys = self.solver._Ys                #Gas-phase (surface) mass fractions [-]
            
            #Washcoat data (None if detailed model is not employed)
            self.coverages_wc = self.solver._covs_wc #Surface coverages (washcoat) [-]
            self.wg_mix_wc = self.solver._wg_mix_wc  #Gas-phase (washcoat) average molar mass [kg/kmol]
            self.Xwc = self.solver._Xwc              #Gas-phase (washcoat) molar fractions [-]
            self.Ywc = self.solver._Ywc              #Gas-phase (washcoat) mass fractions [-]