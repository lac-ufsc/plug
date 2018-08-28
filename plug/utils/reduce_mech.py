import numpy as np

class ReduceMechanism(object):
    def __init__(self,index,gas,surf,bulk=None):
        #Initialize Cantera solution objects containing the full mechanism
        self.gas = gas             #Gas phase
        self.surf = surf           #Surface phase
        
        #Get species objects from each phase 
        self.gas_species = self.gas.species()
        self.surf_species = self.surf.species()
        
        #Get species names 
        self.gas_names = [s.name for s in self.gas_species]
        self.surf_names = [s.name for s in self.surf_species]
        
        #List with all species
        self.species = self.gas_species + self.surf_species
        
        #If there are bulk phases
        if bulk != None:
            #Initialize Cantera solution object
            self.bulk = bulk       #Bulk phase
            
            #Get species objects from each phase 
            self.bulk_species = self.bulk.species()
            
            #Get species names 
            self.bulk_names = [s.name for s in self.bulk_species]
            
            #Add to list of all species
            self.species += self.bulk_species
            
            #Create a flag indicating that there is a bulk phase
            self.flag_bulk = 1
        else:
            #Set the flag off
            self.flag_bulk = 0
            
        #Reduced reaction set index
        self.idx_redux = index
        
        #List of full mechanism reaction objects. Assume mechanism 
        #has only surface reactions
        self.reactions = self.surf.reactions()
        
        #Get inert gases
        self.get_inert_gas_species()
        
        #Check if transport model is on
        if self.gas.transport_model == 'Transport':
            #Turn flag off
            self.flag_transport = 0
        else:
            #Turn flag on
            self.flag_transport = 1
            
        #Physical constants (SI units)
        self.boltzmann = 1.3806487924497037e-23
        self.lightspeed = 299792458.0
        
    def get_inert_gas_species(self):
        
        #Create dict to store species
        species = dict()
        
        #Loop over all reactions
        for r in self.reactions:

            #Get species involved in current reaction
            species.update(r.reactants)
            species.update(r.products)
                    
        #Get species names only
        species = list(species.keys())
        
        #Find inert gas species name
        self.inert_gas = list(set(self.gas_names) - set(species))
        
    def get_redux_mech(self):
        
        #Create lists to store data
        self.r = []
        self.redux_species = dict()
        
        #Loop over reduced reactions set
        for i in self.idx_redux:
            
            #Append reaction object
            self.r.append(self.reactions[i])
            
            #Get species involved in current reaction
            self.redux_species.update(self.reactions[i].reactants)
            self.redux_species.update(self.reactions[i].products)
                    
        #Get species names only
        self.redux_species = list(self.redux_species.keys())
        
        #Add inert species name to list
        self.redux_species += self.inert_gas
        
        #Create lists
        self.s = []
        self.elem = dict()
        self.s_gas = []
        self.s_surf = []
        self.s_bulk = []
        
        #Loop over all species
        for s in self.species:
            
            #If species is in reduced species set
            if s.name in self.redux_species:
                
                #Append to list
                self.s.append(s)
                
                #Append species object to the phase-specific list
                if s.name in self.gas_names:
                    #Gas species
                    self.s_gas.append(s)
                    
                elif s.name in self.surf_names:
                    #Surface species
                    self.s_surf.append(s)
                
                #Check for bulk phases
                if self.flag_bulk == 1:
                    if s.name in self.bulk_names:
                        #Bulk species
                        self.s_bulk.append(s)
                
                #Get elemental composition of current species
                self.elem.update(s.composition)
        
        #Get element names only
        self.elem = list(self.elem.keys())

        #Get species names for each phase
        self.s_gas_names = [s.name for s in self.s_gas]
        self.s_surf_names = [s.name for s in self.s_surf]
        
        #Check the present of a bulk phase
        if self.flag_bulk == 1:
            #Bulk phase species names
            self.s_bulk_names = [s.name for s in self.s_bulk]

    def write_to_cti(self,filename=None):
        
        #Get reduced mechanism data
        self.get_redux_mech()
        
        #Write CTI source file
        lines = []
        
        delimiterLine = '#' + '-'*79
        
        #Units definition
        lines.append("units(length='m', time='s', quantity='kmol', act_energy='J/kmol')")
        lines.append('')
        
        #Gas phase definition
        ideal_gas = []
        ideal_gas.append('ideal_gas(name={0!r},'.format(self.gas.name))
        ideal_gas.append('          elements="{0}",'.format(' '.join(self.elem)))
        ideal_gas.append('          species="""{0}""",'.format(' '.join(self.s_gas_names)))   
        ideal_gas.append("          reactions='{0}-*',".format(self.gas.name))
        if self.flag_transport == 1:
            ideal_gas.append("          transport={0!r},".format(self.gas.transport_model))
        ideal_gas.append('          initial_state=state(temperature=300.0, pressure=OneAtm))')
        
        #Add to lines
        lines += ideal_gas
        lines.append('')
        
        #Bulk phase definition
        if self.flag_bulk == 1:
            bulk_solid = []
            bulk_solid.append('stoichiometric_solid(name={0!r},'.format(self.bulk.name))
            bulk_solid.append('          elements="{0}",'.format(' '.join(self.elem)))
            bulk_solid.append("          density=({0:0.3f}, 'g/cm3'),".format(
                                                        self.bulk.density/1e3))
            bulk_solid.append("          species='{0}',".format(
                                      ' '.join(self.s_bulk_names)))  
            bulk_solid.append('          initial_state=state(temperature=300.0,' 
                                                             'pressure=OneAtm))')
            #Add to lines
            lines +=  bulk_solid
            lines.append('')
        
        #Surface phase definition
        surface = []
        surface.append('ideal_interface(name={0!r},'.format(self.surf.name))
        surface.append('                elements="{0}",'.format(' '.join(self.elem)))
        surface.append('                species="""{0}""",'.format(
                                                  ' '.join(self.s_surf_names)))  
        surface.append('                site_density={0},'.format(self.surf.site_density))
        if self.flag_bulk == 1:
            surface.append('                phases="{0}",'.format(self.gas.name
                                                          +' '+self.bulk.name))
        else:
            surface.append('                phases="{0}",'.format(self.gas.name))   
        surface.append("                reactions='all',")
        surface.append('                initial_state=state(temperature=300.0, pressure=OneAtm))')
            
        #Add to lines
        lines += surface 
        lines.append('')
        
        #Species data
        lines.append(delimiterLine)
        lines.append('# Species data')
        lines.append(delimiterLine)
        lines.append('')
        
        for s in self.s:
            label = s.name
            atoms = ' '.join('{0}:{1:.0f}'.format(*a)
                             for a in s.composition.items())
            
            #Get thermo data
            Tmin = s.thermo.min_temp
            Tmax = s.thermo.max_temp
            Tmid = s.thermo.coeffs[0]
            coeffs_high = s.thermo.coeffs[1:8]
            coeffs_low = s.thermo.coeffs[8:15]
            
            vals = ['{0: 15.8E}'.format(i) for i in coeffs_low]
            lines_low = ['NASA([{0:.2f}, {1:.2f}],'.format(Tmin, Tmid),
                            '     [{0}, {1}, {2},'.format(*vals[0:3]),
                            '      {0}, {1}, {2},'.format(*vals[3:6]),
                            '      {0}]),'.format(vals[6])]
            #lines_low = '\n'.join(lines_low)
            
            vals = ['{0: 15.8E}'.format(i) for i in coeffs_high]
            lines_high = ['NASA([{0:.2f}, {1:.2f}],'.format(Tmid, Tmax),
                            '     [{0}, {1}, {2},'.format(*vals[0:3]),
                            '      {0}, {1}, {2},'.format(*vals[3:6]),
                            '      {0}]),'.format(vals[6])]
            #lines_high = '\n'.join(lines_high)
                
            lines.append('species(name={0!r},'.format(label))
            lines.append('        atoms={0!r},'.format(atoms))
            
            for i,l in enumerate(lines_low):
                if i == 0:
                    lines.append('        thermo=({0}'.format(l))
                else:
                    lines.append('                {0}'.format(l))
                    
            for i,l in enumerate(lines_high):
                    lines.append('                {0}'.format(l))
            lines[-1] = lines[-1][:-1] + '),'            
            #Get transport data
            if s in self.s_gas:
                if self.flag_transport == 1:
                    prefix = ' '*32
                    lines_trans = ['gas_transport(geom={0!r},'
                                                       .format(s.transport.geometry),
                                        prefix+'diam={0:.3f},'
                                        .format(s.transport.diameter*1e10),
                                        prefix+'well_depth={0:.2f},'
                                        .format(s.transport.well_depth/self.boltzmann)]
                    if s.transport.dipole != 0.0:
                        lines_trans.append(prefix+'dipole={0:.3f},'
                                                .format(s.transport.dipole*self.lightspeed/1e-21))
                    if s.transport.polarizability != 0.0:
                        lines_trans.append(prefix+'polar={0},'
                                                .format(s.transport.polarizability*1e30))
                    if s.transport.rotational_relaxation != 0.0:
                        lines_trans.append(prefix+'rot_relax={0},'
                                                .format(s.transport.rotational_relaxation))
                    lines_trans[-1] = lines_trans[-1][:-1] + ')'
                
                #Append individual transport data to species data
                lines.append('        transport={0},'.format('\n'.join(lines_trans)))
            lines.append('        size={0}'.format(s.size))
            lines[-1] += ')'
            lines.append('')
            
        #Reaction data    
        lines.append(delimiterLine)
        lines.append('# Reaction data')
        lines.append(delimiterLine)
        
        for i,r in enumerate(self.r):
            
            if i == 0:
                #Check if Motz-Wise correction is used
                if r.use_motz_wise_correction is True:
                    lines.append('enable_motz_wise()')
                elif r.use_motz_wise_correction is False:
                    lines.append('disable_motz_wise()')
                
            #Write reactions
            lines.append('\n# {0} Reaction {1}'.format(self.surf.name, i + 1))
            
            #Arrhenius parameters
            A = r.rate.pre_exponential_factor
            b = r.rate.temperature_exponent
            Ea = r.rate.activation_energy
            arrhenius_str = '{0}, {1}, {2}'.format(A,b,Ea)
            
             #Reaction rate string   
            if r.is_sticking_coefficient == True:
                rate_str = ' stick({0})'.format(arrhenius_str)
            else:
                rate_str = '[{0}]'.format(arrhenius_str)
                
            lines.append('surface_reaction({0!r},{1},'.format(r.equation, rate_str))
            lines.append("                 id='{0}-{1}'".format(self.surf.name, i + 1))
            lines[-1] += ')'
        
        #CTI source 
        self.cti_source = '\n'.join(lines)
        
        #Write CTI source to file
        if filename != None:
            with open(filename, 'w') as f:
                f.write(self.cti_source)
                
            print('Written file to: {0!r}'.format(filename))
