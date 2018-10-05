import numpy as np

class WashcoatArea(object):
    def __init__(self,**params):
        
        #Washcoat properties
        self.thickness = params.get('thickness',None)    #Washcoat thickness [m]
        self.epsilon = params.get('epsilon',None)        #Washcoat porosity [-]
        self.tortuosity = params.get('tortuosity',None)  #Washcoat tortuosity [-]
        self.pore_diam = params.get('pore_diam',None)    #Washcoat pore diameter [m]       
        
    def washcoat_area_cylinders(self,Se):
        '''########################################################################
        Estimate the washcoat superficial area assuming a flat surface with an 
        array of cylinders protuding inside it.
        
        Input:  Sg          #Total external flat surface area [m**2]
                dp          #Mean pore diameter [m]
                tw          #Washcoat thickness [m]
                epsilon     #Washcoat porosity [-]
                
        Output: Aw          #Total surface area [m**2]                          
        ########################################################################'''
        
        #Total external flat surface area [m**2]
        self.Se = Se
        
        #Mean pore radius [m]
        self.rp = self.pore_diam/2
        
        #Total number of pores [-]
        self.npores = self.epsilon*self.Se/self.rp**2
        
        #Total washcoat area [m**2]
        self.Sw = self.Se + 2*self.npores*np.pi*self.rp*self.thickness
        
        return self.Sw
    
    def washcoat_area_spheres(self,Se):
        '''########################################################################
        Estimate the washcoat superficial area assuming a sphere packing with 
        uniform diameter.
        
        Input:  Se          #Total external flat surface area [m**2]
                dp          #Mean pore diameter [m]
                tw          #Washcoat thickness [m]
                epsilon     #Washcoat porosity [-]
                
        Output: Aw          #Total surface area [m**2]                          
        ########################################################################'''
        
        #Total external flat surface area [m**2]
        self.Se = Se
        
        #Compute sphere diameter using the mean washcoat pore diameter as
        #the hydraulic diameter of the sphere packing
        self.ds = 3/2*self.pore_diam*((1-self.epsilon)/self.epsilon)   #Sphere diameter [m]
    
        #Specific surface per unit volume [m**-1]
        self.Ss = 6*(1-self.epsilon)/self.ds
        
        #Compute total volume of the packing [m**3]
        self.vol = self.Se*self.thickness
        
        #Total washcoat area [m**2]
        self.Sw = self.Ss*self.vol
        
        return self.Sw
    
    def compute_metallic_area(self,**kwargs):
        
        #Get support and catalytist properties
        self.support_density = kwargs.get('support_density',0.0)    #Support density [kg/m**3]
        self.metal_fraction = kwargs.get('metal_fraction',0.0)       #Metallic weight fraction [-]
        self.mean_cristal_size = kwargs.get('mean_cristal_size',0.0) #Mean cristalite size [nm]
        self.cat_mw = kwargs.get('cat_mw',0.0)                       #Metal molecular weigth [kg/kmol]
        self.site_density = kwargs.get('site_density',0.0)           #Site density [kmol/m**2]
        
        #Metal dispersion [-]
        self.metal_dispersion = 1.0/self.mean_cristal_size

        #Relate the metallic area with washcoat geometric area:
        self.Vw = self.Se*self.thickness*(1-self.epsilon) #Washcoat solid volume [m**3]
        self.Mwt = self.Vw*self.support_density           #Washcoat mass including the support [kg]
        self.Mwm = self.Mwt*self.metal_fraction           #Metallic mass in the washcoat [kg]
        self.Mws = self.Mwm*self.metal_dispersion         #Metallic mass dispersed on the surface [kg]
        
        #Number of moles of metal on the surface [kmols]
        self.Nws = self.Mws/self.cat_mw
        self.Aws = self.Nws/self.site_density  #Total metallic surface area [m**2]
        
        #Metal loading [mg]
        self.metal_loading = self.Mwm*1e06
        
        return self.Aws

        
        