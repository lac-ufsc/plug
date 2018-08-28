#Import current package folder to system path (temporary solution)
#import sys
#sys.path.append('/home/tpcarvalho/carva/python_scripts/cantera/pfr/src/plug')

from .reactor.flow_reactor import PlugFlowReactor
from .reactor.reacting_surface import ReactingSurface
from .reactor.reactor_solver import ReactorSolver
from .reactor.washcoat_model import Washcoat
from .kinetics.surf_phase import SurfacePhase
from .utils.reduce_mech import ReduceMechanism
from .utils import pca_reduction



