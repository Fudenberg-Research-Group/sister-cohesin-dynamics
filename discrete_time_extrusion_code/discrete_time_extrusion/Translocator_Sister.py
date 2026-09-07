from . import arrays
import numpy as np

class Translocator_Sister():
    """
    Translocator for a single sister chromatid. Paired with another
    Translocator_Sister instance via shared sister_id indices.
    """
    def __init__(self,
                 extrusion_engine,
                 barrier_engine,
                 type_list,
                 site_types,
                 ctcf_left_positions,
                 ctcf_right_positions,
                 initial_sister_positions, 
                 device='CPU',
                 **kwargs):
        if device == 'CPU':
            import numpy as xp
            
        elif device == 'GPU':
            try:
                import cupy as xp
                use_cuda = xp.cuda.is_available()
        
                if not use_cuda:
                    raise ImportError("Could not load CUDA environment")
    
            except:
                raise
                
        else:
            raise RuntimeError("Unrecognized device %s - use either 'CPU' or 'GPU'" % device)
    
        sites_per_replica = kwargs["region_size_bp"]//kwargs['bp_per_site']
        number_of_LEFs = int(kwargs['number_of_replica'] * kwargs["region_size_bp"] * kwargs['Total_LEF_per_bp'])
        
        assert len(site_types) == sites_per_replica, ("Site type array (%d) doesn't match replica lattice size (%d)"
                                                      % (len(site_types), sites_per_replica))
        
        # 1.25 from single molecule imaging in vitro, 0.25 is the [RN]/[R]
        self.time_unit = 2 * 0.25 * kwargs['bp_per_site'] /kwargs['active_extrusion_rate']
        
        kwargs['time_unit'] = self.time_unit

        lef_arrays = arrays.make_LEF_arrays(xp, type_list, site_types, **kwargs)
        lef_transition_dict = arrays.make_LEF_transition_dict(xp, type_list, site_types, **kwargs)
        
        ctcf_arrays = arrays.make_CTCF_arrays(xp, type_list, site_types, ctcf_left_positions, ctcf_right_positions, **kwargs)
        ctcf_dynamic_arrays = arrays.make_CTCF_dynamic_arrays(xp, type_list, site_types, **kwargs)
        
        self.barrier_engine = barrier_engine(*ctcf_arrays, *ctcf_dynamic_arrays)

        number_of_sisters = kwargs['num_of_sisters']
        
        sister_tau  = kwargs['sister_lifetime']
        sister_damping = kwargs['sister_damping_factor']
        bypass_prob = kwargs['bypass_prob']

        self.extrusion_engine = extrusion_engine(
                               sites_per_replica, 
                               number_of_LEFs, 
                               number_of_sisters,
                               sister_tau,
                               sister_damping,
                               bypass_prob,
                               initial_sister_positions, 
                               self.barrier_engine, 
                               *lef_arrays, **lef_transition_dict)
        
        # convert requested step counts from real time units to lattice steps
        kwargs['steps'] = int(kwargs['steps'] / self.time_unit)
        kwargs['dummy_steps'] = int(kwargs['dummy_steps'] / self.time_unit)
        self.params = kwargs
    
    def get_coupling_status(self):
        """Get sister-extruder coupling information"""
        if hasattr(self.extrusion_engine, 'get_coupling_status'):
            return self.extrusion_engine.get_coupling_status()
        return {'coupled_pairs': [], 'uncoupled_sisters': [], 'total_sisters': 0}
                
    def run(self, N, **kwargs):
        self.extrusion_engine.steps(N, self.params['mode'], **kwargs)
 
    

    