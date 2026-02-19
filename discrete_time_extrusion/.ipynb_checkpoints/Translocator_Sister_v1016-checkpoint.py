from . import arrays
import numpy as np

class Translocator_Sister():
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
    
        sites_per_replica = kwargs['monomers_per_replica'] * kwargs['sites_per_monomer']
        number_of_LEFs = (kwargs['number_of_replica'] * kwargs['monomers_per_replica']) // kwargs['LEF_separation']
        
        assert len(site_types) == sites_per_replica, ("Site type array (%d) doesn't match replica lattice size (%d)"
                                                      % (len(site_types), sites_per_replica))
        self.time_unit = 1. / (kwargs['sites_per_monomer'] * kwargs['velocity_multiplier'])
        
        lef_arrays = arrays.make_LEF_arrays(xp, type_list, site_types, **kwargs)
        lef_transition_dict = arrays.make_LEF_transition_dict(xp, type_list, site_types, **kwargs)
        
        ctcf_arrays = arrays.make_CTCF_arrays(xp, type_list, site_types, ctcf_left_positions, ctcf_right_positions, **kwargs)
        ctcf_dynamic_arrays = arrays.make_CTCF_dynamic_arrays(xp, type_list, site_types, **kwargs)
        
        self.barrier_engine = barrier_engine(*ctcf_arrays, *ctcf_dynamic_arrays)

        number_of_sisters = kwargs['num_of_sisters']
        
        # Define sisiter_lifetime and sister_damping
        sister_tau  = kwargs['sister_lifetime']
        sister_damping = kwargs['sister_damping']
        
        collision_release_prob = kwargs['collision_release_prob']

        self.extrusion_engine = extrusion_engine(
                               number_of_LEFs, 
                               number_of_sisters,
                               sister_tau,
                               sister_damping,
                               collision_release_prob, 
                               initial_sister_positions, 
                               self.barrier_engine, 
                               *lef_arrays, **lef_transition_dict)
          
        kwargs['steps'] = int(kwargs['steps'] / self.time_unit)
        kwargs['dummy_steps'] = int(kwargs['dummy_steps'] / self.time_unit)
        self.params = kwargs
    
    # Add sister helper methods
    def get_sister_positions(self):
        """Get current sister positions"""
        if hasattr(self.extrusion_engine, 'is_sister'):
            sister_positions = []
            for i in range(self.extrusion_engine.number):
                if self.extrusion_engine.is_sister[i] and self.extrusion_engine.states[i] == 1:
                    sister_positions.append({
                        'id': i,
                        'position': int(self.extrusion_engine.positions[i, 0])
                    })
            return sister_positions
        return []
    
    def get_coupling_status(self):
        """Get sister-extruder coupling information"""
        if hasattr(self.extrusion_engine, 'get_coupling_status'):
            return self.extrusion_engine.get_coupling_status()
        return {'coupled_pairs': [], 'uncoupled_sisters': [], 'total_sisters': 0}
                
    def run(self, N, **kwargs):
        self.extrusion_engine.steps(N, self.params['mode'], **kwargs)
        
    def run_trajectory_one_sister(self, period = None, steps = None, 
            prune_unbound_LEFs = True, track_sisters = False, sample_interval = 1, **kwargs):
        self.clear_trajectory()
        steps = int(steps) if steps else self.params['steps']
        period = int(period) if period else self.params['sites_per_monomer']
    
        self.run(self.params['dummy_steps']*period, **kwargs)
    
        for step_idx in range(steps):
            self.run(period, **kwargs)

            # Only save data at sampling intervals
            if step_idx % sample_interval == 0:
                LEF_states = self.extrusion_engine.get_states()
                CTCF_positions = self.barrier_engine.get_bound_positions()
                
                if prune_unbound_LEFs:
                    LEF_positions = self.extrusion_engine.get_bound_positions()
                else:
                    LEF_positions = self.extrusion_engine.get_positions()
            
                self.state_trajectory.append(LEF_states)
                self.lef_trajectory.append(LEF_positions)
                self.ctcf_trajectory.append(CTCF_positions)
        
                # Track sister trajectories
                if track_sisters:
                    # Check if extrusion engine has sister functionality
                    if hasattr(self.extrusion_engine, 'sister_positions') and hasattr(self.extrusion_engine, 'get_coupling_status'):
                        # Make a copy for trajectory
                        sister_positions = self.extrusion_engine.sister_positions.copy()  
                        coupling_status = self.extrusion_engine.get_coupling_status()
                        
                        self.sister_trajectory.append(sister_positions)
                        self.coupling_trajectory.append(coupling_status)
                        
                        if step_idx % sample_interval  == 0:
                            print(f"Step {step_idx}: Saved trajectory point {len(self.sister_trajectory)}")
                else:
                    print(f"Warning: Extrusion engine doesn't have sister functionality")
                    # Initialize empty sister trajectories if not already done
                    if not hasattr(self, 'sister_trajectory'):
                        self.sister_trajectory = []
                    if not hasattr(self, 'coupling_trajectory'):
                        self.coupling_trajectory = []
                    self.sister_trajectory.append([])
                    self.coupling_trajectory.append({'coupled_pairs': [], 'uncoupled_sisters': [], 'total_sisters': 0})
     

    # Add this method to initialize trajectory lists in the translocator
    def clear_trajectory(self):
        """Clear all trajectory data including sister trajectories"""
        # Call parent clear_trajectory if it exists
        if hasattr(super(), 'clear_trajectory'):
            super().clear_trajectory()
        else:
            # Initialize standard trajectories
            self.state_trajectory = []
            self.lef_trajectory = []
            self.ctcf_trajectory = []
        # Initialize sister trajectory tracking
        self.sister_trajectory = []
        self.coupling_trajectory = []