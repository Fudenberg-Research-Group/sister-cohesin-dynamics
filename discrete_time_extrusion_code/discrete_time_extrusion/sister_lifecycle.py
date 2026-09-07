

import numpy as np
import discrete_time_extrusion.arrays as arrays

class TranslocatorFromTrajectory:
    def __init__(self, traj):
        self.sister_trajectory = traj["sister"]
        self.lef_trajectory = traj["lef"]
        self.ctcf_trajectory = traj["ctcf"]


def update(translocator, param_dict, type_list, site_types):
    """Swap in a new parameter set (e.g. switch simulation from WT to dW regime)."""
    lef_arrays = arrays.make_LEF_arrays(np, type_list, site_types, **param_dict)
    lef_transition_dict = arrays.make_LEF_transition_dict(np, type_list, site_types, **param_dict)
    ctcf_dynamic_arrays = arrays.make_CTCF_dynamic_arrays(np, type_list, site_types, **param_dict)

    translocator.extrusion_engine.birth_prob = lef_arrays[0]
    translocator.extrusion_engine.death_prob = lef_arrays[1]
    translocator.extrusion_engine.stalled_death_prob = lef_arrays[2]

    translocator.barrier_engine.birth_prob = ctcf_dynamic_arrays[0]
    translocator.barrier_engine.death_prob = ctcf_dynamic_arrays[1]

    translocator.extrusion_engine.transition_dict = lef_transition_dict['LEF_transitions']
        
def shared_sister_update(translocator1, translocator2, step_number, shared_decay_decisions):
    """
    Update sister states for both translocators using shared decisions
    """
    if step_number >= len(shared_decay_decisions):
        return
    fall_off_mask = shared_decay_decisions[step_number]
    # Apply to translocator1
    if translocator1.extrusion_engine.sister_positions is not None:
        active_mask_1 = translocator1.extrusion_engine.sister_positions != -1
        final_mask_1 = fall_off_mask & active_mask_1
        translocator1.extrusion_engine.sister_positions[final_mask_1] = -1
    # Apply to translocator2 
    if translocator2.extrusion_engine.sister_positions is not None:
        active_mask_2 = translocator2.extrusion_engine.sister_positions != -1
        final_mask_2 = fall_off_mask & active_mask_2
        translocator2.extrusion_engine.sister_positions[final_mask_2] = -1


def shared_sister_collision_decay(translocator1, translocator2, step_number, 
            collision_release_decision_sister1, collision_release_decision_sister2):
    """
    For each sister coupled to an active (moving) extruder, compare with the
    collision-release probability and decide whether the sister falls off.
    """
    decisions_step1 = collision_release_decision_sister1[step_number]
    decisions_step2 = collision_release_decision_sister2[step_number]
    num_sisters = len(decisions_step1)
    
    # arr != -1 marks sisters that are currently coupled to an active extruder
    if len(translocator1.extrusion_engine.coupled_to_extruder) > 0:
        arr = translocator1.extrusion_engine.coupled_to_extruder
        mask = arr != -1
        sisters1 = np.where(mask)[0]
        extruders1 = arr[mask]
        # Vectorized check for active extruders
        active_mask1 = np.array([translocator1.extrusion_engine.states[e] == 1 for e in extruders1], dtype=bool)
        RN_sisters_1 = sisters1[active_mask1]
    else:
        RN_sisters_1 = np.array([], dtype=int)
    
    if len(translocator2.extrusion_engine.coupled_to_extruder) > 0:
        arr = translocator2.extrusion_engine.coupled_to_extruder
        mask = arr != -1
        sisters2 = np.where(mask)[0]
        extruders2 = arr[mask]
        active_mask2 = np.array([translocator2.extrusion_engine.states[e] == 1 for e in extruders2], dtype=bool)
        RN_sisters_2 = sisters2[active_mask2]
    else:
        RN_sisters_2 = np.array([], dtype=int)
    
    collision_mask = np.zeros(num_sisters, dtype=bool)
    
    if len(RN_sisters_1) > 0:
        collision_mask[RN_sisters_1] = decisions_step1[RN_sisters_1]
    if len(RN_sisters_2) > 0:
        collision_mask[RN_sisters_2] |= decisions_step2[RN_sisters_2]
    
    # Apply to both translocators
    if translocator1.extrusion_engine.sister_positions is not None:
        active_mask_1 = translocator1.extrusion_engine.sister_positions != -1
        final_mask_1 = collision_mask & active_mask_1
        translocator1.extrusion_engine.sister_positions[final_mask_1] = -1
    
    if translocator2.extrusion_engine.sister_positions is not None:
        active_mask_2 = translocator2.extrusion_engine.sister_positions != -1
        final_mask_2 = collision_mask & active_mask_2
        translocator2.extrusion_engine.sister_positions[final_mask_2] = -1 



def run_synchronized_trajectories_continue(translocator1, translocator2, sister_lifetime, 
                        period=None, steps=None, 
                        prune_unbound_LEFs=True, track_sisters=False, 
                        sample_interval=1, seed=42, clear_trajectories=False, **kwargs):
    """
    Modified version that optionally preserves existing trajectory data
    """
    # Pre-generate shared decay decisions
    num_sisters = translocator1.extrusion_engine.num_sisters

    lattice_time_unit = translocator1.params['time_unit']

    decay_prob = lattice_time_unit / sister_lifetime
         
    # Calculate total steps needed
    steps = int(steps) if steps else translocator1.params['steps']
    period = int(period) if period else 1 
    assert period == 1, "period != 1 is not currently supported (main loop always advances 1 step per iteration)"
    dummy_steps = translocator1.params['dummy_steps'] * period
    total_steps = steps * period + dummy_steps

    collision_release_prob = translocator1.params['collision_release_prob']
    
    np.random.seed(seed)
    shared_decay_decisions = np.random.random((total_steps, num_sisters)) < decay_prob
    collision_release_decision_sister1 = np.random.random((total_steps, num_sisters)) < collision_release_prob
    collision_release_decision_sister2 = np.random.random((total_steps, num_sisters)) < collision_release_prob
    
    print(f"Generated shared decay schedule for {total_steps} total steps")
    
    # Initialize trajectory lists only if clearing or they don't exist
    if clear_trajectories or not hasattr(translocator1, 'state_trajectory'):
        translocator1.state_trajectory = []
        translocator1.lef_trajectory = []
        translocator1.coupling_trajectory = []
        translocator1.ctcf_trajectory = []
        translocator1.sister_trajectory = []
    
    if clear_trajectories or not hasattr(translocator2, 'state_trajectory'):
        translocator2.state_trajectory = []
        translocator2.lef_trajectory = []
        translocator2.coupling_trajectory = []
        translocator2.ctcf_trajectory = []
        translocator2.sister_trajectory = []
       
    # Dummy steps
    translocator1.run(dummy_steps*period, **kwargs)
    translocator2.run(dummy_steps*period, **kwargs)
    print(f"Running {dummy_steps} dummy steps...")

    for step_idx in range(steps):
        # advance both trajectories by one step
        translocator1.run(1)
        translocator2.run(1)
        if step_idx % 5000 == 0:
            print(f"Step {step_idx} completed")
        # Update sisters after each individual step
        shared_sister_update(translocator1, translocator2, step_idx, shared_decay_decisions)
        shared_sister_collision_decay(translocator1, translocator2, step_idx, collision_release_decision_sister1, collision_release_decision_sister2)
        if step_idx % sample_interval == 0 or step_idx == steps - 1:
            # Collect data for translocator1
            LEF_states_1 = translocator1.extrusion_engine.get_states()
            CTCF_positions_1 = translocator1.barrier_engine.get_bound_positions()
            
            if prune_unbound_LEFs:
                LEF_positions_1 = translocator1.extrusion_engine.get_bound_positions()
            else:
                LEF_positions_1 = translocator1.extrusion_engine.get_positions()
            
            translocator1.state_trajectory.append(LEF_states_1)
            translocator1.lef_trajectory.append(LEF_positions_1)
            translocator1.ctcf_trajectory.append(CTCF_positions_1)
            
            # Collect data for translocator2
            LEF_states_2 = translocator2.extrusion_engine.get_states()
            CTCF_positions_2 = translocator2.barrier_engine.get_bound_positions()
            
            if prune_unbound_LEFs:
                LEF_positions_2 = translocator2.extrusion_engine.get_bound_positions()
            else:
                LEF_positions_2 = translocator2.extrusion_engine.get_positions()
            
            translocator2.state_trajectory.append(LEF_states_2)
            translocator2.lef_trajectory.append(LEF_positions_2)
            translocator2.ctcf_trajectory.append(CTCF_positions_2)
            
            if track_sisters:
                if hasattr(translocator1.extrusion_engine, 'sister_positions'):
                    sister_positions_1 = translocator1.extrusion_engine.sister_positions.copy()
                    coupling_status_1 = translocator1.extrusion_engine.get_coupling_status()
                    translocator1.sister_trajectory.append(sister_positions_1)
                    translocator1.coupling_trajectory.append(coupling_status_1)
                
                if hasattr(translocator2.extrusion_engine, 'sister_positions'):
                    sister_positions_2 = translocator2.extrusion_engine.sister_positions.copy()
                    coupling_status_2 = translocator2.extrusion_engine.get_coupling_status()
                    translocator2.sister_trajectory.append(sister_positions_2)
                    translocator2.coupling_trajectory.append(coupling_status_2)



