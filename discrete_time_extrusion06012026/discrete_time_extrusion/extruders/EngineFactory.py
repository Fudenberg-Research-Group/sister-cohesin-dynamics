import warnings

import numpy as np
# from .engines.DiffusionEngines_Sister import _diffusion_sister_step_cpu, _diffusion_step_gpu
from .engines.DiffusionEngines_Sister import _diffusion_sister_step_cpu, _diffusion_step_gpu
from .engines.SymmetricEngines import _symmetric_step_cpu, _symmetric_step_gpu
from .engines.AsymmetricEngines import _asymmetric_step_cpu, _asymmetric_step_gpu

# Import your sister stepping function
from .engines.SymmetricEngines_Sisters import symmetric_sister_step_cpu

try:
    import cupy as cp
    
    diffusion_step_cuda = cp.RawKernel(_diffusion_step_gpu(), '_diffusion_step_gpu', options=('--use_fast_math',))
    symmetric_step_cuda = cp.RawKernel(_symmetric_step_gpu(), '_symmetric_step_gpu', options=('--use_fast_math',))
    asymmetric_step_cuda = cp.RawKernel(_asymmetric_step_gpu(), '_asymmetric_step_gpu', options=('--use_fast_math',))
    
    cuda_engines = {'diffusion' : diffusion_step_cuda,
                    'symmetric' : symmetric_step_cuda,
                    'asymmetric' : asymmetric_step_cuda}

except:
    pass
    
try:
    import numba as nb
    use_numba = True

    diffusion_sister_step_numba = nb.njit(fastmath=True)(_diffusion_sister_step_cpu)
    symmetric_step_numba = nb.njit(fastmath=True)(_symmetric_step_cpu)
    asymmetric_step_numba = nb.njit(fastmath=True)(_asymmetric_step_cpu)

	# Add sister stepping engine for numba
    symmetric_sister_step_numba = nb.njit(fastmath=True)(symmetric_sister_step_cpu)
    
    numba_engines = {'diffusion' : diffusion_sister_step_numba,
                     'symmetric' : symmetric_step_numba,
                     'asymmetric' : asymmetric_step_numba,
                     'symmetric_sister' : symmetric_sister_step_numba}

except ImportError:
    use_numba = False

    diffusion_sister_step = _diffusion_sister_step_cpu
    symmetric_step = _symmetric_step_cpu
    asymmetric_step = _asymmetric_step_cpu
    symmetric_sister_step = symmetric_sister_step_cpu

    python_engines = {'diffusion_sister' : diffusion_sister_step,
                      'symmetric' : symmetric_step,
                      'asymmetric' : asymmetric_step,
					  'symmetric_sister' : symmetric_sister_step}

	
def SteppingEngine(sim, mode, unbound_state_id, active_state_id, threads_per_block=256, **kwargs):

	rngs_diffusion = sim.xp.random.random((sim.number, 4)).astype(sim.xp.float32)
		
	args_diffusion = tuple([unbound_state_id,
			                rngs_diffusion,
			                sim.number,
			                0, sim.lattice_size,
			                sim.states,
			                sim.occupied,
                            sim.occupied_sister, 
			                sim.stalled,
			                sim.diffusion_prob,
			                sim.positions])

	# Check if this is a sister-aware simulation
	has_sisters = hasattr(sim, 'num_sisters') and sim.num_sisters > 0
	      
	if mode == "symmetric":
        
		if has_sisters:
			rngs = sim.xp.random.random((sim.number, 5)).astype(sim.xp.float32)
			# Use sister-aware stepping engine
			args = tuple([active_state_id,
                          rngs,
                          sim.number,
                          0, sim.lattice_size,
                          sim.states,
                          sim.occupied,
                          sim.barrier_engine.stall_left,
                          sim.barrier_engine.stall_right,
                          sim.pause_prob,
                          sim.positions,
                          sim.stalled,
                          sim.sister_positions,
                          sim.num_sisters,
                          sim.coupled_to_extruder,
                          sim.coupled_to_sister,
						  sim.sister_tau,
						  sim.sister_damping])
			mode_key = "symmetric_sister"
		else: 
			rngs = sim.xp.random.random((sim.number, 4)).astype(sim.xp.float32)
			args = tuple([active_state_id,
			      rngs,
			      sim.number,
			      0, sim.lattice_size,
			      sim.states,
			      sim.occupied,
			      sim.barrier_engine.stall_left,
			      sim.barrier_engine.stall_right,
			      sim.pause_prob,
			      sim.positions,
			      sim.stalled])
			mode_key = "symmetric"

	elif mode == "asymmetric":
		rngs = sim.xp.random.random((sim.number, 2)).astype(sim.xp.float32)
		
		args = tuple([active_state_id,
			      rngs,
			      sim.number,
			      0, sim.lattice_size,
			      sim.states,
			      sim.occupied,
			      sim.directions,
			      sim.barrier_engine.stall_left,
			      sim.barrier_engine.stall_right,
			      sim.pause_prob,
			      sim.positions,
			      sim.stalled])
		mode_key = "asymmetric"

	else:
		raise RuntimeError("Unsupported mode '%s'" % mode)
		
	if sim.xp.__name__ == 'cupy':
		num_blocks = (sim.number+threads_per_block-1) // threads_per_block
		warnings.warn("Running lattice extrusion on the GPU")
		
		cuda_engines[mode_key]((num_blocks,), (threads_per_block,), args)
		sim.update_occupancies()
		
		cuda_engines['diffusion']((num_blocks,), (threads_per_block,), args_diffusion)
    
	elif sim.xp.__name__ == 'numpy':
		if use_numba:
			
			warnings.warn("Running lattice extrusion on the CPU using Numba")
			
			### Used in DEBUG: Check sister_positions before step
			# spos_before = sim.sister_positions.copy()
			# ext_before = sim.positions.copy()
		    
			# numba_engines[mode_key](*args)
			numba_engines[mode_key].py_func(*args)
			
			### DEBUG: Check if any sister entered loop during symmetric step
			# spos_after = sim.sister_positions.copy()
			# ext_after = sim.positions.copy()
			# left_after = ext_after[:,0]
			# right_after = ext_after[:,1]
			# valid = spos_before >= 0
			# Find sisters that were OK before but inside loop now
			# contained_before = ((spos_before[:, None] > ext_before[:,0][None, :]) & (spos_before[:, None] < ext_before[:,1][None, :])).any(axis=1)
			# contained_after = ((spos_after[:, None] > left_after[None, :]) & (spos_after[:, None] < right_after[None, :])).any(axis=1)
			# newly_trapped = (contained_after & ~contained_before) & valid
			# newly_trapped_idx = np.nonzero(newly_trapped)[0]
			
			# if newly_trapped_idx.size:
			#	print(f"\n[DIAGNOSIS] After SYMMETRIC_STEP: {newly_trapped_idx.size} sister(s) newly trapped in loop!")
			#	for s_idx in newly_trapped_idx:
			#		s_pos_before = spos_before[s_idx]
			#		s_pos_after = spos_after[s_idx]
			#		# Find which extruder trapped her
			#		ext_idx = np.nonzero((s_pos_after > left_after) & (s_pos_after < right_after))[0]
			#		# Find if she was coupled
			#		coupled_to = sim.coupled_to_extruder[s_idx]
			#		print(f"  sister {s_idx}: {s_pos_before}->{s_pos_after}, coupled_to_extruder={coupled_to}, now inside extruder(s) {ext_idx.tolist()}")
			#		# Print that extruder's movement
			#		for ei in ext_idx[:1]:  # Just first one
			#			print(f"    extruder {ei}: [{ext_before[ei,0]},{ext_before[ei,1]}]->[{ext_after[ei,0]},{ext_after[ei,1]}]")
            
			spos = sim.sister_positions
			valid = spos >= 0 
			left = sim.positions[:,0]
			right = sim.positions[:,1]
			contained = ((spos[:, None] > left[None, :]) & (spos[:, None] < right[None, :])).any(axis=1)
			contained = contained & valid
			bad_idx = np.nonzero(contained)[0]
			if bad_idx.size:
				print(f"\n[TOTAL] After symmetric_step: {bad_idx.size} sister(s) inside loop")

			sim.update_occupancies()
        
			sim.occupied_sister.fill(False)
			ids = sim.sister_positions[sim.xp.greater_equal(sim.sister_positions, 0)]
			sim.occupied_sister[ids.astype(np.int32)] = True
		
			# Store state before diffusion
			# spos_before_diff = sim.sister_positions.copy()
			# ext_before_diff = sim.positions.copy()
			
			numba_engines['diffusion'].py_func(*args_diffusion)
			
			# Check if diffusion created new trapped sisters
			# spos_after_diff = sim.sister_positions.copy()
			# ext_after_diff = sim.positions.copy()
			
			# left_after_diff = ext_after_diff[:,0]
			# right_after_diff = ext_after_diff[:,1]
			# valid = spos_before_diff >= 0
			
			# contained_before_diff = ((spos_before_diff[:, None] > ext_before_diff[:,0][None, :]) & (spos_before_diff[:, None] < ext_before_diff[:,1][None, :])).any(axis=1)
			# contained_after_diff = ((spos_after_diff[:, None] > left_after_diff[None, :]) & (spos_after_diff[:, None] < right_after_diff[None, :])).any(axis=1)
			
			# newly_trapped_diff = (contained_after_diff & ~contained_before_diff) & valid
			# newly_trapped_diff_idx = np.nonzero(newly_trapped_diff)[0]
			
			# if newly_trapped_diff_idx.size:
			#	print(f"\n[DIAGNOSIS] After DIFFUSION: {newly_trapped_diff_idx.size} sister(s) newly trapped in loop!")
			#	for s_idx in newly_trapped_diff_idx:
			#		s_pos_before = spos_before_diff[s_idx]
			#		s_pos_after = spos_after_diff[s_idx]
			#		ext_idx = np.nonzero((s_pos_after > left_after_diff) & (s_pos_after < right_after_diff))[0]
			#		print(f"  sister {s_idx}: {s_pos_before}->{s_pos_after}, inside extruder(s) {ext_idx.tolist()}")
			#		for ei in ext_idx[:1]:
			#			print(f"    extruder {ei}: [{ext_before_diff[ei,0]},{ext_before_diff[ei,1]}]->[{ext_after_diff[ei,0]},{ext_after_diff[ei,1]}]")
		
			
			# spos = sim.sister_positions
			# valid = spos >= 0 
			# left = sim.positions[:,0]
			# right = sim.positions[:,1]
			# contained = ((spos[:, None] > left[None, :]) & (spos[:, None] < right[None, :])).any(axis=1)
			# contained = contained & valid
			# bad_idx = np.nonzero(contained)[0]
			# if bad_idx.size:
			#	print(f"\n[FINAL] After diffusion: {bad_idx.size} sister(s) remain inside loop")
				
			sim.update_occupancies()

		else:
			warnings.warn("Running lattice extrusion on the CPU using pure Python backend")
			
			python_engines[mode_key](*args)
			sim.update_occupancies()

			python_engines['diffusion'](*args_diffusion)
			sim.update_occupancies()
