
import numpy as np
# import matplotlib.pyplot as plt 
from . import NullExtruder, EngineFactory

class BaseExtruder_Sister(NullExtruder.NullExtruder):

    def __init__(self,
                 number_of_extruders,
                 number_of_sisters, 
                 sister_tau,
                 sister_damping,
                 initial_sister_positions, 
                 barrier_engine,
                 birth_prob,
                 death_prob,
                 stalled_death_prob,
                 diffusion_prob,
                 pause_prob,
                 *args, **kwargs):
    
        self.num_extruders = number_of_extruders
        self.num_sisters = number_of_sisters

        self.sister_positions = None
        self.extruder_sister_counts = None  # Array: extruder_id -> number of coupled sisters
        
        # Change dictionary format: sister_id -> extruder_id [many:1] into matrix with -1 for no coupling
        self.coupled_to_extruder = np.full(number_of_sisters, -1, dtype=int)
        # extruder_id -> [sister_ids] [1:many] of size (number_of_extruders, number_of_sisters) with -1 for no coupling   
        self.coupled_to_sister = np.full((number_of_extruders, number_of_sisters), -1, dtype=int)     
        
        # Pre-computed position lookup for faster coupling checks
        self._extruder_position_to_ID = {}    # Cache for extruder position -> ID mapping
        self._position_cache_valid = False

        super().__init__(number_of_extruders, barrier_engine)
        
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        self.stalled_death_prob = stalled_death_prob
        self.diffusion_prob = diffusion_prob
        self.pause_prob = pause_prob
        self.sister_tau = sister_tau
        self.sister_damping = sister_damping
        self.stepping_engine = EngineFactory.SteppingEngine

        if initial_sister_positions is not None:
            self._initialize_sisters_fix(initial_sister_positions)
            self.sister_positions = self.xp.array(initial_sister_positions, dtype=self.xp.int32)
        else:
            self._initialize_sisters()
        
        ## For testing, set up specific scenarios 
        # self.setup_test_scenario_mean_field_alpha()

    
    def _initialize_sisters_fix(self, initial_positions):
        """Initialize sisters either randomly or from saved file"""
        if self.num_sisters <= 0:
            print("No sisters to initialize")
            return
        # Initialize arrays
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        self.extruder_sister_counts = self.xp.zeros(self.num_extruders, dtype=self.xp.int32)

        if len(initial_positions) >= self.num_sisters:
            self.sister_positions = initial_positions[:self.num_sisters]
        print(f"Loaded sisters from fixed positions ")


    def _initialize_sisters(self):
        """Optimized sister initialization using vectorized operations"""
        if self.num_sisters <= 0:
            print("No sisters to initialize")
            return
        
        lattice_size = len(self.occupied)
        print(f"Initializing {self.num_sisters} sisters on lattice of size {lattice_size}")

        # Initialize arrays
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        self.extruder_sister_counts = self.xp.zeros(self.num_extruders, dtype=self.xp.int32)

        # Vectorized initialization: find all free positions at once
        free_positions = self.xp.where(~self.occupied)[0]
        
        if len(free_positions) >= self.num_sisters:
            # Randomly select from free positions
            selected_positions = self.xp.random.choice(
                free_positions, size = self.num_sisters, replace=False
            )
            self.sister_positions = selected_positions
        else:
            print(f"WARNING: Not enough free positions ({len(free_positions)}) for {self.num_sisters} sisters")
            # Place as many as possible
            num_to_place = min(self.num_sisters, len(free_positions))
            self.sister_positions[:num_to_place] = free_positions[:num_to_place]
            # Mark excess sisters as inactive
            self.sister_positions[num_to_place:] = -1
        
        self.xp.save("sister.npy", self.sister_positions) 
        print(f"Initialized {self.num_sisters} sisters at positions: {self.sister_positions}")


    def _update_extruder_position_to_ID_cache(self):
        """Vectorized extruder position cache update"""
        ### Build position to extruder ID mapping
        self._extruder_position_to_ID = {}

        # VECTORIZED: Find all active extruders (state != 0) in one operation
        active_extruder_mask = self.states != 0
        active_ids = self.xp.where(active_extruder_mask)[0]
    
        # Exit early if no active extruders
        if len(active_ids) == 0:
            self._position_cache_valid = True
            return
        
        # VECTORIZED: Get all active extruder positions at once, shape: (n_active, 2)
        active_positions = self.positions[active_ids]  
        
        # Build cache dictionary - loop only over active extruders
        for idx, extruder_id in enumerate(active_ids):
            pos1 = int(active_positions[idx, 0])
            pos2 = int(active_positions[idx, 1])
            # Add first leg position
            if pos1 not in self._extruder_position_to_ID:
                self._extruder_position_to_ID[pos1] = []
            self._extruder_position_to_ID[pos1].append(int(extruder_id))
        
            # Add second leg position if different
            if pos2 != pos1:
                if pos2 not in self._extruder_position_to_ID:
                    self._extruder_position_to_ID[pos2] = []
                self._extruder_position_to_ID[pos2].append(int(extruder_id))

        self._position_cache_valid = True


    def update_sister_extruder_coupling(self):
        """Optimized coupling check with immediate dict sync and safe caching"""
        if self.sister_positions is None: 
            return
        ## Update position cache for extruders if invalid
        self._update_extruder_position_to_ID_cache()
       
        # Find new couplings, first identify uncoupled sisters
        uncoupled_sisters = self.xp.where(self.coupled_to_extruder == -1)[0]
        
        if len(uncoupled_sisters) == 0:
            return 
        
        for sister_id in uncoupled_sisters:
            sister_pos = int(self.sister_positions[sister_id])
            ## if sister pos in the extruder positions, couple them and upadate the coupling info
            if sister_pos in self._extruder_position_to_ID:
                extruder_id = self._extruder_position_to_ID[sister_pos][0]
                self.extruder_sister_counts[extruder_id] += 1
                ## update coupling info 
                self.coupled_to_extruder[sister_id] = extruder_id
                self.coupled_to_sister[extruder_id, int(sister_id)] = int(sister_id)
            else:
                continue

    
    def birth(self, unbound_state_id):
        """Optimized birth function - unchanged but benefits from improved coupling"""
        self._position_cache_valid = False  # Invalidate cache
        
        free_sites = self.sites[~self.occupied]
        if len(free_sites) == 0:
            return self.xp.array([])
            
        binding_sites = self.xp.random.choice(free_sites, size = self.number, replace = False)

        rng = self.xp.less(self.xp.random.random(self.number), self.birth_prob[binding_sites])
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, unbound_state_id))
                
        if len(ids) > 0:
            binding_sites = binding_sites[ids]
            self.positions[ids] = binding_sites[:, None]
        
            rng_dir = self.xp.less(self.xp.random.random(len(ids)), 0.5)
            rng_stagger = self.xp.less(self.xp.random.random(len(ids)), 0.5) * ~self.occupied[binding_sites+1]
            
            self.positions[ids, 1] = self.xp.where(rng_stagger,
                                                   self.positions[ids, 1] + 1,
                                                   self.positions[ids, 1])
            self.directions[ids] = rng_dir.astype(self.xp.uint32)
                                                           
        return ids    

    def death(self, bound_state_id):
        """Optimized death function"""
        death_prob = self.xp.where(self.stalled,
                                   self.stalled_death_prob[self.positions],
                                   self.death_prob[self.positions])
        death_prob = self.xp.max(death_prob, axis=1)
        
        rng = self.xp.less(self.xp.random.random(self.number), death_prob)
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, bound_state_id))
        
        return ids

    def unload(self, ids_death):
        """Optimized unload using vectorized operations"""
        if len(ids_death) == 0:
            return
        # Vectorized uncoupling: find all sisters coupled to fallen off extruders
        coupled_sisters_death = self.xp.where(self.xp.isin(self.coupled_to_extruder, ids_death))[0]
        if len(coupled_sisters_death) > 0:
            # Reset sister coupling status to -1 for coupled sisters where extruder has fallen off 
            self.coupled_to_extruder[coupled_sisters_death] = -1 
            # Reset extruder coupling status to -1 for fallen off extruders
            for ext_id in ids_death:
                self.coupled_to_sister[ext_id, :] = -1
        # Reset sister counts to 0 for fallen off extruders
        self.extruder_sister_counts[ids_death] = 0
        # Standard unload for extruders
        self.stalled[ids_death] = False
        self.positions[ids_death] = -1
        # Invalidate position cache
        self._position_cache_valid = False
    
    # Not used in multistate extruder class, used only in base extruder class
    def update_states(self, unbound_state_id, bound_state_id):
        ids_birth = self.birth(unbound_state_id)
        ids_death = self.death(bound_state_id)
        
        self.states[ids_birth] = bound_state_id
        self.states[ids_death] = unbound_state_id
        
        self.unload(ids_death)
    
    def get_coupling_status(self):
        """Optimized coupling status using vectorized operations"""
        # Safety check for initialization
        if self.sister_positions is None or self.xp.all(self.coupled_to_extruder == -1) :
            return {
                'coupled_pairs': [],
                'uncoupled_sisters': [],
                'total_sisters': self.num_sisters
            }
            
        coupled_mask = self.coupled_to_extruder != -1
        coupled_sisters = self.xp.where(coupled_mask)[0]
        uncoupled_sisters = self.xp.where(~coupled_mask)[0]
        
        coupled_pairs = []
        if len(coupled_sisters) > 0:
            for sister_id in coupled_sisters:
                extruder_id = int(self.coupled_to_extruder[sister_id])
                coupled_pairs.append({
                    'sister_id': int(sister_id),
                    'extruder_id': extruder_id,
                    'sister_pos': int(self.sister_positions[sister_id]),
                    'extruder_pos1': int(self.positions[extruder_id, 0]),
                    'extruder_pos2': int(self.positions[extruder_id, 1])
                })
        
        uncoupled_sisters_list = []
        for sister_id in uncoupled_sisters:
            uncoupled_sisters_list.append({
                'sister_id': int(sister_id),
                'sister_pos': int(self.sister_positions[sister_id])
            })
        
        return {
            'coupled_pairs': coupled_pairs,
            'uncoupled_sisters': uncoupled_sisters_list,
            'total_sisters': self.num_sisters
        }
    

    def step(self, mode, unbound_state_id = 0, bound_state_id = 1, active_state_id = 1, **kwargs):
        """Optimized step function"""
       
        ## Update extruders states
        self.update_states(unbound_state_id, bound_state_id)
        
        ## Update sister and extruder coupling
        self.update_sister_extruder_coupling()  
    
        self.update_occupancies()
        # Parent class step
        super().step(**kwargs)
    
        # Update kwargs with coupling info
        kwargs.update({
            'coupled_to_extruder': self.coupled_to_extruder,
            'coupled_to_sister': self.coupled_to_sister})
        
        # Movement step of sisters and extruders with coupling information
        self.stepping_engine(self, mode, unbound_state_id, active_state_id, **kwargs)


    
