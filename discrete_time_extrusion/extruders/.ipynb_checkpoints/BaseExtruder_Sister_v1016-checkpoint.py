
import numpy as np
# import matplotlib.pyplot as plt 
from . import NullExtruder, EngineFactory

class BaseExtruder_Sister(NullExtruder.NullExtruder):

    def __init__(self,
                 number_of_extruders,
                 number_of_sisters, 
                 sister_tau,
                 sister_damping,
                 collision_release_prob, 
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
        self.sister_coupled_to = None       # Array: sister_id -> extruder_id (-1 if uncoupled)
        self.extruder_sister_counts = None  # Array: extruder_id -> number of coupled sisters
        
        # Maintain backward compatibility with dictionary format
        self.coupled_to_extruder = {}       # sister_id -> extruder_id [many:1]
        self.coupled_to_sister = {}         # extruder_id -> [sister_ids] [1:many]
        
        # Pre-computed position lookup for faster coupling checks
        self._position_to_extruders = {}    # Cache for position -> extruder mapping
        self._position_cache_valid = False

        super().__init__(number_of_extruders, barrier_engine)
        
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        self.stalled_death_prob = stalled_death_prob
        self.diffusion_prob = diffusion_prob
        self.pause_prob = pause_prob
        self.sister_tau = sister_tau
        self.sister_damping = sister_damping
        self.collision_release_prob = collision_release_prob 
        self.stepping_engine = EngineFactory.SteppingEngine
        
        # Initialize sisters 
        # Way1: randomly 
        if initial_sister_positions is not None:
            self._initialize_sisters_fix(initial_sister_positions)
            self.sister_positions = self.xp.array(initial_sister_positions, dtype=self.xp.int32)
        else:
            self._initialize_sisters()
        # Way2: load sister positions
        # self._initialize_sisters_load()
        # Test function of loading sisterCs
        # self._test_single_position()

        # self.setup_test_scenario()
    def _initialize_sisters_fix(self, initial_positions):
        """Initialize sisters either randomly or from saved file"""
        if self.num_sisters <= 0:
            print("No sisters to initialize")
            return
        # Initialize arrays
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        self.sister_coupled_to = self.xp.full(self.num_sisters, -1, dtype=self.xp.int32)
        self.extruder_sister_counts = self.xp.zeros(self.num_extruders, dtype=self.xp.int32)

        if len(initial_positions) >= self.num_sisters:
            self.sister_positions = initial_positions[:self.num_sisters]
        print(f"Loaded sisters from fixed positions ")


    def _initialize_sisters_load(self, sister_file_path = None):
        """Initialize sisters either randomly or from saved file"""
        if self.num_sisters <= 0:
            print("No sisters to initialize")
            return
        # Initialize arrays
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        self.sister_coupled_to = self.xp.full(self.num_sisters, -1, dtype=self.xp.int32)
        self.extruder_sister_counts = self.xp.zeros(self.num_extruders, dtype=self.xp.int32)

        loaded_positions = self.xp.load(sister_file_path)
        if len(loaded_positions) >= self.num_sisters:
            self.sister_positions = loaded_positions[:self.num_sisters]
        print(f"Loaded sisters from {sister_file_path}")

    
    def _initialize_sisters(self):
        """Optimized sister initialization using vectorized operations"""
        if self.num_sisters <= 0:
            print("No sisters to initialize")
            return
        
        lattice_size = len(self.occupied)
        print(f"Initializing {self.num_sisters} sisters on lattice of size {lattice_size}")

        # Initialize arrays
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        self.sister_coupled_to = self.xp.full(self.num_sisters, -1, dtype=self.xp.int32)
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
    
    def _test_single_position(self):
        # Initialize arrays
        lattice_size = len(self.occupied)
        print(f"Initializing {self.num_sisters} sisters on lattice of size {lattice_size}")
        # Initialize arrays
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        self.sister_coupled_to = self.xp.full(self.num_sisters, -1, dtype=self.xp.int32)
        self.extruder_sister_counts = self.xp.zeros(self.num_extruders, dtype=self.xp.int32)
        # Manually put all sisters at position 1000
        # self.sister_positions = [2500, 3500, 4500, 4500, 4500, 4500]
        print(f"Result: {self.sister_positions}")
    
    def _update_extruder_position_cache(self):
        ### Build the extruder two leg postions -- id map 
        ### Position to extruder lef and right legs (both are indexed by the extruder ID)
        self._position_to_extruders = {}
        active_extruders = []
        for extruder_id in range(self.number):
            if self.states[extruder_id] != 0:  # Extruder is on chromosome -- active 
                active_extruders.append(extruder_id)
                pos1 = int(self.positions[extruder_id, 0])
                pos2 = int(self.positions[extruder_id, 1])
                ## Add both leg positions
                if pos1 not in self._position_to_extruders:
                    self._position_to_extruders[pos1] = []
                self._position_to_extruders[pos1].append(extruder_id)
                if pos2 != pos1:
                    if pos2 not in self._position_to_extruders:
                        self._position_to_extruders[pos2] = []
                    self._position_to_extruders[pos2].append(extruder_id)

        self._position_cache_valid = True
   
    def _sync_coupling_dicts(self):
        """Sync array-based coupling data to dictionary format for backward compatibility"""
        self.coupled_to_extruder.clear()
        self.coupled_to_sister.clear()
        # Only sync if sisters are initialized
        if self.sister_coupled_to is not None:
            coupled_mask = self.sister_coupled_to != -1
            coupled_sisters = self.xp.where(coupled_mask)[0]
            
            for sister_id in coupled_sisters:
                extruder_id = int(self.sister_coupled_to[sister_id])
                self.coupled_to_extruder[int(sister_id)] = extruder_id
                
                if extruder_id not in self.coupled_to_sister:
                    self.coupled_to_sister[extruder_id] = []
                self.coupled_to_sister[extruder_id].append(int(sister_id))


    def check_sister_coupling(self):
        """Optimized coupling check with immediate dict sync and safe caching"""
        if self.sister_positions is None or self.sister_coupled_to is None:
            return
        ## Update position cache for extruders if invalid
        self._update_extruder_position_cache()
        ## First, uncouple any sisters whose extruders have died
        # self._uncouple_dead_extruders()
        self._check_extruder_deaths()
        # Second pass: find new couplings
        uncoupled_sisters = self.xp.where(self.sister_coupled_to == -1)[0]
        
        for sister_id in uncoupled_sisters:
            sister_pos = int(self.sister_positions[sister_id])
            ## if sister pos in the extruder positions, couple them 
            if sister_pos in self._position_to_extruders:
                extruder_id = self._position_to_extruders[sister_pos][0]
                ## update the arrary sister_couple_to 
                self.sister_coupled_to[sister_id] = extruder_id
                self.extruder_sister_counts[extruder_id] += 1
                ## update dictionaries 
                self.coupled_to_extruder[sister_id] = extruder_id

                if extruder_id not in self.coupled_to_sister:
                    self.coupled_to_sister[extruder_id] = []
                self.coupled_to_sister[extruder_id].append(int(sister_id))
            else:
                continue

    def _check_extruder_deaths(self):
        """Uncouple sisters whose extruders have died"""
        dead_couplings = []
        
        for sister_id, extruder_id in self.coupled_to_extruder.items():
            if self.states[extruder_id] == 0:  # Extruder died
                dead_couplings.append((sister_id, extruder_id))
        
        # Uncouple dead extruders but keep sisters alive
        for sister_id, extruder_id in dead_couplings:
            del self.coupled_to_extruder[sister_id]
            if extruder_id in self.coupled_to_sister:
                self.coupled_to_sister[extruder_id].remove(sister_id)
                if not self.coupled_to_sister[extruder_id]:  # Empty list
                    del self.coupled_to_sister[extruder_id]
    
    def _uncouple_dead_extruders(self):
        """Vectorized uncoupling of sisters from dead extruders"""
        if self.sister_coupled_to is None:
            return
    
        coupled_mask = self.sister_coupled_to != -1
        if not self.xp.any(coupled_mask):
            return
            
        coupled_sisters = self.xp.where(coupled_mask)[0]
        extruder_ids = self.sister_coupled_to[coupled_sisters]
        
        # Check which extruders are dead (vectorized), if 0 dead, if 1 is active 
        dead_mask = self.states[extruder_ids] == 0 
        dead_couplings = coupled_sisters[dead_mask]
        
        if len(dead_couplings) > 0:
            # Uncouple dead extruders (vectorized)
            dead_extruder_ids = self.sister_coupled_to[dead_couplings]
            self.extruder_sister_counts[dead_extruder_ids] -= 1
            self.sister_coupled_to[dead_couplings] = -1

            for sister_id in dead_couplings:
                sister_id_int = int(sister_id)
                extruder_id = int(dead_extruder_ids[sister_id == dead_couplings][0])
                # Remove from dicts
                if sister_id_int in self.coupled_to_extruder:
                    del self.coupled_to_extruder[sister_id_int]
                if extruder_id in self.coupled_to_sister:
                    if sister_id_int in self.coupled_to_sister[extruder_id]:
                        self.coupled_to_sister[extruder_id].remove(sister_id_int)
                    if not self.coupled_to_sister[extruder_id]:
                        del self.coupled_to_sister[extruder_id]
    
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
        # Vectorized uncoupling: find all sisters coupled to dying extruders
        coupled_sisters_death = self.xp.where(self.xp.isin(self.sister_coupled_to, ids_death))[0]
        if len(coupled_sisters_death) > 0:
            # Reset sister coupling status to -1 for unload coupled sisters
            self.sister_coupled_to[coupled_sisters_death] = -1
        # Reset sister counts for dying extruders
        self.extruder_sister_counts[ids_death] = 0
        # Standard unload for extruders
        self.stalled[ids_death] = False
        self.positions[ids_death] = -1
        # Invalidate position cache
        self._position_cache_valid = False

    def update_states(self, unbound_state_id, bound_state_id):
        ids_birth = self.birth(unbound_state_id)
        ids_death = self.death(bound_state_id)
        
        self.states[ids_birth] = bound_state_id
        self.states[ids_death] = unbound_state_id
        
        self.unload(ids_death)
    
    def update_sister_active_states(self):
        if self.sister_positions is None:
            return
        else:
            ## time_step is 1s 
            decay_prob = 1 - np.exp(-1 / self.sister_tau)
            random_vals = np.random.random(len(self.sister_positions))
            fall_off_mask = random_vals < decay_prob
            ## Set fallen off positions to -1
            self.sister_positions[fall_off_mask] = -1

    def get_coupling_status(self):
        """Optimized coupling status using vectorized operations"""
        # Safety check for initialization
        if self.sister_positions is None or self.sister_coupled_to is None:
            return {
                'coupled_pairs': [],
                'uncoupled_sisters': [],
                'total_sisters': self.num_sisters
            }
            
        coupled_mask = self.sister_coupled_to != -1
        coupled_sisters = self.xp.where(coupled_mask)[0]
        uncoupled_sisters = self.xp.where(~coupled_mask)[0]
        
        coupled_pairs = []
        if len(coupled_sisters) > 0:
            for sister_id in coupled_sisters:
                extruder_id = int(self.sister_coupled_to[sister_id])
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
    

    def setup_test_scenario(self):
        """Setup a single permanent LEF at position [50, 50]"""
        if hasattr(self, '_test_initialized') and self._test_initialized:
            return
        # Find one unbound LEF
        unbound_ids = self.xp.where(self.states == 0)[0]

        if len(unbound_ids) == 0:
            print("No unbound LEFs available")
            return
    
        # Take the first unbound LEF
        # lef_id = unbound_ids[0]
        
        # Set it at position [1, 1] (left) as one test case and make it bound
        # self.positions[lef_id] = self.xp.array([1, 1])
        lef_id = unbound_ids[0:1000]
        starts = np.random.choice(np.arange(32000), size=1000, replace=False)
        self.positions[lef_id] = self.xp.stack([starts, starts], axis=1)
        self.states[lef_id] = 1  # bound state
        self.directions[lef_id] = 0
        self.stalled[lef_id] = False
    
        # Mark as initialized
        self._test_initialized = True
        self._position_cache_valid = False
    
        print(f"Test LEF {lef_id} initialized at position [50, 50]")

    def step(self, mode, unbound_state_id = 0, bound_state_id = 1, active_state_id = 1, **kwargs):
        """Optimized step function"""
       
        ## test simple cases for extruders 
        # self.setup_test_scenario()

        ## Update extruders
        self.update_states(unbound_state_id, bound_state_id)
        
        ## Update sister states for decaying
        # self.update_sister_active_states()
        
        self.check_sister_coupling()      
    
        self.update_occupancies()
        # Parent class step
        super().step(**kwargs)
        
        # Update kwargs with coupling info
        kwargs.update({
            'coupled_to_extruder': self.coupled_to_extruder,
            'coupled_to_sister': self.coupled_to_sister})
        
        # Movement step of sisters and extruders with coupling information
        self.stepping_engine(self, mode, unbound_state_id, active_state_id, **kwargs)


    