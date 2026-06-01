class NullBoundary():
     
    def __init__(self,
                 stall_left,
                 stall_right,
                 *args, **kwargs):

        try:
            import cupy as cp
            self.xp = cp.get_array_module(stall_left)
            
        except:
            import numpy as np
            self.xp = np

        self.number = 0
        self.lattice_size = len(stall_left)
        
        self.stall_left = self.xp.zeros_like(stall_left)
        self.stall_right = self.xp.zeros_like(stall_right)
        
        self.get_list = lambda x: x.get().tolist() if self.xp.__name__ == 'cupy' else x.tolist()


    def step(self, *args, **kwargs):
    
        pass
        
        
    def get_bound_positions(self):

        bound_left_positions = self.xp.flatnonzero(self.stall_left)
        bound_right_positions = self.xp.flatnonzero(self.stall_right)
    
        return self.get_list(bound_left_positions) + self.get_list(bound_right_positions)
