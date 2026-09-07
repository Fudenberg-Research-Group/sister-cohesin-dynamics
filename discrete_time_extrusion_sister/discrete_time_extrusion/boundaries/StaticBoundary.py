from . import NullBoundary

        
class StaticBoundary(NullBoundary.NullBoundary):
    """CTCF barrier with fixed positions -- step() is a no-op, so CTCF sites never change over the simulation."""
    def __init__(self,
                 stall_left,
                 stall_right,
                 *args, **kwargs):
        
        super().__init__(stall_left, stall_right)

        self.stall_left = stall_left
        self.stall_right = stall_right
        
        self.number = len(self.get_bound_positions())
