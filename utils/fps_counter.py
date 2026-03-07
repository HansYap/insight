import time
from collections import deque

class FPSCounter:
    def __init__(self, window=30):
        self.timestamps = deque(maxlen=window)
    
    def tick(self):
        self.timestamps.append(time.perf_counter())
    
    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
    
    def __str__(self):
        return f"{self.fps:.1f} FPS"