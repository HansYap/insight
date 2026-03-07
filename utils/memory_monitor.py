import psutil
import os

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def process_mb(self):
        """RAM used by this Python process in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def system_available_mb(self):
        """Free RAM on the whole Pi"""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def system_percent(self):
        return psutil.virtual_memory().percent
    
    def report(self):
        return (
            f"Process: {self.process_mb():.0f}MB | "
            f"Available: {self.system_available_mb():.0f}MB | "
            f"System: {self.system_percent():.1f}%"
        )