import time
import psutil
import os
import gc
import tracemalloc

class PerformanceMeasurer:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def measure(self, action):
        gc.collect()

        tracemalloc.start()
        memory_before = self.process.memory_info().rss
        allocated_before, _ = tracemalloc.get_traced_memory()

        cpu_time_before = time.process_time()
        start_time = time.perf_counter()

        result = action()

        end_time = time.perf_counter()
        cpu_time_after = time.process_time()
        memory_after = self.process.memory_info().rss
        allocated_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_ms = (end_time - start_time) * 1000
        cpu_time_ms = (cpu_time_after - cpu_time_before) * 1000
        memory_kb = (memory_after - memory_before) / 1024
        allocated_mb = (allocated_after - allocated_before) / (1024 * 1024)

        return result, {
            "elapsed_ms": elapsed_ms,
            "cpu_time_ms": cpu_time_ms,
            "memory_kb": memory_kb,
            "allocated_memory_mb": allocated_mb
        }
