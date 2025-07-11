using System.Diagnostics;

namespace ClassifierBench.Experiment;

public class PerformanceMeasurer
{
    private readonly Process _process;

    public PerformanceMeasurer()
    {
        _process = Process.GetCurrentProcess();
    }

    public PerformanceMetrics Measure(Action action)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        long memoryBefore = GC.GetTotalMemory(forceFullCollection: true);
        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
        TimeSpan cpuBefore = _process.TotalProcessorTime;
        var stopwatch = Stopwatch.StartNew();

        action();

        stopwatch.Stop();
        _process.Refresh(); 
        long memoryAfter = GC.GetTotalMemory(forceFullCollection: true);
        long allocatedAfter = GC.GetAllocatedBytesForCurrentThread();
        TimeSpan cpuAfter = _process.TotalProcessorTime;

        return new PerformanceMetrics
        {
            ElapsedMilliseconds = stopwatch.ElapsedMilliseconds,
            MemoryKb = (memoryAfter - memoryBefore) / 1024.0,
            AllocatedMemoryMb = (allocatedAfter - allocatedBefore) / 1024.0 / 1024.0,
            CpuTimeMs = (cpuAfter - cpuBefore).TotalMilliseconds
        };
    }
}