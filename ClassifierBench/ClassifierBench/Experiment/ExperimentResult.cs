namespace ClassifierBench.Experiment;

public class ExperimentResult
{
    public PerformanceMetrics Performance { get; set; } = null!;
    public IDictionary<string, double>? ModelMetrics { get; set; }
}