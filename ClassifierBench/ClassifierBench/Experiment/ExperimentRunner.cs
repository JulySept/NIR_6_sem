using ClassifierBench.Algorithms;
using Microsoft.ML;

namespace ClassifierBench.Experiment;

public class ExperimentRunner(IDataView trainData, IDataView testData)
{
    private readonly PerformanceMeasurer _performanceMeasurer = new();

    public PerformanceMetrics RunExperiment(AlgorithmRunner algorithmRunner, int runsCount = 5)
    {
        var metricsList = new List<PerformanceMetrics>();

        for (int i = 0; i < runsCount; i++)
        {
            var trainDataPrepared = algorithmRunner.PrepareData(trainData);
            var testDataPrepared = algorithmRunner.PrepareData(testData);
            
            var metrics = _performanceMeasurer.Measure(() =>
            {
                algorithmRunner.Train(trainDataPrepared);
                algorithmRunner.Predict(testDataPrepared);
            });
            
            metricsList.Add(metrics);
        }

        return new PerformanceMetrics
        {
            ElapsedMilliseconds = (long)metricsList.Average(m => m.ElapsedMilliseconds),
            CpuTimeMs = metricsList.Average(m => m.CpuTimeMs),
            MemoryKb = metricsList.Average(m => m.MemoryKb)
        };
    }
}