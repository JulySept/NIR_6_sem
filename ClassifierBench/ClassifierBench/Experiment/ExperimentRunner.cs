using ClassifierBench.Algorithms;
using Microsoft.ML;

namespace ClassifierBench.Experiment;

public class ExperimentRunner(IDataView trainData, IDataView testData)
{
    private readonly PerformanceMeasurer _performanceMeasurer = new();

    public ExperimentResult RunExperiment(AlgorithmRunner algorithmRunner, int runsCount = 5)
    {
        var perfMetricsList = new List<PerformanceMetrics>();
        var modelMetricsList = new List<Dictionary<string, double>>();

        for (int i = 0; i < runsCount; i++)
        {
            var trainDataPrepared = algorithmRunner.PrepareData(trainData);
            var testDataPrepared = algorithmRunner.PrepareData(testData);
            IDataView predictions = null!;

            var metrics = _performanceMeasurer.Measure(() =>
            {
                algorithmRunner.Train(trainDataPrepared);
                predictions = algorithmRunner.Predict(testDataPrepared);
            });

            perfMetricsList.Add(metrics);
            var currentModelMetrics = algorithmRunner.Evaluate(predictions);
            modelMetricsList.Add(currentModelMetrics);
        }

        return new ExperimentResult
        {
            Performance = new PerformanceMetrics
            {
                ElapsedMilliseconds = (long)perfMetricsList.Average(m => m.ElapsedMilliseconds),
                CpuTimeMs = perfMetricsList.Average(m => m.CpuTimeMs),
                MemoryKb = perfMetricsList.Average(m => m.MemoryKb),
                AllocatedMemoryMb = perfMetricsList.Average(m => m.AllocatedMemoryMb)
            },
            ModelMetrics = AverageModelMetrics(modelMetricsList)
        };
    }

    private static Dictionary<string, double> AverageModelMetrics(List<Dictionary<string, double>> metricsList)
    {
        return metricsList
            .SelectMany(dict => dict)
            .GroupBy(kv => kv.Key)
            .ToDictionary(
                g => g.Key,
                g => g.Average(kv => kv.Value)
            );
    }
}