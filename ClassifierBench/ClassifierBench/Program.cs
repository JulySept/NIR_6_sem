using ClassifierBench.Algorithms;
using ClassifierBench.Experiment;
using Microsoft.ML;

namespace ClassifierBench;

public class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();
        string dataPath = "Data/mushrooms_cleaned.csv";
        IDataView rawData = mlContext.Data.LoadFromTextFile<MushroomData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',');
        var split = mlContext.Data.TrainTestSplit(rawData, testFraction: 0.2);
        var trainData = split.TrainSet;
        var testData = split.TestSet;
        var algorithms = new List<IAlgorithmRunner>
        {
            new LdSvmRunner(mlContext),
            new PriorTrainerRunner(mlContext),
            new NaiveBayesRunner(mlContext),
            new LinearSvmRunner(mlContext),
        };

        int runsCount = 5;
        var experimentRunner = new ExperimentRunner(trainData, testData);
        foreach (var algorithm in algorithms)
        {
            Console.WriteLine($"Запускаем алгоритм: {algorithm.Name}");

            var avgMetrics = experimentRunner.RunExperiment(algorithm, runsCount);
            
            Console.WriteLine($"Средние показатели для {algorithm.Name}:");
            Console.WriteLine($"  Время выполнения: {avgMetrics.ElapsedMilliseconds} ms");
            Console.WriteLine($"  CPU время: {avgMetrics.CpuTimeMs} ms");
            Console.WriteLine($"  Память: {avgMetrics.MemoryKb:F2} Kb");
            Console.WriteLine();
        }
    }
}