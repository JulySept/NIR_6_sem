using ClassifierBench.Algorithms;
using ClassifierBench.Experiment;
using Microsoft.ML;

namespace ClassifierBench;

public static class Program
{
    static void Main()
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
        var algorithms = new List<AlgorithmRunner>
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

            var experimentResult = experimentRunner.RunExperiment(algorithm, runsCount);

            Console.WriteLine($"Результаты эксперимента для {algorithm.Name}:");
            Console.WriteLine($"  Время выполнения: {experimentResult.Performance.ElapsedMilliseconds} ms");
            Console.WriteLine($"  CPU время: {experimentResult.Performance.CpuTimeMs} ms");
            Console.WriteLine($"  Память: {experimentResult.Performance.MemoryKb:F2} Kb");
            Console.WriteLine($"  Аллоцировано памяти: {experimentResult.Performance.AllocatedMemoryMb:F2} Mb");

            if (experimentResult.ModelMetrics != null)
            {
                Console.WriteLine("  Метрики модели:");
                foreach (var (name, value) in experimentResult.ModelMetrics)
                {
                    Console.WriteLine($"    {name}: {value:F4}");
                }
            }
            Console.WriteLine();

        }
    }
}