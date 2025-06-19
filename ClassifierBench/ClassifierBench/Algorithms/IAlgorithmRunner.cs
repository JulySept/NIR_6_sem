using Microsoft.ML;

namespace ClassifierBench.Algorithms;

public interface IAlgorithmRunner
{
    string Name { get; }
    IDataView PrepareData(IDataView rawData);
    void Train(IDataView trainData);
    IDataView Predict(IDataView testData);
}