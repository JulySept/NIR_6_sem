using Microsoft.ML;

namespace ClassifierBench.Algorithms;

public abstract class AlgorithmRunner(MLContext mlContext)
{
    protected MLContext MlContext => mlContext;
    private ITransformer? _model;
    protected abstract IEstimator<ITransformer> DataPrepEstimator { get; }
    protected abstract IEstimator<ITransformer> Estimator { get; }
    public abstract string Name { get; }

    public IDataView PrepareData(IDataView data)
    {
        return DataPrepEstimator.Fit(data).Transform(data);
    }

    public void Train(IDataView trainData)
    {
        _model = Estimator.Fit(trainData);
    }

    public IDataView Predict(IDataView testData)
    {
        if (_model == null)
            throw new InvalidOperationException("Model is not trained.");

        return _model.Transform(testData);
    }
    
    public abstract Dictionary<string, double> Evaluate(IDataView predictions);
}