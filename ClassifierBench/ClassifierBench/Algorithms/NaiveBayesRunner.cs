using Microsoft.ML;

namespace ClassifierBench.Algorithms;

public class NaiveBayesRunner(MLContext mlContext) : IAlgorithmRunner
{
    private ITransformer? _model;

    private readonly IEstimator<ITransformer> _dataPrepEstimator =
        mlContext.Transforms.Conversion.MapValueToKey("Label").Append(
                mlContext.Transforms.Concatenate("Features", "CapDiameter", "CapShape", "GillAttachment", "GillColor",
                    "StemHeight", "StemWidth", "StemColor", "Season")
            )
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

    private readonly IEstimator<ITransformer> _estimator = mlContext.MulticlassClassification.Trainers.NaiveBayes();
    private bool _isModelTrained = false;
    public string Name => "NaiveBayes";

    public IDataView PrepareData(IDataView data)
    {
        return _dataPrepEstimator.Fit(data).Transform(data);
    }

    public void Train(IDataView trainData)
    {
        _model = _estimator.Fit(trainData);
        _isModelTrained = true;
    }

    public IDataView Predict(IDataView testData)
    {
        return _isModelTrained ? _model.Transform(testData) : null;
    }
}