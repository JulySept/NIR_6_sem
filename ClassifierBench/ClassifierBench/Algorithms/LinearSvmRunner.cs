using Microsoft.ML;

namespace ClassifierBench.Algorithms;

public class LinearSvmRunner(MLContext mlContext) : AlgorithmRunner(mlContext)
{
    protected override IEstimator<ITransformer> DataPrepEstimator => MlContext.Transforms.Concatenate("Features",
            "CapDiameter", "CapShape", "GillAttachment", "GillColor",
            "StemHeight", "StemWidth", "StemColor", "Season")
        .Append(MlContext.Transforms.NormalizeMinMax("Features"));

    protected override IEstimator<ITransformer> Estimator => MlContext.BinaryClassification.Trainers.LinearSvm();
    public override string Name => "LinearSvm";
    public override IDictionary<string, double> Evaluate(IDataView predictions)
    {
        var metrics = MlContext.BinaryClassification.EvaluateNonCalibrated(predictions);
        return new Dictionary<string, double>
        {
            ["Accuracy"] = metrics.Accuracy,
            ["AUC"] = metrics.AreaUnderRocCurve,
            ["F1Score"] = metrics.F1Score,
        };
    }
}