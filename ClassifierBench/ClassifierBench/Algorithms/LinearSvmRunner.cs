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
}