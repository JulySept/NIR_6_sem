using Microsoft.ML;

namespace ClassifierBench.Algorithms;

public class LdSvmRunner(MLContext mlContext) : AlgorithmRunner(mlContext)
{
    protected override IEstimator<ITransformer> DataPrepEstimator => MlContext.Transforms.Concatenate("Features",
            "CapDiameter", "CapShape", "GillAttachment", "GillColor",
            "StemHeight", "StemWidth", "StemColor", "Season")
        .Append(MlContext.Transforms.NormalizeMinMax("Features"));

    protected override IEstimator<ITransformer> Estimator => MlContext.BinaryClassification.Trainers.LdSvm();
    public override string Name => "LdSvm";
}