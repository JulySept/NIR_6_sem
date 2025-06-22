using Microsoft.ML;

namespace ClassifierBench.Algorithms;

public class NaiveBayesRunner(MLContext mlContext) : AlgorithmRunner(mlContext)
{
    protected override IEstimator<ITransformer> DataPrepEstimator =>
        MlContext.Transforms.Conversion.MapValueToKey("Label").Append(
                MlContext.Transforms.Concatenate("Features", "CapDiameter", "CapShape", "GillAttachment", "GillColor",
                    "StemHeight", "StemWidth", "StemColor", "Season")
            )
            .Append(MlContext.Transforms.NormalizeMinMax("Features"));

    protected override IEstimator<ITransformer> Estimator => MlContext.MulticlassClassification.Trainers.NaiveBayes();
    public override string Name => "NaiveBayes";
}