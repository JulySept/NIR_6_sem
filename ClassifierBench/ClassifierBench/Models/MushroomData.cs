using Microsoft.ML.Data;

namespace ClassifierBench;

public class MushroomData
{
    [LoadColumn(0)]
    public float CapDiameter { get; set; }

    [LoadColumn(1)]
    public float CapShape { get; set; }

    [LoadColumn(2)]
    public float GillAttachment { get; set; }

    [LoadColumn(3)]
    public float GillColor { get; set; }

    [LoadColumn(4)]
    public float StemHeight { get; set; }

    [LoadColumn(5)]
    public float StemWidth { get; set; }

    [LoadColumn(6)]
    public float StemColor { get; set; }

    [LoadColumn(7)]
    public float Season { get; set; }

    [LoadColumn(8)]
    [ColumnName("Label")]
    public bool IsPoison { get; set; }
}