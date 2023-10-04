using Microsoft.ML.Data;

namespace ML_Regression_H5.ML.Objects
{
    internal class EmploymentHistoryPrediction
    {
        [ColumnName("Score")]
        public float durationInMonths { get; set; }
    }
}
