using Microsoft.ML.Data;

namespace ML_Regression_H5.ML.Objects
{
    internal class EmploymentHistoryPrediction
    {
        [ColumnName("durationInMonths")]
        public float durationInMonths { get; set; }
    }
}
