using Microsoft.ML;

namespace ML_Regression_H5.ML.Base
{
    public class BaseML 
    {
        protected readonly MLContext mlContext;
        protected BaseML()
        {
            mlContext = new MLContext();
        }


    }
}
