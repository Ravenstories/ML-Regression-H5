﻿using Microsoft.ML;
using ML_Regression_H5.ML.Base;
using ML_Regression_H5.ML.Objects;
using Newtonsoft.Json;
using Constants = ML_Regression_H5.Common.Constants;

namespace ML_Regression_H5.ML
{
    class Predictor : BaseML
    {
        public void Predict(string inputData)
        {
            if (!File.Exists(Constants.modelFile))
            {
                Console.WriteLine($"Failed to find model at {Constants.modelFile}");
                return;
            }else if (!File.Exists(Constants.sampleData))
            {
                Console.WriteLine($"Failed to find model at {Constants.sampleData}");
                return;
            }

            ITransformer mlModel;

            using (var stream = new FileStream(Constants.modelFile, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }
            if (mlModel == null)
            {
                Console.WriteLine("Failed to load model");
                return;
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<EmploymentHistory, EmploymentHistoryPrediction>(mlModel);

            // Læs JSON-data fra filen
            var json = File.ReadAllText(inputData);
            // Deserialiser JSON til EmploymentHistory objekt
            var employmentHistory = JsonConvert.DeserializeObject<EmploymentHistory>(json);
            // Udfør forudsigelsen
            var prediction = predictionEngine.Predict(employmentHistory);


            Console.WriteLine($"Based on input JSON:{Environment.NewLine}{json}{Environment.NewLine}");
            Console.WriteLine($"The employee is predicted to work {prediction.DurationInMonths} months");
        }
    }
}