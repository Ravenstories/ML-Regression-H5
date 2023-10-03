using Microsoft.ML;
using ML_Regression_H5.ML.Base;
using ML_Regression_H5.ML.Objects;
using ML_Regression_H5.Common;

namespace ML_Regression_H5.ML
{
    class Trainer : BaseML
    {
        public void Train(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file {trainingFileName}");
                return;
            }

            var trainingDataView = mlContext.Data.LoadFromTextFile<EmploymentHistory>(trainingFileName, ',');

            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", nameof(EmploymentHistory.durationInMonths))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.isMarried)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.bsDegree)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.msDegree)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.yearsExperience)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.ageAtHire)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.hasKids)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.withinMonthOfVesting)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.deskDecorations)))
                .Append(mlContext.Transforms.NormalizeMeanVariance(nameof(EmploymentHistory.longCommute)))
                .Append(mlContext.Transforms.Concatenate("Features",
                    typeof(EmploymentHistory).ToPropertyList<EmploymentHistory>(nameof(EmploymentHistory.durationInMonths))));


            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, Constants.modelFile);

            var testSetTransform = trainedModel.Transform(trainingDataView);

            var modelMetrics = mlContext.Regression.Evaluate(testSetTransform);

            Console.WriteLine($"Loss Function: {modelMetrics.LossFunction:0.##}{Environment.NewLine}" +
                $"Mean Absolute Error: {modelMetrics.MeanAbsoluteError:#.##}{Environment.NewLine}" +
                $"Mean Squared Error: {modelMetrics.MeanSquaredError:#.##}{Environment.NewLine}" +
                $"RSquared: {modelMetrics.RSquared:0.##}{Environment.NewLine}" +
                $"Root Mean Squared Error: {modelMetrics.RootMeanSquaredError:#.##}");
        }
    }
}
