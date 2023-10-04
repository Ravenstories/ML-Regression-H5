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
            
            //var sampleData = mlContext.Data.LoadFromTextFile<EmploymentHistory>(trainingFileName, ',');
            var trainingDataView = mlContext.Data.LoadFromTextFile<EmploymentHistory>(trainingFileName, ',');
            
            var dataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

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
            
            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, Constants.modelFile);

            var testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            var modelMetrics = mlContext.Regression.Evaluate(testSetTransform);

            Console.WriteLine($"Loss Function: {modelMetrics.LossFunction:0.##}{Environment.NewLine}" +
                $"Mean Absolute Error: {modelMetrics.MeanAbsoluteError:#.##}{Environment.NewLine}" +
                $"Mean Squared Error: {modelMetrics.MeanSquaredError:#.##}{Environment.NewLine}" +
                $"RSquared: {modelMetrics.RSquared:0.##}{Environment.NewLine}" +
                $"Root Mean Squared Error: {modelMetrics.RootMeanSquaredError:#.##} \n");
        }
    }
}
