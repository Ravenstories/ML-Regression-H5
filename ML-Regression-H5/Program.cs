using Microsoft.VisualBasic;
using ML_Regression_H5.ML;

internal class Program
{
    private static void Main(string[] args)
    {
        while (true)
        {
            Console.WriteLine("Please provide a valid option predict/train/exit");

            var option = Console.ReadLine();


            switch (option)
            {
                case "predict":
                    if (!File.Exists(ML_Regression_H5.Common.Constants.inputData))
                    {
                        Console.WriteLine($"Couldn't find file: {ML_Regression_H5.Common.Constants.inputData}");
                        return;
                    }
                    new Predictor().Predict(ML_Regression_H5.Common.Constants.inputData);
                    break;
                case "train":
                    new Trainer().Train(ML_Regression_H5.Common.Constants.sampleData);
                    break;
                case "exit":
                    return;

                default:
                    Console.WriteLine($"{args[0]} is an invalid option");
                    break;
            }

        }
    }
}