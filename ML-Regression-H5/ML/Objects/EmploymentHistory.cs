using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Regression_H5.ML.Objects
{
    internal class EmploymentHistory
    {
        [LoadColumn(0)] 
        public float durationInMonths { get; set; }

        [LoadColumn(1)] 
        public float isMarried { get; set; }

        [LoadColumn(2)]
        public float bsDegree { get; set; }

        [LoadColumn(3)]
        public float msDegree { get; set; }

        [LoadColumn(4)] 
        public float yearsExperience { get; set; }

        [LoadColumn(5)] 
        public float ageAtHire { get; set; }

        [LoadColumn(6)]
        public float hasKids { get; set; }

        [LoadColumn(7)] 
        public float withinMonthOfVesting { get; set; }

        [LoadColumn(8)] 
        public float deskDecorations { get; set; }

        [LoadColumn(9)] 
        public float longCommute { get; set; }
    }

}
