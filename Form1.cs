using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Iris;
using System.IO;
using static Iris.Iris_dataset;

namespace Iris
{
    
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            
        }

        private string[] _labels = { "Iris-verginica", "Iris-versiocolor", "Iris-setosa" };

        private void input_ValueChanged(object sender, EventArgs e)
        {
            Iris_dataset.ModelInput sampleData = new Iris_dataset.ModelInput()
            {
                Sepal_length = (float)sepal_length_input.Value / 10.0f,
                Sepal_width = (float)sepal_width_input.Value / 10.0f,
                Petal_length = (float)petal_length_input.Value / 10.0f,
                Petal_width = (float)petal_width_input.Value / 10.0f,
            };

            var predictionResult = Iris_dataset.Predict(sampleData);

            string strResult = "";

            int num = 0;
            for (int i = 0; i < predictionResult.Score.Length; i++)
            {
                var score = predictionResult.Score[i];
                //if (score > max_value) { max_value = score; max_index = num; }
                strResult += $"Case #{num + 1} {_labels.ElementAt(num)} Score: {score * 100}\n";
                num++;
            }

            result.Text = strResult;
            model_output.Text = predictionResult.PredictedLabel;
       
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Iris_dataset.ModelInput sampleData = new Iris_dataset.ModelInput()
            {
                Sepal_length = (float)sepal_length_input.Value / 10.0f,
                Sepal_width = (float)sepal_width_input.Value / 10.0f,
                Petal_length = (float)petal_length_input.Value / 10.0f,
                Petal_width = (float)petal_width_input.Value / 10.0f,
            };

            var stream = Convert(sampleData);

            using (SaveFileDialog dr = new SaveFileDialog())
            {
                dr.Filter = "ONNX file(*.onnx)|*.onnx";
                if (dr.ShowDialog() == DialogResult.OK)
                {
                    //using (Stream inputStream = File.OpenRead("inputfile.txt"))
                    using (Stream outputStream = File.Create(dr.FileName))
                    {
                        stream.Seek(0, SeekOrigin.Begin);
                        stream.CopyTo(outputStream);
                    }

                }
            }
        }

        public static Stream Convert(ModelInput input)
        {
            var mlContext = new MLContext();
            var mlModel = mlContext.Model.Load(Iris_dataset.MLNetModelPath, out var _);

            List<ModelInput> l = new List<ModelInput>();
            l.Add(input);
            var inputIDataview = mlContext.Data.LoadFromEnumerable<ModelInput>(l);
            var stream = new MemoryStream();
            mlContext.Model.ConvertToOnnx(mlModel, inputIDataview, stream);
            return stream;
        }
    }
}
