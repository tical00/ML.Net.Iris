namespace Iris
{
    partial class Form1
    {
        /// <summary>
        /// 필수 디자이너 변수입니다.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 사용 중인 모든 리소스를 정리합니다.
        /// </summary>
        /// <param name="disposing">관리되는 리소스를 삭제해야 하면 true이고, 그렇지 않으면 false입니다.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form 디자이너에서 생성한 코드

        /// <summary>
        /// 디자이너 지원에 필요한 메서드입니다. 
        /// 이 메서드의 내용을 코드 편집기로 수정하지 마세요.
        /// </summary>
        private void InitializeComponent()
        {
            this.petal_width_input = new System.Windows.Forms.TrackBar();
            this.petal_length_input = new System.Windows.Forms.TrackBar();
            this.sepal_width_input = new System.Windows.Forms.TrackBar();
            this.sepal_length_input = new System.Windows.Forms.TrackBar();
            this.model_output = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.result = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.petal_width_input)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.petal_length_input)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.sepal_width_input)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.sepal_length_input)).BeginInit();
            this.SuspendLayout();
            // 
            // petal_width_input
            // 
            this.petal_width_input.AutoSize = false;
            this.petal_width_input.Location = new System.Drawing.Point(270, 155);
            this.petal_width_input.Maximum = 50;
            this.petal_width_input.Minimum = 1;
            this.petal_width_input.Name = "petal_width_input";
            this.petal_width_input.Size = new System.Drawing.Size(280, 20);
            this.petal_width_input.TabIndex = 26;
            this.petal_width_input.TickStyle = System.Windows.Forms.TickStyle.None;
            this.petal_width_input.Value = 1;
            this.petal_width_input.ValueChanged += new System.EventHandler(this.input_ValueChanged);
            // 
            // petal_length_input
            // 
            this.petal_length_input.AutoSize = false;
            this.petal_length_input.Location = new System.Drawing.Point(270, 130);
            this.petal_length_input.Maximum = 100;
            this.petal_length_input.Minimum = 5;
            this.petal_length_input.Name = "petal_length_input";
            this.petal_length_input.Size = new System.Drawing.Size(280, 20);
            this.petal_length_input.TabIndex = 25;
            this.petal_length_input.TickStyle = System.Windows.Forms.TickStyle.None;
            this.petal_length_input.Value = 5;
            this.petal_length_input.ValueChanged += new System.EventHandler(this.input_ValueChanged);
            // 
            // sepal_width_input
            // 
            this.sepal_width_input.AutoSize = false;
            this.sepal_width_input.Location = new System.Drawing.Point(270, 105);
            this.sepal_width_input.Maximum = 80;
            this.sepal_width_input.Minimum = 10;
            this.sepal_width_input.Name = "sepal_width_input";
            this.sepal_width_input.Size = new System.Drawing.Size(280, 20);
            this.sepal_width_input.TabIndex = 24;
            this.sepal_width_input.TickStyle = System.Windows.Forms.TickStyle.None;
            this.sepal_width_input.Value = 10;
            this.sepal_width_input.ValueChanged += new System.EventHandler(this.input_ValueChanged);
            // 
            // sepal_length_input
            // 
            this.sepal_length_input.AutoSize = false;
            this.sepal_length_input.Location = new System.Drawing.Point(270, 80);
            this.sepal_length_input.Maximum = 100;
            this.sepal_length_input.Minimum = 10;
            this.sepal_length_input.Name = "sepal_length_input";
            this.sepal_length_input.Size = new System.Drawing.Size(280, 20);
            this.sepal_length_input.TabIndex = 23;
            this.sepal_length_input.TickStyle = System.Windows.Forms.TickStyle.None;
            this.sepal_length_input.Value = 10;
            this.sepal_length_input.ValueChanged += new System.EventHandler(this.input_ValueChanged);
            // 
            // model_output
            // 
            this.model_output.AutoSize = true;
            this.model_output.Font = new System.Drawing.Font("맑은 고딕", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.model_output.Location = new System.Drawing.Point(340, 210);
            this.model_output.Name = "model_output";
            this.model_output.Size = new System.Drawing.Size(95, 19);
            this.model_output.TabIndex = 22;
            this.model_output.Text = "Model output";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Font = new System.Drawing.Font("맑은 고딕", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label8.Location = new System.Drawing.Point(5, 155);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(237, 19);
            this.label8.TabIndex = 21;
            this.label8.Text = "sepal width in mm [range of 1 - 50]:";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Font = new System.Drawing.Font("맑은 고딕", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label7.Location = new System.Drawing.Point(5, 130);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(249, 19);
            this.label7.TabIndex = 20;
            this.label7.Text = "petal length in mm [range of 5 - 100]:";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("맑은 고딕", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label6.Location = new System.Drawing.Point(5, 105);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(245, 19);
            this.label6.TabIndex = 19;
            this.label6.Text = "sepal width in mm [range of 10 - 80]:";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("맑은 고딕", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label5.Location = new System.Drawing.Point(5, 80);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(258, 19);
            this.label5.TabIndex = 18;
            this.label5.Text = "sepal length in mm [range of 10 - 100]:";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("맑은 고딕", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label4.Location = new System.Drawing.Point(5, 210);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(326, 19);
            this.label4.TabIndex = 17;
            this.label4.Text = "Based on the information provided, the Iris type is:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("맑은 고딕", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label3.Location = new System.Drawing.Point(5, 180);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(68, 21);
            this.label3.TabIndex = 16;
            this.label3.Text = "Output:";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("맑은 고딕", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label2.Location = new System.Drawing.Point(5, 50);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(155, 21);
            this.label2.TabIndex = 15;
            this.label2.Text = "Provide the input :";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("맑은 고딕", 20F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Underline))), System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.label1.Location = new System.Drawing.Point(5, 5);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(445, 37);
            this.label1.TabIndex = 14;
            this.label1.Text = "Data Analysis App - Windows ML";
            // 
            // button1
            // 
            this.button1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.button1.Location = new System.Drawing.Point(690, 275);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(105, 48);
            this.button1.TabIndex = 27;
            this.button1.Text = "convert ONNX";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // result
            // 
            this.result.AutoSize = true;
            this.result.Font = new System.Drawing.Font("맑은 고딕", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(129)));
            this.result.Location = new System.Drawing.Point(5, 235);
            this.result.Name = "result";
            this.result.Size = new System.Drawing.Size(0, 19);
            this.result.TabIndex = 28;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 329);
            this.Controls.Add(this.result);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.petal_width_input);
            this.Controls.Add(this.petal_length_input);
            this.Controls.Add(this.sepal_width_input);
            this.Controls.Add(this.sepal_length_input);
            this.Controls.Add(this.model_output);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.petal_width_input)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.petal_length_input)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.sepal_width_input)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.sepal_length_input)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.TrackBar petal_width_input;
        private System.Windows.Forms.TrackBar petal_length_input;
        private System.Windows.Forms.TrackBar sepal_width_input;
        private System.Windows.Forms.TrackBar sepal_length_input;
        private System.Windows.Forms.Label model_output;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Label result;
    }
}

