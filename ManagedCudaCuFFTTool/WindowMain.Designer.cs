namespace ManagedCudaCuFFTTool
{
    partial class WindowMain
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			listBox_log = new ListBox();
			pictureBox_waveform = new PictureBox();
			button_move = new Button();
			button_fft = new Button();
			button_normalize = new Button();
			button_playback = new Button();
			comboBox_cudaDevices = new ComboBox();
			label_vramUsage = new Label();
			progressBar_vram = new ProgressBar();
			numericUpDown_chunkSize = new NumericUpDown();
			button_exportWav = new Button();
			button_kernelCompile = new Button();
			button_kernelLoad = new Button();
			button_kernelRun = new Button();
			numericUpDown_kernelParam1 = new NumericUpDown();
			label_kernelName = new Label();
			button_stretch = new Button();
			numericUpDown_factor = new NumericUpDown();
			((System.ComponentModel.ISupportInitialize) pictureBox_waveform).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_chunkSize).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_kernelParam1).BeginInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_factor).BeginInit();
			SuspendLayout();
			// 
			// listBox_log
			// 
			listBox_log.FormattingEnabled = true;
			listBox_log.ItemHeight = 15;
			listBox_log.Location = new Point(12, 335);
			listBox_log.Name = "listBox_log";
			listBox_log.Size = new Size(440, 94);
			listBox_log.TabIndex = 0;
			// 
			// pictureBox_waveform
			// 
			pictureBox_waveform.BackColor = Color.White;
			pictureBox_waveform.Location = new Point(12, 229);
			pictureBox_waveform.Name = "pictureBox_waveform";
			pictureBox_waveform.Size = new Size(360, 100);
			pictureBox_waveform.TabIndex = 1;
			pictureBox_waveform.TabStop = false;
			// 
			// button_move
			// 
			button_move.Location = new Point(378, 229);
			button_move.Name = "button_move";
			button_move.Size = new Size(74, 20);
			button_move.TabIndex = 2;
			button_move.Text = "Move";
			button_move.UseVisualStyleBackColor = true;
			button_move.Click += button_move_Click;
			// 
			// button_fft
			// 
			button_fft.Location = new Point(378, 255);
			button_fft.Name = "button_fft";
			button_fft.Size = new Size(74, 20);
			button_fft.TabIndex = 3;
			button_fft.Text = "FFT";
			button_fft.UseVisualStyleBackColor = true;
			button_fft.Click += button_fft_Click;
			// 
			// button_normalize
			// 
			button_normalize.Location = new Point(378, 281);
			button_normalize.Name = "button_normalize";
			button_normalize.Size = new Size(74, 20);
			button_normalize.TabIndex = 4;
			button_normalize.Text = "Normalize";
			button_normalize.UseVisualStyleBackColor = true;
			button_normalize.Click += button_normalize_Click;
			// 
			// button_playback
			// 
			button_playback.Location = new Point(378, 307);
			button_playback.Name = "button_playback";
			button_playback.Size = new Size(20, 20);
			button_playback.TabIndex = 5;
			button_playback.Text = "▶";
			button_playback.UseVisualStyleBackColor = true;
			button_playback.Click += button_playback_Click;
			// 
			// comboBox_cudaDevices
			// 
			comboBox_cudaDevices.FormattingEnabled = true;
			comboBox_cudaDevices.Location = new Point(12, 12);
			comboBox_cudaDevices.Name = "comboBox_cudaDevices";
			comboBox_cudaDevices.Size = new Size(200, 23);
			comboBox_cudaDevices.TabIndex = 6;
			// 
			// label_vramUsage
			// 
			label_vramUsage.AutoSize = true;
			label_vramUsage.Location = new Point(12, 38);
			label_vramUsage.Name = "label_vramUsage";
			label_vramUsage.Size = new Size(90, 15);
			label_vramUsage.TabIndex = 7;
			label_vramUsage.Text = "VRAM: 0 / 0 MB";
			// 
			// progressBar_vram
			// 
			progressBar_vram.Location = new Point(12, 56);
			progressBar_vram.Name = "progressBar_vram";
			progressBar_vram.Size = new Size(200, 10);
			progressBar_vram.TabIndex = 8;
			// 
			// numericUpDown_chunkSize
			// 
			numericUpDown_chunkSize.Location = new Point(378, 200);
			numericUpDown_chunkSize.Maximum = new decimal(new int[] { 16777216, 0, 0, 0 });
			numericUpDown_chunkSize.Minimum = new decimal(new int[] { 1024, 0, 0, 0 });
			numericUpDown_chunkSize.Name = "numericUpDown_chunkSize";
			numericUpDown_chunkSize.Size = new Size(74, 23);
			numericUpDown_chunkSize.TabIndex = 9;
			numericUpDown_chunkSize.ThousandsSeparator = true;
			numericUpDown_chunkSize.Value = new decimal(new int[] { 2097152, 0, 0, 0 });
			numericUpDown_chunkSize.ValueChanged += numericUpDown_chunkSize_ValueChanged;
			// 
			// button_exportWav
			// 
			button_exportWav.Location = new Point(404, 307);
			button_exportWav.Name = "button_exportWav";
			button_exportWav.Size = new Size(48, 20);
			button_exportWav.TabIndex = 10;
			button_exportWav.Text = "WAV";
			button_exportWav.UseVisualStyleBackColor = true;
			button_exportWav.Click += button_exportWav_Click;
			// 
			// button_kernelCompile
			// 
			button_kernelCompile.Location = new Point(377, 12);
			button_kernelCompile.Name = "button_kernelCompile";
			button_kernelCompile.Size = new Size(75, 23);
			button_kernelCompile.TabIndex = 13;
			button_kernelCompile.Text = "Compile";
			button_kernelCompile.UseVisualStyleBackColor = true;
			button_kernelCompile.Click += button_kernelCompile_Click;
			// 
			// button_kernelLoad
			// 
			button_kernelLoad.Location = new Point(377, 41);
			button_kernelLoad.Name = "button_kernelLoad";
			button_kernelLoad.Size = new Size(75, 23);
			button_kernelLoad.TabIndex = 14;
			button_kernelLoad.Text = "Load";
			button_kernelLoad.UseVisualStyleBackColor = true;
			button_kernelLoad.Click += button_kernelLoad_Click;
			// 
			// button_kernelRun
			// 
			button_kernelRun.Location = new Point(377, 70);
			button_kernelRun.Name = "button_kernelRun";
			button_kernelRun.Size = new Size(75, 23);
			button_kernelRun.TabIndex = 15;
			button_kernelRun.Text = "Run";
			button_kernelRun.UseVisualStyleBackColor = true;
			button_kernelRun.Click += button_kernelRun_Click;
			// 
			// numericUpDown_kernelParam1
			// 
			numericUpDown_kernelParam1.DecimalPlaces = 4;
			numericUpDown_kernelParam1.Increment = new decimal(new int[] { 5, 0, 0, 131072 });
			numericUpDown_kernelParam1.Location = new Point(311, 70);
			numericUpDown_kernelParam1.Maximum = new decimal(new int[] { 2, 0, 0, 0 });
			numericUpDown_kernelParam1.Minimum = new decimal(new int[] { 5, 0, 0, 131072 });
			numericUpDown_kernelParam1.Name = "numericUpDown_kernelParam1";
			numericUpDown_kernelParam1.Size = new Size(60, 23);
			numericUpDown_kernelParam1.TabIndex = 16;
			numericUpDown_kernelParam1.Value = new decimal(new int[] { 1, 0, 0, 0 });
			// 
			// label_kernelName
			// 
			label_kernelName.AutoSize = true;
			label_kernelName.Location = new Point(377, 96);
			label_kernelName.Name = "label_kernelName";
			label_kernelName.Size = new Size(73, 15);
			label_kernelName.TabIndex = 17;
			label_kernelName.Text = "Kernel name";
			// 
			// button_stretch
			// 
			button_stretch.Location = new Point(12, 145);
			button_stretch.Name = "button_stretch";
			button_stretch.Size = new Size(55, 23);
			button_stretch.TabIndex = 18;
			button_stretch.Text = "Stretch";
			button_stretch.UseVisualStyleBackColor = true;
			button_stretch.Click += button_stretch_Click;
			// 
			// numericUpDown_factor
			// 
			numericUpDown_factor.DecimalPlaces = 10;
			numericUpDown_factor.Location = new Point(73, 145);
			numericUpDown_factor.Maximum = new decimal(new int[] { 5, 0, 0, 0 });
			numericUpDown_factor.Minimum = new decimal(new int[] { 5, 0, 0, 131072 });
			numericUpDown_factor.Name = "numericUpDown_factor";
			numericUpDown_factor.Size = new Size(90, 23);
			numericUpDown_factor.TabIndex = 19;
			numericUpDown_factor.Value = new decimal(new int[] { 1, 0, 0, 0 });
			// 
			// WindowMain
			// 
			AutoScaleDimensions = new SizeF(7F, 15F);
			AutoScaleMode = AutoScaleMode.Font;
			ClientSize = new Size(464, 441);
			Controls.Add(numericUpDown_factor);
			Controls.Add(button_stretch);
			Controls.Add(label_kernelName);
			Controls.Add(numericUpDown_kernelParam1);
			Controls.Add(button_kernelRun);
			Controls.Add(button_kernelLoad);
			Controls.Add(button_kernelCompile);
			Controls.Add(button_exportWav);
			Controls.Add(numericUpDown_chunkSize);
			Controls.Add(progressBar_vram);
			Controls.Add(label_vramUsage);
			Controls.Add(comboBox_cudaDevices);
			Controls.Add(button_playback);
			Controls.Add(button_normalize);
			Controls.Add(button_fft);
			Controls.Add(button_move);
			Controls.Add(pictureBox_waveform);
			Controls.Add(listBox_log);
			MaximizeBox = false;
			MaximumSize = new Size(480, 480);
			MinimumSize = new Size(480, 480);
			Name = "WindowMain";
			Text = "ManagedCuda-12 CuFFT Tool";
			((System.ComponentModel.ISupportInitialize) pictureBox_waveform).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_chunkSize).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_kernelParam1).EndInit();
			((System.ComponentModel.ISupportInitialize) numericUpDown_factor).EndInit();
			ResumeLayout(false);
			PerformLayout();
		}

		#endregion

		private ListBox listBox_log;
		private PictureBox pictureBox_waveform;
		private Button button_move;
		private Button button_fft;
		private Button button_normalize;
		private Button button_playback;
		private ComboBox comboBox_cudaDevices;
		private Label label_vramUsage;
		private ProgressBar progressBar_vram;
		private NumericUpDown numericUpDown_chunkSize;
		private Button button_exportWav;
		private Button button_kernelCompile;
		private Button button_kernelLoad;
		private Button button_kernelRun;
		private NumericUpDown numericUpDown_kernelParam1;
		private Label label_kernelName;
		private Button button_stretch;
		private NumericUpDown numericUpDown_factor;
	}
}
