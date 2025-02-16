
namespace ManagedCudaCuFFTTool
{
	public partial class WindowMain : Form
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\
		public string Repopath;

		public AudioHandling? AudioH = null;
		public CudaHandling CudaH;


		private int lastChunkSize = 2097152;


		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTORS ~~~~~ ~~~~~ ~~~~~ \\
		public WindowMain()
		{
			InitializeComponent();

			// Window position
			this.StartPosition = FormStartPosition.Manual;
			this.Location = new Point(0, 0);

			// Set repo path
			Repopath = GetRepopath(true);

			// Initialize classes
			AudioH = null;
			CudaH = new CudaHandling(Repopath, 0, listBox_log, comboBox_cudaDevices, label_vramUsage, progressBar_vram, label_kernelName);

			// Register events
			this.pictureBox_waveform.DoubleClick += ImportTrack;

			// Setup GUI
			SetWaveform();
		}





		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public string GetRepopath(bool root = false)
		{
			string repo = AppDomain.CurrentDomain.BaseDirectory;

			if (root)
			{
				repo += @"..\..\..\";
			}

			repo = Path.GetFullPath(repo);

			return repo;
		}

		public void Log(string message, string inner = "", int layer = 0)
		{
			CudaH.Log(message, inner, layer);
		}

		public void SetWaveform()
		{
			ToggleButtons();

			if (AudioH == null || AudioH.Floats.Length == 0)
			{
				pictureBox_waveform.Image = null;
				pictureBox_waveform.Refresh();
				return;
			}

			// Draw waveform
			int res = AudioH.GetFitResolution(pictureBox_waveform.Width);
			AudioH.DrawWaveformSmooth(pictureBox_waveform, 0, res, true);
		}

		public void ToggleButtons()
		{
			// Move button
			button_move.Enabled = AudioH != null && (AudioH.Floats.Length > 0 || CudaH.FloatVariables.Count > 0);
			button_move.Text = CudaH.FloatVariables.Count > 0 ? "Host <-" : "-> CUDA";

			// FFT button
			button_fft.Enabled = AudioH != null && (CudaH.FloatVariables.Count > 0 || CudaH.ComplexVariables.Count > 0);
			button_fft.Text = CudaH.ComplexVariables.Count > 0 ? "IFFT" : "FFT";

			// Normalize button
			button_normalize.Enabled = AudioH != null && AudioH.Floats.Length > 0;

			// Playback button
			button_playback.Enabled = AudioH != null && AudioH.Floats.Length > 0;
			button_playback.Text = AudioH?.IsPlaying == true ? "⏹" : "▶";

			// Export button
			button_exportWav.Enabled = AudioH != null && AudioH.Floats.Length > 0;

			// Compile button
			button_kernelCompile.Enabled = CudaH.Ctx != null;

			// Load kernel button
			button_kernelLoad.Enabled = CudaH.Ctx != null;

			// Run kernel button
			button_kernelRun.Enabled = CudaH.Ctx != null && CudaH.KernelH != null && CudaH.KernelH.Kernel != null && (CudaH.FloatVariables.Count > 0 || CudaH.ComplexVariables.Count > 0);

			// Stretch button
			button_stretch.Enabled = CudaH.Ctx != null && AudioH != null && (AudioH.Floats.Length > 0 || CudaH.FloatVariables.Count > 0);
		}







		// ~~~~~ ~~~~~ ~~~~~ EVENTS ~~~~~ ~~~~~ ~~~~~ \\
		private void ImportTrack(object? sender, EventArgs e)
		{
			// OFD at MyMusic, single audio file
			OpenFileDialog ofd = new();
			ofd.Title = "Import audio file";
			ofd.Multiselect = false;
			ofd.Filter = "Audio files|*.wav;*.mp3;*.flac";
			ofd.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);

			// OFD show -> AudioHandling
			if (ofd.ShowDialog() == DialogResult.OK)
			{
				AudioH = new AudioHandling(ofd.FileName);
			}

			// Update GUI
			SetWaveform();
		}

		private void button_move_Click(object sender, EventArgs e)
		{
			// If no audio or Ctx is null, return
			if (AudioH == null || CudaH.Ctx == null)
			{
				return;
			}

			// If data on Host, move to CUDA
			if (AudioH.Floats.Length > 0)
			{
				CudaH.PushArraysToCuda(AudioH.MakeChunks((int) numericUpDown_chunkSize.Value));

				AudioH.Floats = [];
			}
			// If data on CUDA, move to Host
			else if (CudaH.FloatVariables.Count > 0)
			{
				AudioH.Floats = AudioH.AggregateChunks(CudaH.PullArraysFromCuda<float>());

				CudaH.FloatVariables.Clear();
			}

			// Update GUI
			SetWaveform();
		}

		private void numericUpDown_chunkSize_ValueChanged(object sender, EventArgs e)
		{
			// If chunk size increased, *2
			if (numericUpDown_chunkSize.Value > lastChunkSize)
			{
				numericUpDown_chunkSize.Value = Math.Min(numericUpDown_chunkSize.Maximum, lastChunkSize * 2);
			}

			// If chunk size decreased, /2
			if (numericUpDown_chunkSize.Value < lastChunkSize)
			{
				numericUpDown_chunkSize.Value = Math.Max(numericUpDown_chunkSize.Minimum, lastChunkSize / 2);
			}

			// Update last chunk size
			lastChunkSize = (int) numericUpDown_chunkSize.Value;
		}

		private void button_fft_Click(object sender, EventArgs e)
		{
			// If no audio or Ctx is null, return
			if (AudioH == null || CudaH.Ctx == null)
			{
				return;
			}

			// If no data on CUDA, return
			if (CudaH.FloatVariables.Count == 0 && CudaH.ComplexVariables.Count == 0)
			{
				return;
			}

			// If FloatVariables, FFT
			if (CudaH.FloatVariables.Count > 0)
			{
				CudaH.PerformFFT();
			}

			// If ComplexVariables, IFFT
			else if (CudaH.ComplexVariables.Count > 0)
			{
				CudaH.PerformIFFT();
			}

			// Update GUI
			SetWaveform();
		}

		private void button_normalize_Click(object sender, EventArgs e)
		{
			// If no audio, return
			if (AudioH == null || AudioH.Floats.Length == 0)
			{
				return;
			}

			// Normalize audio
			AudioH.Normalize();

			// Update GUI
			SetWaveform();
		}

		private void button_playback_Click(object sender, EventArgs e)
		{
			// If no audio, return
			if (AudioH == null || AudioH.Floats.Length == 0)
			{
				return;
			}

			// Play or stop audio
			AudioH.PlayStop(button_playback);

			// Update GUI
			SetWaveform();
		}

		private void button_exportWav_Click(object sender, EventArgs e)
		{
			// Abort if no audio
			if (AudioH == null || AudioH.Floats.Length == 0)
			{
				return;
			}

			// SFD at MyMusic, single audio file
			SaveFileDialog sfd = new();
			sfd.Title = "Export audio file";
			sfd.FileName = AudioH.Name + "_cufft.wav";
			sfd.Filter = "WAV files|*.wav";
			sfd.InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyMusic);

			// SFD show -> AudioHandling
			if (sfd.ShowDialog() == DialogResult.OK)
			{
				AudioH.ExportAudioWav(sfd.FileName);
			}

			// Update GUI
			Log("Exported audio to WAV", sfd.FileName);
			SetWaveform();
		}

		private void button_kernelCompile_Click(object sender, EventArgs e)
		{
			// Abort if no Ctx
			if (CudaH.Ctx == null || CudaH.KernelH == null)
			{
				return;
			}

			// OFD at Kernels, single kernel file (.c / .txt / .cu)
			OpenFileDialog ofd = new();
			ofd.Title = "Import kernel file";
			ofd.Multiselect = false;
			ofd.Filter = "Kernel files|*.c;*.txt;*.cu";
			ofd.InitialDirectory = Path.Combine(Repopath, "Resources\\Kernels\\C");

			// OFD show -> KernelHandling
			if (ofd.ShowDialog() == DialogResult.OK)
			{
				CudaH.KernelH.CompileKernel(ofd.FileName, false, true);
			}

			// Update GUI
			SetWaveform();
		}

		private void button_kernelLoad_Click(object sender, EventArgs e)
		{
			// Abort if no Ctx or KernelH
			if (CudaH.Ctx == null || CudaH.KernelH == null)
			{
				return;
			}

			// OFD at PTX, single PTX file
			OpenFileDialog ofd = new();
			ofd.Title = "Import PTX file";
			ofd.Multiselect = false;
			ofd.Filter = "PTX files|*.ptx";
			ofd.InitialDirectory = Path.Combine(Repopath, "Resources\\Kernels\\PTX");

			// OFD show -> KernelHandling
			if (ofd.ShowDialog() == DialogResult.OK)
			{
				CudaH.KernelH.LoadKernel(ofd.FileName);
			}

			// Update GUI
			SetWaveform();
		}

		private void button_kernelRun_Click(object sender, EventArgs e)
		{
			// Abort if no Ctx or KernelH
			if (CudaH.Ctx == null || CudaH.KernelH == null || CudaH.KernelH.Kernel == null || (CudaH.FloatVariables.Count == 0 && CudaH.ComplexVariables.Count == 0))
			{
				return;
			}

			if (CudaH.KernelH.Kernel.KernelName.ToLower().Contains("normalize"))
			{
				// Run kernel
				Log("Started running kernel", CudaH.KernelH.Kernel.KernelName, 1);
				CudaH.KernelH.RunKernelNormalize(CudaH.FloatVariables, (float) numericUpDown_kernelParam1.Value);
			}
			else if (CudaH.KernelH.Kernel.KernelName.ToLower().Contains("stretch"))
			{
				// Run kernel
				Log("Started running kernel", CudaH.KernelH.Kernel.KernelName, 1);
				CudaH.KernelH.RunKernelStretch(CudaH.ComplexVariables, (float) numericUpDown_kernelParam1.Value);
			}
			else
			{
				// Log error
				Log("Kernel not recognized", CudaH.KernelH.Kernel.KernelName, 1);
			}

			// Update GUI
			SetWaveform();
		}

		private void button_stretch_Click(object sender, EventArgs e)
		{
			// Abort if no audio or Ctx
			if (AudioH == null || CudaH.Ctx == null || CudaH.KernelH == null)
			{
				return;
			}

			// If no data on Cuda, move audio to Cuda
			if (CudaH.FloatVariables.Count == 0)
			{
				if (AudioH.Floats.Length == 0)
				{
					Log("No audio data", "Stretching", 1);
					return;
				}

				CudaH.PushArraysToCuda(AudioH.MakeChunks((int) numericUpDown_chunkSize.Value));
			}

			// Perform FFT
			CudaH.PerformFFT();

			// Run kernel
			string ptxPath = CudaH.KernelH.CompileKernel(Path.Combine(Repopath, "Resources\\Kernels\\C\\StretchKernel.c"), true, true);
			CudaH.KernelH.LoadKernel(ptxPath);
			CudaH.KernelH.RunKernelStretch(CudaH.ComplexVariables, (float) numericUpDown_factor.Value);

			// Perform IFFT
			CudaH.PerformIFFT();

			// Normalize audio
			ptxPath = CudaH.KernelH.CompileKernel(Path.Combine(Repopath, "Resources\\Kernels\\C\\NormalizeKernel.c"), true, true);
			CudaH.KernelH.LoadKernel(ptxPath);
			CudaH.KernelH.RunKernelNormalize(CudaH.FloatVariables, 1.0f);

			// Move audio to Host
			AudioH.Floats = AudioH.AggregateChunks(CudaH.PullArraysFromCuda<float>());

			// Normalize audio
			AudioH.Normalize();

			// Update GUI
			SetWaveform();
		}
	}
}
