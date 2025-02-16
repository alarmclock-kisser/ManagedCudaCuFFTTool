using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;

namespace ManagedCudaCuFFTTool
{
	public class CudaHandling
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\
		public string Repopath;
		public ListBox LogBox;
		public ComboBox DevicesBox;
		public Label VramLabel;
		public ProgressBar VramPbar;
		public Label KernelLabel;

		public int DeviceId = -1;
		public PrimaryContext? Ctx = null;


		public List<CudaDeviceVariable<float>> FloatVariables = [];
		public List<CudaDeviceVariable<float2>> ComplexVariables = [];

		public KernelHandling? KernelH = null;


		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTORS ~~~~~ ~~~~~ ~~~~~ \\
		public CudaHandling(string repopath, int deviceId = 0, ListBox? logbox = null, ComboBox? devicesbox = null, Label? vramlabel = null, ProgressBar? vrampbar = null, Label? kernellabel = null)
		{
			Repopath = repopath;
			LogBox = logbox ?? new ListBox();
			DevicesBox = devicesbox ?? new ComboBox();
			VramLabel = vramlabel ?? new Label();
			VramPbar = vrampbar ?? new ProgressBar();
			KernelLabel = kernellabel ?? new Label();

			DevicesBox.SelectedIndexChanged += (sender, e) => SetDevice(DevicesBox.SelectedIndex);

			FillDeviceNames(deviceId);
		}





		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public void Log(string message, string inner = "", int layer = 1)
		{
			string msg = "[" + DateTime.Now.ToString("HH:mm:ss") + "] ";

			for (int i = 0; i < layer; i++)
			{
				msg += "\t";
			}

			msg += message;

			if (inner != "")
			{
				msg += " (" + inner + ")";
			}

			LogBox.Items.Add(msg);
			LogBox.SelectedIndex = LogBox.Items.Count - 1;
		}

		public int GetDeviceCount()
		{
			return CudaContext.GetDeviceCount();
		}

		public string[] FillDeviceNames(int init = -1)
		{
			string[] names = new string[GetDeviceCount()];

			for (int i = 0; i < names.Length; i++)
			{
				names[i] = CudaContext.GetDeviceInfo(i).DeviceName;
			}

			DevicesBox.Items.Clear();
			DevicesBox.Items.AddRange(names);
			DevicesBox.Items.Add("Use no CUDA device");
			DevicesBox.SelectedIndex = init;

			return names;
		}

		public void SetDevice(int deviceid)
		{
			DeviceId = deviceid;
			Dispose();

			if (deviceid < 0 || deviceid >= GetDeviceCount())
			{
				DeviceId = -1;
				Ctx = null;
				GetMemoryUsage(true);
				Log("No CUDA device selected", "SetDevice", 1);
				return;
			}

			Ctx = new PrimaryContext(deviceid);
			Ctx.SetCurrent();
			KernelH = new KernelHandling(Repopath, Ctx, LogBox, KernelLabel);

			Log("CUDA device set to " + CudaContext.GetDeviceInfo(deviceid).DeviceName, "SetDevice", 1);
			GetMemoryUsage(true);
		}

		public void Dispose()
		{
			Ctx?.Dispose();
			Ctx = null;

			GC.Collect();
		}

		public long[] GetMemoryUsage(bool readable = false)
		{
			if (Ctx == null)
			{
				VramLabel.Text = "VRAM: 0 / 0 MB";
				VramPbar.Maximum = 0;
				VramPbar.Value = 0;
				return [0, 0, 0];
			}

			// Update UI elements
			VramLabel.Text = "VRAM: " + (Ctx.GetTotalDeviceMemorySize() - Ctx.GetFreeDeviceMemorySize()) / 1024 / 1024 + " / " + Ctx.GetTotalDeviceMemorySize() / 1024 / 1024 + " MB";
			VramPbar.Maximum = (int) (Ctx.GetTotalDeviceMemorySize() / 1024 / 1024);
			VramPbar.Value = (int) (Ctx.GetTotalDeviceMemorySize() - Ctx.GetFreeDeviceMemorySize()) / 1024 / 1024;

			if (readable)
			{
				return [Ctx.GetTotalDeviceMemorySize() / 1024 / 1024, Ctx.GetFreeDeviceMemorySize() / 1024 / 1024];
			}

			return [Ctx.GetTotalDeviceMemorySize(), Ctx.GetFreeDeviceMemorySize()];
		}

		public Dictionary<long, int> PushArraysToCuda<T>(List<T[]> arrays) where T : unmanaged
		{
			List<CudaDeviceVariable<T>> variables = [];

			foreach (T[] array in arrays)
			{
				CudaDeviceVariable<T> variable = new(array.Length);
				variable.CopyToDevice(array);
				variables.Add(variable);

				Log("Pushed array to CUDA: " + array.Length + " elements", "PushArraysToCuda", 1);
			}

			if (typeof(T) == typeof(float))
			{
				// Float variables
				foreach (CudaDeviceVariable<float> variable in FloatVariables)
				{
					variable.Dispose();
					Log("Disposed of float variable", "PushArraysToCuda", 1);
				}
				FloatVariables.Clear();

				FloatVariables.AddRange(variables.Cast<CudaDeviceVariable<float>>());
				Log("Float variables count: " + FloatVariables.Count, "PushArraysToCuda", 1);
			}
			else if (typeof(T) == typeof(float2))
			{
				// Float2 variables
				foreach (CudaDeviceVariable<float2> variable in ComplexVariables)
				{
					variable.Dispose();
					Log("Disposed of complex variable", "PushArraysToCuda", 1);
				}
				ComplexVariables.Clear();

				ComplexVariables.AddRange(variables.Cast<CudaDeviceVariable<float2>>());
				Log("Complex variables count: " + ComplexVariables.Count, "PushArraysToCuda", 1);
			}

			// Make dict of long Pointers & sizes
			Dictionary<long, int> pointers = [];
			foreach (CudaDeviceVariable<T> variable in variables)
			{
				pointers.Add(variable.DevicePointer.Pointer, variable.Size);
			}

			// Return pointers
			GetMemoryUsage(true);
			return pointers;
		}

		public List<T[]> PullArraysFromCuda<T>() where T : unmanaged
		{
			List<T[]> arrays = [];

			foreach (var ptr in FloatVariables)
			{
				T[] array = new T[ptr.Size];
				ptr.CopyToHost(array);
				arrays.Add(array);

				ptr.Dispose();
				Log("Pulled array from CUDA: " + array.Length + " elements", "PullArraysFromCuda", 1);
			}
			FloatVariables.Clear();

			GetMemoryUsage(true);
			return arrays;
		}

		public void PerformFFT()
		{
			// Dispose of ComplexVariables
			foreach (CudaDeviceVariable<float2> variable in ComplexVariables)
			{
				variable.Dispose();
				Log("Disposed of complex variable", "PerformFFT", 1);
			}
			ComplexVariables.Clear();

			// Take float variables and FFT forward to complex variables
			foreach (CudaDeviceVariable<float> variable in FloatVariables)
			{
				CudaDeviceVariable<float2> complex = new(variable.Size);
				CudaFFTPlan1D plan = new(variable.Size, cufftType.R2C, 1);
				plan.Exec(variable.DevicePointer, complex.DevicePointer, TransformDirection.Forward);
				plan.Dispose();
				ComplexVariables.Add(complex);

				variable.Dispose();
				Log("Performed FFT on float variable", "PerformFFT", 1);
			}
			FloatVariables.Clear();

			// Update GUI
			GetMemoryUsage(true);
		}

		public void PerformIFFT()
		{
			// Dispose of FloatVariables
			foreach (CudaDeviceVariable<float> variable in FloatVariables)
			{
				variable.Dispose();
				Log("Disposed of float variable", "PerformIFFT", 1);
			}
			FloatVariables.Clear();

			// Take complex variables and IFFT backward to float variables
			foreach (CudaDeviceVariable<float2> variable in ComplexVariables)
			{
				CudaDeviceVariable<float> fft = new(variable.Size);
				CudaFFTPlan1D plan = new(variable.Size, cufftType.C2R, 1);
				plan.Exec(variable.DevicePointer, fft.DevicePointer, TransformDirection.Inverse);
				plan.Dispose();
				FloatVariables.Add(fft);
				variable.Dispose();

				Log("Performed IFFT on complex variable", "PerformIFFT", 1);
			}
			ComplexVariables.Clear();

			// Update GUI
			GetMemoryUsage(true);
		}

		public void StretchComplex(float stretchFactor = 1.0f)
		{
			// Überprüfen, ob der Stretch-Faktor gültig ist
			if (stretchFactor <= 0)
			{
				Log("Ungültiger Stretch-Faktor", "StretchComplex", 1);
				return;
			}

			// Parameter für den Phase Vocoder
			int windowSize = 1024; // Größe des Analysefensters
			int hopSizeInput = windowSize / 4; // Schrittweite im Originalsignal
			int hopSizeOutput = (int) (hopSizeInput * stretchFactor); // Schrittweite im gestreckten Signal

			// Für jede komplexe Variable den Phase Vocoder anwenden
			for (int idx = 0; idx < ComplexVariables.Count; idx++)
			{
				var variable = ComplexVariables[idx];

				int signalLength = variable.Size;
				int numFrames = (signalLength - windowSize) / hopSizeInput + 1;

				// Host-Array für das komplexe Signal
				float2[] complexSignal = new float2[signalLength];
				variable.CopyToHost(complexSignal);

				// Listen für die gestreckten Magnituden und Phasen
				List<float[]> magnitudes = new List<float[]>();
				List<float[]> phases = new List<float[]>();

				// Vorherige Phase für die Phasendifferenzberechnung
				float[] previousPhase = new float[windowSize];
				float[] sumPhase = new float[windowSize];

				// Analysephase: Berechnung der Magnitude und Phase
				for (int i = 0; i < numFrames; i++)
				{
					int offset = i * hopSizeInput;
					if (offset + windowSize > signalLength)
						break;

					// Fenster des Signals extrahieren
					float2[] windowedSegment = new float2[windowSize];
					Array.Copy(complexSignal, offset, windowedSegment, 0, windowSize);

					// Berechnung von Magnitude und Phase
					float[] mag = new float[windowSize];
					float[] phase = new float[windowSize];

					for (int k = 0; k < windowSize; k++)
					{
						mag[k] = (float) Math.Sqrt(windowedSegment[k].x * windowedSegment[k].x + windowedSegment[k].y * windowedSegment[k].y);
						phase[k] = (float) Math.Atan2(windowedSegment[k].y, windowedSegment[k].x);
					}

					// Phasendifferenz berechnen
					float[] deltaPhase = new float[windowSize];
					for (int k = 0; k < windowSize; k++)
					{
						float phaseDiff = phase[k] - previousPhase[k];
						previousPhase[k] = phase[k];

						// Erwartete Phasendifferenz für die aktuelle Frequenz
						float expectedPhaseDiff = (2 * (float) Math.PI * k) / windowSize * hopSizeInput;
						deltaPhase[k] = phaseDiff - expectedPhaseDiff;

						// Phasenwrap
						deltaPhase[k] = (float) (deltaPhase[k] - 2 * Math.PI * Math.Round(deltaPhase[k] / (2 * Math.PI)));

						// Korrigierte Phasendifferenz
						deltaPhase[k] = expectedPhaseDiff + deltaPhase[k];

						// Akkumulierte Phase
						sumPhase[k] += deltaPhase[k] * (hopSizeOutput / (float) hopSizeInput);
					}

					magnitudes.Add(mag);
					phases.Add((float[]) sumPhase.Clone());
				}

				// Rekonstruktion des gestreckten Signals
				int outputLength = (numFrames - 1) * hopSizeOutput + windowSize;
				float2[] stretchedSignal = new float2[outputLength];

				for (int i = 0; i < magnitudes.Count; i++)
				{
					int offset = i * hopSizeOutput;

					float[] mag = magnitudes[i];
					float[] phase = phases[i];

					for (int k = 0; k < windowSize; k++)
					{
						float real = mag[k] * (float) Math.Cos(phase[k]);
						float imag = mag[k] * (float) Math.Sin(phase[k]);

						if (offset + k < outputLength)
						{
							stretchedSignal[offset + k].x += real;
							stretchedSignal[offset + k].y += imag;
						}
					}
				}

				// Erstellen einer neuen CudaDeviceVariable für das gestreckte Signal
				CudaDeviceVariable<float2> stretchedVariable = new(outputLength);
				stretchedVariable.CopyToDevice(stretchedSignal);

				// Alte Variable freigeben und durch die neue ersetzen
				variable.Dispose();
				ComplexVariables[idx] = stretchedVariable;

				Log($"Stretched complex signal by x{stretchFactor}", "StretchComplex", 1);
			}

			// GUI aktualisieren
			GetMemoryUsage(true);
		}


	}
}