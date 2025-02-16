using ManagedCuda;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;

namespace ManagedCudaCuFFTTool
{
	public class KernelHandling
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\
		public string Repopath;
		public ListBox ListBox;
		public Label KernelLabel;

		public PrimaryContext Ctx;
		public CudaKernel? Kernel = null;



		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTORS ~~~~~ ~~~~~ ~~~~~ \\
		public KernelHandling(string repopath, PrimaryContext context, ListBox? logbox = null, Label? kernelLabel = null)
		{
			Repopath = repopath;
			Ctx = context;
			ListBox = logbox ?? new ListBox();
			KernelLabel = kernelLabel ?? new Label();
		}




		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public void Log(string message, string inner = "", int layer = 0)
		{
			string indent = new('\t', layer);
			string msg = "[" + DateTime.Now.ToString("HH:mm:ss") + "] " + indent + message;

			if (inner != "")
			{
				msg += " (" + inner + ")";
			}

			ListBox.Items.Add(msg);
			ListBox.SelectedIndex = ListBox.Items.Count - 1;
		}

		public string CompileKernel(string filepath, bool silent = false, bool export = false)
		{
			string kernelName = Path.GetFileNameWithoutExtension(filepath);
			kernelName = kernelName.Replace("Kernel", "");

			// Load kernel file
			string kernelCode = File.ReadAllText(filepath);

			// Compile kernel
			var rtc = new CudaRuntimeCompiler(kernelCode, kernelName);
			rtc.Compile([]);
			string log = rtc.GetLogAsString();

			// Get ptx code
			byte[] ptxCode = rtc.GetPTX();
			Log("Kernel compiled successfully!", log, 1);

			// Msg box
			if (!silent)
			{
				MessageBox.Show("Kernel compiled successfully!\n\n" + log, "Kernel Compilation", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}

			// Export ptx
			if (export)
			{
				string ptxPath = Path.Combine(Repopath, "Resources\\Kernels\\PTX", kernelName + "Kernel" + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				Log("PTX exported to " + ptxPath, "Kernel Compilation", 1);

				return ptxPath;
			}

			return "";
		}

		public void LoadKernel(string filepath)
		{
			// Get kernel name
			string kernelName = Path.GetFileNameWithoutExtension(filepath);
			kernelName = kernelName.Replace("Kernel", "");

			// Load ptx code
			byte[] ptxCode = File.ReadAllBytes(filepath);
			
			// Load kernel
			Kernel = Ctx.LoadKernelPTX(ptxCode, kernelName);

			// Update GUI
			KernelLabel.Text = '"' + kernelName + '"';

			Log("Kernel loaded successfully!", "Kernel Loading", 1);
		}

		public void RunKernelNormalize(List<CudaDeviceVariable<float>> data, float maxAmplitude)
		{
			if (Kernel == null || !Kernel.KernelName.ToLower().Contains("normalize"))
			{
				Log("No kernel loaded!", "Kernel Running", 1);
				return;
			}

			foreach (var d in data)
			{
				// Get data size
				int size = d.Size;
				
				// Set kernel parameters
				Kernel.BlockDimensions = new dim3(256);
				Kernel.GridDimensions = new dim3((size + 255) / 256);
				
				// Run kernel
				Kernel.Run(size, d.DevicePointer, maxAmplitude);

				Log("Kernel ran successfully!", "Kernel Running", 1);
			}
		}

		public void RunKernelStretch(List<CudaDeviceVariable<float2>> data, float factor)
		{
			if (Kernel == null || !Kernel.KernelName.ToLower().Contains("stretch"))
			{
				Log("No kernel loaded!", "Kernel Running", 1);
				return;
			}

			foreach (var d in data)
			{
				// Get data size
				int size = d.Size;

				// Set kernel parameters
				Kernel.BlockDimensions = new dim3(256);
				Kernel.GridDimensions = new dim3((size + 255) / 256);

				// Run kernel
				Kernel.Run(size, d.DevicePointer, factor);
				Log("Kernel ran successfully!", "Kernel Running", 1);
			}
		}




	}
}
