using NAudio.Wave;
using System.Drawing.Drawing2D;
using System.Numerics;

namespace ManagedCudaCuFFTTool
{
	public class AudioHandling
	{
		// ~~~~~ ~~~~~ ~~~~~ ATTRIBUTES ~~~~~ ~~~~~ ~~~~~ \\
		public string Filepath;
		public string Name;

		public int Samplerate;
		public int Bitdepth;
		public int Channels;

		public long Length;
		public double Duration;

		public float[] Floats;

		public WaveOutEvent Player;
		public bool IsPlaying => Player.PlaybackState == PlaybackState.Playing;


		// ~~~~~ ~~~~~ ~~~~~ CONSTRUCTORS ~~~~~ ~~~~~ ~~~~~ \\
		public AudioHandling(string filepath)
		{
			Player = new WaveOutEvent();
			Filepath = filepath;

			if (CheckFilepath(Filepath) == false)
			{
				Name = "Invalid";
				Samplerate = 0;
				Bitdepth = 0;
				Channels = 0;
				Length = 0;
				Duration = 0;
				Floats = [];
				return;
			}

			Name = Path.GetFileNameWithoutExtension(Filepath);

			// Get audiofilereader
			AudioFileReader reader = new(Filepath);

			Samplerate = reader.WaveFormat.SampleRate;
			Bitdepth = reader.WaveFormat.BitsPerSample;
			Channels = reader.WaveFormat.Channels;

			Length = reader.Length;
			Duration = reader.TotalTime.TotalSeconds;
			Floats = new float[Length];

			reader.Read(Floats, 0, Floats.Length);

			reader.Close();
			reader.Dispose();
		}





		// ~~~~~ ~~~~~ ~~~~~ METHODS ~~~~~ ~~~~~ ~~~~~ \\
		public bool CheckFilepath(string filepath)
		{
			// Check if file exists
			if (!File.Exists(filepath))
			{
				return false;
			}

			// Check if file is audio
			string ext = Path.GetExtension(filepath).ToLower();
			if (ext != ".wav" && ext != ".mp3" && ext != ".flac")
			{
				return false;
			}

			// Check if file is larger than 0 bytes
			FileInfo fi = new FileInfo(filepath);
			if (fi.Length == 0)
			{
				return false;
			}

			return true;
		}

		public void PlayStop(Button? playbackButton = null)
		{
			if (Player.PlaybackState == PlaybackState.Playing)
			{
				if (playbackButton != null)
				{
					playbackButton.Text = "⏵";
				}
				Player.Stop();
			}
			else
			{
				byte[] bytes = GetBytes();

				MemoryStream ms = new(bytes);
				RawSourceWaveStream raw = new(ms, new WaveFormat(Samplerate, Bitdepth, Channels));

				Player.Init(raw);

				if (playbackButton != null)
				{
					playbackButton.Text = "⏹";
				}
				Player.Play();
			}
		}

		public void Stop(Button? playbackButton = null)
		{
			if (playbackButton != null)
			{
				playbackButton.Text = "⏵";
			}
			Player.Stop();
		}

		public byte[] GetBytes()
		{
			int bytesPerSample = Bitdepth / 8;
			byte[] bytes = new byte[Floats.Length * bytesPerSample];

			for (int i = 0; i < Floats.Length; i++)
			{
				byte[] byteArray;
				float sample = Floats[i];

				switch (Bitdepth)
				{
					case 16:
						short shortSample = (short) (sample * short.MaxValue);
						byteArray = BitConverter.GetBytes(shortSample);
						break;
					case 24:
						int intSample24 = (int) (sample * (1 << 23));
						byteArray = new byte[3];
						byteArray[0] = (byte) (intSample24 & 0xFF);
						byteArray[1] = (byte) ((intSample24 >> 8) & 0xFF);
						byteArray[2] = (byte) ((intSample24 >> 16) & 0xFF);
						break;
					case 32:
						int intSample32 = (int) (sample * int.MaxValue);
						byteArray = BitConverter.GetBytes(intSample32);
						break;
					default:
						throw new ArgumentException("Unsupported bit depth");
				}

				Buffer.BlockCopy(byteArray, 0, bytes, i * bytesPerSample, bytesPerSample);
			}

			return bytes;
		}

		public void Normalize(float maxAmplitude = 1.0f)
		{
			// Get length
			long length = Floats.Length;

			// Normalize to max amplitude
			float max = length > 0 ? Floats.Max() : 1.0f;
			for (int i = 0; i < length; i++)
			{
				Floats[i] *= maxAmplitude / max;
			}
		}

		public string ExportAudioWav(string filepath)
		{
			// Create new wave file
			WaveFormat format = new(Samplerate, Bitdepth, Channels);
			WaveFileWriter writer = new(filepath, format);

			// Write audio data
			byte[] bytes = GetBytes();
			writer.Write(bytes, 0, bytes.Length);

			// Close writer
			writer.Close();

			return filepath;
		}
		
		public Bitmap DrawWaveformSmooth(PictureBox wavebox, long offset = 0, int samplesPerPixel = 1, bool update = false, Color? graph = null)
		{
			// Überprüfen, ob floats und die PictureBox gültig sind
			if (Floats.Length == 0 || wavebox.Width <= 0 || wavebox.Height <= 0)
			{
				// Empty picturebox
				if (update)
				{
					wavebox.Image = null;
					wavebox.Refresh();
				}

				return new Bitmap(1, 1);
			}

			// Colors (background depends on graph brightness)
			Color waveformColor = graph ?? Color.FromName("HotTrack");
			Color backgroundColor = waveformColor.GetBrightness() < 0.5 ? Color.White : Color.Black;


			Bitmap bmp = new(wavebox.Width, wavebox.Height);
			using Graphics gfx = Graphics.FromImage(bmp);
			using Pen pen = new(waveformColor);
			gfx.SmoothingMode = SmoothingMode.AntiAlias;
			gfx.Clear(backgroundColor);

			float centerY = wavebox.Height / 2f;
			float yScale = wavebox.Height / 2f;

			for (int x = 0; x < wavebox.Width; x++)
			{
				long sampleIndex = offset + (long) x * samplesPerPixel;

				if (sampleIndex >= Floats.Length)
				{
					break;
				}

				float maxValue = float.MinValue;
				float minValue = float.MaxValue;

				for (int i = 0; i < samplesPerPixel; i++)
				{
					if (sampleIndex + i < Floats.Length)
					{
						maxValue = Math.Max(maxValue, Floats[sampleIndex + i]);
						minValue = Math.Min(minValue, Floats[sampleIndex + i]);
					}
				}

				float yMax = centerY - maxValue * yScale;
				float yMin = centerY - minValue * yScale;

				// Überprüfen, ob die Werte innerhalb des sichtbaren Bereichs liegen
				if (yMax < 0) yMax = 0;
				if (yMin > wavebox.Height) yMin = wavebox.Height;

				// Zeichne die Linie nur, wenn sie sichtbar ist
				if (Math.Abs(yMax - yMin) > 0.01f)
				{
					gfx.DrawLine(pen, x, yMax, x, yMin);
				}
				else if (samplesPerPixel == 1)
				{
					// Zeichne einen Punkt, wenn samplesPerPixel 1 ist und die Linie zu klein ist
					gfx.DrawLine(pen, x, centerY, x, centerY - Floats[sampleIndex] * yScale);
				}
			}

			// Update PictureBox
			if (update)
			{
				wavebox.Image = bmp;
				wavebox.Refresh();
			}

			return bmp;
		}

		public int GetFitResolution(int width)
		{
			// Gets pixels per sample for a given width to fit the whole waveform
			int samplesPerPixel = (int) Math.Ceiling((double) Floats.Length / width) / 4;
			return samplesPerPixel;
		}

		public List<float[]> MakeChunks(int chunkSize)
		{
			long length = Floats.Length;

			int count = (int) Math.Ceiling((double) length / chunkSize);
			List<float[]> chunks = new List<float[]>(count);

			for (int i = 0; i < count; i++)
			{
				int start = i * chunkSize;
				int end = Math.Min(start + chunkSize, (int) length);
				float[] split = new float[end - start];
				Array.Copy(Floats, start, split, 0, end - start);
				chunks.Add(split);
			}

			return chunks;
		}

		public int[] GetChunkSizes(List<float[]> chunks)
		{
			int[] sizes = new int[chunks.Count];

			for (int i = 0; i < chunks.Count; i++)
			{
				sizes[i] = chunks[i].Length;
			}

			return sizes;
		}

		public float[] AggregateChunks(List<float[]> chunks)
		{
			long length = chunks.Sum(split => split.Length);
			float[] aggregated = new float[length];

			int index = 0;
			foreach (float[] split in chunks)
			{
				Array.Copy(split, 0, aggregated, index, split.Length);
				index += split.Length;
			}

			return aggregated;
		}

		

		

		
	}
}