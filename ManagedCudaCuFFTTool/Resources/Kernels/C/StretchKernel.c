extern "C"
{
    __global__ void Stretch(const int N, cuFloatComplex* __restrict data, float factor)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N)
        {
            // Phase vocoder algorithm
            float magnitude = cuCabsf(data[idx]);
            float phase = atan2f(cuCimagf(data[idx]), cuCrealf(data[idx]));

            // Stretching phase
            phase *= factor;

			// Reconstruction
            data[idx] = make_cuFloatComplex(magnitude * cosf(phase), magnitude * sinf(phase));
        }
    }
}

