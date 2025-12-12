# Monte_Carlo_Pi
This project reflect the course work that touch the classic *Monte Carlo simulation*.

## General Logic
![Graphic Illustration](illustration.png)
- Area of the square: $$4r^2$$
- Area of the circle: $$\pi r^2$$

The probability of randomly having a point falls inside the circle:

$$P = \frac{\pi r^2}{4r^2} = \frac{\pi}{4}$$
We can then calculate π as:

$$\pi = 4P$$

## Quick Start
Clone the repository
```bash
git clone https://github.com/AnaOnTram/Monte_Carlo_Pi.git
```

Compile the scripts
* You should have cuda toolkit installed
```bash
#Pi calculation
nvcc monte_pi.cu -o monte_pi -lm

#GPU Floatpoint Benchmark (RTX30xx: sm80 RTX40xx: sm89 Jetson Orin: SM87 RTX50xx: sm120)
nvcc -O3 -use_fast_math -arch=sm_[YOUR_GPU_ARCH] benchmark.cu -o benchmark
```

