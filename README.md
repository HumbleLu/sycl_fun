# sycl_fun
A repository to dump my sycl-implemented functions. The code is(can) run in CSC Mahti (https://docs.csc.fi/computing/systems-mahti/).

- mlf_approx.cpp: a sycl implementation for approximating the Mittag-Leffler function (MLF)
- unif_act_fun.cpp: a sycl implementation of the unified activate function proposed in [1]

[1] Mostafanejad, M. (2023). Unification of popular artificial neural network activation functions. arXiv preprint arXiv:2302.11007.

## Running

To run tests, run the following command (in CSC Mahti)

- mlf_approx.cpp

compile .cpp file into a executive:
```bash
icpx -fuse-ld=lld -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 mlf_approx.cpp -o mlf_approx.x
```

run the compiled executive file:
```bash
srun --account=project_2008874 --nodes=1 --partition=gputest --gres=gpu:a100:1 --time=00:05:00 ./mlf_approx.x
```

output:
```bash
srun: job 3246471 queued and waiting for resources
srun: job 3246471 has been allocated resources
z: 1.4
Device: NVIDIA A100-SXM4-40GB
mlf approximated cos(z): 0.169967
cos(z): 0.169967

Device: NVIDIA A100-SXM4-40GB
mlf approximated exp(z): 4.0552
exp(z): 4.0552
```

- unif_act_fun.cpp

compile .cpp file into a executive:
```bash
icpx -fuse-ld=lld -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 unif_act_fun.cpp -o unif_act_fun.x
```

run the compiled executive file:
```bash
srun --account=project_2008874 --nodes=1 --partition=gputest --gres=gpu:a100:1 --time=00:05:00 ./unif_act_fun.x
```

output:
```bash
srun: job 3246475 queued and waiting for resources
srun: job 3246475 has been allocated resources
Device: NVIDIA A100-SXM4-40GB
x = 1.5
approximated tanh(x): 0.905148
std::tanh(x): 0.905148

Device: NVIDIA A100-SXM4-40GB
approximated sigmoid(x): 0.817574
1 / (1 + exp(-x)): 0.817574
```

## Author

- [@humblelu](https://humblelu.github.io/personal_website/)

