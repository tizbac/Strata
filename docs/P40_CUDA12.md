# DeepSeek-V4.1 on a Tesla P40 (sm_61) with CUDA 12

This fork (`tizbac/Strata`) builds `strata-dsv41`, Strata's DeepSeek-V4.1-Flash engine, for **Pascal compute
capability 6.1**: Tesla P40, Tesla P4 and the GTX 10 series. Upstream ships DeepSeek packages for sm_86, sm_89 and
sm_120 only, and its `src/dsv41` sources use WMMA unconditionally, which does not exist below sm_70.

Nothing here changes the engine for sm_70 and newer cards.

## What had to change

Two device kernels use `nvcuda::wmma`, whose header is declared only when `__CUDA_ARCH__ >= 700`
(`cuda/mma.h`). A CUDA 12 sm_61 build stops at "name must be a namespace name" without a guard.

| File | Kernel | Change |
| --- | --- | --- |
| `src/dsv41/ds4_gguf.cu` | `mx_set_tc_kernel` (MXFP4 gate/up on the tensor cores) | compiled only for `__CUDA_ARCH__ >= 700`; an empty stub is emitted for older passes |
| `src/dsv41/ds4_iq.cu` | `set_tc_kernel` (IQ gate/up on the tensor cores) | the same |

Both files already had a runtime check, `tc_ok()`, that returns false below compute capability 7.0, and a scalar
fallback kernel (`mx_set_kernel` / `set_kernel`). On sm_61 the stub is never launched: the scalar kernels run.
No other dsv41 source uses WMMA, so the rest of the engine builds unchanged.

The build also needs `-DSTRATA_EXPERIMENTAL_SM60=ON`: CMake refuses any architecture below sm_75 without it, and
that flag compiles the Pascal code paths (the #540 attention kernel, the Pascal BF16 projection path).

## Build

Requirements: Linux x86_64, CMake 3.24 or newer, Ninja, a C++ compiler, the CUDA **12.x** toolkit (12.9 tested;
CUDA 13 dropped sm_61), and the llama.cpp checkout at the commit pinned in `setup.py`
(`python3 -c "import setup; setup.get_llama_cpp()"`).

```sh
cmake -G Ninja -S . -B build-p40 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
  "-DCMAKE_CUDA_ARCHITECTURES=61-real;61-virtual" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DSTRATA_GGML_DIR=third_party/llama.cpp \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF '-DCMAKE_INSTALL_RPATH=$ORIGIN'
cmake --build build-p40 --target strata-dsv41 -j 8
./build-p40/strata-dsv41 --version      # needs the toolkit's lib dir on LD_LIBRARY_PATH before packaging
```

`61-real` is the P40's machine code. `61-virtual` adds PTX, so the same binary also starts (JIT-compiled) on newer
NVIDIA cards. Drop `;61-virtual` if you want a smaller file.

### Portable build (older glibc)

A binary built on a new distribution inherits that glibc. To produce the same archive the CI workflow makes, build
inside Ubuntu 22.04 (glibc 2.35, libstdc++ from GCC 11), which is what `packaging/build_cuda_linux.sh` targets for
the released CUDA 12 engine:

```sh
docker run --rm -v "$PWD:/src" -w /src ubuntu:22.04 bash -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg ninja-build \
      build-essential python3 python3-pip unzip
  python3 -m pip install --no-cache-dir "cmake>=3.31,<3.32"
  curl -fsSL -o /tmp/keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i /tmp/keyring.deb && apt-get update
  apt-get install -y --no-install-recommends cuda-nvcc-12-9 cuda-cudart-dev-12-9 libcublas-dev-12-9 cuda-cccl-12-9
  cmake -G Ninja -S . -B build-p40 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=61-real;61-virtual" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF "-DCMAKE_INSTALL_RPATH=\$ORIGIN"
  cmake --build build-p40 --target strata-dsv41 -j 4
'
```

## Package it like the released DeepSeek engine

The released packages are a folder `<version>-linux-x86_64-cuda-cuda12/` with `strata-dsv41`, the CUDA runtime
libraries it loads from beside itself (RUNPATH `$ORIGIN`), `BUILD.json`, `SYSTEM-LIBS.txt` and a signed
`engine-component.json`; the archive `strata-dsv41-<version>-linux-x86_64-cuda-cuda12.tar.gz` holds that folder's
files at its root. This fork reproduces that, unsigned:

```sh
stage=stage-dsv41
mkdir -p "$stage"
cp build-p40/strata-dsv41 "$stage/"
printf '%s\n' '{"version": "0.1.53", "source": "build_dsv41_cuda.sh sm_61 (CUDA 12.9)", "archs": "61", "cuda": "12.9", "portable": true}' > "$stage/BUILD.json"
python3 packaging/collect_libs.py --binary "$stage/strata-dsv41" --search /usr/local/cuda-12.9/lib64 \
    --out "$stage" --also 'libcublasLt.so*'
python3 packaging/package_engine.py --src "$stage" --backend cuda --version 0.1.53 --out out \
    --arch sm_61 --arch compute_61 --min-driver nvidia=575.51 --variant cuda12 --platform linux-x86_64 \
    --build-id "$(git rev-parse --short=9 HEAD)-cuda12.9-sm61" --entrypoint strata-dsv41 \
    --include SYSTEM-LIBS.txt --require-dsv41
python3 packaging/package_engine.py --verify out/0.1.53-linux-x86_64-cuda-cuda12
python3 packaging/binary_deps.py out/0.1.53-linux-x86_64-cuda-cuda12
python3 packaging/feed_tool.py pack-engine out/0.1.53-linux-x86_64-cuda-cuda12 --out release --prefix strata-dsv41
```

The last command writes `release/strata-dsv41-0.1.53-linux-x86_64-cuda-cuda12.tar.gz` and its `.entry.json`
(sha256, size, archs, minimum driver).

## Run it on the P40

```sh
tar xf strata-dsv41-0.1.53-linux-x86_64-cuda-cuda12.tar.gz
./0.1.53-linux-x86_64-cuda-cuda12/strata-dsv41 --version   # loads libcudart/libcublas from its own folder
./0.1.53-linux-x86_64-cuda-cuda12/strata-dsv41 --pack <pack> --pack-info
```

Strata's setup looks for `strata-dsv41` beside the `strata` engine (`engine/`), or in `engine-dsv41/`. You can also
run the binary directly, as above, and point the server at it with `STRATA_DSV41_EXE=<path>/strata-dsv41`.

On a P40:

* **Pictures: use `--vision off`.** The vision tower's projections are BF16 GEMMs; cuBLAS has no BF16 GEMM on
  compute capability 6.1, so the tower cannot run there. Text is unaffected.
* **The tensor-core expert kernels are off automatically** (`tc_ok()` is false below sm_70); the scalar MXFP4/IQ
  kernels run. `STRATA_DS4_NO_TC=1` is not needed but does no harm.
* **The model is large.** The 2-bit sets are 239-274 GB on disk (model + 111 GB Engram table + 8 GB drafter). The
  24 GB of a P40 holds the dense weights and a share of the routed experts; the rest stream from RAM and SSD.
  Keep the files on an NVMe drive.
* **Driver:** built with CUDA 12.9, so `min_driver` is NVIDIA 575.51. A CUDA 12.4/12.6 toolkit build lowers that
  floor if the box runs an older driver.

### Requirements of the archive

| | |
| --- | --- |
| OS | Linux x86_64 |
| CPU | AVX2 (`-DSTRATA_PORTABLE=ON`); AVX-512 is used through runtime dispatch where available |
| glibc | 2.35 or newer for the Ubuntu 22.04 / CI build; 2.38 for a build made on Debian 13 |
| libstdc++ | GCC 11 runtime (`GLIBCXX_3.4.30`) for the Ubuntu 22.04 build |
| GPU driver | NVIDIA 575.51 or newer (CUDA 12.9); the driver's own `libcuda.so.1` is not bundled |
| CUDA runtime | bundled (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`) |

### Building on a distribution with glibc 2.41 (Debian 13)

CUDA 12.9's `crt/math_functions.h` declares `sinpi`, `cospi`, `sinpif` and `cospif` without an exception
specification, and glibc 2.41 declares the same functions `noexcept`, so nvcc stops with
"exception specification is incompatible with that of previous function". This is a toolkit-header clash, not a
Strata problem. Two ways past it: add `noexcept (true)` to those four declarations in the toolkit's
`math_functions.h`, or build with CUDA 12.8 (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)). The CI workflow uses
Ubuntu 22.04 (glibc 2.35), where the stock headers are fine.

## What is not verified

The engine compiles for sm_61, starts, loads its bundled libraries, and passes `package_engine.py --verify` and
`binary_deps.py`. It has **not** been run against a model on P40 hardware by this fork's author, and no tokens/s
number is claimed. The upstream project lists no Pascal DeepSeek package for the same reason. Please post a chat
and a code answer, the driver, the model set and the engine log if you test it.

## CI

`.github/workflows/p40-cuda12.yml` builds this combination on `ubuntu-22.04` with CUDA 12.9 for
`61-real;61-virtual`, packages and verifies the archive, checks every imported library is bundled or belongs to the
OS/driver, runs `--version` from the unpacked folder, and uploads the `.tar.gz` and `.entry.json` as artifacts.
GitHub-hosted runners have no GPU, so the workflow builds and validates the package but cannot run inference.
