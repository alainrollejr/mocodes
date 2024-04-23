<br/>
<p align="center">
  <a href="https://github.com/alainrollejr/mocodes">
    <img src="https://github.com/alainrollejr/mocodes/blob/main/mocodeslogo.png" alt="Logo" width="200" height="200">
  </a>

  <h1 align="center">MoCodes</h1>

  <p align="center">
    An Error Correction (De)Coding library in pure Mojo 🔥
  </p>
</p>




## About The Project

MoCodes is a stand-alone Error Correction (De)Coding framework that leverages the power of Mojo.

As [discussed](https://docs.modular.com/mojo/why-mojo) by Modular, Mojo is a language for the future of AI development. Built on top of MLIR technology, rather than existing GCC and LLVM approaches, Mojo looks and feels like Python code, yet performs much closer to languages like Rust or C++. 

Error Correction Codes are being used in domains such as Wireless Communication and Quantum Computing. They are known to be very compute intensive, so much so that until recently they were implemented in dedicated ASIC silicon or programmed on FPGA accelerators.

In recent years with the advent of wide-vector SIMD CPU architectures and affordable yet powerful GPU cores they have been also implemented in C++ with a lot of vendor-specific intrinsics on CPU or with CUDA on GPU as well.

About time then to take a stab at how well Mojo lives up to the challenge (and how well the author lives up to the challenge of understanding how Mojo is meant to be used)

## Benchmark

We've been on a voyage of  exploration to find out how platform-independent frameworks originally meant for machine learning can be repurposed for error correction decoding. We have had reasonable results on GPU and TPU but not until Mojo came along we've reached decent throughputs on CPU. Results ofc vary with platform specifics, we have tried Intel, AMD and Macbook M3.

![ldpc_benchmark](https://github.com/alainrollejr/mocodes/blob/main/mocodesbenchmark.png)

While that looks awesome we estimate that there is a performance gap of about a factor 2 yet to be closed wrt C++ code that uses vendor specific intrinsics eg from the avx instruction set on Intel. Game on !

For now we only support generic (ir)regular  LDPC codes and we have committed just one example (1512 x 1872) LDPC Parity Check Matrix to this repo. This sparse parity check matrix has 7092 non-zero elements and is shown hereafter.

![ldpc_pcm](https://github.com/alainrollejr/mocodes/blob/main/codebook/example_pcm.png)

For now, this parity check matrix gets translated to look-up tables by an offline scipy script that takes an .npz file as input. The look-up tables get stored in the /codebook/ subdirectory.


## Quick Start

Try out the LDPC benchmark for yourself, on your own platform:

```
mojo build ldpcdec.mojo
```
```
./ldpcdec
```
You can tweak the following parameters in the main() function of ldpcdec.mojo: "intra_codeword_parallellism_factor", "ncodewordperthread", "nthread". Currently committed defaults seem to be close to optimal regardless the platform we have tried. 



## Roadmap

### v1.0 ✅
- [x] support for irregular LDPC decoding
- [x] Add profiling and additional performance tests

### v1.1 (WIP)
- [ ] Improve throughput by community expertise injection (target: factor 2)
- [ ] Add profiling and additional performance tests
- [ ] Add a serving functionality (preferably gRPC based, ideally leveraging MAX serving)


### v1.2 (WIP)
- [ ] incorporate generation of Look-Up Tables in the mojo code, such that the .npz file becomes the only configuration input that defines the code
- [ ] add an LDPC encoder
- [ ] add a script to simulate and visualise BER and BLER codes
- [ ] Autotuning and related features


## Contributing

The way we set this repo up should allow Mojo experts to contribute without necessarily being Error Correction Coding specialists.
Notably, the LDPC heavy lifting is done by a handful of functions in [!heavy](https://github.com/alainrollejr/mocodes/blob/main/mdpc/types.mojo), i.e. fn all_Lqij() and fn cnpu(). Memory load and store determine the throughput so all tips and tricks to speed up that memory access would much appreciated. 

If you are considering larger contributions, feel free to contact us for a smoother communication channel on Discord. If you find a bug or have an idea for a feature, please use our issue tracker. Before creating a new issue, please:
* Check if the issue already exists. If an issue is already reported, you can contribute by commenting on the existing issue.
* If not, create a new issue and include all the necessary details to understand/recreate the problem or feature request.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request
> Once your changes are pushed, navigate to your fork on GitHub. And create a pull request against the original  repository.
> - Before creating a PR make sure the functional text output of ./ldpcdec is the same as the one on the main branch
> - In the pull request, provide a detailed description of the changes and why they're needed. Link any relevant issues.


## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See LLVM [License](https://llvm.org/LICENSE.txt) for more information.

## Acknowledgements

* Built with [Mojo](https://github.com/modularml/mojo) created by [Modular](https://github.com/modularml)

