<br/>
<p align="center">
  <a href="https://github.com/alainrollejr/mocodes">
    <img src="https://github.com/alainrollejr/mocodes/mocodeslogo" alt="Logo" width="200" height="200">
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

![basalt_benchmark](https://github.com/basalt-org/basalt/assets/46826967/83037770-a9e3-440d-bdca-f51af0aebee0)


## Quick Start

Try out the benchmarks yourself:

```
mojo -I . examples/housing.mojo
```
```
mojo -I . examples/sin_estimate.mojo
```
```
mojo -I . examples/mnist.mojo
```

Compare to the alternative PyTorch implementation:  
Make sure to install the requirements in `python-requirements.txt` in your python environment.

```
python examples/housing.py
python examples/sin_estimate.py
python examples/mnist.py
```

## Roadmap

### v0.1.0 ✅
- [x] Improve matrix multiplication and convolution kernels
- [x] Switch to custom Tensor and TensorShape implementations
- [x] Improve benchmarks and overall model execution performance
- [x] Add profiling and additional performance tests

### v0.2.0 (WIP)
- [ ] Add additional operators: Slice, (Un)Squeeze, Concat, Clip, Gather, Split, FMA ...
- [ ] Better layer support and more activation functions
- [ ] Graph submodules & graph concatenation
- [ ] Computer vision benchmark. 

### Long-Term
- [ ] Better parallelization
- [ ] GPU support
- [ ] Reworked Dataloader
- [ ] Autotuning and related features
- [ ] Graph compilation optimizations
- [ ] Operator fusion
- [ ] ONNX / Max compatibility

## Contributing

Basalt is built by community efforts and relies on your expertise and enthousiasm!  
Small fixes and improvements are much appreciated. If you are considering larger contributions, feel free to contact us for a smoother communication channel on Discord. If you find a bug or have an idea for a feature, please use our issue tracker. Before creating a new issue, please:
* Check if the issue already exists. If an issue is already reported, you can contribute by commenting on the existing issue.
* If not, create a new issue and include all the necessary details to understand/recreate the problem or feature request.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request
> Once your changes are pushed, navigate to your fork on GitHub. And create a pull request against the original basalt-org/basalt repository.
> - Before creating a PR make sure it doesn't break any of the unit-tests. (e.g. `mojo run -I . test/test_ops.mojo`)
> - Introducing new big features requires a new test!
> - In the pull request, provide a detailed description of the changes and why they're needed. Link any relevant issues.
> - If there are any specific instructions for testing or validating your changes, include those as well.

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See LLVM [License](https://llvm.org/LICENSE.txt) for more information.

## Acknowledgements

* Built with [Mojo](https://github.com/modularml/mojo) created by [Modular](https://github.com/modularml)

