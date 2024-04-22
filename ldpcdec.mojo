from sys.info import simdwidthof
from codebook.luts import nc,nnz, ncnpu, nvnpu,  vnpu_sizes, vnpu_indices, cnpu_sizes, cnpu_indices
from mdpc.types import nelts,dtype,Tensor3D, parallel_decode
from tensor import Tensor, TensorSpec, TensorShape
from random import rand, seed
from sys.info import num_physical_cores, num_logical_cores, num_performance_cores

from python import Python
from python.object import PythonObject


from sys.info import (
    os_is_linux,
    os_is_windows,
    os_is_macos,
    has_sse4,
    has_avx,
    has_avx2,
    has_avx512f,
    has_vnni,
    has_neon,
    is_apple_m1,
    has_intel_amx,
    _current_target,
    _current_cpu,
    _triple_attr,
)


from time import now
def main():
    
    var os = ""
    if os_is_linux():
        os = "linux"
    elif os_is_macos():
        os = "macOS"
    else:
        os = "windows"
    var cpu = String(_current_cpu())
    var arch = String(_triple_attr())
    var cpu_features = String("")
    if has_sse4():
        cpu_features += " sse4"
    if has_avx():
        cpu_features += " avx"
    if has_avx2():
        cpu_features += " avx2"
    if has_avx512f():
        cpu_features += " avx512f"
    if has_vnni():
        if has_avx512f():
            cpu_features += " avx512_vnni"
        else:
            cpu_features += " avx_vnni"
    if has_intel_amx():
        cpu_features += " intel_amx"
    if has_neon():
        cpu_features += " neon"
    if is_apple_m1():
        cpu_features += " Apple M1"

    print("System information: ")
    print("    OS          : ", os)
    print("    CPU         : ", cpu)
    print("    Arch        : ", arch)
    print("    Num Cores   : ", num_logical_cores())
    print("    CPU Features:", cpu_features)
    
    print("SIMD width of ",dtype, " = ",nelts)
   
    alias base_niter = 10
    alias max_niter = 20
    var nvcpu = num_logical_cores()

    # between following two we must balance the available nvcpu
    var intra_codeword_parallellism_factor = 1
    var nthread = nvcpu//2 +2

    alias ncodewordperthread = 4*nelts # must be a multiple of simdwidth = nelts    

      


    # get binary heavy LUTs
    var rep_spec = TensorSpec(DType.uint16, nc, 1)
    var rep_indices_tensor = Tensor[DType.uint16](rep_spec)
    rep_indices_tensor = rep_indices_tensor.fromfile('codebook/repeats.bin')

    var rep_offsets_tensor = Tensor[DType.uint16](rep_spec)
    rep_offsets_tensor = rep_offsets_tensor.fromfile('codebook/repeats_offsets.bin')

    var c2r_spec = TensorSpec(DType.uint16, nnz, 1)
    var c2r_indices_tensor = Tensor[DType.uint16](c2r_spec)
    c2r_indices_tensor = c2r_indices_tensor.fromfile('codebook/c2r.bin')

    var r2c_indices_tensor = Tensor[DType.uint16](c2r_spec)
    r2c_indices_tensor = r2c_indices_tensor.fromfile('codebook/r2c.bin')



    # var batch_size: Int
    var X: Tensor3D  # columns are codewords

    # warmup
    print("warmup...")
    var batch_syndrome: Int = -1
    var batch_size: Int

    batch_size = ncodewordperthread*nthread
    print("batch_size: ", batch_size)
    X = Tensor3D(nc,batch_size)
    X.rand_init()
 

    print("ncodewordperthread: ", ncodewordperthread)
    print("nthread: ",nthread)

    # prepare an output buffer for the decoded batch (warning: now rows are codewords)
    var Y = Tensor3D(batch_size, nc)   
     
    X.print()

    parallel_decode[base_niter, max_niter](X, Y, batch_syndrome,intra_codeword_parallellism_factor, nthread, nc,nnz, ncnpu, nvnpu, ncodewordperthread, rep_indices_tensor, rep_offsets_tensor,c2r_indices_tensor, r2c_indices_tensor, vnpu_sizes, vnpu_indices, cnpu_sizes, cnpu_indices)
    print("Y after decoding")
    Y.print[3,3]()
    print("warmup Batch syndrome: ",batch_syndrome)


    # # now benchmark
    print("now timeit: ...")
    var ncall = 10
    var tot_msecs = 0.0
    for i in range(ncall):
        # avoid that the compiler thinks there is no sense decoding same input X time and time again
        X.rand_init()

        var start = now()           
        parallel_decode[base_niter, max_niter](X, Y, batch_syndrome,intra_codeword_parallellism_factor, nthread, nc,nnz, ncnpu, nvnpu, ncodewordperthread, rep_indices_tensor, rep_offsets_tensor,c2r_indices_tensor, r2c_indices_tensor, vnpu_sizes, vnpu_indices, cnpu_sizes, cnpu_indices)
        var dt = (1e-6*(now()-start))
        tot_msecs += dt
    var msecs = tot_msecs/ncall

    
    
    var mbps: Float32 = 1e-3*batch_size * nc/msecs
    print("Throughput is ", batch_size," codewords in ",msecs, "msec, equivalent to ",mbps," Mbps")
    





