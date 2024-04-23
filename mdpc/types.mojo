from math import trunc, mod,min,abs,max
from memory.unsafe import DTypePointer
from sys.info import simdwidthof
from python import Python
from python.object import PythonObject
from random import rand, randint
from algorithm import vectorize,  parallelize
from testing import assert_true
from tensor import Tensor, TensorSpec, TensorShape
from sys.intrinsics import strided_load, PrefetchOptions
from sys.info import num_physical_cores, num_logical_cores, num_performance_cores
from collections import List
from collections.vector import InlinedFixedVector


alias dtype = DType.int8
alias nelts = 2*simdwidthof[dtype]()  # The SIMD vector width, for some reason overbooking by factor 2 helps

alias max_allowed = SIMD[DType.int16,nelts](127)
alias min_allowed = SIMD[DType.int16,nelts](-127)



@parameter
fn all_cnpu_compiled(inout ip: Tensor3D, nnz: Int, ncnpu: Int, cnpu_sizes: VariadicList[Int],cnpu_indices: VariadicList[Int]) raises:        
    var slice_start : Int = 0       

    for idx in range(ncnpu):
        var slice_stop  : Int
        var n = cnpu_sizes[idx]
        if (idx == (ncnpu - 1)):
            slice_stop = nnz
        else:
            slice_stop  = slice_start + (cnpu_indices[idx+1] - cnpu_indices[idx])*n
        var block = ip.get_reference_to_row_slice(slice_start, slice_stop)
        # block.print()
        # print("cnpu [",slice_start,":",slice_stop,"] reshape to ",n)
        block.reshape_dim0(n)
        block.cnpu()
        block.flatten_dim2()
        slice_start = slice_stop

@parameter
fn all_syndromes_compiled(inout ip: Tensor3D,nnz: Int, ncnpu: Int, cnpu_sizes: VariadicList[Int], cnpu_indices: VariadicList[Int]) raises:        
    var slice_start : Int = 0       

    for idx in range(ncnpu):
        var slice_stop  : Int
        var n = cnpu_sizes[idx]
        if (idx == (ncnpu - 1)):
            slice_stop = nnz
        else:
            slice_stop  = slice_start + (cnpu_indices[idx+1] - cnpu_indices[idx])*n
        var block = ip.get_reference_to_row_slice(slice_start, slice_stop)
        # block.print()
        # print("cnpu [",slice_start,":",slice_stop,"] reshape to ",n)
        block.reshape_dim0(n)
        block.syndromes()
        block.flatten_dim2()
        slice_start = slice_stop

@parameter
fn all_Lqij(inout ip: Tensor3D, llr: Tensor3D, start_codeword_idx: Int, intra_codeword_parallellism_factor: Int,nc: Int,rep_indices: Tensor[DType.uint16], rep_offsets: Tensor[DType.uint16], r2c_indices: Tensor[DType.uint16]) raises:        
    # this version just loops over columns of the virtual column-first stored LLRs

    # below prefetch lines compile but don't seem to make a performance difference
    # alias r_opt = PrefetchOptions().high_locality()
    # llr.data().prefetch[r_opt]()
    # ip.data().prefetch[r_opt]()

       

    @parameter
    fn vnpu(column_idx:Int):
        var n_simd = rep_indices[column_idx]
        var n = n_simd.to_int()
        var slice_start = rep_offsets[column_idx].to_int()
        

        var lut_idx_cache = InlinedFixedVector[Int](n)
        
        for j in range(0,n):
            var m = r2c_indices[slice_start + j].to_int()
            lut_idx_cache[j] = m

        @parameter
        fn vp[nelts_local: Int](k: Int):
            
            # if n > 1:
            var sum = llr.load[nelts](column_idx, k+start_codeword_idx)                                
            var sum_int16 = sum.cast[DType.int16]()
            
            for j in range(0,n):
                var m = lut_idx_cache[j]
                var el = ip.load[nelts](m, k)  
                var el_16 =  el.cast[DType.int16]() 
                sum_int16 += el_16
                
            
            var tmp = sum_int16
            
            for j in range(n):
                # var m = r2c_indices[slice_start+j]
                var m = lut_idx_cache[j]
                                    
                # TODO: find a way to avoid the second load from memory, perhaps by making use of memory > buffer struct
                var el = ip.load[nelts](m, k)
                tmp  = sum_int16 - el.cast[DType.int16]()
                
                tmp = tmp.min(max_allowed)
                tmp = tmp.max(min_allowed)

                var tmp_dtype = tmp.cast[dtype]()

                ip.store[nelts](m,k,tmp_dtype) 



        vectorize[vp, nelts](ip.dim1)
        #vectorize_unroll[nelts,8,vp](ip.dim1) # unroll factor ideally set to ip.dim1//nelts
    parallelize[vnpu](nc,intra_codeword_parallellism_factor)

@parameter
fn all_LQ(inout ip: Tensor3D, llr: Tensor3D, inout op: Tensor3D, start_codeword_idx: Int, intra_codeword_parallellism_factor: Int,nc: Int, ncodeword: Int, rep_indices: Tensor[DType.uint16], rep_offsets: Tensor[DType.uint16], r2c_indices: Tensor[DType.uint16]) raises:        

    @parameter
    fn vnpu(codebit_idx:Int):  # equivalent of : "for codebit_idx in range(nc):"
        var n = rep_indices[codebit_idx].to_int()
        var slice_start = rep_offsets[codebit_idx].to_int()

        var lut_idx_cache = InlinedFixedVector[Int](n)
        for j in range(0,n):
            var m = r2c_indices[slice_start+j].to_int()
            lut_idx_cache[j] = m

        @parameter
        fn vp[nelts_local: Int](k: Int): # equivalent of : "for k in range(ncodeword):"" but we are taking nelts = simdwidth codewords at a time
            var sum = llr.load[nelts](codebit_idx, k+start_codeword_idx)
            var tmp = sum.cast[DType.int16]()
            for j in range(0,n):
                var m = lut_idx_cache[j]
                var el = ip.load[nelts](m, k)                  
                tmp += el.cast[DType.int16]()               
           

            tmp = tmp.min(max_allowed)
            tmp = tmp.max(min_allowed)

            var tmp_dtype = tmp.cast[dtype]()

            @unroll
            for j in range(nelts): # subtoptimal, should be strided store but syntax eludes me a bit 
                op.store[1](start_codeword_idx+k+j,codebit_idx,tmp_dtype[j])            

        vectorize[vp, nelts](ncodeword)
        #vectorize_unroll[nelts,8,vp](ip.dim1) # unroll factor ideally set to ip.dim1//nelts
    parallelize[vnpu](nc,intra_codeword_parallellism_factor)

@parameter
fn all_LQ_for_syndrome(inout ip: Tensor3D, llr: Tensor3D, inout op: Tensor3D, start_codeword_idx: Int, intra_codeword_parallellism_factor: Int,nc: Int,  rep_indices:Tensor[DType.uint16], rep_offsets: Tensor[DType.uint16], r2c_indices: Tensor[DType.uint16]) raises:        

    @parameter
    fn vnpu(column_idx:Int):  # equivalent of : "for column_idx in range(nc):"
        var n = rep_indices[column_idx].to_int()
        var slice_start = rep_offsets[column_idx].to_int()

        var lut_idx_cache = InlinedFixedVector[Int](n)
        for j in range(0,n):
            var m = r2c_indices[slice_start+j].to_int()
            lut_idx_cache[j] = m

        @parameter
        fn vp[nelts_local: Int](k: Int):
            # TODO: if n == 1 we can at once set the result to zero across the board
            # also TODO : don't load the same values twice, create a local array of SIMD for the n values
            var sum = llr.load[nelts](column_idx, k+start_codeword_idx)
            var sum_int16 = sum.cast[DType.int16]()
            for j in range(0,n):
                var m = lut_idx_cache[j]
                var el = ip.load[nelts](m, k)                  
                sum_int16 += el.cast[DType.int16]()
                
            var tmp = sum_int16
            for j in range(n):
                var m = lut_idx_cache[j]

                tmp = tmp.min(max_allowed)
                tmp = tmp.max(min_allowed)

                var tmp_dtype = tmp.cast[dtype]()

                op.store[nelts](m,k,tmp_dtype)                       

        vectorize[vp, nelts](ip.dim1)
    parallelize[vnpu](nc,intra_codeword_parallellism_factor)

           


@parameter
fn decode_compiled_v5[ base_niter: Int, max_niter: Int](ip : Tensor3D, inout op : Tensor3D, inout batch_syndrome: Int, intra_codeword_parallellism_factor: Int, start_codeword_idx: Int, nc: Int,nnz: Int, ncnpu: Int, nvnpu: Int, ncodeword: Int, rep_indices: Tensor[DType.uint16], rep_offsets: Tensor[DType.uint16], c2r_indices: Tensor[DType.uint16], r2c_indices: Tensor[DType.uint16], vnpu_sizes: VariadicList[Int], vnpu_indices: VariadicList[Int], cnpu_sizes: VariadicList[Int], cnpu_indices: VariadicList[Int]) raises:
    #print("decode_compiled_v5 on start_codeword_idx = ",start_codeword_idx)
    var ip_llr_rep = repeat_compiled_offset(ip, start_codeword_idx,rep_indices, nnz, ncodeword)
    ip_llr_rep.set_num_workers(intra_codeword_parallellism_factor)
    
    var syndrome_scratchpad = Tensor3D(nnz, ncodeword)
    syndrome_scratchpad.set_num_workers(intra_codeword_parallellism_factor)

    # first iteration, declares the variables
    var reliabilities = c2r(ip_llr_rep,c2r_indices) # a permuted copy  # ip_llr_rep_rfirst  
    reliabilities.set_num_workers(intra_codeword_parallellism_factor)

    all_cnpu_compiled(reliabilities,nnz, ncnpu, cnpu_sizes,cnpu_indices) # modifies in-place  
    all_Lqij(reliabilities, ip,start_codeword_idx,intra_codeword_parallellism_factor,nc,rep_indices,rep_offsets,r2c_indices) # modifies in-place

    # remaining iterations
    #@unroll
    for idx in range(1,base_niter-1):
        all_cnpu_compiled(reliabilities,nnz, ncnpu, cnpu_sizes,cnpu_indices) # modifies in-place    
        all_Lqij(reliabilities, ip,start_codeword_idx,intra_codeword_parallellism_factor,nc,rep_indices,rep_offsets,r2c_indices) # modifies in-place
        

    # final iteration
    all_cnpu_compiled(reliabilities,nnz, ncnpu, cnpu_sizes,cnpu_indices) # modifies in-place    
    all_LQ(reliabilities, ip, op, start_codeword_idx, intra_codeword_parallellism_factor,nc,ncodeword,rep_indices,rep_offsets,r2c_indices) # modifies in-place    
    
    # batch syndrome computation  (inefficient, but leaves reliabilities untouched for continued iteration if must be )
    all_LQ_for_syndrome(reliabilities, ip, syndrome_scratchpad, start_codeword_idx,intra_codeword_parallellism_factor,nc,rep_indices,rep_offsets,r2c_indices) # modifies syndrome_scratchpad in-place            
    all_syndromes_compiled(syndrome_scratchpad, nnz, ncnpu, cnpu_sizes, cnpu_indices) 
    
    batch_syndrome = syndrome_scratchpad.reduce_syndrome()

    var other_niter = max_niter - base_niter
    if other_niter > 0:
        if batch_syndrome != 0:
            all_Lqij(reliabilities, ip,start_codeword_idx,intra_codeword_parallellism_factor,nc,rep_indices,rep_offsets,r2c_indices) # modifies in-place

            for idx in range(1,other_niter-1):
                all_cnpu_compiled(reliabilities,nnz, ncnpu, cnpu_sizes,cnpu_indices) # modifies in-place    
                all_Lqij(reliabilities, ip,start_codeword_idx,intra_codeword_parallellism_factor,nc,rep_indices,rep_offsets,r2c_indices) # modifies in-place
                

            # final iteration
            all_cnpu_compiled(reliabilities,nnz, ncnpu, cnpu_sizes,cnpu_indices) # modifies in-place    
            all_LQ(reliabilities, ip, op, start_codeword_idx, intra_codeword_parallellism_factor,nc,ncodeword,rep_indices,rep_offsets,r2c_indices) # modifies in-place    
            
            # batch syndrome computation  (inefficient, but leaves reliabilities untouched for continued iteration if must be )
            all_LQ_for_syndrome(reliabilities, ip, syndrome_scratchpad, start_codeword_idx,intra_codeword_parallellism_factor,nc,rep_indices,rep_offsets,r2c_indices) # modifies syndrome_scratchpad in-place            
            all_syndromes_compiled(syndrome_scratchpad, nnz, ncnpu, cnpu_sizes, cnpu_indices) 
            
            batch_syndrome = syndrome_scratchpad.reduce_syndrome()



fn parallel_decode[base_niter: Int, max_niter: Int](ip : Tensor3D, inout op : Tensor3D, inout batch_syndrome: Int, intra_codeword_parallellism_factor: Int, nthread: Int, nc: Int,nnz: Int, ncnpu: Int, nvnpu: Int, ncodewordperthread: Int, rep_indices: Tensor[DType.uint16], rep_offsets: Tensor[DType.uint16], c2r_indices: Tensor[DType.uint16], r2c_indices: Tensor[DType.uint16], vnpu_sizes: VariadicList[Int], vnpu_indices: VariadicList[Int], cnpu_sizes: VariadicList[Int], cnpu_indices: VariadicList[Int]) raises:
    var syndromes = InlinedFixedVector[Int](nthread)
    
    @parameter
    fn decode_slice(s: Int):
        var slice_syndrome: Int = -1
        try:
            decode_compiled_v5[base_niter, max_niter](ip, op, slice_syndrome, intra_codeword_parallellism_factor, s*ncodewordperthread, nc,nnz, ncnpu, nvnpu, ncodewordperthread, rep_indices, rep_offsets, c2r_indices, r2c_indices, vnpu_sizes, vnpu_indices, cnpu_sizes,cnpu_indices)
            syndromes[s] = slice_syndrome
        except:
            print("try failed on decode_compiled_v5")

    parallelize[decode_slice](nthread, nthread)

    # if any of the slices had a failed syndrome, then the whole batch syndrome is flagged as failed
    batch_syndrome = 0
    for i in range(nthread):
        # print("slice ",i,": syndrome",syndromes[i])
        if syndromes[i] > 0:
            batch_syndrome = 1
            
   

# repeats the first dimension according to repeat indices input
@parameter
fn repeat_compiled_offset(inpv: Tensor3D, start_codeword_idx: Int, indices: Tensor[DType.uint16], total_repeats: Int, ncodeword: Int)  -> Tensor3D:       
    var Z = Tensor3D(total_repeats, ncodeword)
    @parameter
    fn rep[nelts: Int](k: Int):
        var idx = 0
        for j in range(inpv.dim0):
            var x =  inpv.load[nelts](j, k+start_codeword_idx)  
            var nrep = indices[j].to_int()
            for n in range(nrep): 
                Z.store[nelts](idx,k,x)
                idx += 1

    vectorize[rep, nelts](ncodeword)        
    return Z 







# permutes the first dimension according to given indices
@parameter
fn c2r(inpv: Tensor3D, c2r_indices: Tensor[DType.uint16])  -> Tensor3D:
    var Z = Tensor3D(inpv.dim0, inpv.dim1)
    @parameter
    fn gthr_c2r[nelts: Int](k: Int):
        for j in range(inpv.dim0):                 
            var idx = c2r_indices[j].to_int()
            var x =  inpv.load[nelts](idx, k)                    
            Z.store[nelts](j,k,x)

    vectorize[gthr_c2r, nelts](inpv.dim1)        
    return Z

@always_inline
fn pe[nelts_pe: Int](x: SIMD[dtype,nelts_pe], y: SIMD[dtype,nelts_pe]) -> SIMD[dtype,nelts_pe]:
    var ax = abs(x)
    var ay = abs(y)
    var m = min(ax,ay)
    var sx = x < 0
    var sy = y < 0
    var sd = sx == sy
    var v = sd.select(m,-m)
    return v




struct Tensor3D:
    var dim0: Int
    var dim1: Int
    var dim2: Int # optional 3rd dimension
    var dimprod: Int # dim0 * dim1
    var num_workers: Int
    var _data: DTypePointer[dtype]
    #var _tensor: Tensor[dtype]
    alias simd_width: Int = nelts

    fn __init__(inout self, *dims: Int):
        self.dim0 = dims[0]
        self.dim1 = dims[1]
        self.dimprod = self.dim1*self.dim0
        if len(dims) > 2:
            self.dim2 = dims[2]
        else:
            self.dim2 = 1
        var size = self.dim0 * self.dim1 * self.dim2
        self.num_workers = num_logical_cores()
        self._data = DTypePointer[dtype].alloc(size)
        
        #self._tensor = Tensor[dtype](self.dim0, self.dim2, self.dim1) # note the swap of dim1 and dim2, for now needed to align interpretations ??
        #self._data = self._tensor.data()

        # random initialisation in the interval centered on +16, mimicking roughly an all-zeros codeword plus mild noise
        # 

    fn rand_init(inout self):
        var size = self.dim0 * self.dim1 * self.dim2
        randint(self._data, size,-2,10)

        
    fn __copyinit__(inout self, other: Self):
        #self._tensor = other._tensor
        self._data = other._data
        self.dim0 = other.dim0
        self.dim1 = other.dim1
        self.dim2 = other.dim2
        self.dimprod = other.dimprod
        self.num_workers = other.num_workers
        

    # Initialize taking a pointer, don't set any elements
    fn __init__(
        inout self, data: DTypePointer[dtype],*dims: Int
    ):
        self._data = data
        self.num_workers = 1
        self.dim0 = dims[0]
        self.dim1 = dims[1]
        self.dimprod = self.dim1*self.dim0
        if len(dims) > 2:
            self.dim2 = dims[2]
        else:
            self.dim2 = 1

 
    
    fn set_num_workers(inout self, num_workers: Int):
        self.num_workers = num_workers

    fn free(inout self):
        self._data.free()


    # add a third dimension and arteficially reduce the first dimension to new_dim
    # note no actual data is being moved
    fn reshape_dim0(inout self, new_dim0: Int) raises:
        assert_true((self.dim0 % new_dim0) == 0, "reshape new_dim0 must divide original dim0")
        var f = self.dim0 // new_dim0
        self.dim2 = f * self.dim2
        self.dim0 = new_dim0
        self.dimprod = self.dim1*self.dim0

    # this is the 'undo' operation of reshape_dim0
    # note no actual data is being moved
    fn flatten_dim2(inout self):
        self.dim0 = self.dim0 * self.dim2
        self.dim2 = 1
        self.dimprod = self.dim1*self.dim0

    # the below code is not safe unless run as last thing in the program, when the
    # Tensor3D is no longer used or needed. Otherwise we can get following error:
    # double free or corruption (top)
    fn save(self, fpath: String) raises  -> String: 
        var spec = TensorSpec(dtype, self.dim0, self.dim1)
        var _tensor = Tensor[dtype](self._data,spec)       
        _tensor.tofile(fpath)
        print('File saved:',fpath)
        return fpath


    #@staticmethod
    fn load(inout self, fpath:String) raises:
        var load_mat = Tensor[dtype].fromfile(fpath) 
        var size = self.dim0 * self.dim1 * self.dim2       
        assert_true(load_mat.num_elements() == size,"nr of elements in file do not match self.num_elements()")
        memcpy(self._data,load_mat.data(),size)
        _ = load_mat # dunno why this line would be needed tbh, nicked it from https://www.modular.com/blog/whats-new-in-mojo-sdk-v0-5

        
    fn reduce_syndrome(self) -> Int: 
        for j in range(self.dim0):
            for k in range(self.dim1):                 
                var v = self.load[1](j, k)
                if v > 0:
                    return 1
        return 0
        
    

    @always_inline
    fn __getitem__(self, j: Int, k: Int) -> SIMD[dtype,1]:
        return self._data.load[width=1](j * self.dim1 + k)

    @always_inline
    fn __getitem__(self, j: Int, k: Int, l: Int) -> SIMD[dtype,1]:
        return self._data.load[width=1](j * self.dim1 + l*self.dimprod + k)

    fn data(self) -> DTypePointer[dtype]:
        return self._data

    @always_inline
    fn get_reference_to_row_slice(inout self,slice_start: Int, slice_end: Int) -> Self:
        var slice_len = slice_end - slice_start
        var src_ptr = self._data.offset(slice_start*self.dim1)
        return Self(src_ptr,slice_len,self.dim1)
    


    fn print[maxel_x: Int = 3, maxel_y: Int = 3](self)->None:
        var rank:Int = 2
        var dim0:Int = 0
        var dim1:Int = 0
        var dim2:Int = 0
        var val:SIMD[dtype, 1]=0
        

        if self.dim0 == 1:
            rank = 1
            dim0 = 1
            dim1 = self.dim1
            dim2 = self.dim2
        else:
            dim0 = self.dim0
            dim1 = self.dim1
            dim2 = self.dim2
        if dim0>0 and dim1>0:
            for l in range(dim2):
                print("Page: ",l)
                for j in range(dim0):
                    if (j < maxel_x) | (j > (dim0 - (maxel_x+1))):
                        if rank>1:
                            if j==0:
                                print("  [",end="")
                            else:
                                print("\n   ",end="")
                        print("[",end="")
                        for k in range(dim1):
                            if rank==1:
                                val = self[j,k,l]
                            if rank==2:
                                val = self[j,k,l]
                            if k==0:
                                print(val,end="")
                            elif (k < maxel_y) | (k > (dim1 - (maxel_y+1))):
                                print("  ",val,end="")
                            elif k == maxel_y:
                                print("...",end="")

                        print("]",end="")
                    elif j == maxel_x:
                        print()
                        print("    ...     ")
                if rank>1:
                    print("]",end="")
                print()


            print()
            # if rank>2:
            #     print("]")
        print("  Tensor3D:",self.dim0,'x',self.dim1,'x',self.dim2,",","DType:", dtype.__str__())
        print()

    fn print_simd(self):
        for j in range(self.dim0):
            @parameter
            fn pr[nelts: Int](k: Int):
                print(self.load[nelts](j, k))            

            vectorize[pr, nelts](self.dim1)

    @always_inline
    fn load[nelts: Int](self, j: Int, k: Int) -> SIMD[dtype, nelts]:
        return self._data.load[width=nelts](j * self.dim1 + k)

    @always_inline
    fn store[nelts: Int](self, j: Int, k: Int, val: SIMD[dtype, nelts]):
        return self._data.store[width=nelts](j * self.dim1 + k, val)

    

    # variants in case the 3D view on the data is used
    @always_inline
    fn load[nelts: Int](self, j: Int, k: Int, l: Int) -> SIMD[dtype, nelts]:
        return self._data.load[width=nelts](j * self.dim1 + l*self.dimprod + k)

    @always_inline
    fn store[nelts: Int](self, j: Int, k: Int, l: Int, val: SIMD[dtype, nelts]):
        return self._data.store[width=nelts](j * self.dim1 + l*self.dimprod + k, val)

        
    

    # perform a syndrome_N operation whereby N is assumed to equal dim0
    fn syndromes(inout self):
        @parameter
        fn calc_page(l:Int):
            @parameter
            fn sd[nelts: Int](k: Int):
                # perform xor of all sign bits
                var v = self.load[nelts](0, k,l)
                var ok = v*0
                var nok = ok + 1
                var tmp = v < 0
                for j in range(1,self.dim0):                 
                    tmp = tmp ^ (self.load[nelts](j, k,l) < 0)
                # at this point tmp shall have False values where the parity checks hold (xor of all sign bits should be zero)
                for j in range(self.dim0):
                    var s = tmp.select(nok,ok)
                    self.store[nelts](j,k,l,s)

            vectorize[sd, nelts](self.dim1)
        parallelize[calc_page](self.dim2,self.num_workers)



    
    fn cnpu(inout self) raises:
        if self.dim0 == 3:
            self.cnpu_3()
        elif self.dim0 == 4:
            self.cnpu_4()
        elif self.dim0 == 5:
            self.cnpu_5()
        elif self.dim0 == 6:
            self.cnpu_6()
        elif self.dim0 == 7:
            self.cnpu_7()
        elif self.dim0 == 8:
            self.cnpu_8()
        elif self.dim0 == 9:
            self.cnpu_9()
        elif self.dim0 == 10:
            self.cnpu_10()
        elif self.dim0 == 11:
            self.cnpu_11()
        elif self.dim0 == 12:
            self.cnpu_12()
        elif self.dim0 == 13:
            self.cnpu_13()
        elif self.dim0 == 14:
            self.cnpu_14()
        elif self.dim0 == 15:
            self.cnpu_15()
        elif self.dim0 == 16:
            self.cnpu_16()
        elif self.dim0 == 32:
            self.cnpu_32()
        else:
            assert_true(False, "unsupported CNPU size")

    fn cnpu_3(inout self) raises:        
        assert_true(self.dim0 == 3,"Calling cnpu_3 on a Tensor3D with incompatible shape")     
        
        @parameter
        fn calc_page(l:Int):
            @parameter
            fn cnpu_3_loc[nelts: Int](k: Int):
                var r0 =  self.load[nelts](0, k,l)    
                var r1 =  self.load[nelts](1, k,l) 
                var r2 =  self.load[nelts](2, k,l)

                # get the first row of the result for the currently fetched SIMD vector
                var x = r1
                var y = r2
                var v = pe[nelts](x,y)        
                self.store[nelts](0,k,l,v) 

                # get the 2nd row of the result for the currently fetched SIMD vector
                x = r0
                y = r2
                v = pe[nelts](x,y)
                self.store[nelts](1,k,l,v)

                # get the 3rd row of the result for the currently fetched SIMD vector
                x = r0
                y = r1
                v = pe[nelts](x,y)
                self.store[nelts](2,k,l,v)

            # vectorize
            vectorize[cnpu_3_loc, nelts](self.dim1)
        parallelize[calc_page](self.dim2,self.num_workers)

    fn cnpu_4(inout self) raises:
        assert_true(self.dim0 == 4,"Calling cnpu_4 on a matrix with incompatible shape")
        @parameter
        fn calc_page(l:Int):
            @parameter
            fn cnpu_4_loc[nelts: Int](k: Int):
                var x_4_1 = self.load[nelts](0, k, l)
                var x_4_2 = self.load[nelts](1, k, l)
                var x_4_3 = self.load[nelts](2, k, l)
                var x_4_4 = self.load[nelts](3, k, l)
                var x_duo_1 = x_4_1
                var y_duo_1 = x_4_2
                var pc_4_1_to_2 = pe[nelts](x_duo_1,y_duo_1)
                var x_duo_2 = pc_4_1_to_2
                var y_duo_2 = x_4_3
                var pc_4_1_to_3 = pe[nelts](x_duo_2,y_duo_2)
                var x_duo_rev_4 = x_4_4
                var y_duo_rev_4 = x_4_3
                var pc_4_3_to_4 = pe[nelts](x_duo_rev_4,y_duo_rev_4)
                var x_duo_rev_3 = pc_4_3_to_4
                var y_duo_rev_3 = x_4_2
                var pc_4_2_to_4 = pe[nelts](x_duo_rev_3,y_duo_rev_3)
                var x_op_duo_2 = x_4_1
                var y_op_duo_2 = pc_4_3_to_4
                var x_op_duo_3 = pc_4_1_to_2
                var y_op_duo_3 = x_4_4
                var op_4_1 = pc_4_2_to_4
                var op_4_2 = pe[nelts](x_op_duo_2,y_op_duo_2)
                var op_4_3 = pe[nelts](x_op_duo_3,y_op_duo_3)
                var op_4_4 = pc_4_1_to_3
                self.store[nelts](0,k,l,op_4_1)
                self.store[nelts](1,k,l,op_4_2)
                self.store[nelts](2,k,l,op_4_3)
                self.store[nelts](3,k,l,op_4_4)
            # vectorize
            vectorize[cnpu_4_loc, nelts](self.dim1)
        parallelize[calc_page](self.dim2,self.num_workers)

    fn cnpu_5(inout self) raises:
        assert_true(self.dim0 == 5,"Calling cnpu_5 on a matrix with incompatible shape")
        @parameter
        fn calc_page(l:Int):
            @parameter
            fn cnpu_5_loc[nelts: Int](k: Int):
                var x_5_1 = self.load[nelts](0, k, l)
                var x_5_2 = self.load[nelts](1, k, l)
                var x_5_3 = self.load[nelts](2, k, l)
                var x_5_4 = self.load[nelts](3, k, l)
                var x_5_5 = self.load[nelts](4, k, l)
                var x_duo_1 = x_5_1
                var y_duo_1 = x_5_2
                var pc_5_1_to_2 = pe[nelts](x_duo_1,y_duo_1)
                var x_duo_2 = pc_5_1_to_2
                var y_duo_2 = x_5_3
                var pc_5_1_to_3 = pe[nelts](x_duo_2,y_duo_2)
                var x_duo_3 = pc_5_1_to_3
                var y_duo_3 = x_5_4
                var pc_5_1_to_4 = pe[nelts](x_duo_3,y_duo_3)
                var x_duo_rev_5 = x_5_5
                var y_duo_rev_5 = x_5_4
                var pc_5_4_to_5 = pe[nelts](x_duo_rev_5,y_duo_rev_5)
                var x_duo_rev_4 = pc_5_4_to_5
                var y_duo_rev_4 = x_5_3
                var pc_5_3_to_5 = pe[nelts](x_duo_rev_4,y_duo_rev_4)
                var x_duo_rev_3 = pc_5_3_to_5
                var y_duo_rev_3 = x_5_2
                var pc_5_2_to_5 = pe[nelts](x_duo_rev_3,y_duo_rev_3)
                var x_op_duo_2 = x_5_1
                var y_op_duo_2 = pc_5_3_to_5
                var x_op_duo_3 = pc_5_1_to_2
                var y_op_duo_3 = pc_5_4_to_5
                var x_op_duo_4 = pc_5_1_to_3
                var y_op_duo_4 = x_5_5
                var op_5_1 = pc_5_2_to_5
                var op_5_2 = pe[nelts](x_op_duo_2,y_op_duo_2)
                var op_5_3 = pe[nelts](x_op_duo_3,y_op_duo_3)
                var op_5_4 = pe[nelts](x_op_duo_4,y_op_duo_4)
                var op_5_5 = pc_5_1_to_4
                self.store[nelts](0,k,l,op_5_1)
                self.store[nelts](1,k,l,op_5_2)
                self.store[nelts](2,k,l,op_5_3)
                self.store[nelts](3,k,l,op_5_4)
                self.store[nelts](4,k,l,op_5_5)
            # vectorize
            vectorize[cnpu_5_loc, nelts](self.dim1)
        parallelize[calc_page](self.dim2,self.num_workers)

    fn cnpu_6(inout self) raises:
        assert_true(self.dim0 == 6,"Calling cnpu_6 on a matrix with incompatible shape")
        @parameter
        fn calc_page(l:Int):
            @parameter
            fn cnpu_6_loc[nelts: Int](k: Int):
                var x_6_1 = self.load[nelts](0, k, l)
                var x_6_2 = self.load[nelts](1, k, l)
                var x_6_3 = self.load[nelts](2, k, l)
                var x_6_4 = self.load[nelts](3, k, l)
                var x_6_5 = self.load[nelts](4, k, l)
                var x_6_6 = self.load[nelts](5, k, l)
                var x_duo_1 = x_6_1
                var y_duo_1 = x_6_2
                var pc_6_1_to_2 = pe[nelts](x_duo_1,y_duo_1)
                var x_duo_2 = pc_6_1_to_2
                var y_duo_2 = x_6_3
                var pc_6_1_to_3 = pe[nelts](x_duo_2,y_duo_2)
                var x_duo_3 = pc_6_1_to_3
                var y_duo_3 = x_6_4
                var pc_6_1_to_4 = pe[nelts](x_duo_3,y_duo_3)
                var x_duo_4 = pc_6_1_to_4
                var y_duo_4 = x_6_5
                var pc_6_1_to_5 = pe[nelts](x_duo_4,y_duo_4)
                var x_duo_rev_6 = x_6_6
                var y_duo_rev_6 = x_6_5
                var pc_6_5_to_6 = pe[nelts](x_duo_rev_6,y_duo_rev_6)
                var x_duo_rev_5 = pc_6_5_to_6
                var y_duo_rev_5 = x_6_4
                var pc_6_4_to_6 = pe[nelts](x_duo_rev_5,y_duo_rev_5)
                var x_duo_rev_4 = pc_6_4_to_6
                var y_duo_rev_4 = x_6_3
                var pc_6_3_to_6 = pe[nelts](x_duo_rev_4,y_duo_rev_4)
                var x_duo_rev_3 = pc_6_3_to_6
                var y_duo_rev_3 = x_6_2
                var pc_6_2_to_6 = pe[nelts](x_duo_rev_3,y_duo_rev_3)
                var x_op_duo_2 = x_6_1
                var y_op_duo_2 = pc_6_3_to_6
                var x_op_duo_3 = pc_6_1_to_2
                var y_op_duo_3 = pc_6_4_to_6
                var x_op_duo_4 = pc_6_1_to_3
                var y_op_duo_4 = pc_6_5_to_6
                var x_op_duo_5 = pc_6_1_to_4
                var y_op_duo_5 = x_6_6
                var op_6_1 = pc_6_2_to_6
                var op_6_2 = pe[nelts](x_op_duo_2,y_op_duo_2)
                var op_6_3 = pe[nelts](x_op_duo_3,y_op_duo_3)
                var op_6_4 = pe[nelts](x_op_duo_4,y_op_duo_4)
                var op_6_5 = pe[nelts](x_op_duo_5,y_op_duo_5)
                var op_6_6 = pc_6_1_to_5
                self.store[nelts](0,k,l,op_6_1)
                self.store[nelts](1,k,l,op_6_2)
                self.store[nelts](2,k,l,op_6_3)
                self.store[nelts](3,k,l,op_6_4)
                self.store[nelts](4,k,l,op_6_5)
                self.store[nelts](5,k,l,op_6_6)
            # vectorize
            vectorize[cnpu_6_loc, nelts](self.dim1)
        parallelize[calc_page](self.dim2,self.num_workers)

    

    fn cnpu_8(inout self) raises:
        assert_true(self.dim0 == 8,"Calling cnpu_8 on a matrix with incompatible shape")
        @parameter
        fn calc_page(l:Int):
            @parameter
            fn cnpu_8_loc[nelts: Int](k: Int):
                var x_8_1 = self.load[nelts](0, k, l)
                var x_8_2 = self.load[nelts](1, k, l)
                var x_8_3 = self.load[nelts](2, k, l)
                var x_8_4 = self.load[nelts](3, k, l)
                var x_8_5 = self.load[nelts](4, k, l)
                var x_8_6 = self.load[nelts](5, k, l)
                var x_8_7 = self.load[nelts](6, k, l)
                var x_8_8 = self.load[nelts](7, k, l)
                var x_duo_1 = x_8_1
                var y_duo_1 = x_8_2
                var pc_8_1_to_2 = pe[nelts](x_duo_1,y_duo_1)
                var x_duo_2 = pc_8_1_to_2
                var y_duo_2 = x_8_3
                var pc_8_1_to_3 = pe[nelts](x_duo_2,y_duo_2)
                var x_duo_3 = pc_8_1_to_3
                var y_duo_3 = x_8_4
                var pc_8_1_to_4 = pe[nelts](x_duo_3,y_duo_3)
                var x_duo_4 = pc_8_1_to_4
                var y_duo_4 = x_8_5
                var pc_8_1_to_5 = pe[nelts](x_duo_4,y_duo_4)
                var x_duo_5 = pc_8_1_to_5
                var y_duo_5 = x_8_6
                var pc_8_1_to_6 = pe[nelts](x_duo_5,y_duo_5)
                var x_duo_6 = pc_8_1_to_6
                var y_duo_6 = x_8_7
                var pc_8_1_to_7 = pe[nelts](x_duo_6,y_duo_6)
                var x_duo_rev_8 = x_8_8
                var y_duo_rev_8 = x_8_7
                var pc_8_7_to_8 = pe[nelts](x_duo_rev_8,y_duo_rev_8)
                var x_duo_rev_7 = pc_8_7_to_8
                var y_duo_rev_7 = x_8_6
                var pc_8_6_to_8 = pe[nelts](x_duo_rev_7,y_duo_rev_7)
                var x_duo_rev_6 = pc_8_6_to_8
                var y_duo_rev_6 = x_8_5
                var pc_8_5_to_8 = pe[nelts](x_duo_rev_6,y_duo_rev_6)
                var x_duo_rev_5 = pc_8_5_to_8
                var y_duo_rev_5 = x_8_4
                var pc_8_4_to_8 = pe[nelts](x_duo_rev_5,y_duo_rev_5)
                var x_duo_rev_4 = pc_8_4_to_8
                var y_duo_rev_4 = x_8_3
                var pc_8_3_to_8 = pe[nelts](x_duo_rev_4,y_duo_rev_4)
                var x_duo_rev_3 = pc_8_3_to_8
                var y_duo_rev_3 = x_8_2
                var pc_8_2_to_8 = pe[nelts](x_duo_rev_3,y_duo_rev_3)
                var x_op_duo_2 = x_8_1
                var y_op_duo_2 = pc_8_3_to_8
                var x_op_duo_3 = pc_8_1_to_2
                var y_op_duo_3 = pc_8_4_to_8
                var x_op_duo_4 = pc_8_1_to_3
                var y_op_duo_4 = pc_8_5_to_8
                var x_op_duo_5 = pc_8_1_to_4
                var y_op_duo_5 = pc_8_6_to_8
                var x_op_duo_6 = pc_8_1_to_5
                var y_op_duo_6 = pc_8_7_to_8
                var x_op_duo_7 = pc_8_1_to_6
                var y_op_duo_7 = x_8_8
                var op_8_1 = pc_8_2_to_8
                var op_8_2 = pe[nelts](x_op_duo_2,y_op_duo_2)
                var op_8_3 = pe[nelts](x_op_duo_3,y_op_duo_3)
                var op_8_4 = pe[nelts](x_op_duo_4,y_op_duo_4)
                var op_8_5 = pe[nelts](x_op_duo_5,y_op_duo_5)
                var op_8_6 = pe[nelts](x_op_duo_6,y_op_duo_6)
                var op_8_7 = pe[nelts](x_op_duo_7,y_op_duo_7)
                var op_8_8 = pc_8_1_to_7
                self.store[nelts](0,k,l,op_8_1)
                self.store[nelts](1,k,l,op_8_2)
                self.store[nelts](2,k,l,op_8_3)
                self.store[nelts](3,k,l,op_8_4)
                self.store[nelts](4,k,l,op_8_5)
                self.store[nelts](5,k,l,op_8_6)
                self.store[nelts](6,k,l,op_8_7)
                self.store[nelts](7,k,l,op_8_8)
            # vectorize
            vectorize[cnpu_8_loc, nelts](self.dim1)
        parallelize[calc_page](self.dim2,self.num_workers)

  

    fn cnpu_10(inout self) raises:
        assert_true(self.dim0 == 10,"Calling cnpu_10 on a matrix with incompatible shape")
        @parameter
        fn calc_page(l:Int):
            @parameter
            fn cnpu_10_loc[nelts: Int](k: Int):
                var x_10_1 = self.load[nelts](0, k, l)
                var x_10_2 = self.load[nelts](1, k, l)
                var x_10_3 = self.load[nelts](2, k, l)
                var x_10_4 = self.load[nelts](3, k, l)
                var x_10_5 = self.load[nelts](4, k, l)
                var x_10_6 = self.load[nelts](5, k, l)
                var x_10_7 = self.load[nelts](6, k, l)
                var x_10_8 = self.load[nelts](7, k, l)
                var x_10_9 = self.load[nelts](8, k, l)
                var x_10_10 = self.load[nelts](9, k, l)
                var x_duo_1 = x_10_1
                var y_duo_1 = x_10_2
                var pc_10_1_to_2 = pe[nelts](x_duo_1,y_duo_1)
                var x_duo_2 = pc_10_1_to_2
                var y_duo_2 = x_10_3
                var pc_10_1_to_3 = pe[nelts](x_duo_2,y_duo_2)
                var x_duo_3 = pc_10_1_to_3
                var y_duo_3 = x_10_4
                var pc_10_1_to_4 = pe[nelts](x_duo_3,y_duo_3)
                var x_duo_4 = pc_10_1_to_4
                var y_duo_4 = x_10_5
                var pc_10_1_to_5 = pe[nelts](x_duo_4,y_duo_4)
                var x_duo_5 = pc_10_1_to_5
                var y_duo_5 = x_10_6
                var pc_10_1_to_6 = pe[nelts](x_duo_5,y_duo_5)
                var x_duo_6 = pc_10_1_to_6
                var y_duo_6 = x_10_7
                var pc_10_1_to_7 = pe[nelts](x_duo_6,y_duo_6)
                var x_duo_7 = pc_10_1_to_7
                var y_duo_7 = x_10_8
                var pc_10_1_to_8 = pe[nelts](x_duo_7,y_duo_7)
                var x_duo_8 = pc_10_1_to_8
                var y_duo_8 = x_10_9
                var pc_10_1_to_9 = pe[nelts](x_duo_8,y_duo_8)
                var x_duo_rev_10 = x_10_10
                var y_duo_rev_10 = x_10_9
                var pc_10_9_to_10 = pe[nelts](x_duo_rev_10,y_duo_rev_10)
                var x_duo_rev_9 = pc_10_9_to_10
                var y_duo_rev_9 = x_10_8
                var pc_10_8_to_10 = pe[nelts](x_duo_rev_9,y_duo_rev_9)
                var x_duo_rev_8 = pc_10_8_to_10
                var y_duo_rev_8 = x_10_7
                var pc_10_7_to_10 = pe[nelts](x_duo_rev_8,y_duo_rev_8)
                var x_duo_rev_7 = pc_10_7_to_10
                var y_duo_rev_7 = x_10_6
                var pc_10_6_to_10 = pe[nelts](x_duo_rev_7,y_duo_rev_7)
                var x_duo_rev_6 = pc_10_6_to_10
                var y_duo_rev_6 = x_10_5
                var pc_10_5_to_10 = pe[nelts](x_duo_rev_6,y_duo_rev_6)
                var x_duo_rev_5 = pc_10_5_to_10
                var y_duo_rev_5 = x_10_4
                var pc_10_4_to_10 = pe[nelts](x_duo_rev_5,y_duo_rev_5)
                var x_duo_rev_4 = pc_10_4_to_10
                var y_duo_rev_4 = x_10_3
                var pc_10_3_to_10 = pe[nelts](x_duo_rev_4,y_duo_rev_4)
                var x_duo_rev_3 = pc_10_3_to_10
                var y_duo_rev_3 = x_10_2
                var pc_10_2_to_10 = pe[nelts](x_duo_rev_3,y_duo_rev_3)
                var x_op_duo_2 = x_10_1
                var y_op_duo_2 = pc_10_3_to_10
                var x_op_duo_3 = pc_10_1_to_2
                var y_op_duo_3 = pc_10_4_to_10
                var x_op_duo_4 = pc_10_1_to_3
                var y_op_duo_4 = pc_10_5_to_10
                var x_op_duo_5 = pc_10_1_to_4
                var y_op_duo_5 = pc_10_6_to_10
                var x_op_duo_6 = pc_10_1_to_5
                var y_op_duo_6 = pc_10_7_to_10
                var x_op_duo_7 = pc_10_1_to_6
                var y_op_duo_7 = pc_10_8_to_10
                var x_op_duo_8 = pc_10_1_to_7
                var y_op_duo_8 = pc_10_9_to_10
                var x_op_duo_9 = pc_10_1_to_8
                var y_op_duo_9 = x_10_10
                var op_10_1 = pc_10_2_to_10
                var op_10_2 = pe[nelts](x_op_duo_2,y_op_duo_2)
                var op_10_3 = pe[nelts](x_op_duo_3,y_op_duo_3)
                var op_10_4 = pe[nelts](x_op_duo_4,y_op_duo_4)
                var op_10_5 = pe[nelts](x_op_duo_5,y_op_duo_5)
                var op_10_6 = pe[nelts](x_op_duo_6,y_op_duo_6)
                var op_10_7 = pe[nelts](x_op_duo_7,y_op_duo_7)
                var op_10_8 = pe[nelts](x_op_duo_8,y_op_duo_8)
                var op_10_9 = pe[nelts](x_op_duo_9,y_op_duo_9)
                var op_10_10 = pc_10_1_to_9
                self.store[nelts](0,k,l,op_10_1)
                self.store[nelts](1,k,l,op_10_2)
                self.store[nelts](2,k,l,op_10_3)
                self.store[nelts](3,k,l,op_10_4)
                self.store[nelts](4,k,l,op_10_5)
                self.store[nelts](5,k,l,op_10_6)
                self.store[nelts](6,k,l,op_10_7)
                self.store[nelts](7,k,l,op_10_8)
                self.store[nelts](8,k,l,op_10_9)
                self.store[nelts](9,k,l,op_10_10)
            # vectorize
            vectorize[cnpu_10_loc, nelts](self.dim1)
        parallelize[calc_page](self.dim2,self.num_workers)

    





   




    
