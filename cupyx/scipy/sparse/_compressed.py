from __future__ import annotations

import string
import warnings

import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
import cupyx

from cupy import _core
from cupy._core import _scalar
from cupy._creation import basic
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util

from cupyx.scipy.sparse import _index


class _compressed_sparse_matrix(sparse_data._data_matrix,
                                sparse_data._minmax_mixin,
                                _index.IndexMixin):

    _max_min_reduction_code = r'''
        template<typename TI> __global__
        void ${func}(double* data, TI* x, TI* y, TI length, double* z) {
            // Get the index of the block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Calculate the block length
            TI block_length = y[tid] - x[tid];

            // Select initial value based on the block density
            double running_value = 0;
            if (${cond}){
                running_value = data[x[tid]];
            } else {
                running_value = 0;
            }

            // Iterate over the block and update
            for (TI entry = x[tid]; entry < y[tid]; entry++){
                if (data[entry] != data[entry]){
                    // Check for NaN
                    running_value = nan("");
                    break;
                } else {
                    // Check for a value update
                    if (data[entry] ${op} running_value){
                        running_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = running_value;
        }'''

    # Index type specializations: int (int32) and long long (int64).
    _idx_types = ('int', 'long long')

    _max_reduction_mod = _core.RawModule(
        code=string.Template(_max_min_reduction_code).substitute(
            func='max_reduction', op='>', cond='block_length == length'),
        name_expressions=[
            f'max_reduction<{t}>' for t in _idx_types])

    _max_nonzero_reduction_mod = _core.RawModule(
        code=string.Template(_max_min_reduction_code).substitute(
            func='max_nonzero_reduction', op='>',
            cond='block_length > 0'),
        name_expressions=[
            f'max_nonzero_reduction<{t}>' for t in _idx_types])

    _min_reduction_mod = _core.RawModule(
        code=string.Template(_max_min_reduction_code).substitute(
            func='min_reduction', op='<', cond='block_length == length'),
        name_expressions=[
            f'min_reduction<{t}>' for t in _idx_types])

    _min_nonzero_reduction_mod = _core.RawModule(
        code=string.Template(_max_min_reduction_code).substitute(
            func='min_nonzero_reduction', op='<',
            cond='block_length > 0'),
        name_expressions=[
            f'min_nonzero_reduction<{t}>' for t in _idx_types])

    # For _max_arg_reduction_mod and _min_arg_reduction_mod below, we pick
    # the right template specialization according to input dtypes at runtime.
    # The distinction in int types (T2) is important for portability in OS.

    _argmax_argmin_code = r'''
        template<typename T1, typename T2, typename TI> __global__ void
        ${func}_arg_reduction(T1* data, TI* indices, TI* x, TI* y,
                              TI length, T2* z) {
            // Get the index of the block
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Calculate the block length
            TI block_length = y[tid] - x[tid];

            // Select initial value based on the block density
            TI data_index = 0;
            double data_value = 0;

            if (block_length == length){
                // Block is dense. Fill the first value
                data_value = data[x[tid]];
                data_index = indices[x[tid]];
            } else if (block_length > 0)  {
                // Block has at least one zero. Assign first occurrence as the
                // starting reference
                data_value = 0;
                for (data_index = 0; data_index < length; data_index++){
                    if (data_index != indices[x[tid] + data_index] ||
                        x[tid] + data_index >= y[tid]){
                        break;
                    }
                }
            } else {
                // Zero valued array
                data_value = 0;
                data_index = 0;
            }

            // Iterate over the section of the sparse matrix
            for (TI entry = x[tid]; entry < y[tid]; entry++){
                if (data[entry] != data[entry]){
                    // Check for NaN
                    data_value = nan("");
                    data_index = 0;
                    break;
                } else {
                    // Check for a value update
                    if (data[entry] ${op} data_value){
                        data_index = indices[entry];
                        data_value = data[entry];
                    }
                }
            }

            // Store in the return function
            z[tid] = (T2)data_index;
        }'''

    # T1=data type, T2=output type, TI=index type
    _arg_reduction_types = [
        f'{{func}}_arg_reduction<{t1}, {t2}, {ti}>'
        for t1 in ('float', 'double')
        for t2 in ('int', 'long long')
        for ti in ('int', 'long long')
    ]

    _max_arg_reduction_mod = _core.RawModule(
        code=string.Template(_argmax_argmin_code).substitute(
            func='max', op='>'),
        name_expressions=[
            t.format(func='max') for t in _arg_reduction_types])

    _min_arg_reduction_mod = _core.RawModule(
        code=string.Template(_argmax_argmin_code).substitute(
            func='min', op='<'),
        name_expressions=[
            t.format(func='min') for t in _arg_reduction_types])

    # TODO(leofang): rewrite a more load-balanced approach than this naive one?
    _has_sorted_indices_kern = _core.ElementwiseKernel(
        'raw T indptr, raw T indices',
        'bool diff',
        '''
        bool diff_out = true;
        for (T jj = indptr[i]; jj < indptr[i+1] - 1; jj++) {
            if (indices[jj] > indices[jj+1]){
                diff_out = false;
            }
        }
        diff = diff_out;
        ''', 'cupyx_scipy_sparse_has_sorted_indices')

    # TODO(leofang): rewrite a more load-balanced approach than this naive one?
    _has_canonical_format_kern = _core.ElementwiseKernel(
        'raw T indptr, raw T indices',
        'bool diff',
        '''
        bool diff_out = true;
        if (indptr[i] > indptr[i+1]) {
            diff = false;
            return;
        }
        for (T jj = indptr[i]; jj < indptr[i+1] - 1; jj++) {
            if (indices[jj] >= indices[jj+1]) {
                diff_out = false;
            }
        }
        diff = diff_out;
        ''', 'cupyx_scipy_sparse_has_canonical_format')

    def __init__(self, arg1, shape=None, dtype=None, copy=False,
                 *, maxprint=None):
        from cupyx import cusparse

        if maxprint is not None:
            self.maxprint = maxprint

        if shape is not None:
            if not _util.isshape(shape, nonneg=True):
                raise ValueError(
                    'invalid shape (must be a 2-tuple of non-negative '
                    'int)')
            shape = int(shape[0]), int(shape[1])

        if _base.issparse(arg1):
            x = arg1.asformat(self.format)
            data = x.data
            indices = x.indices
            indptr = x.indptr
            # Preserve index dtype from source matrix.
            idx_dtype = indices.dtype

            if arg1.format != self.format:
                # When formats are different, all arrays are already copied
                copy = False

            if shape is None:
                shape = arg1.shape

        elif _util.isshape(arg1):
            # ``isshape`` here intentionally allows negatives so that
            # ``csr_array((-3, 5))`` is dispatched as a shape -- and
            # rejected by the explicit ``nonneg=True`` check below
            # rather than mis-dispatched to the (data, (row, col)) /
            # tuple-2 branches and surfacing as a confusing
            # "invalid input format" error.
            if not _util.isshape(arg1, nonneg=True):
                raise ValueError(
                    'invalid shape (must be a 2-tuple of non-negative '
                    'int)')
            m, n = arg1
            m, n = int(m), int(n)
            idx_dtype = _sputils.get_index_dtype(maxval=max(m, n))
            data = basic.zeros(0, dtype if dtype else 'd')
            indices = basic.zeros(0, idx_dtype)
            indptr = basic.zeros(self._swap(m, n)[0] + 1, dtype=idx_dtype)
            # shape and copy argument is ignored
            shape = (m, n)
            copy = False

        elif scipy_available and scipy.sparse.issparse(arg1):
            # Convert scipy.sparse to cupyx.scipy.sparse.
            # Preserve scipy's index dtype (scipy uses
            # get_index_dtype internally).
            x = arg1.asformat(self.format)
            data = cupy.array(x.data)
            idx_dtype = x.indices.dtype
            indices = cupy.array(x.indices, dtype=idx_dtype)
            indptr = cupy.array(x.indptr, dtype=idx_dtype)
            copy = False

            if shape is None:
                shape = arg1.shape

        elif isinstance(arg1, tuple) and len(arg1) == 2:
            # Note: This implementation is not efficient, as it first
            # constructs a sparse object with coo format, then converts it to
            # compressed format.  Use ``self._coo_container`` so the array
            # vs. matrix kind (and the array path's ``check_contents=False``
            # int64 preservation) carries through to the intermediate.
            sp_coo = self._coo_container(
                arg1, shape=shape, dtype=dtype, copy=copy)
            sp_compressed = sp_coo.asformat(self.format)
            data = sp_compressed.data
            indices = sp_compressed.indices
            indptr = sp_compressed.indptr
            # Derived from COO's get_index_dtype call.
            idx_dtype = indices.dtype

        elif isinstance(arg1, tuple) and len(arg1) == 3:
            data, indices, indptr = arg1
            if not (_base.isdense(data) and data.ndim == 1 and
                    _base.isdense(indices) and indices.ndim == 1 and
                    _base.isdense(indptr) and indptr.ndim == 1):
                raise ValueError(
                    'data, indices, and indptr should be 1-D')

            if len(data) != len(indices):
                raise ValueError('indices and data should have the same size')

            # V2-24: validate ``indptr[0] == 0`` (matches scipy).
            # Without this guard, an `indptr` that doesn't start at 0
            # silently produces wrong results (or, in cuSPARSE-backed
            # downstream ops, can crash the CUDA context with
            # ``cudaErrorInvalidAddressSpace``).  ``check_contents=True``
            # below already syncs, so folding in one more D2H read
            # for the head element is essentially free.
            if indptr.size > 0 and int(indptr[0]) != 0:
                raise ValueError(
                    'index pointer should start with 0 (got {})'
                    .format(int(indptr[0])))

            # Mirror scipy: choose int32 when values fit,
            # int64 when they don't.
            # maxval may be None if shape is not yet known; contents check
            # handles that case.
            maxval = max(shape) if shape is not None else None
            idx_dtype = self._get_index_dtype(
                (indices, indptr), maxval=maxval, check_contents=True)

        elif _base.isdense(arg1):
            if arg1.ndim > 2:
                raise ValueError('expected dimension <= 2 array or matrix')
            elif arg1.ndim == 1:
                arg1 = arg1[None]
            elif arg1.ndim == 0:
                arg1 = arg1[None, None]
            data, indices, indptr = self._convert_dense(arg1)
            idx_dtype = indices.dtype  # respect what _convert_dense chose
            copy = False
            if shape is None:
                shape = arg1.shape

        else:
            # Match scipy's error message for the most common surprise
            # (scalar input).  scipy: ``csr_matrix(5)`` builds a
            # (1, 1) matrix, but ``csr_array(5)`` raises "scipy sparse
            # array classes do not support instantiation from a
            # scalar".  CuPy raises in both cases (no scalar fast path
            # for matrices), but the message should at least name
            # scalars when that's what was passed.
            if numpy.isscalar(arg1):
                if isinstance(self, _base.sparray):
                    raise ValueError(
                        'sparse array classes do not support '
                        'instantiation from a scalar')
                raise ValueError(
                    'unrecognized {}_matrix constructor usage'
                    .format(self.format))
            # NumPy ndarrays are *not* implicitly accepted by CuPy
            # sparse constructors (no implicit host->device transfer);
            # mention that explicitly so a user porting from scipy
            # knows to call ``cupy.asarray`` first.
            if isinstance(arg1, numpy.ndarray):
                raise ValueError(
                    'numpy.ndarray input is not supported; '
                    'wrap with cupy.asarray() first')
            raise ValueError(
                'Unsupported initializer format')

        if dtype is None:
            dtype = data.dtype
        else:
            dtype = numpy.dtype(dtype)

        if dtype.char not in '?fdFD':
            raise ValueError(
                'Only bool, float32, float64, complex64 and complex128 '
                'are supported')

        data = data.astype(dtype, copy=copy)
        sparse_data._data_matrix.__init__(self, data)

        self.indices = indices.astype(idx_dtype, copy=copy)
        self.indptr = indptr.astype(idx_dtype, copy=copy)

        if shape is None:
            shape = self._swap(
                len(indptr) - 1, int(indices.max()) + 1)  # synchronize!

        major, minor = self._swap(*shape)
        if len(indptr) != major + 1:
            raise ValueError('index pointer size (%d) should be (%d)'
                             % (len(indptr), major + 1))

        self._descr = cusparse.MatDescriptor.create()
        self._shape = shape

    @classmethod
    def _from_parts(cls, data, indices, indptr, shape,
                    has_canonical_format=None,
                    has_sorted_indices=None,
                    _skip_buffer_check=False):
        """Construct from pre-validated arrays (no check_contents).

        Internal API for building sparse matrices when the caller has
        already determined the correct index dtype.  Skips the
        check_contents=True downcast that the tuple-3 constructor
        applies.

        Caller must ensure *indices* and *indptr* share the same
        integer dtype and are within bounds for *shape*.

        Args:
            has_canonical_format (bool or None): If ``True`` or
                ``False``, cache the flag directly (avoids the lazy
                GPU kernel on first access).  ``None`` (default)
                leaves the flag unset for lazy computation.
                ``True`` implies ``has_sorted_indices=True``.
            has_sorted_indices (bool or None): Same semantics.
            _skip_buffer_check (bool): **Internal use only.**  Opt-out
                for callers that have already validated
                ``data.size == indptr[-1]`` (e.g. ``_with_data``,
                ``_empty_like``, all ``cupyx.cusparse`` wraps that
                produce tight buffers by construction).  Default
                ``False`` validates the invariant with one D2H read of
                ``indptr[-1]``.  External callers should leave this
                ``False`` -- the validation closes a silent
                cuSPARSE-corruption path (slack data treated as live).

        Raises:
            ValueError: If *indices* and *indptr* dtypes differ, the
                ``has_canonical_format`` / ``has_sorted_indices`` flags
                are inconsistent, ``data`` and ``indices`` lengths
                differ, ``indptr`` length does not match the major
                axis of *shape*, the index dtype is too narrow for
                *shape*, or (when ``_skip_buffer_check`` is False) the
                live nnz from ``indptr[-1]`` does not match
                ``data.size``.
        """
        # V2-8: ndim guard.  The public constructor enforces 1-D for
        # ``data`` / ``indices`` / ``indptr``; the ``_from_parts``
        # bypass had no such check, letting a 2-D buffer slip through
        # and break downstream ops with confusing errors.
        if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
            raise ValueError(
                'data, indices, and indptr must be 1-D, got ndim '
                '{}, {}, {}'.format(data.ndim, indices.ndim, indptr.ndim))
        if indices.dtype != indptr.dtype:
            raise ValueError(
                'indices and indptr must have the same dtype, '
                'got {} and {}'.format(indices.dtype, indptr.dtype))
        if has_canonical_format is True and has_sorted_indices is False:
            raise ValueError(
                'has_canonical_format=True implies sorted indices, '
                'but has_sorted_indices=False was passed')
        if data.size != indices.size:
            raise ValueError(
                'data and indices must have the same length, '
                'got {} and {}'.format(data.size, indices.size))
        # Major axis: shape[0] for CSR (_major_axis=0), shape[1] for CSC.
        # Subclasses must define _major_axis.
        major = shape[cls._major_axis]
        if indptr.size != major + 1:
            raise ValueError(
                'indptr has length {}, expected {} (major axis + 1)'
                .format(indptr.size, major + 1))
        # Index dtype must be wide enough for the shape: cuSPARSE will
        # raise/UB if max(shape) exceeds iinfo(idx_dtype).max, and
        # narrower-than-shape dtypes are a footgun even pure-CuPy.
        idx_max = numpy.iinfo(indices.dtype).max
        if shape[0] > idx_max or shape[1] > idx_max:
            raise ValueError(
                'shape {} too large for index dtype {} (max {})'
                .format(shape, indices.dtype, idx_max))
        # CuPy's nnz semantic is data.size, but cuSPARSE descriptors
        # built from this object will pass that nnz to createCsr/Csc.
        # If indptr[-1] disagrees (intentional slack from the caller),
        # cuSPARSE will silently consume the slack and produce wrong
        # arithmetic.  Validate at construction unless the caller can
        # prove the buffer is tight.  Cost: one D2H read of indptr[-1].
        # ``indptr.size >= 1`` is already guaranteed by the
        # ``indptr.size != major + 1`` check above (major >= 0); the
        # ``> 0`` guard is purely defensive.  synchronize!
        if not _skip_buffer_check and indptr.size > 0:
            live_nnz = int(indptr[-1])
            if data.size != live_nnz:
                raise ValueError(
                    'data has length {} but indptr[-1] == {}; '
                    'cuSPARSE would treat trailing buffer entries '
                    'as live data.  Trim with .prune() before '
                    'construction, or pass _skip_buffer_check=True '
                    'if the slack is intentional (e.g. a transient '
                    'build buffer).'
                    .format(data.size, live_nnz))
        A = cls.__new__(cls)
        sparse_data._data_matrix.__init__(A, data)
        A.indices = indices
        A.indptr = indptr
        from cupyx.cusparse import MatDescriptor
        A._descr = MatDescriptor.create()
        A._shape = shape
        if has_canonical_format is not None:
            A._has_canonical_format = has_canonical_format
            if has_canonical_format:
                A._has_sorted_indices = True
        if has_sorted_indices is not None:
            A._has_sorted_indices = has_sorted_indices
        return A

    def _with_data(self, data, copy=True):
        """Return a matrix with the same sparsity structure but
        different data.  Preserves sort/canonical flags.
        """
        # Read private attrs to avoid the property getter, which
        # launches a GPU kernel when the flag has not been computed.
        # _skip_buffer_check: indptr is unchanged from self and
        # data.size matches self.data.size (caller invariant), so the
        # tight-buffer invariant is inherited inductively without a
        # D2H sync per arithmetic op.
        return self.__class__._from_parts(
            data,
            self.indices.copy() if copy else self.indices,
            self.indptr.copy() if copy else self.indptr,
            self.shape,
            has_canonical_format=getattr(
                self, '_has_canonical_format', None),
            has_sorted_indices=getattr(
                self, '_has_sorted_indices', None),
            _skip_buffer_check=True)

    def _empty_like(self, shape):
        """Return an empty matrix with the same index dtype."""
        idx = self.indices.dtype
        major = self._swap(*shape)[0]
        # data.size == 0 == zeros(major+1)[-1], tight by construction.
        return self.__class__._from_parts(
            cupy.empty(0, self.dtype),
            cupy.empty(0, idx),
            cupy.zeros(major + 1, idx),
            shape,
            _skip_buffer_check=True)

    def _convert_dense(self, x):
        raise NotImplementedError

    def _swap(self, x, y):
        raise NotImplementedError

    def _add_sparse(self, other, alpha, beta):
        raise NotImplementedError

    def _add(self, other, lhs_negative, rhs_negative):
        if cupy.isscalar(other):
            if other == 0:
                if lhs_negative:
                    return -self
                else:
                    return self.copy()
            else:
                raise NotImplementedError(
                    'adding a nonzero scalar to a sparse matrix is not '
                    'supported')
        elif _base.issparse(other):
            alpha = -1 if lhs_negative else 1
            beta = -1 if rhs_negative else 1
            return self._add_sparse(other, alpha, beta)
        elif _base.isdense(other):
            if lhs_negative:
                if rhs_negative:
                    return -self.todense() - other
                else:
                    return other - self.todense()
            else:
                if rhs_negative:
                    return self.todense() - other
                else:
                    return self.todense() + other
        else:
            return NotImplemented

    def __add__(self, other):
        return self._add(other, False, False)

    def __radd__(self, other):
        return self._add(other, False, False)

    def __sub__(self, other):
        return self._add(other, False, True)

    def __rsub__(self, other):
        return self._add(other, True, False)

    def _get_intXint(self, row, col):
        major, minor = self._swap(row, col)
        data, indices, _ = _index._get_csr_submatrix_major_axis(
            self.data, self.indices, self.indptr, major, major + 1)
        dtype = data.dtype
        res = cupy.zeros((), dtype=dtype)
        if dtype.kind == 'c':
            _index._compress_getitem_complex_kern(
                data.real, data.imag, indices, minor, res.real, res.imag)
        else:
            _index._compress_getitem_kern(data, indices, minor, res)
        return res

    def _get_sliceXslice(self, row, col):
        major, minor = self._swap(row, col)
        copy = major.step in (1, None)
        return self._major_slice(major)._minor_slice(minor, copy=copy)

    def _get_arrayXarray(self, row, col, not_found_val=0):
        # inner indexing
        idx_dtype = self.indices.dtype
        M, N = self._swap(*self.shape)
        major, minor = self._swap(row, col)
        major = major.astype(idx_dtype, copy=False)
        minor = minor.astype(idx_dtype, copy=False)

        val = _index._csr_sample_values(
            M, N, self.indptr, self.indices, self.data,
            major.ravel(), minor.ravel(),
            not_found_val)

        if major.ndim == 1:
            # Sparse arrays return a 1-D ndarray; sparse matrices keep
            # the legacy 2-D shape (matching scipy).
            if isinstance(self, _base.sparray):
                return val
            return cupy.expand_dims(val, 0)
        return self.__class__(val.reshape(major.shape))

    def _get_columnXarray(self, row, col):
        # outer indexing
        major, minor = self._swap(row, col)
        return self._major_index_fancy(major)._minor_index_fancy(minor)

    def _major_index_fancy(self, idx):
        """Index along the major axis where idx is an array of ints.
        """
        _, N = self._swap(*self.shape)
        M = idx.size
        new_shape = self._swap(M, N)
        if self.nnz == 0 or M == 0:
            return self._empty_like(new_shape)

        return self.__class__._from_parts(
            *_index._csr_row_index(
                self.data, self.indices, self.indptr, idx),
            shape=new_shape)

    _bincount_kernel = r"""
    template<typename I>
    __global__
    void bincount_idx_global(const int  n_idx,
                            const I* __restrict__ idx,
                            int*       __restrict__ col_cnt)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_idx) return;
        atomicAdd(col_cnt + idx[k], 1);
    }
    """

    _bincount_mod = _core.RawModule(
        code=_bincount_kernel,
        options=('-std=c++17',),
        name_expressions=[
            f'bincount_idx_global<{t}>' for t in _idx_types
        ],
    )

    _calc_Bp_kernel = r"""
    template<typename I>
    __global__
    void row_kept_count(const int  n_row,
                        const I* __restrict__ Ap,
                        const I* __restrict__ Aj,
                        const int* __restrict__ col_cnt,
                        I*       __restrict__ Bp)
{
    // 1 block = 1 row
    const int row = blockIdx.x;
    if (row >= n_row) return;

    int local = 0;
    for (I p = Ap[row] + threadIdx.x; p < Ap[row + 1]; p += blockDim.x)
        local += col_cnt[Aj[p]];

    #pragma unroll
    for (int offs = 16; offs; offs >>= 1)
        local += __shfl_down_sync(0xffffffff, local, offs);

    static __shared__ int s[32];              // one per warp
    if ((threadIdx.x & 31) == 0) s[threadIdx.x>>5] = local;
    __syncthreads();

    if (threadIdx.x < 32) {
        int val = (threadIdx.x < (blockDim.x>>5)) ? s[threadIdx.x] : int(0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) Bp[row + 1] = I(val);
    }
}
"""

    _calc_Bp_mod = _core.RawModule(
        code=_calc_Bp_kernel,
        options=('-std=c++17',),
        name_expressions=[
            f'row_kept_count<{t}>' for t in _idx_types
        ],
    )

    _fill_B_kernel = r"""
    template<typename T, typename I> __global__ void
    fill_B(const int  n_row,
                        const I* __restrict__ Ap,
                        const I* __restrict__ Aj,
                        const   T* __restrict__ Ax,
                        const int* __restrict__ col_offset,
                        const int* __restrict__ col_order,
                        const I* __restrict__ Bp,
                        I*       __restrict__ Bj,
                        T*       __restrict__ Bx)
    {
        // 1 block = 1 row
        const int row = blockIdx.x;
        if (row >= n_row) return;

        // Atomic write pointer -- use unsigned long long for atomicAdd
        // (CUDA supports atomicAdd on unsigned long long but not long long)
        __shared__ unsigned long long row_ptr;
        if (threadIdx.x == 0)
            row_ptr = static_cast<unsigned long long>(Bp[row]);
        __syncthreads();

        for (I p = Ap[row] +threadIdx.x; p < Ap[row + 1]; p +=blockDim.x)
        {
            I col     = Aj[p];
            int stop  = col_offset[col];
            int start = (col == 0) ? 0 : col_offset[col - 1];
            int cnt   = stop - start;
            if (cnt == 0) continue;

            T v = Ax[p];
            // unique slice for this thread
            unsigned long long my_out = atomicAdd(
                &row_ptr, static_cast<unsigned long long>(cnt));
            for (int k = 0; k < cnt; ++k)
            {
                Bj[my_out + k] = I(col_order[start + k]);
                Bx[my_out + k] = v;
            }
        }
    }
    """

    _fill_B = _core.RawModule(
        code=_fill_B_kernel,
        options=('-std=c++17',),
        name_expressions=[
            f'fill_B<{dt}, {it}>'
            for dt in ('float', 'double')
            for it in ('int', 'long long')
        ],
    )

    _fill_B_kernel_complex = r"""
    template<typename T, typename I> __global__ void
    fill_B_complex(const int  n_row,
                        const I* __restrict__ Ap,
                        const I* __restrict__ Aj,
                        const   T* __restrict__ Ax,
                        const int* __restrict__ col_offset,
                        const int* __restrict__ col_order,
                        const I* __restrict__ Bp,
                        I*       __restrict__ Bj,
                        T*       __restrict__ Bx)
    {
        // 1 block = 1 row
        const int row = blockIdx.x;
        if (row >= n_row) return;

        __shared__ unsigned long long row_ptr;
        if (threadIdx.x == 0)
            row_ptr = static_cast<unsigned long long>(Bp[row]);
        __syncthreads();

        for (I p = Ap[row] +threadIdx.x; p < Ap[row + 1]; p +=blockDim.x)
        {
            I col     = Aj[p];
            int stop  = col_offset[col];
            int start = (col == 0) ? 0 : col_offset[col - 1];
            int cnt   = stop - start;
            if (cnt == 0) continue;

            T v = Ax[p*2];
            T i = Ax[p*2+1];
            unsigned long long my_out = atomicAdd(
                &row_ptr, static_cast<unsigned long long>(cnt));
            for (int k = 0; k < cnt; ++k)
            {
                Bj[my_out + k] = I(col_order[start + k]);
                Bx[(my_out + k)*2] = v;
                Bx[(my_out + k)*2 + 1] = i;
            }
        }
    }
    """

    _fill_B_complex = _core.RawModule(
        code=_fill_B_kernel_complex,
        options=('-std=c++17',),
        name_expressions=[
            f'fill_B_complex<{dt}, {it}>'
            for dt in ('float', 'double')
            for it in ('int', 'long long')
        ],
    )

    @staticmethod
    def _idx_type_name(dtype):
        if dtype == cupy.int64:
            return 'long long'
        return 'int'

    def _minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """
        M, N = self._swap(*self.shape)
        n_idx = idx.size
        new_shape = self._swap(M, n_idx)
        if self.nnz == 0 or n_idx == 0:
            return self._empty_like(new_shape)

        # V2-13: both branches below allocate ``col_counts`` /
        # ``col_order`` as int32 and rely on cumulative sums fitting
        # in int32.  The fast path is unsafe when either:
        #   * ``N > INT32_MAX`` (the count buffer would be > 8 GB and
        #     the row offsets in it can't fit), or
        #   * ``n_idx > INT32_MAX`` (``col_order = argsort(idx)
        #     .astype(int32)`` truncates and ``cumsum(col_counts)``
        #     overflows since the sum equals ``n_idx``).
        # Either condition forces the O(nnz) sort-based fallback.
        # No direct test for the second arm because it requires an
        # > 8 GB idx array.
        i32max = numpy.iinfo(numpy.int32).max
        if N > i32max or n_idx > i32max:
            return self._minor_index_fancy_sorted(
                idx, M, n_idx, new_shape)

        idx_dtype = self.indices.dtype
        idx_tname = self._idx_type_name(idx_dtype)

        # Create buffers
        col_counts = cupy.zeros(N, dtype=cupy.int32)
        Bp = cupy.empty(M + 1, dtype=idx_dtype)
        Bp[0] = 0

        # Count occurrences of each column
        thread_count = 256

        block_count = (n_idx + thread_count - 1) // thread_count

        bincount_ker = self._bincount_mod.get_function(
            'bincount_idx_global<{}>'.format(idx_tname))
        bincount_ker((block_count,),
                     (thread_count,),
                     (n_idx, idx, col_counts))

        # Compute Bp
        calc_Bp_ker = self._calc_Bp_mod.get_function(
            'row_kept_count<{}>'.format(idx_tname))
        calc_Bp_ker((M,),
                    (thread_count,),
                    (M,
                     self.indptr,
                     self.indices,
                     col_counts,
                     Bp)
                    )

        # Compute col_order and col_offset
        col_order = cupy.argsort(idx).astype(cupy.int32)
        col_offset = cupy.cumsum(col_counts, dtype=cupy.int32)

        # Compute Bp
        Bp[1:] = cupy.cumsum(Bp[1:], dtype=idx_dtype)
        nnzB = int(Bp[-1])  # synchronize!

        Bj = cupy.empty(nnzB, dtype=idx_dtype)
        Bx = cupy.empty(nnzB, dtype=self.data.dtype)

        # Compute Bj and Bx
        if self.dtype.kind == 'c':
            ker_name = 'fill_B_complex<{}, {}>'.format(
                _scalar.get_typename(self.data.real.dtype),
                idx_tname,
            )
            fillB = self._fill_B_complex.get_function(ker_name)
        else:
            ker_name = 'fill_B<{}, {}>'.format(
                _scalar.get_typename(self.data.dtype),
                idx_tname,
            )
            fillB = self._fill_B.get_function(ker_name)
        threads = 32
        fillB((M,),
              (threads,),
              (M,
               self.indptr,
               self.indices,
               self.data,
               col_offset,
               col_order,
               Bp,
               Bj,
               Bx),
              )

        # _skip_buffer_check: ``Bp[-1]`` is the live nnz that the
        # ``fill_B`` kernel just emitted; ``Bx`` and ``Bj`` were sized
        # to that ``nnzB`` above.  Tight by construction.
        return self.__class__._from_parts(
            Bx, Bj, Bp, new_shape, _skip_buffer_check=True)

    def _minor_index_fancy_sorted(self, idx, M, n_idx, new_shape):
        """Sort-based fancy minor-axis indexing for large minor axis.

        O(nnz + n_idx) space instead of O(N). Used when N > INT32_MAX
        where the histogram-based path would require a prohibitive
        O(N) allocation.
        """
        idx_dtype = self.indices.dtype
        idx = cupy.asarray(idx, dtype=idx_dtype)

        sort_order = cupy.argsort(idx)
        sorted_idx = idx[sort_order]

        lo = cupy.searchsorted(sorted_idx, self.indices, side='left')
        hi = cupy.searchsorted(sorted_idx, self.indices, side='right')
        cnt = (hi - lo).astype(cupy.int64)

        total_nnz = int(cnt.sum())  # synchronize!
        if total_nnz == 0:
            return self._empty_like(new_shape)

        out_src = cupy.repeat(
            cupy.arange(self.nnz, dtype=cupy.int64), cnt)
        # offset: position within each repeated group (grouped arange)
        cum_cnt = cupy.zeros(self.nnz + 1, dtype=cupy.int64)
        cupy.cumsum(cnt, out=cum_cnt[1:])
        offset = cupy.arange(total_nnz, dtype=cupy.int64) \
            - cum_cnt[out_src]

        out_minor = sort_order[lo[out_src] + offset]
        out_data = self.data[out_src]

        from cupyx import cusparse as _cusparse_mod
        major_of_each = _cusparse_mod._indptr_to_coo(self.indptr)
        out_major = major_of_each[out_src]

        sort_key = cupy.lexsort(cupy.stack([out_minor, out_major]))
        out_major = out_major[sort_key]
        out_minor = out_minor[sort_key]
        out_data = out_data[sort_key]

        out_idx_dtype = _sputils.get_index_dtype(
            arrays=(self.indices,), maxval=max(M, n_idx))
        out_indptr = _cusparse_mod._build_indptr(
            out_major, M, out_idx_dtype)

        # _skip_buffer_check: ``_build_indptr(out_major, M, ...)``
        # produces ``out_indptr[-1] == out_major.size == out_data.size``
        # (the histogram + prefix-sum recipe).  Tight by construction.
        return self.__class__._from_parts(
            out_data, out_minor.astype(out_idx_dtype),
            out_indptr, new_shape,
            has_canonical_format=True,
            has_sorted_indices=True,
            _skip_buffer_check=True)

    def _major_slice(self, idx, copy=False):
        """Index along the major axis where idx is a slice object.
        """
        M, N = self._swap(*self.shape)
        start, stop, step = idx.indices(M)

        if start == 0 and stop == M and step == 1:
            return self.copy() if copy else self

        M = len(range(start, stop, step))
        new_shape = self._swap(M, N)

        if step == 1:
            if M == 0 or self.nnz == 0:
                return self._empty_like(new_shape)
            data, indices, indptr = \
                _index._get_csr_submatrix_major_axis(
                    self.data, self.indices, self.indptr,
                    start, stop)
            if copy:
                data = data.copy()
                indices = indices.copy()
                # indptr is already a new array (Ap[start:stop+1] -
                # Ap[start]), so no copy needed.
            # _skip_buffer_check: ``_get_csr_submatrix_major_axis``
            # returns tight slices: ``data`` / ``indices`` are sized
            # to ``indptr[-1]`` by construction (sub-indptr derived
            # from the original indptr offset).
            return self.__class__._from_parts(
                data, indices, indptr, new_shape,
                has_canonical_format=getattr(
                    self, '_has_canonical_format', None),
                has_sorted_indices=getattr(
                    self, '_has_sorted_indices', None),
                _skip_buffer_check=True)
        rows = cupy.arange(start, stop, step, dtype=self.indptr.dtype)
        return self._major_index_fancy(rows)

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """
        M, N = self._swap(*self.shape)
        start, stop, step = idx.indices(N)

        if start == 0 and stop == N and step == 1:
            return self.copy() if copy else self

        N = len(range(start, stop, step))
        new_shape = self._swap(M, N)

        if N == 0 or self.nnz == 0:
            return self._empty_like(new_shape)
        cols = cupy.arange(start, stop, step, dtype=self.indices.dtype)
        return self._minor_index_fancy(cols)

    def _set_intXint(self, row, col, x):
        i, j = self._swap(row, col)
        self._set_many(i, j, x)

    def _set_arrayXarray(self, row, col, x):
        i, j = self._swap(row, col)
        self._set_many(i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        # clear entries that will be overwritten
        self._zero_many(*self._swap(row, col))

        M, N = row.shape  # matches col.shape
        broadcast_row = M != 1 and x.shape[0] == 1
        broadcast_col = N != 1 and x.shape[1] == 1
        r, c = x.row, x.col
        x = cupy.asarray(x.data, dtype=self.dtype)
        if broadcast_row:
            r = cupy.repeat(cupy.arange(M), r.size)
            c = cupy.tile(c, M)
            x = cupy.tile(x, M)
        if broadcast_col:
            r = cupy.repeat(r, N)
            c = cupy.tile(cupy.arange(N), c.size)
            x = cupy.repeat(x, N)
        # only assign entries in the new sparsity structure
        i, j = self._swap(row[r, c], col[r, c])
        self._set_many(i, j, x)

    def _prepare_indices(self, i, j):
        M, N = self._swap(*self.shape)

        def check_bounds(indices, bound):
            idx = indices.max()
            if idx >= bound:
                raise IndexError('index (%d) out of range (>= %d)' %
                                 (idx, bound))
            idx = indices.min()
            if idx < -bound:
                raise IndexError('index (%d) out of range (< -%d)' %
                                 (idx, bound))

        i = cupy.array(i, dtype=self.indptr.dtype,
                       copy=True, ndmin=1).ravel()
        j = cupy.array(j, dtype=self.indices.dtype,
                       copy=True, ndmin=1).ravel()
        check_bounds(i, M)
        check_bounds(j, N)
        return i, j, M, N

    def _set_many(self, i, j, x):
        """Sets value at each (i, j) to x
        Here (i,j) index major and minor respectively, and must not contain
        duplicate entries.
        """
        i, j, M, N = self._prepare_indices(i, j)
        x = cupy.array(x, dtype=self.dtype, copy=True, ndmin=1).ravel()

        # Temporary CSR mapping each stored element to its flat offset.
        # Use _from_parts to avoid check_contents D2H syncs; the
        # indices/indptr are already validated (they come from self).
        # Use the indices dtype for the offset array so we get exact
        # integer arithmetic at any nnz (float64 loses precision past
        # 2**53, and -1 is a valid sentinel for any signed integer).
        idx_dtype = self.indices.dtype
        # _skip_buffer_check: ``arange(self.nnz)`` has length self.nnz,
        # which equals self.indices.size and self.indptr[-1] (the
        # tight-buffer invariant inherited from ``self``).
        new_sp = cupyx.scipy.sparse.csr_matrix._from_parts(
            cupy.arange(self.nnz, dtype=idx_dtype),
            self.indices, self.indptr, shape=(M, N),
            _skip_buffer_check=True)

        offsets = new_sp._get_arrayXarray(
            i, j, not_found_val=-1).astype(idx_dtype).ravel()

        mask = offsets > -1
        self.data[offsets[mask]] = x[mask]

        if mask.all():  # synchronize!
            # only affects existing non-zero cells
            return

        # only insertions remain
        warnings.warn('Changing the sparsity structure of a '
                      '{}_matrix is expensive.'.format(self.format),
                      _base.SparseEfficiencyWarning)
        mask = ~mask
        i = i[mask]
        i[i < 0] += M
        j = j[mask]
        j[j < 0] += N
        self._insert_many(i, j, x[mask])

    def _zero_many(self, i, j):
        """Sets value at each (i, j) to zero, preserving sparsity structure.
        Here (i,j) index major and minor respectively.
        """
        i, j, M, N = self._prepare_indices(i, j)

        # Use indices dtype for exact integer offsets (see _set_many).
        # _skip_buffer_check: same arange-length-equals-self.nnz
        # invariant as in _set_many above.
        idx_dtype = self.indices.dtype
        new_sp = cupyx.scipy.sparse.csr_matrix._from_parts(
            cupy.arange(self.nnz, dtype=idx_dtype),
            self.indices, self.indptr, shape=(M, N),
            _skip_buffer_check=True)

        offsets = new_sp._get_arrayXarray(
            i, j, not_found_val=-1).astype(idx_dtype).ravel()

        # only assign zeros to the existing sparsity structure
        self.data[offsets[offsets > -1]] = 0

    def _perform_insert(self, indices_inserts, data_inserts,
                        rows, row_counts, idx_dtype):
        """Insert new elements into current sparse matrix in sorted order"""
        indptr_diff = cupy.diff(self.indptr)
        indptr_diff[rows] += row_counts

        new_indptr = cupy.empty(self.indptr.shape, dtype=idx_dtype)
        new_indptr[0] = idx_dtype(0)
        new_indptr[1:] = indptr_diff

        # Build output arrays
        cupy.cumsum(new_indptr, out=new_indptr)
        out_nnz = int(new_indptr[-1])  # synchronize!

        new_indices = cupy.empty(out_nnz, dtype=idx_dtype)
        new_data = cupy.empty(out_nnz, dtype=self.data.dtype)

        # Build an indexed indptr that contains the offsets for each
        # row but only for in i, j, and x.
        new_indptr_lookup = cupy.zeros(new_indptr.size, dtype=idx_dtype)
        new_indptr_lookup[1:][rows] = row_counts
        cupy.cumsum(new_indptr_lookup, out=new_indptr_lookup)

        _index._insert_many_populate_arrays(
            indices_inserts, data_inserts, new_indptr_lookup,
            self.indptr, self.indices, self.data, new_indptr, new_indices,
            new_data, size=self.indptr.size-1)

        self.indptr = new_indptr
        self.indices = new_indices
        self.data = new_data

    def _insert_many(self, i, j, x):
        """Inserts new nonzero at each (i, j) with value x
        Here (i,j) index major and minor respectively.
        i, j and x must be non-empty, 1d arrays.
        Inserts each major group (e.g. all entries per row) at a time.
        Maintains has_sorted_indices property.
        Modifies i, j, x in place.
        """

        order = cupy.argsort(i)  # stable for duplicates
        i = i.take(order)
        j = j.take(order)
        x = x.take(order)

        # Update index data type

        idx_dtype = _sputils.get_index_dtype(
            (self.indices, self.indptr), maxval=(
                self.nnz + x.size))

        self.indptr = self.indptr.astype(idx_dtype)
        self.indices = self.indices.astype(idx_dtype)
        self.data = self.data.astype(self.dtype)

        indptr_inserts, indices_inserts, data_inserts = \
            _index._select_last_indices(i, j, x, idx_dtype)

        rows, ui_indptr = cupy.unique(indptr_inserts, return_index=True)

        to_add = cupy.empty(ui_indptr.size+1, ui_indptr.dtype)
        to_add[-1] = j.size
        to_add[:-1] = ui_indptr
        ui_indptr = to_add

        # Compute the counts for each row in the insertion array.
        row_counts = cupy.zeros(ui_indptr.size-1, dtype=idx_dtype)
        search_idx = cupy.searchsorted(rows, indptr_inserts)
        cupy.add.at(row_counts, search_idx, 1)

        self._perform_insert(indices_inserts, data_inserts,
                             rows, row_counts, idx_dtype)

    def __get_has_canonical_format(self):
        """Determine whether the matrix has sorted indices and no duplicates.

        Returns
            bool: ``True`` if the above applies, otherwise ``False``.

        .. note::
            :attr:`has_canonical_format` implies :attr:`has_sorted_indices`, so
            if the latter flag is ``False``, so will the former be; if the
            former is found ``True``, the latter flag is also set.

        .. warning::
            Getting this property might synchronize the device.

        """
        # Modified from the SciPy counterpart.

        # In CuPy the implemented conversions do not exactly match those of
        # SciPy's, so it's hard to put this exactly as where it is in SciPy,
        # but this should do the job.
        if self.data.size == 0:
            self._has_canonical_format = True
            # Empty -> trivially sorted too; mirror the setter so a
            # subsequent ``has_sorted_indices`` read avoids a kernel.
            self._has_sorted_indices = True
        # check to see if result was cached
        elif not getattr(self, '_has_sorted_indices', True):
            # not sorted => not canonical
            self._has_canonical_format = False
        elif not hasattr(self, '_has_canonical_format'):
            is_canonical = self._has_canonical_format_kern(
                self.indptr, self.indices, size=self.indptr.size-1)
            self._has_canonical_format = bool(
                is_canonical.all())  # synchronize!
        # canonical => sorted; propagate so a later
        # has_sorted_indices read avoids re-running the kernel.
        if self._has_canonical_format:
            self._has_sorted_indices = True
        return self._has_canonical_format

    def __set_has_canonical_format(self, val):
        """Taken from SciPy as is."""
        self._has_canonical_format = bool(val)
        if val:
            self.has_sorted_indices = True

    has_canonical_format = property(fget=__get_has_canonical_format,
                                    fset=__set_has_canonical_format)

    def __get_sorted(self):
        """Determine whether the matrix has sorted indices.

        Returns
            bool:
                ``True`` if the indices of the matrix are in sorted order,
                otherwise ``False``.

        .. warning::
            Getting this property might synchronize the device.

        """
        # Modified from the SciPy counterpart.

        # In CuPy the implemented conversions do not exactly match those of
        # SciPy's, so it's hard to put this exactly as where it is in SciPy,
        # but this should do the job.
        if self.data.size == 0:
            self._has_sorted_indices = True
        # check to see if result was cached
        elif not hasattr(self, '_has_sorted_indices'):
            is_sorted = self._has_sorted_indices_kern(
                self.indptr, self.indices, size=self.indptr.size-1)
            self._has_sorted_indices = bool(is_sorted.all())  # synchronize!
        return self._has_sorted_indices

    def __set_sorted(self, val):
        self._has_sorted_indices = bool(val)

    has_sorted_indices = property(fget=__get_sorted, fset=__set_sorted)

    def _getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Args:
            axis ({None, 0, 1, -1, -2}): Axis along which to count.
                ``None`` (default) returns the total stored count
                (``data.size``).  ``0`` / ``1`` return per-column /
                per-row stored counts; negatives are interpreted
                ``axis + 2``.  Mirrors :meth:`scipy.sparse._spbase.getnnz`.

        Returns:
            int or cupy.ndarray: Scalar when ``axis is None``,
            otherwise a 1-D cupy ``intp`` array of length
            ``shape[1 - axis]``.

        Notes:
            Counts every entry in the buffer (including explicit zeros);
            use :meth:`count_nonzero` to skip explicit zeros.
        """
        if axis is None:
            return self.data.size
        if axis < 0:
            axis += 2
        if axis < 0 or axis >= 2:
            raise ValueError('axis out of bounds')
        # ``major_axis`` is the axis along which the indptr partitions:
        # rows for CSR (_swap is identity), cols for CSC (_swap swaps).
        major_axis, _ = self._swap(axis, 1 - axis)
        major_dim, minor_dim = self._swap(*self.shape)
        if major_axis == 1:
            # axis-along-minor: per-major-index count == diff(indptr).
            return cupy.diff(self.indptr).astype(cupy.intp)
        # axis-along-major: per-minor-index count from indices.
        if self.indices.size == 0:
            return cupy.zeros(minor_dim, dtype=cupy.intp)
        return cupy.bincount(
            self.indices.astype(cupy.int64),
            minlength=minor_dim).astype(cupy.intp, copy=False)

    def count_nonzero(self, axis=None):
        """Number of non-zero entries.

        Unlike :attr:`nnz`, this counts only true non-zero values
        (explicit zeros are excluded).  Duplicates are summed first.

        Args:
            axis ({-2, -1, 0, 1, ``None``}):
                Count over the whole matrix, or along a logical
                row/col axis.

        Returns:
            int or cupy.ndarray: Scalar count when ``axis=None``,
            otherwise a 1-D ``cupy.ndarray`` of length
            ``shape[1 - axis]``.

        .. seealso:: :meth:`scipy.sparse.csr_matrix.count_nonzero`
        """
        # Match scipy: dedup in place, then count.  ``self.indices``
        # and ``self.indptr`` then directly give per-axis counts via
        # bincount / np.diff with no COO round-trip.
        self.sum_duplicates()
        if axis is None:
            return int(cupy.count_nonzero(self.data))
        if axis < -2 or axis > 1:
            raise ValueError('axis out of bounds')
        if axis < 0:
            axis += 2
        # Translate (row, col) → (major, minor).  CSR's _swap is
        # identity; CSC's swaps.
        major_axis, _ = self._swap(axis, 1 - axis)
        major_dim, minor_dim = self._swap(*self.shape)
        out_dim = minor_dim if major_axis == 0 else major_dim
        # Empty short-circuit -- ``cupy.bincount`` raises
        # ``zero-size array to reduction`` when the input is empty,
        # even with ``minlength`` set, so bypass that path for the
        # trivial cases.
        if self.data.size == 0:
            return cupy.zeros(out_dim, dtype=cupy.intp)
        mask = self.data != 0
        # V2-5: read the popcount of ``mask`` once.  ``mask.all()``
        # and ``mask.any()`` are independent reductions; running both
        # forced two D2H syncs in the partially-zero case.  A single
        # ``int(mask.sum())`` answers both questions: ``all`` ↔
        # ``popcount == size``, ``any`` ↔ ``popcount > 0``.
        nnz_kept = int(mask.sum())  # synchronize!
        if nnz_kept == 0:
            return cupy.zeros(out_dim, dtype=cupy.intp)
        all_kept = nnz_kept == mask.size
        if major_axis == 0:
            # axis-along-major: per-minor-index count → bincount(indices)
            idx = self.indices if all_kept else self.indices[mask]
            return cupy.bincount(
                idx.astype(cupy.int64), minlength=minor_dim)
        # axis-along-minor: per-major-index count
        if all_kept:
            return cupy.diff(self.indptr).astype(cupy.intp)
        from cupyx.cusparse import _indptr_to_coo
        major = _indptr_to_coo(self.indptr)
        return cupy.bincount(
            major[mask].astype(cupy.int64), minlength=major_dim)

    def sorted_indices(self):
        """Return a copy of this matrix with sorted indices

        .. warning::
            Calling this function might synchronize the device.
        """
        # Taken from SciPy as is.
        A = self.copy()
        A.sort_indices()
        return A

    def check_format(self, full_check=True):
        """Validate the CSR/CSC internal invariants (F15).

        Args:
            full_check (bool): If ``True`` (default), scan ``indices``
                and ``indptr`` for out-of-bounds entries and
                non-monotonic ``indptr``.  If ``False``, only the
                cheap structural checks (dtypes, lengths, ndims) are
                performed.  ``full_check=True`` issues GPU reductions
                and synchronizes (~5 D2H syncs).

        Mirrors :meth:`scipy.sparse.csr_matrix.check_format`.

        Raises:
            ValueError: If the matrix is structurally inconsistent.
        """
        # Cheap structural checks first.
        if self.indptr.dtype.kind != 'i':
            warnings.warn(
                f'indptr array has non-integer dtype '
                f'({self.indptr.dtype.name})',
                stacklevel=3)
        if self.indices.dtype.kind != 'i':
            warnings.warn(
                f'indices array has non-integer dtype '
                f'({self.indices.dtype.name})',
                stacklevel=3)
        if (self.data.ndim != 1 or self.indices.ndim != 1
                or self.indptr.ndim != 1):
            raise ValueError(
                'data, indices, and indptr should be 1-D')
        major_dim, minor_dim = self._swap(*self.shape)
        if len(self.indptr) != major_dim + 1:
            raise ValueError(
                f'index pointer size {len(self.indptr)} should be '
                f'{major_dim + 1}')
        if int(self.indptr[0]) != 0:  # synchronize!
            raise ValueError('index pointer should start with 0')
        if len(self.indices) != len(self.data):
            raise ValueError(
                'indices and data should have the same size')
        live = int(self.indptr[-1])  # synchronize!
        if live > len(self.indices):
            raise ValueError(
                'Last value of index pointer should be less than '
                'the size of index and data arrays')
        if not full_check:
            return
        if live > 0:
            # Slice to live nnz so a slack-buffered ``_from_parts``
            # build doesn't surface garbage indices.
            indices = (self.indices[:live] if live < self.indices.size
                       else self.indices)
            if int(indices.max()) >= minor_dim:  # synchronize!
                raise ValueError(f'indices must be < {minor_dim}')
            if int(indices.min()) < 0:  # synchronize!
                raise ValueError('indices must be >= 0')
        if int(cupy.diff(self.indptr).min()) < 0:  # synchronize!
            raise ValueError(
                'indptr must be a non-decreasing sequence')

    def resize(self, *shape):
        """Resize the array/matrix in-place to the given shape (F15).

        Entries that fall outside the new shape are dropped; new
        entries are not added (the result is the truncation of the
        original).  ``shape`` may be passed as ``(m, n)`` or ``m, n``.

        Note: ``has_canonical_format`` and ``has_sorted_indices`` are
        preserved across resize.  Resize is a structural truncation
        / pad operation: it does not reorder entries (sortedness
        preserved) and does not introduce duplicates (canonical
        preserved if it was set, non-canonical preserved if it was
        not).  Verified empirically against scipy 1.17.

        .. seealso:: :meth:`scipy.sparse.csr_matrix.resize`
        """
        from cupyx import cusparse

        new_shape = _sputils.check_shape(shape)
        new_M, new_N = self._swap(*new_shape)
        old_M, old_N = self._swap(*self.shape)

        if new_M < old_M:
            # Drop trailing major-axis entries.  ``int(indptr[new_M])``
            # is the live count after truncation -- synchronize!
            keep = int(self.indptr[new_M])
            self.indices = self.indices[:keep].copy()
            self.data = self.data[:keep].copy()
            self.indptr = self.indptr[:new_M + 1].copy()
        elif new_M > old_M:
            # Pad indptr with the current ``indptr[-1]``: every new
            # major-axis row/col is empty.
            extra = cupy.full(
                new_M - old_M, self.indptr[-1],
                dtype=self.indptr.dtype)
            self.indptr = cupy.concatenate([self.indptr, extra])

        if new_N < old_N:
            # Drop entries with minor index >= new_N.
            mask = self.indices < new_N
            if not bool(mask.all()):  # synchronize!
                # Rebuild indptr from surviving entries.
                major = cusparse._indptr_to_coo(
                    self.indptr, self.indptr.dtype)
                kept_major = major[mask]
                self.indices = self.indices[mask].copy()
                self.data = self.data[mask].copy()
                self.indptr = cusparse._build_indptr(
                    kept_major, new_M, self.indptr.dtype)

        self._shape = (new_shape[0], new_shape[1])
        # NB: ``has_canonical_format`` / ``has_sorted_indices`` are
        # NOT invalidated -- resize never reorders entries and never
        # introduces duplicates, so canonical-ness and sortedness are
        # preserved.  Verified against scipy 1.17.

    def prune(self):
        """Remove empty space after all non-zero elements.

        After certain operations the ``indices`` and ``data`` buffers
        may be longer than the logical entry count (``indptr[-1]``).
        ``prune`` shrinks them so the buffers exactly match.  CuPy's
        own helpers don't produce oversized buffers, so this is mostly
        useful when arrays were assembled directly via ``_from_parts``.

        .. seealso:: :meth:`scipy.sparse.csr_matrix.prune`
        """
        major_dim = self._swap(*self.shape)[0]
        if len(self.indptr) != major_dim + 1:
            raise ValueError('index pointer has invalid length')
        # indptr[-1] is the logical entry count; data/indices may be
        # longer than that if the caller built them with slack.
        nnz = int(self.indptr[-1])  # synchronize!
        if len(self.indices) < nnz:
            raise ValueError('indices array has fewer than nnz elements')
        if len(self.data) < nnz:
            raise ValueError('data array has fewer than nnz elements')
        if len(self.indices) > nnz:
            self.indices = self.indices[:nnz].copy()
        if len(self.data) > nnz:
            self.data = self.data[:nnz].copy()

    def sort_indices(self):
        # Unlike in SciPy, here this is implemented in child classes because
        # each child needs to call its own sort function from cuSPARSE
        raise NotImplementedError

    def sum_duplicates(self):
        """Eliminate duplicate matrix entries by adding them together.

        .. note::
            This is an *in place* operation.

        .. warning::
            Calling this function might synchronize the device.

        .. seealso::
           :meth:`scipy.sparse.csr_matrix.sum_duplicates`,
           :meth:`scipy.sparse.csc_matrix.sum_duplicates`
        """
        if self.has_canonical_format:
            return
        # TODO(leofang): add a kernel for compressed sparse matrices without
        # converting to coo
        coo = self.tocoo()
        coo.sum_duplicates()
        self.__init__(coo.asformat(self.format))
        self.has_canonical_format = True

    #####################
    # Reduce operations #
    #####################

    def _minor_reduce(self, ufunc, axis, nonzero):
        """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.
        Warning: this does not call sum_duplicates()

        Args:
            ufunc (object): Function handle giving the operation to be
                conducted.
            axis (int): Matrix over which the reduction should be
                conducted.

        Returns:
            (cupy.ndarray): Reduce result for nonzeros in each
            major_index.

        """
        out_shape = self.shape[1 - axis]
        # TI (index type) must match self.indptr.dtype for correct
        # int64 indptr.
        idx_dtype = self.indptr.dtype
        tname = _scalar.get_typename(idx_dtype)
        out = cupy.zeros(out_shape).astype(cupy.float64)
        mod, fname = {
            (cupy.amax, False): (self._max_reduction_mod, 'max_reduction'),
            (cupy.amin, False): (self._min_reduction_mod, 'min_reduction'),
            (cupy.amax, True): (self._max_nonzero_reduction_mod,
                                'max_nonzero_reduction'),
            (cupy.amin, True): (self._min_nonzero_reduction_mod,
                                'min_nonzero_reduction'),
        }[(ufunc, nonzero)]
        ker = mod.get_function('{}<{}>'.format(fname, tname))
        ker((out_shape,), (1,),
            (self.data.astype(cupy.float64),
             self.indptr[:-1], self.indptr[1:],
             idx_dtype.type(self.shape[axis]),
             out))

        return out

    def _arg_minor_reduce(self, ufunc, axis):
        """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.
        Warning: this does not call sum_duplicates()

        Args:
            ufunc (object): Function handle giving the operation to be
                conducted.
            axis (int): Maxtrix over which the reduction should be conducted

        Returns:
            (cupy.ndarray): Reduce result for nonzeros in each
            major_index

        """

        # Call to the appropriate kernel function
        # Create the vector to hold output
        # Note: it's important to set "int" here, following what SciPy
        # does, as the outcome dtype is platform dependent
        out_shape = self.shape[1 - axis]
        out = cupy.zeros(out_shape, dtype=int)

        # Perform the calculation
        # TI (index type) must match self.indices.dtype for correct int64
        # column values. indptr slices are cast to match indices dtype.
        idx_dtype = self.indices.dtype
        indptr_x = self.indptr[:len(self.indptr) - 1].astype(idx_dtype,
                                                             copy=False)
        indptr_y = self.indptr[1:].astype(idx_dtype, copy=False)
        ker_name = '_arg_reduction<{}, {}, {}>'.format(
            _scalar.get_typename(self.data.dtype),
            _scalar.get_typename(out.dtype),
            _scalar.get_typename(idx_dtype))

        if ufunc == cupy.argmax:
            ker = self._max_arg_reduction_mod.get_function('max' + ker_name)
        elif ufunc == cupy.argmin:
            ker = self._min_arg_reduction_mod.get_function('min' + ker_name)

        ker((out_shape,), (1,),
            (self.data, self.indices,
             indptr_x, indptr_y,
             idx_dtype.type(self.shape[axis]),
             out))

        return out
