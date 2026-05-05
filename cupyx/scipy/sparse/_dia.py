from __future__ import annotations

try:
    import scipy.sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False

import numpy

import cupy
from cupy import _core
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _data
from cupyx.scipy.sparse import _sputils
from cupyx.scipy.sparse import _util


# TODO(leofang): The current implementation is CSC-based, which is troublesome
# on ROCm/HIP. We should convert it to CSR-based for portability.
class _dia_base(_data._data_matrix):
    """DIA format base (shared by dia_matrix and dia_array).

    Sparse matrix with DIAgonal storage.

    Now it has only one initializer format below:

    ``dia_matrix((data, offsets))``

    Args:
        arg1: Arguments for the initializer.
        shape (tuple): Shape of a matrix. Its length must be two.
        dtype: Data type. It must be an argument of :class:`numpy.dtype`.
        copy (bool): If ``True``, copies of given arrays are always used.

    .. seealso::
       :class:`scipy.sparse.dia_matrix`

    """

    format = 'dia'

    def __init__(self, arg1, shape=None, dtype=None, copy=False,
                 *, maxprint=None):
        if maxprint is not None:
            self.maxprint = maxprint
        if shape is not None and not _util.isshape(shape, nonneg=True):
            raise ValueError(
                'invalid shape (must be a 2-tuple of non-negative int)')
        if _scipy_available and scipy.sparse.issparse(arg1):
            x = arg1.todia()
            data = x.data
            offsets = x.offsets
            shape = x.shape
            # Preserve the caller-supplied dtype so that
            # ``dia_array(scipy_obj, dtype=cupy.float32)`` actually
            # casts (the prior code unconditionally rebound dtype to
            # the scipy source's dtype, silently dropping the kwarg).
            if dtype is None:
                dtype = x.dtype
            copy = False
        elif _base.issparse(arg1):
            # CuPy sparse object (matrix or array, any format).
            # Round-trip through ``todia()`` so we extract data /
            # offsets in the canonical layout regardless of the input
            # format.  Preserves dtype unless the user asked for a
            # specific one.
            x = arg1.todia()
            data = x.data
            offsets = x.offsets
            shape = x.shape
            if dtype is None:
                dtype = x.dtype
            copy = False
        elif (isinstance(arg1, tuple) and len(arg1) == 2
                and _util.isshape(arg1)):
            # See _compressed.py: dispatch on shape-likeness regardless
            # of sign so negatives raise a precise message instead of
            # falling through to the (data, offsets) branch.
            if not _util.isshape(arg1, nonneg=True):
                raise ValueError(
                    'invalid shape (must be a 2-tuple of non-negative '
                    'int)')
            # ``dia_array((M, N))`` -- empty DIA with the given shape.
            shape = arg1
            data = cupy.zeros((0, 0), dtype=dtype or cupy.float64)
            offsets = cupy.zeros(0, dtype=cupy.intp)
        elif isinstance(arg1, tuple):
            data, offsets = arg1
            if shape is None:
                raise ValueError('expected a shape argument')
        elif _base.isdense(arg1):
            # 2-D cupy ndarray -- harvest the non-zero diagonals.
            # NumPy ndarrays are *not* accepted (per CuPy convention,
            # no implicit numpy<->cupy conversion for sparse construction;
            # callers must ``cupy.asarray`` first).
            # SciPy DIA layout: ``data[i, j]`` is the value at
            # ``(j - offsets[i], j)`` and ``data.shape[1] == col.max() + 1``
            # (only as wide as the rightmost stored entry, mirrors
            # scipy's COO -> DIA recipe).
            arr = cupy.asarray(arg1)
            if arr.ndim != 2:
                raise ValueError(
                    'dia_matrix from dense array requires 2-D input, '
                    f'got {arr.ndim}-D')
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            shape = arr.shape
            rows, cols = cupy.nonzero(arr)
            if rows.size:
                # Each non-zero (r, c) sits on diagonal k = c - r.
                # int64 cast: ``c - r`` could overflow int32 for shapes
                # near 2**31.
                diags = (cols.astype(cupy.int64)
                         - rows.astype(cupy.int64))
                unique_diags, diag_idx = cupy.unique(
                    diags, return_inverse=True)
                n_diags = int(unique_diags.size)
                width = int(cols.max()) + 1  # synchronize!
                data = cupy.zeros((n_diags, width), dtype=arr.dtype)
                data[diag_idx.astype(cupy.intp),
                     cols.astype(cupy.intp)] = arr[rows, cols]
                offsets = unique_diags
            else:
                data = cupy.zeros((0, 0), dtype=arr.dtype)
                offsets = cupy.zeros(0, dtype=cupy.intp)
            copy = False
        else:
            raise ValueError(
                'unrecognized form for dia_matrix constructor')

        # ``cupy.array(..., dtype=dtype, copy=False)`` raises when
        # ``dtype`` requires a cast (CuPy disallows dtype-changing
        # zero-copy creation).  Force ``copy=True`` for the cast to
        # match scipy's "always honor the dtype kwarg" behavior.
        if dtype is not None and cupy.dtype(dtype) != data.dtype:
            data = cupy.asarray(data, dtype=dtype)
        else:
            data = cupy.array(data, dtype=dtype, copy=copy)
        data = cupy.atleast_2d(data)
        off_dtype = _sputils.get_index_dtype(maxval=max(shape))
        offsets = cupy.array(offsets, dtype=off_dtype)
        offsets = cupy.atleast_1d(offsets)

        if offsets.ndim != 1:
            raise ValueError('offsets array must have rank 1')

        if data.ndim != 2:
            raise ValueError('data array must have rank 2')

        if data.shape[0] != len(offsets):
            raise ValueError(
                'number of diagonals (%d) does not match the number of '
                'offsets (%d)'
                % (data.shape[0], len(offsets)))

        sorted_offsets = cupy.sort(offsets)
        if (sorted_offsets[:-1] == sorted_offsets[1:]).any():  # synchronize!
            raise ValueError('offset array contains duplicate values')

        self.data = data
        self.offsets = offsets
        if not _util.isshape(shape, nonneg=True):
            raise ValueError(
                'invalid shape (must be a 2-tuple of non-negative int)')
        self._shape = int(shape[0]), int(shape[1])

    def _with_data(self, data, copy=True):
        """Return a sparse object with the same sparsity structure as self,
        but with different data.  By default the structure arrays are copied.

        Preserves the concrete leaf type (``dia_array`` vs ``dia_matrix``).
        """
        if copy:
            return type(self)(
                (data, self.offsets.copy()), shape=self.shape)
        else:
            return type(self)((data, self.offsets), shape=self.shape)

    def __add__(self, other):
        """DIA + DIA preserves DIA format (matches scipy).

        Other operand types (scalar, dense, non-DIA sparse) fall back
        to the base CSR-routed path, which densifies / converts as
        before.  The DIA-DIA fast path mirrors
        :meth:`scipy.sparse._dia._dia_base._add_sparse`.
        """
        if _util.isscalarlike(other):
            if other == 0:
                # ``0`` is the additive identity for sparse arithmetic
                # -- preserve format and value.
                return self.copy()
            # Adding a non-zero scalar densifies; route through CSR
            # for the existing fallback to handle it.
            return self.tocsr() + other
        if _base.issparse(other) and other.format == 'dia':
            if other.shape != self.shape:
                raise ValueError('inconsistent shapes')
            return self._dia_add_sparse(other, sub=False)
        return self.tocsr() + other

    def __sub__(self, other):
        """DIA - DIA preserves DIA format.  See :meth:`__add__`."""
        if _util.isscalarlike(other):
            if other == 0:
                return self.copy()
            return self.tocsr() - other
        if _base.issparse(other) and other.format == 'dia':
            if other.shape != self.shape:
                raise ValueError('inconsistent shapes')
            return self._dia_add_sparse(other, sub=True)
        return self.tocsr() - other

    def _dia_add_sparse(self, other, sub):
        """Add or subtract two DIA matrices, preserving DIA format.

        Mirrors :meth:`scipy.sparse._dia._dia_base._add_sparse`.  Same
        offsets fast-path skips the offset-union scan; otherwise the
        result holds the union of offsets.
        """
        # Fast path: identical offsets (most common when ``A + A``).
        # synchronize!
        if (self.offsets.size == other.offsets.size
                and bool(cupy.array_equal(
                    self.offsets, other.offsets))):
            new_data = (self.data - other.data
                        if sub else self.data + other.data)
            return self._with_data(new_data)

        # General path: union of offsets, project each operand's
        # diagonals into the unified buffer, accumulate.
        new_offsets = cupy.union1d(self.offsets, other.offsets)
        self_idx = cupy.searchsorted(new_offsets, self.offsets)
        other_idx = cupy.searchsorted(new_offsets, other.offsets)

        # Result diag length: needs to hold the largest diagonal that
        # fits in the matrix (max(M+offset, N) trimmed to N).
        last_offset = int(new_offsets[-1])  # synchronize!
        d = min(self.shape[0] + last_offset, self.shape[1])
        if d <= 0:
            d = 0

        new_dtype = numpy.result_type(self.data.dtype, other.data.dtype)
        new_data = cupy.zeros(
            (len(new_offsets), d), dtype=new_dtype)

        self_d = min(self.data.shape[1], d)
        other_d = min(other.data.shape[1], d)
        new_data[self_idx, :self_d] = self.data[:, :self_d]
        if sub:
            new_data[other_idx, :other_d] -= other.data[:, :other_d]
        else:
            new_data[other_idx, :other_d] += other.data[:, :other_d]

        return self._dia_container(
            (new_data, new_offsets), shape=self.shape)

    def __repr__(self):
        # Match scipy's DIA repr which annotates the diagonal count.
        format_name = _base._format_names.get(self.format, self.format)
        sparse_cls = (
            'array' if isinstance(self, _base.sparray) else 'matrix')
        return (
            f"<{format_name} sparse {sparse_cls} of dtype '{self.dtype}'\n"
            f"\twith {self.nnz} stored elements ({len(self.offsets)} "
            f"diagonals) and shape {self.shape}>")

    def get(self, stream=None):
        """Returns a copy of the array on host memory.

        Args:
            stream (cupy.cuda.Stream): CUDA stream object. If it is given, the
                copy runs asynchronously. Otherwise, the copy is synchronous.

        Returns:
            scipy.sparse.dia_matrix: Copy of the array on host memory.

        """
        if not _scipy_available:
            raise RuntimeError('scipy is not available')
        data = self.data.get(stream)
        offsets = self.offsets.get(stream)
        if isinstance(self, _base.sparray):
            sp_cls = scipy.sparse.dia_array
        else:
            sp_cls = scipy.sparse.dia_matrix
        return sp_cls((data, offsets), shape=self._shape)

    def _getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Args:
            axis: Not supported.

        Returns:
            int: Number of stored values.
        """
        if axis is not None:
            raise NotImplementedError(
                'getnnz over an axis is not implemented for DIA format')

        m, n = self.shape
        # Bound by the actual data buffer length so an "empty" DIA
        # (data.shape[1] == 0 with non-empty offsets) reports 0,
        # matching scipy 1.17 (gh-23055).
        L = min(self.data.shape[1], n)
        it = self.offsets.dtype.type
        # Use int64 accumulator: per-diagonal counts fit int32, but the
        # sum across (m + n - 1) diagonals can exceed INT32_MAX even when
        # the offsets dtype is int32 (e.g., a dense 2**15 x 2**15 matrix).
        nnz = _core.ReductionKernel(
            'I offsets, I m, I L', 'int64 nnz',
            'max(min(m + offsets, L) - max(offsets, (I)0), (I)0)',
            'a + b', 'nnz = a', '0', 'dia_nnz')(
                self.offsets, it(m), it(L))
        return int(nnz)

    def _data_mask(self):
        """Boolean mask the same shape as ``self.data`` indicating
        which entries fall inside the matrix.

        ``mask[i, j]`` is True iff position ``(j - offsets[i], j)``
        lies within ``self.shape``.  Used to filter ``data`` for
        operations that should ignore the buffer's pad columns
        (``count_nonzero``, dense expansion, etc.).

        Mirrors :meth:`scipy.sparse._dia._dia_base._data_mask`.
        """
        num_rows, num_cols = self.shape
        offset_inds = cupy.arange(self.data.shape[1])
        row = offset_inds - self.offsets[:, None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        return mask

    def count_nonzero(self, axis=None):
        """Number of non-zero entries.

        Excludes explicit zeros and entries that lie outside the
        matrix shape (DIA's ``data`` buffer can be wider than
        ``num_cols``; those pad entries don't count).  DIA doesn't
        support per-axis counts — use ``tocsr().count_nonzero(axis)``.

        .. seealso:: :meth:`scipy.sparse.dia_matrix.count_nonzero`
        """
        if axis is not None:
            raise NotImplementedError(
                'count_nonzero over an axis is not implemented for '
                'DIA format')
        mask = self._data_mask()
        return int(cupy.count_nonzero(self.data[mask]))

    def toarray(self, order=None, out=None):
        """Returns a dense matrix representing the same value."""
        return self.tocsc().toarray(order=order, out=out)

    def todia(self, copy=False):
        """Return this object unchanged (already in DIA format).

        The base ``_spbase.todia`` would round-trip via CSR, which is
        unnecessary work and currently raises ``NotImplementedError``
        because :meth:`csr_matrix.todia` is unimplemented.

        Args:
            copy (bool): If ``True``, return a copy.
        """
        if copy:
            return self.copy()
        return self

    def tocsc(self, copy=False):
        """Converts the matrix to Compressed Sparse Column format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in dia to csc conversion.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        if self.data.size == 0:
            return self._csc_container(self.shape, dtype=self.dtype)

        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        idx_dtype = _sputils.get_index_dtype(maxval=max(self.shape))

        it = idx_dtype
        row, mask = _core.ElementwiseKernel(
            'I offset_len, I offsets, I num_rows, '
            'I num_cols, T data',
            'I row, bool mask',
            '''
            I offset_inds = (I)(i % offset_len);
            row = offset_inds - offsets;
            mask = (row >= 0 && row < num_rows
                    && offset_inds < num_cols
                    && data != T(0));
            ''',
            'cupyx_scipy_sparse_dia_tocsc')(
                it(offset_len),
                self.offsets[:, None].astype(idx_dtype, copy=False),
                it(num_rows), it(num_cols), self.data)
        indptr = cupy.zeros(num_cols + 1, dtype=idx_dtype)
        # Each column of ``data`` contributes one count to the matching
        # output column.  When ``offset_len`` exceeds ``num_cols`` (data
        # buffer wider than the matrix), the extra trailing columns lie
        # outside the matrix and their mask entries are all False, so
        # truncate to ``num_cols`` for the indptr write.
        col_counts = mask.sum(axis=0)
        eff_len = min(offset_len, num_cols)
        indptr[1: eff_len + 1] = cupy.cumsum(col_counts[:eff_len])
        indptr[eff_len + 1:] = indptr[eff_len]
        indices = row.T[mask.T].astype(idx_dtype, copy=False)
        data = self.data.T[mask.T]
        return self._csc_container(
            (data, indices, indptr), shape=self.shape, dtype=self.dtype)

    def tocsr(self, copy=False):
        """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in dia to csr conversion.

        Returns:
            cupyx.scipy.sparse.csc_matrix: Converted matrix.

        """
        return self.tocsc().tocsr()

    def diagonal(self, k=0):
        """Returns the k-th diagonal of the matrix.

        Args:
            k (int, optional): Which diagonal to get, corresponding to elements
            a[i, i+k]. Default: 0 (the main diagonal).

        Returns:
            cupy.ndarray : The k-th diagonal.
        """
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return cupy.empty(0, dtype=self.data.dtype)
        idx, = cupy.nonzero(self.offsets == k)
        first_col, last_col = max(0, k), min(rows + k, cols)
        if idx.size == 0:
            return cupy.zeros(last_col - first_col, dtype=self.data.dtype)
        return self.data[idx[0], first_col:last_col]


class dia_matrix(_base.spmatrix, _dia_base):
    """Sparse matrix with DIAgonal storage.

    .. seealso:: :class:`scipy.sparse.dia_matrix`
    """
    pass


class dia_array(_dia_base, _base.sparray):
    """Sparse array with DIAgonal storage.

    .. seealso:: :class:`scipy.sparse.dia_array`
    """
    pass


def isspmatrix_dia(x):
    """Checks if a given matrix is of DIA format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.dia_matrix`.

    """
    return isinstance(x, dia_matrix)
