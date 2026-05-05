"""Tests for sparse array classes (csr_array, csc_array, coo_array, dia_array).
"""
from __future__ import annotations

import warnings

import numpy
import pytest
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupyx.scipy import sparse


def _make_csr(xp, sp, dtype, *, array=False):
    """3x4 CSR with 4 nonzeros."""
    data = xp.array([0, 1, 2, 3], dtype)
    indices = xp.array([0, 1, 3, 2], 'i')
    indptr = xp.array([0, 2, 3, 4], 'i')
    cls = sp.csr_array if array else sp.csr_matrix
    return cls((data, indices, indptr), shape=(3, 4))


def _make_csr_sq(xp, sp, dtype, *, array=False):
    """3x3 square CSR."""
    data = xp.array([1, 2, 3], dtype)
    indices = xp.array([0, 1, 2], 'i')
    indptr = xp.array([0, 1, 2, 3], 'i')
    cls = sp.csr_array if array else sp.csr_matrix
    return cls((data, indices, indptr), shape=(3, 3))


def _make_csr_sq2(xp, sp, dtype, *, array=False):
    """Another 3x3 square CSR (different values)."""
    data = xp.array([4, 5, 6], dtype)
    indices = xp.array([2, 0, 1], 'i')
    indptr = xp.array([0, 1, 2, 3], 'i')
    cls = sp.csr_array if array else sp.csr_matrix
    return cls((data, indices, indptr), shape=(3, 3))


def _make_for_matmul(xp, sp, dtype, *, array=False):
    """4x3 CSR for matmul with 3x4."""
    data = xp.array([1, 2, 3, 4, 5], dtype)
    indices = xp.array([0, 2, 1, 0, 2], 'i')
    indptr = xp.array([0, 1, 3, 3, 5], 'i')
    cls = sp.csr_array if array else sp.csr_matrix
    return cls((data, indices, indptr), shape=(4, 3))


class TestSparseArrayTypeIdentity:
    """issparse, isspmatrix, isinstance checks for all formats."""

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_array_issparse(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        A = cls((2, 3), dtype=numpy.float64)
        assert sparse.issparse(A)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_array_not_isspmatrix(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        A = cls((2, 3), dtype=numpy.float64)
        assert not sparse.isspmatrix(A)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_array_isinstance_sparray(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        A = cls((2, 3), dtype=numpy.float64)
        assert isinstance(A, sparse.sparray)
        assert not isinstance(A, sparse.spmatrix)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_matrix_isinstance_spmatrix(self, fmt):
        cls = getattr(sparse, f'{fmt}_matrix')
        M = cls((2, 3), dtype=numpy.float64)
        assert isinstance(M, sparse.spmatrix)
        assert not isinstance(M, sparse.sparray)

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_matrix_issparse(self, fmt):
        cls = getattr(sparse, f'{fmt}_matrix')
        M = cls((2, 3), dtype=numpy.float64)
        assert sparse.issparse(M)
        assert sparse.isspmatrix(M)

    def test_dense_not_sparse(self):
        assert not sparse.issparse(cupy.array([1, 2]))
        assert not sparse.isspmatrix(cupy.array([1, 2]))

    def test_coo_array_coords_property(self):
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype='i')
        col = cupy.array([2, 0, 1], dtype='i')
        A = sparse.coo_array((data, (row, col)), shape=(3, 3))
        assert isinstance(A.coords, tuple)
        assert len(A.coords) == 2
        assert A.coords[0] is A.row
        assert A.coords[1] is A.col

    def test_repr_array(self):
        A = sparse.csr_array(
            (cupy.array([1.0, 2.0]), cupy.array([0, 1], 'i'),
             cupy.array([0, 1, 2], 'i')), shape=(2, 2))
        r = repr(A)
        assert 'sparse array' in r
        assert 'sparse matrix' not in r
        assert 'Compressed Sparse Row' in r

    def test_repr_matrix(self):
        M = sparse.csr_matrix(
            (cupy.array([1.0, 2.0]), cupy.array([0, 1], 'i'),
             cupy.array([0, 1, 2], 'i')), shape=(2, 2))
        r = repr(M)
        assert 'sparse matrix' in r
        assert 'sparse array' not in r

    def test_mT_property(self):
        A = sparse.csr_array(
            (cupy.array([1.0, 2.0]), cupy.array([0, 1], 'i'),
             cupy.array([0, 1, 2], 'i')), shape=(2, 2))
        assert A.mT.shape == (2, 2)
        assert isinstance(A.mT, sparse.sparray)

    def test_block_array_returns_sparray(self):
        A = sparse.csr_array(cupy.array([[1, 0], [0, 1]], dtype='d'))
        result = sparse.block_array([[A, A], [A, A]])
        assert isinstance(result, sparse.sparray)
        assert result.shape == (4, 4)

    def test_block_diag_arrays(self):
        A = sparse.csr_array(cupy.array([[1, 2]], dtype='d'))
        B = sparse.csr_array(cupy.array([[3]], dtype='d'))
        result = sparse.block_diag((A, B))
        assert isinstance(result, sparse.sparray)
        assert result.shape == (2, 3)

    def test_block_diag_matrices(self):
        A = sparse.csr_matrix(cupy.array([[1, 2]], dtype='d'))
        B = sparse.csr_matrix(cupy.array([[3]], dtype='d'))
        result = sparse.block_diag((A, B))
        assert isinstance(result, sparse.spmatrix)

    def test_safely_cast_index_arrays_csr(self):
        A = sparse.csr_array(cupy.array([[1, 0], [0, 2]], dtype='d'))
        ind, ptr = sparse.safely_cast_index_arrays(A, numpy.int32)
        assert ind.dtype == numpy.int32
        assert ptr.dtype == numpy.int32

    def test_safely_cast_index_arrays_coo(self):
        A = sparse.coo_array(
            (cupy.array([1.0]), (cupy.array([0], 'i'),
             cupy.array([0], 'i'))), shape=(2, 2))
        row, col = sparse.safely_cast_index_arrays(A, numpy.int32)
        assert row.dtype == numpy.int32
        assert col.dtype == numpy.int32

    def test_get_index_dtype_export(self):
        assert sparse.get_index_dtype(maxval=10) == numpy.int32
        assert (sparse.get_index_dtype(maxval=2**33)
                == numpy.int64)

    def test_nonzero_array(self):
        A = sparse.csr_array(
            cupy.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]))
        row, col = A.nonzero()
        assert row.shape == (3,)
        assert col.shape == (3,)
        cupy.testing.assert_array_equal(row, cupy.array([0, 0, 1]))
        cupy.testing.assert_array_equal(col, cupy.array([0, 1, 2]))

    def test_round_array(self):
        A = sparse.csr_array(
            cupy.array([[1.4, 2.6, 0.0], [0.0, 0.0, 3.5]]))
        R = round(A)
        assert isinstance(R, sparse.sparray)
        cupy.testing.assert_array_equal(
            R.toarray(),
            cupy.array([[1.0, 3.0, 0.0], [0.0, 0.0, 4.0]]))

    def test_matrix_transpose_function(self):
        A = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        T = sparse.matrix_transpose(A)
        assert T.shape == (2, 2)
        # csr_array.T is csc_array (same data, different format).
        assert isinstance(T, sparse.sparray)

    def test_swapaxes(self):
        A = sparse.csr_array(cupy.array([[1., 2., 3.], [4., 5., 6.]]))
        T = sparse.swapaxes(A, 0, 1)
        assert T.shape == (3, 2)
        assert isinstance(T, sparse.sparray)
        T_neg = sparse.swapaxes(A, -2, -1)
        assert T_neg.shape == (3, 2)
        # Identity swap.
        T_id = sparse.swapaxes(A, 0, 0)
        assert T_id.shape == A.shape

    def test_permute_dims(self):
        A = sparse.csr_array(cupy.array([[1., 2., 3.], [4., 5., 6.]]))
        # Default reverses axes.
        P = sparse.permute_dims(A)
        assert P.shape == (3, 2)
        # Identity.
        P_id = sparse.permute_dims(A, (0, 1))
        assert P_id.shape == A.shape
        # Explicit reversal.
        P_rev = sparse.permute_dims(A, (1, 0))
        assert P_rev.shape == (3, 2)
        # Bad permutation raises.
        with pytest.raises(ValueError):
            sparse.permute_dims(A, (0, 0))

    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_array_maxprint_kwarg(self, fmt):
        cls = getattr(sparse, f'{fmt}_array')
        A = cls(cupy.array([[1., 2.]]), maxprint=10)
        assert A.maxprint == 10
        # Default
        B = cls(cupy.array([[1., 2.]]))
        assert B.maxprint == 50

    def test_dia_array_maxprint_kwarg(self):
        data = cupy.array([[1., 2., 3.]])
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3), maxprint=10)
        assert A.maxprint == 10

    def test_negate_bool_array_raises(self):
        # Match scipy 1.17: NotImplementedError, not TypeError.
        B = sparse.csr_array(
            cupy.array([[True, False], [False, True]]))
        with pytest.raises(NotImplementedError, match='boolean'):
            -B

    def test_bool_sparse_mask_indexing_returns_dense(self):
        # Boolean sparse-array indexing returns a 1-D dense ndarray
        # (matches scipy 1.17).
        A = sparse.csr_array(
            cupy.array([[1., 2., 0.], [0., 0., 3.]]))
        mask = sparse.csr_array(
            cupy.array([[True, False, True], [True, False, True]]))
        res = A[mask]
        assert isinstance(res, cupy.ndarray)
        cupy.testing.assert_array_equal(
            res, cupy.array([1., 0., 0., 3.]))

    def test_csc_array_matmul_preserves_array_type(self):
        # SciPy gh-fix: csc_array @ csc_array (or csr_array) returns
        # csr_array, not csr_matrix.
        A = sparse.csc_array(cupy.array([[1., 2.], [3., 4.]]))
        B = sparse.csc_array(cupy.array([[1., 2.], [3., 4.]]))
        assert isinstance(A @ B, sparse.sparray)
        C = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        assert isinstance(A @ C, sparse.sparray)

    def test_setitem_scalar_from_sparse_rhs(self):
        # Assigning a 1x1 sparse RHS to a scalar position works
        # (densifies the RHS first).  Used to crash with TypeError.
        A = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        rhs = sparse.csr_array(cupy.array([[7.]]))
        A[0, 0] = rhs
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[7., 2.], [3., 4.]]))

    def test_get_array_module_sparray(self):
        # cupy.get_array_module and cupyx.scipy.get_array_module recognize
        # sparray, not just spmatrix.
        import cupyx.scipy as cscipy
        A = sparse.csr_array(cupy.array([[1., 2.]]))
        M = sparse.csr_matrix(cupy.array([[1., 2.]]))
        assert cupy.get_array_module(A) is cupy
        assert cupy.get_array_module(M) is cupy
        assert cscipy.get_array_module(A).__name__ == 'cupyx.scipy'
        assert cscipy.get_array_module(M).__name__ == 'cupyx.scipy'

    def test_dia_tocsc_data_wider_than_matrix(self):
        # Regression: DIA with data buffer wider than the matrix used to
        # raise a broadcast-shape ValueError in tocsc().
        data = cupy.array([[1., 2., 3., 4., 5., 6.]])
        offsets = cupy.array([0])
        m = sparse.dia_array((data, offsets), shape=(3, 4))
        cupy.testing.assert_array_equal(
            m.tocsc().toarray(),
            cupy.array([[1., 0., 0., 0.],
                        [0., 2., 0., 0.],
                        [0., 0., 3., 0.]]))

    def test_csc_setdiag(self):
        A = sparse.csc_matrix(cupy.zeros((3, 3)))
        A.setdiag(cupy.array([1., 2., 3.]))
        cupy.testing.assert_array_equal(
            A.toarray(),
            cupy.array([[1., 0., 0.],
                        [0., 2., 0.],
                        [0., 0., 3.]]))

    def test_prune_csr(self):
        # Construct a CSR with extra slack at the end of indices/data.
        # CuPy's ``nnz`` reports the buffer length (not indptr[-1]) so
        # ``prune()`` is what actually trims the buffers down to
        # ``indptr[-1]``.  ``_from_parts`` validates ``data.size ==
        # int(indptr[-1])`` by default (cuSPARSE descriptors otherwise
        # silently consume slack as live data); the
        # ``_skip_buffer_check=True`` opt-out is the documented escape
        # hatch for callers who intentionally construct with slack and
        # plan to ``prune()``.
        data = cupy.array([1.0, 2.0, 3.0, 99.0, 99.0])
        indices = cupy.array([0, 1, 2, 99, 99], dtype='i')
        indptr = cupy.array([0, 1, 2, 3], dtype='i')
        A = sparse.csr_matrix._from_parts(
            data, indices, indptr, (3, 3), _skip_buffer_check=True)
        assert int(A.indptr[-1]) == 3  # logical entry count
        assert A.indices.shape == (5,)  # buffer has slack
        A.prune()
        assert A.indices.shape == (3,)
        assert A.data.shape == (3,)
        cupy.testing.assert_array_equal(A.indices, cupy.array([0, 1, 2]))

    def test_astype_copy_param(self):
        A = sparse.csr_array(cupy.array([[1., 2.]]))
        # No-op when dtype matches and copy=False
        B = A.astype(cupy.float64, copy=False)
        assert B is A
        # New object when copy=True
        C = A.astype(cupy.float64, copy=True)
        assert C is not A
        # Different dtype always returns new object
        D = A.astype(cupy.float32)
        assert D.dtype == cupy.float32
        assert D is not A

    def test_dia_todia_returns_self(self):
        # ``_dia_base.todia`` overrides ``_spbase.todia`` so that an
        # already-DIA input returns ``self`` (or a copy) without
        # round-tripping through CSR -> COO -> DIA.  ``csr.todia`` /
        # ``coo.todia`` are now real implementations (not
        # NotImplementedError as before), so the round-trip would
        # produce a correct result, but ``self`` is still much faster.
        data = cupy.array([[1., 2., 3.]])
        offsets = cupy.array([0])
        m = sparse.dia_array((data, offsets), shape=(3, 3))
        assert m.todia() is m
        # copy=True returns a new object
        n = m.todia(copy=True)
        assert n is not m
        cupy.testing.assert_array_equal(n.toarray(), m.toarray())

    def test_block_diag_int_dense(self):
        # block_diag with integer-typed Python list/scalar/tuple input
        # used to fail because cuSPARSE rejects the int dtype; now we
        # promote to float64 first.
        res = sparse.block_diag([[[1, 2], [3, 4]], [[5]]])
        cupy.testing.assert_array_equal(
            res.toarray(),
            cupy.array([[1., 2., 0.],
                        [3., 4., 0.],
                        [0., 0., 5.]]))

    def test_block_diag_tuple_input(self):
        res = sparse.block_diag([(1, 2), [[3]]])
        cupy.testing.assert_array_equal(
            res.toarray(),
            cupy.array([[1., 2., 0.],
                        [0., 0., 3.]]))

    def test_block_diag_scalars(self):
        res = sparse.block_diag([1.0, 2.0, 3.0])
        cupy.testing.assert_array_equal(
            res.toarray(),
            cupy.array([[1., 0., 0.],
                        [0., 2., 0.],
                        [0., 0., 3.]]))

    def test_count_nonzero_csr_axis(self):
        # CSR/CSC count_nonzero(axis=...) uses the bincount fast path
        # (no tocoo round-trip).  Result matches per-row / per-col
        # counts of the dense form.
        A = sparse.csr_array(
            cupy.array([[1., 0., 2., 0.],
                        [0., 0., 0., 3.],
                        [4., 5., 0., 6.]]))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=0), cupy.array([2, 1, 1, 2]))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1), cupy.array([2, 1, 3]))
        # Dedupes first: stored zero excluded.
        data = cupy.array([1.0, 2.0, 0.0, 3.0])
        ind = cupy.array([0, 1, 2, 0], dtype='i')
        ptr = cupy.array([0, 1, 3, 4], dtype='i')
        B = sparse.csr_array._from_parts(data, ind, ptr, (3, 3))
        assert B.count_nonzero() == 3  # explicit zero excluded

    def test_count_nonzero_coo_axis(self):
        A = sparse.coo_array(
            cupy.array([[1., 0., 2.],
                        [0., 3., 0.]]))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=0), cupy.array([1, 1, 1]))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1), cupy.array([2, 1]))

    def test_count_nonzero_dia_axis_raises(self):
        # DIA axis-aware count_nonzero matches scipy: NotImplementedError.
        # Users can convert to CSR/CSC for per-axis counts.
        m = sparse.dia_array(
            (cupy.array([[1., 2., 3.]]), cupy.array([0])),
            shape=(3, 3))
        assert m.count_nonzero() == 3
        with pytest.raises(NotImplementedError):
            m.count_nonzero(axis=0)

    def test_count_nonzero_dia_data_wider_than_matrix(self):
        # When ``data`` is wider than the matrix, the pad columns lie
        # outside the matrix and must be excluded — matching scipy.
        m = sparse.dia_array(
            (cupy.array([[1., 2., 3., 4., 5., 6.]]),
             cupy.array([0])),
            shape=(3, 4))
        # Only data[0:3] correspond to in-matrix positions (0,0), (1,1),
        # (2,2); the pad columns 3..5 are outside the (3, 4) matrix.
        assert m.count_nonzero() == 3

    def test_count_nonzero_dia_excludes_explicit_zero(self):
        # Explicit zero in the diagonal data is excluded.
        m = sparse.dia_array(
            (cupy.array([[0., 1., 2.]]), cupy.array([0])),
            shape=(3, 3))
        assert m.count_nonzero() == 2  # the 0 at (0, 0) doesn't count
        assert m.nnz == 3                # but it is stored

    @testing.with_requires('scipy')
    @pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo'])
    def test_type_system_matches_scipy(self, fmt):
        """CuPy and SciPy type predicates should agree."""
        sp_arr_cls = getattr(scipy.sparse, f'{fmt}_array')
        sp_mat_cls = getattr(scipy.sparse, f'{fmt}_matrix')
        cp_arr_cls = getattr(sparse, f'{fmt}_array')
        cp_mat_cls = getattr(sparse, f'{fmt}_matrix')

        sp_a = sp_arr_cls((2, 3))
        sp_m = sp_mat_cls((2, 3))
        cp_a = cp_arr_cls((2, 3))
        cp_m = cp_mat_cls((2, 3))

        assert sparse.issparse(cp_a) == scipy.sparse.issparse(sp_a)
        assert sparse.issparse(cp_m) == scipy.sparse.issparse(sp_m)
        assert sparse.isspmatrix(cp_a) == scipy.sparse.isspmatrix(sp_a)
        assert sparse.isspmatrix(cp_m) == scipy.sparse.isspmatrix(sp_m)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy')
class TestCsrArrayConstruction:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_data_indices_indptr(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        assert m.format == 'csr'
        assert isinstance(m, sp.sparray)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_dense(self, xp, sp):
        dense = xp.array([[1, 0, 2], [0, 3, 0]], dtype=self.dtype)
        m = sp.csr_array(dense)
        assert isinstance(m, sp.sparray)
        return m

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_from_coo_tuple(self, xp, sp):
        data = xp.array([1, 2, 3], self.dtype)
        row = xp.array([0, 1, 2], 'i')
        col = xp.array([2, 0, 1], 'i')
        m = sp.csr_array((data, (row, col)), shape=(3, 3))
        assert isinstance(m, sp.sparray)
        return m

    def test_empty(self):
        m = sparse.csr_array((3, 4), dtype=numpy.float64)
        assert m.shape == (3, 4)
        assert m.nnz == 0
        assert isinstance(m, sparse.sparray)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrArrayStarIsElementwise:
    """Verify that * is element-wise for csr_array (matching scipy.sparse)."""

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_sparse(self, xp, sp):
        """array * array should be element-wise."""
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        b = _make_csr_sq(xp, sp, self.dtype, array=True)
        return a * b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_scalar(self, xp, sp):
        """array * scalar should be scalar multiplication."""
        a = _make_csr(xp, sp, self.dtype, array=True)
        return a * self.dtype(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_rstar_scalar(self, xp, sp):
        """scalar * array should be scalar multiplication."""
        a = _make_csr(xp, sp, self.dtype, array=True)
        return self.dtype(3.0) * a


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrArrayMatmul:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_matmul_sparse(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        b = _make_for_matmul(xp, sp, self.dtype, array=True)
        return a @ b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_matmul_dense_vector(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        x = xp.arange(4).astype(self.dtype)
        return a @ x

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_matmul_dense_matrix(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        x = xp.arange(8).reshape(4, 2).astype(self.dtype)
        return a @ x

    def test_matmul_scalar_raises(self):
        a = _make_csr_sq(cupy, sparse, numpy.float64, array=True)
        with pytest.raises(ValueError):
            a @ 5.0


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrArrayPower:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_elementwise(self, xp, sp):
        """array ** n should be element-wise, not matrix power."""
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        return a ** 2

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_matches_dense(self, xp, sp):
        """Verify ** gives same result as dense element-wise power."""
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        result_sparse = (a ** 2).toarray()
        result_dense = a.toarray() ** 2
        xp.testing.assert_allclose(result_sparse, result_dense)
        return result_sparse


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrMatrixStarIsMatmul:
    """Verify that * is still matmul for csr_matrix (unchanged)."""

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_matmul(self, xp, sp):
        """matrix * matrix should be matmul."""
        a = _make_csr(xp, sp, self.dtype)
        b = _make_for_matmul(xp, sp, self.dtype)
        return a * b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_star_scalar(self, xp, sp):
        """matrix * scalar should still work."""
        a = _make_csr(xp, sp, self.dtype)
        return a * self.dtype(2.0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_pow_matrix_power(self, xp, sp):
        """matrix ** n should be matrix power."""
        a = _make_csr_sq(xp, sp, self.dtype)
        return a ** 2

    def test_power_zero_raises(self):
        """Array ** 0 raises NotImplementedError (would densify)."""
        a = _make_csr_sq(cupy, sparse, numpy.float64, array=True)
        with pytest.raises(NotImplementedError):
            a ** 0


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
class TestCsrArrayTypePreservation:
    """Operations on csr_array should return csr_array (not csr_matrix)."""

    def _check_array(self, result):
        assert isinstance(result, sparse.sparray), (
            f'Expected sparray, got {type(result).__name__}')
        assert not isinstance(result, sparse.spmatrix)

    def test_star(self):
        a = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        b = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        self._check_array(a * b)

    def test_matmul(self):
        a = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        b = _make_csr_sq2(cupy, sparse, self.dtype, array=True)
        self._check_array(a @ b)

    def test_add(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        b = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(a + b)

    def test_sub(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        b = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(a - b)

    def test_neg(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(-a)

    def test_scalar_mul(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(a * self.dtype(2.0))

    def test_pow(self):
        a = _make_csr_sq(cupy, sparse, self.dtype, array=True)
        self._check_array(a ** 2)

    def test_copy(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(a.copy())

    def test_abs(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        self._check_array(abs(a))

    def test_T(self):
        a = _make_csr(cupy, sparse, self.dtype, array=True)
        result = a.T
        assert isinstance(result, sparse.sparray)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCsrArrayConversions:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocsc(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        result = m.tocsc()
        assert isinstance(result, sp.sparray)
        assert result.format == 'csc'
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_tocoo(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        result = m.tocoo()
        assert isinstance(result, sp.sparray)
        assert result.format == 'coo'
        return result

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        return m.toarray()

    def test_tocsr_returns_self(self):
        m = _make_csr(cupy, sparse, numpy.float64, array=True)
        assert m.tocsr() is m

    def test_tocsr_copy(self):
        m = _make_csr(cupy, sparse, numpy.float64, array=True)
        n = m.tocsr(copy=True)
        assert n is not m
        assert isinstance(n, sparse.sparray)


@testing.with_requires('scipy')
class TestCsrArrayGet:

    def test_array_get_returns_scipy_array(self):
        m = _make_csr(cupy, sparse, numpy.float64, array=True)
        sp_m = m.get()
        assert isinstance(sp_m, scipy.sparse.csr_array)
        assert isinstance(sp_m, scipy.sparse.sparray)

    def test_matrix_get_returns_scipy_matrix(self):
        m = _make_csr(cupy, sparse, numpy.float64, array=False)
        sp_m = m.get()
        assert isinstance(sp_m, scipy.sparse.csr_matrix)
        assert isinstance(sp_m, scipy.sparse.spmatrix)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_array_get_values(self, xp, sp):
        m = _make_csr(xp, sp, numpy.float64, array=True)
        return m.toarray()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@testing.with_requires('scipy')
class TestCsrArrayArithmeticSciPyComparison:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        b = _make_csr(xp, sp, self.dtype, array=True)
        return a + b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sub(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        b = _make_csr(xp, sp, self.dtype, array=True)
        return (a - b).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_neg(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return (-a).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_elementwise(self, xp, sp):
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        b = _make_csr_sq(xp, sp, self.dtype, array=True)
        return a * b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mul_scalar(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return a * self.dtype(2.5)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_matmul(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        b = _make_for_matmul(xp, sp, self.dtype, array=True)
        return a @ b

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_power_elementwise(self, xp, sp):
        a = _make_csr_sq(xp, sp, self.dtype, array=True)
        return a ** 2

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_abs(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return abs(a)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_transpose(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return a.T

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_conj(self, xp, sp):
        a = _make_csr(xp, sp, self.dtype, array=True)
        return a.conj()


class TestCsrArrayRemovedMethods:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.arr = sparse.csr_array(
            (cupy.array([1.0]), cupy.array([0], dtype='i'),
             cupy.array([0, 1], dtype='i')), shape=(1, 2))

    def test_no_A(self):
        with pytest.raises(AttributeError):
            self.arr.A

    def test_no_H(self):
        with pytest.raises(AttributeError):
            self.arr.H

    def test_no_getrow(self):
        with pytest.raises(AttributeError):
            self.arr.getrow(0)

    def test_no_getcol(self):
        with pytest.raises(AttributeError):
            self.arr.getcol(0)

    def test_no_getH(self):
        with pytest.raises(AttributeError):
            self.arr.getH()

    def test_no_asfptype(self):
        with pytest.raises(AttributeError):
            self.arr.asfptype()

    def test_no_getformat(self):
        with pytest.raises(AttributeError):
            self.arr.getformat()

    def test_no_getmaxprint(self):
        with pytest.raises(AttributeError):
            self.arr.getmaxprint()

    def test_no_shape_setter(self):
        with pytest.raises(AttributeError):
            self.arr.shape = (2, 1)

    def test_has_shape(self):
        assert self.arr.shape == (1, 2)

    def test_has_nnz(self):
        assert self.arr.nnz == 1

    def test_has_format(self):
        assert self.arr.format == 'csr'

    def test_has_ndim(self):
        assert self.arr.ndim == 2


# Matrix-only APIs that don't exist on sparray.  In SciPy 1.14 most of
# these were deprecated; SciPy 1.17 un-deprecated everything except
# ``A`` and ``H`` (those still emit DeprecationWarning on CuPy until
# users have a release cycle to migrate).  The ``A`` / ``H``
# DeprecationWarnings are silenced here; warning behaviour itself is
# exercised by ``test_base.py::TestDeprecatedSpmatrixApi``.
@pytest.mark.filterwarnings(
    "ignore:`spmatrix\\.(A|H)`:DeprecationWarning"
)
class TestCsrMatrixLegacyMethods:

    @pytest.fixture(autouse=True)
    def setUp(self):
        self.mat = sparse.csr_matrix(
            (cupy.array([1.0]), cupy.array([0], dtype='i'),
             cupy.array([0, 1], dtype='i')), shape=(1, 2))

    def test_has_A(self):
        result = self.mat.A
        assert isinstance(result, cupy.ndarray)

    def test_has_H(self):
        result = self.mat.H
        assert sparse.issparse(result)

    def test_has_getH(self):
        result = self.mat.getH()
        assert sparse.issparse(result)

    def test_has_asfptype(self):
        result = self.mat.asfptype()
        assert sparse.issparse(result)

    def test_has_getformat(self):
        assert self.mat.getformat() == 'csr'

    def test_has_getrow(self):
        result = self.mat.getrow(0)
        assert sparse.issparse(result)

    def test_has_getcol(self):
        result = self.mat.getcol(0)
        assert sparse.issparse(result)

    def test_has_shape_setter(self):
        # shape setter exists on matrices (even if reshape is no-op here)
        self.mat.shape = (1, 2)


# Index dtype policy: arrays preserve int64, matrices may downcast

@testing.with_requires('scipy')
class TestCsrArrayIndexDtype:

    def test_array_preserves_int64(self):
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        A = sparse.csr_array((data, indices, indptr), shape=(3, 3))
        assert A.indices.dtype == cupy.int64
        assert A.indptr.dtype == cupy.int64

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_array_int64_matches_scipy(self, xp, sp):
        data = xp.array([1.0, 2.0, 3.0])
        indices = xp.array([0, 1, 2], dtype='int64')
        indptr = xp.array([0, 1, 2, 3], dtype='int64')
        A = sp.csr_array((data, indices, indptr), shape=(3, 3))
        return A.indices.dtype == 'int64'

    def test_matrix_may_downcast(self):
        data = cupy.array([1.0, 2.0, 3.0])
        indices = cupy.array([0, 1, 2], dtype=cupy.int64)
        indptr = cupy.array([0, 1, 2, 3], dtype=cupy.int64)
        M = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
        # matrix may downcast small int64 values to int32
        assert M.indices.dtype in (cupy.int32, cupy.int64)

    def test_coo_array_preserves_int64(self):
        data = cupy.array([1.0, 2.0, 3.0])
        row = cupy.array([0, 1, 2], dtype=cupy.int64)
        col = cupy.array([0, 1, 2], dtype=cupy.int64)
        A = sparse.coo_array((data, (row, col)), shape=(3, 3))
        assert A.row.dtype == cupy.int64
        assert A.col.dtype == cupy.int64


# CSC/COO array conversion type preservation

class TestNonCsrArrayConversions:

    def test_coo_array_tocsr_type(self):
        A = sparse.coo_array(
            (cupy.array([1.0]), (cupy.array([0], dtype='i'),
             cupy.array([0], dtype='i'))), shape=(2, 2))
        B = A.tocsr()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csr'

    def test_coo_array_tocsc_type(self):
        A = sparse.coo_array(
            (cupy.array([1.0]), (cupy.array([0], dtype='i'),
             cupy.array([0], dtype='i'))), shape=(2, 2))
        B = A.tocsc()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csc'

    def test_csc_array_tocsr_type(self):
        data = cupy.array([1.0, 2.0], dtype='d')
        indices = cupy.array([0, 1], dtype='i')
        indptr = cupy.array([0, 1, 2], dtype='i')
        A = sparse.csc_array((data, indices, indptr), shape=(2, 2))
        B = A.tocsr()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csr'

    def test_csc_array_tocoo_type(self):
        data = cupy.array([1.0, 2.0], dtype='d')
        indices = cupy.array([0, 1], dtype='i')
        indptr = cupy.array([0, 1, 2], dtype='i')
        A = sparse.csc_array((data, indices, indptr), shape=(2, 2))
        B = A.tocoo()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'coo'

    def test_csc_array_transpose_type(self):
        data = cupy.array([1.0, 2.0], dtype='d')
        indices = cupy.array([0, 1], dtype='i')
        indptr = cupy.array([0, 1, 2], dtype='i')
        A = sparse.csc_array((data, indices, indptr), shape=(2, 2))
        AT = A.T
        assert isinstance(AT, sparse.sparray)


# Construction functions

class TestConstructionFunctions:

    def test_eye_array_exists(self):
        A = sparse.eye_array(3)
        assert isinstance(A, sparse.sparray)
        assert A.shape == (3, 3)

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eye_array_values(self, xp, sp):
        return sp.eye_array(4, k=1, dtype='d').toarray()

    def test_eye_array_format(self):
        A = sparse.eye_array(3, format='csc')
        assert isinstance(A, sparse.sparray)
        assert A.format == 'csc'

    def test_diags_array(self):
        A = sparse.diags_array([1, 2, 3])
        assert isinstance(A, sparse.sparray)
        assert A.shape == (3, 3)

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_diags_array_values(self, xp, sp):
        # Use explicit float dtype to sidestep the SciPy 1.17 FutureWarning
        # for integer-input ``diags_array`` calls (will become an error in
        # SciPy 1.19).  CuPy's ``diags_array`` already requires a float
        # storage dtype.
        return sp.diags_array([1, 2, 3], offsets=0, dtype=float).toarray()

    def test_random_array(self):
        A = sparse.random_array((10, 10), density=0.5)
        assert isinstance(A, sparse.sparray)
        assert A.shape == (10, 10)

    def test_random_array_format(self):
        A = sparse.random_array((5, 5), format='csr')
        assert isinstance(A, sparse.sparray)
        assert A.format == 'csr'


# Type-aware construction: kron, hstack, vstack, tril, triu

class TestTypeAwareConstruct:

    @pytest.fixture
    def arr_pair(self):
        d = cupy.array([[1, 0], [0, 2]], dtype='d')
        return sparse.csr_array(d), sparse.csr_array(d)

    @pytest.fixture
    def mat_pair(self):
        d = cupy.array([[1, 0], [0, 2]], dtype='d')
        return sparse.csr_matrix(d), sparse.csr_matrix(d)

    def test_hstack_arrays(self, arr_pair):
        result = sparse.hstack(list(arr_pair))
        assert isinstance(result, sparse.sparray)

    def test_hstack_matrices(self, mat_pair):
        result = sparse.hstack(list(mat_pair))
        assert isinstance(result, sparse.spmatrix)

    def test_vstack_arrays(self, arr_pair):
        result = sparse.vstack(list(arr_pair))
        assert isinstance(result, sparse.sparray)

    def test_vstack_matrices(self, mat_pair):
        result = sparse.vstack(list(mat_pair))
        assert isinstance(result, sparse.spmatrix)

    def test_kron_arrays(self, arr_pair):
        result = sparse.kron(*arr_pair)
        assert isinstance(result, sparse.sparray)

    def test_kron_matrices(self, mat_pair):
        result = sparse.kron(*mat_pair)
        assert isinstance(result, sparse.spmatrix)

    def test_tril_array(self, arr_pair):
        result = sparse.tril(arr_pair[0])
        assert isinstance(result, sparse.sparray)

    def test_tril_matrix(self, mat_pair):
        result = sparse.tril(mat_pair[0])
        assert isinstance(result, sparse.spmatrix)

    def test_triu_array(self, arr_pair):
        result = sparse.triu(arr_pair[0])
        assert isinstance(result, sparse.sparray)

    def test_triu_matrix(self, mat_pair):
        result = sparse.triu(mat_pair[0])
        assert isinstance(result, sparse.spmatrix)


# CSC array arithmetic

@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestCscArrayArithmetic:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_add(self, xp, sp):
        data = xp.array([1, 2, 3], self.dtype)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 1, 2, 3], 'i')
        a = sp.csc_array((data, indices, indptr), shape=(3, 3))
        b = sp.csc_array((data, indices, indptr), shape=(3, 3))
        return a + b

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_sub(self, xp, sp):
        data = xp.array([1, 2, 3], self.dtype)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 1, 2, 3], 'i')
        a = sp.csc_array((data, indices, indptr), shape=(3, 3))
        b = sp.csc_array((data, indices, indptr), shape=(3, 3))
        return (a - b).toarray()

    def test_add_preserves_type(self):
        data = cupy.array([1, 2, 3], numpy.float64)
        indices = cupy.array([0, 1, 2], 'i')
        indptr = cupy.array([0, 1, 2, 3], 'i')
        a = sparse.csc_array((data, indices, indptr), shape=(3, 3))
        b = sparse.csc_array((data, indices, indptr), shape=(3, 3))
        result = a + b
        assert isinstance(result, sparse.sparray)


# Cross-format multiply

@testing.with_requires('scipy')
class TestCrossFormatMultiply:

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_coo_star_coo(self, xp, sp):
        """COO * COO element-wise should work."""
        data = xp.array([1, 2, 3], numpy.float64)
        row = xp.array([0, 1, 2], 'i')
        col = xp.array([0, 1, 2], 'i')
        a = sp.coo_array((data, (row, col)), shape=(3, 3))
        b = sp.coo_array((data, (row, col)), shape=(3, 3))
        return (a * b).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp', contiguous_check=False)
    def test_csc_star_csc(self, xp, sp):
        """CSC * CSC element-wise should work."""
        data = xp.array([1, 2, 3], numpy.float64)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 1, 2, 3], 'i')
        a = sp.csc_array((data, indices, indptr), shape=(3, 3))
        b = sp.csc_array((data, indices, indptr), shape=(3, 3))
        return (a * b).toarray()

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_csr_star_coo(self, xp, sp):
        """CSR * COO cross-format multiply should work."""
        data = xp.array([1, 2, 3], numpy.float64)
        indices = xp.array([0, 1, 2], 'i')
        indptr = xp.array([0, 1, 2, 3], 'i')
        a = sp.csr_array((data, indices, indptr), shape=(3, 3))
        row = xp.array([0, 1, 2], 'i')
        col = xp.array([0, 1, 2], 'i')
        b = sp.coo_array((data, (row, col)), shape=(3, 3))
        return (a * b).toarray()


# Reduction 1D shaping

@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestArrayReductions:

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_sum_axis0_ndim(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        result = m.sum(axis=0)
        return result.ndim

    @testing.numpy_cupy_equal(sp_name='sp')
    def test_sum_axis1_ndim(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        result = m.sum(axis=1)
        return result.ndim

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_axis0_values(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        return m.sum(axis=0)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_sum_axis1_values(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        return m.sum(axis=1)

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_mean_axis0(self, xp, sp):
        m = _make_csr(xp, sp, self.dtype, array=True)
        return m.mean(axis=0)

    def test_matrix_sum_stays_2d(self):
        """Matrix sum(axis=0) should still be 2D."""
        m = _make_csr(cupy, sparse, numpy.float64, array=False)
        result = m.sum(axis=0)
        assert result.ndim == 2


# DIA array

class TestDiaArrayBasic:

    def test_construction(self):
        data = cupy.array([[1, 2, 3]], dtype=numpy.float64)
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3))
        assert isinstance(A, sparse.sparray)
        assert A.format == 'dia'

    def test_tocsr(self):
        data = cupy.array([[1, 2, 3]], dtype=numpy.float64)
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3))
        B = A.tocsr()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csr'

    def test_tocsc(self):
        data = cupy.array([[1, 2, 3]], dtype=numpy.float64)
        offsets = cupy.array([0])
        A = sparse.dia_array((data, offsets), shape=(3, 3))
        B = A.tocsc()
        assert isinstance(B, sparse.sparray)
        assert B.format == 'csc'

    @testing.with_requires('scipy')
    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_toarray_matches_scipy(self, xp, sp):
        data = xp.array([[1, 2, 3]], dtype=numpy.float64)
        offsets = xp.array([0])
        A = sp.dia_array((data, offsets), shape=(3, 3))
        return A.toarray()


# LinearOperator from array

class TestLinearOperatorFromArray:

    def test_aslinearoperator_csr_array(self):
        from cupyx.scipy.sparse.linalg import aslinearoperator
        m = _make_csr_sq(cupy, sparse, numpy.float64, array=True)
        op = aslinearoperator(m)
        v = cupy.ones(3, dtype=numpy.float64)
        result = op @ v
        expected = m @ v
        cupy.testing.assert_allclose(result, expected)


# Linalg solvers accept arrays

class TestSpsolveArray:

    def test_spsolve_csr_array(self):
        from cupyx.scipy.sparse.linalg import spsolve
        n = 8
        A_dense = cupy.zeros((n, n), dtype=numpy.float64)
        A_dense[cupy.arange(n), cupy.arange(n)] = 4
        A_dense[cupy.arange(n - 1), cupy.arange(1, n)] = 1
        A_dense[cupy.arange(1, n), cupy.arange(n - 1)] = 1
        A = sparse.csr_array(A_dense)
        b = cupy.arange(1, n + 1, dtype=numpy.float64)
        x = spsolve(A, b)
        cupy.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_spsolve_csr_matrix(self):
        from cupyx.scipy.sparse.linalg import spsolve
        n = 8
        A_dense = cupy.zeros((n, n), dtype=numpy.float64)
        A_dense[cupy.arange(n), cupy.arange(n)] = 4
        A_dense[cupy.arange(n - 1), cupy.arange(1, n)] = 1
        A_dense[cupy.arange(1, n), cupy.arange(n - 1)] = 1
        M = sparse.csr_matrix(A_dense)
        b = cupy.arange(1, n + 1, dtype=numpy.float64)
        x = spsolve(M, b)
        cupy.testing.assert_allclose(M * x, b, rtol=1e-10)


# ---------------------------------------------------------------
# Red-team follow-up regression tests (sparray red-team report,
# round 1-3; int64-branch redteam_findings.md F-series)
# ---------------------------------------------------------------

class TestRedTeamFollowups:
    """Regressions covering the round-3 red-team findings."""

    # C1 -- tuple-2 path silently downcast int64 -> int32 for arrays.
    @pytest.mark.parametrize('cls', [sparse.csr_array, sparse.csc_array])
    def test_csr_csc_tuple2_preserves_int64(self, cls):
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, 1], dtype=cupy.int64)
        A = cls((data, (row, col)), shape=(3, 3))
        assert A.indices.dtype == cupy.int64
        assert A.indptr.dtype == cupy.int64

    def test_coo_tuple2_preserves_int64(self):
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, 1], dtype=cupy.int64)
        A = sparse.coo_array((data, (row, col)), shape=(3, 3))
        assert A.row.dtype == cupy.int64
        assert A.col.dtype == cupy.int64

    def test_csr_matrix_tuple2_still_downcasts(self):
        # Sanity: matrix kind keeps the existing check_contents
        # downcast behaviour -- we only changed the array path.
        data = cupy.array([1.0, 2.0])
        row = cupy.array([0, 1], dtype=cupy.int64)
        col = cupy.array([0, 1], dtype=cupy.int64)
        M = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
        assert M.indices.dtype == cupy.int32

    # C2 -- dia_array constructor accepts dense + sparse cupy inputs.
    def test_dia_array_from_cupy_dense(self):
        A = sparse.dia_array(cupy.eye(3))
        assert A.format == 'dia'
        cupy.testing.assert_array_equal(A.toarray(), cupy.eye(3))

    def test_dia_array_from_cupy_sparse(self):
        # CSR/CSC/COO -> DIA via the new constructor entry.
        A = sparse.dia_array(sparse.csr_array(cupy.eye(3)))
        assert A.format == 'dia'
        cupy.testing.assert_array_equal(A.toarray(), cupy.eye(3))
        B = sparse.dia_array(sparse.coo_array(cupy.eye(3)))
        assert B.format == 'dia'
        cupy.testing.assert_array_equal(B.toarray(), cupy.eye(3))

    def test_dia_array_from_dia_matrix(self):
        # Promotes matrix -> array, preserving the diagonal layout.
        M = sparse.dia_matrix(cupy.eye(3))
        A = sparse.dia_array(M)
        assert isinstance(A, sparse.sparray)
        assert A.format == 'dia'

    def test_dia_array_from_shape_tuple(self):
        A = sparse.dia_array((3, 4))
        assert A.format == 'dia'
        assert A.shape == (3, 4)
        assert A.nnz == 0

    def test_dia_array_dense_layout_matches_scipy(self):
        # SciPy DIA layout: data[i, j] is the value at column j on the
        # i-th stored diagonal -- verify by round-trip.
        A_np = numpy.array(
            [[1, 2, 0, 0],
             [3, 4, 5, 0],
             [0, 6, 7, 8],
             [0, 0, 9, 10]], dtype=float)
        A_cp = cupy.asarray(A_np)
        A_dia = sparse.dia_array(A_cp)
        cupy.testing.assert_array_equal(A_dia.toarray(), A_cp)

    # C3 -- eye_array / diags_array now return dia.
    def test_eye_array_returns_dia(self):
        A = sparse.eye_array(5)
        assert A.format == 'dia'
        assert isinstance(A, sparse.sparray)

    def test_diags_array_returns_dia(self):
        A = sparse.diags_array([1, 2, 3], dtype=float)
        assert A.format == 'dia'

    def test_eye_array_format_override_still_works(self):
        # Caller can still ask for csr/coo via format=...
        assert sparse.eye_array(5, format='csr').format == 'csr'
        assert sparse.eye_array(5, format='coo').format == 'coo'

    def test_eye_array_storage_savings(self):
        # The whole point of DIA: storage is O(1) for diagonal, not O(N).
        A = sparse.eye_array(1000)
        assert A.data.size == 1000   # one diagonal of length N
        assert A.offsets.size == 1   # one stored diagonal

    # C4 -- DIA scalar mul/div preserves format.
    def test_dia_array_scalar_mul_preserves_format(self):
        A = sparse.dia_array(cupy.eye(3))
        B = A * 2.0
        assert B.format == 'dia'
        assert isinstance(B, sparse.dia_array)
        cupy.testing.assert_array_equal(B.toarray(), 2 * cupy.eye(3))

    def test_dia_array_scalar_truediv_preserves_format(self):
        A = sparse.dia_array(cupy.eye(3) * 4.0)
        B = A / 2.0
        assert B.format == 'dia'
        assert isinstance(B, sparse.dia_array)
        cupy.testing.assert_array_equal(B.toarray(), 2 * cupy.eye(3))

    # C5b / T3 -- fancy indexing must raise on out-of-bounds rather
    # than silently wrapping via modulo (silent data corruption on
    # ``__setitem__``).
    def test_fancy_getitem_raises_on_out_of_range(self):
        A = sparse.csr_array(cupy.array(
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
        with pytest.raises(IndexError):
            A[cupy.array([10, 20]), :].toarray()
        with pytest.raises(IndexError):
            A[cupy.array([-100]), :].toarray()
        with pytest.raises(IndexError):
            A[:, cupy.array([5])].toarray()

    def test_fancy_setitem_raises_on_out_of_range(self):
        # Worst case: silently writing to wrong rows.  T3 reproducer.
        A = sparse.csr_array(cupy.zeros((3, 3)))
        with pytest.raises(IndexError):
            A[cupy.array([10, 20]), cupy.array([0, 1])] = 99.0

    def test_fancy_indexing_inrange_negative_still_wraps(self):
        # In-range negatives must continue wrapping -- only
        # *out-of-range* indices raise.
        A = sparse.csr_array(cupy.array(
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
        # A[-1] == A[2]
        out = A[cupy.array([-1]), :].toarray()
        cupy.testing.assert_array_equal(out, cupy.array([[7., 8., 9.]]))

    def test_fancy_indexing_nonneg_skips_modulo(self):
        # L4c -- when all indices are non-negative and in range, the
        # ``x % length`` kernel is skipped (no observable behavior
        # change, just a perf win).  Lock in by checking that
        # non-negative inputs round-trip exactly.
        A = sparse.csr_array(cupy.array(
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
        out = A[cupy.array([0, 2]), :].toarray()
        cupy.testing.assert_array_equal(
            out, cupy.array([[1., 2., 3.], [7., 8., 9.]]))

    # C8 -- ``_from_parts`` validates ``data.size == int(indptr[-1])``
    # by default; ``_skip_buffer_check=True`` is the documented escape
    # hatch for callers that intentionally construct with slack.
    def test_from_parts_validates_buffer_size(self):
        data = cupy.array([1., 2., 3., 99., 88.])
        indices = cupy.array([0, 1, 2, 0, 1], dtype='i')
        indptr = cupy.array([0, 1, 2, 3], dtype='i')
        with pytest.raises(ValueError, match='indptr'):
            sparse.csr_array._from_parts(data, indices, indptr, (3, 3))
        # Same for CSC.
        with pytest.raises(ValueError, match='indptr'):
            sparse.csc_array._from_parts(data, indices, indptr, (3, 3))

    def test_from_parts_skip_buffer_check_allows_slack(self):
        # ``prune()`` is the documented escape: build with slack,
        # trim later.  Caller uses ``_skip_buffer_check=True``.
        data = cupy.array([1., 2., 3., 99.])
        indices = cupy.array([0, 1, 2, 0], dtype='i')
        indptr = cupy.array([0, 1, 2, 3], dtype='i')
        A = sparse.csr_array._from_parts(
            data, indices, indptr, (3, 3),
            _skip_buffer_check=True)
        assert A.indices.size == 4    # buffer has slack
        assert int(A.indptr[-1]) == 3   # logical entry count
        A.prune()
        assert A.indices.size == 3

    # F3 -- coo._from_parts validates length / dtype invariants.
    def test_coo_from_parts_length_validation(self):
        with pytest.raises(ValueError, match='same length'):
            sparse.coo_array._from_parts(
                cupy.array([1., 2.]),
                cupy.array([0], dtype=cupy.int64),
                cupy.array([0], dtype=cupy.int64),
                (2, 3))

    def test_coo_from_parts_dtype_validation(self):
        with pytest.raises(ValueError, match='same dtype'):
            sparse.coo_array._from_parts(
                cupy.array([1., 2.]),
                cupy.array([0, 1], dtype=cupy.int32),
                cupy.array([0, 1], dtype=cupy.int64),
                (2, 3))

    def test_coo_from_parts_nonneg_shape(self):
        # CSR/CSC ``_from_parts`` reject negative shape implicitly via
        # the ``indptr.size != major + 1`` check; COO has no indptr so
        # an explicit shape>=0 guard is needed to keep the contract
        # symmetric.
        with pytest.raises(ValueError, match='non-negative'):
            sparse.coo_array._from_parts(
                cupy.array([1.]),
                cupy.array([0], dtype=cupy.int32),
                cupy.array([0], dtype=cupy.int32),
                (-3, 5))
        with pytest.raises(ValueError, match='non-negative'):
            sparse.coo_array._from_parts(
                cupy.array([1.]),
                cupy.array([0], dtype=cupy.int32),
                cupy.array([0], dtype=cupy.int32),
                (3, -5))

    # F9 -- getnnz(axis=) handles degenerate (zero-row / zero-col)
    # shapes without crashing.  The empty-axis branch returns an empty
    # vector; the non-empty axis returns a zero vector of the right
    # length.
    def test_getnnz_axis_degenerate_shape(self):
        for cls in (sparse.csr_array, sparse.csc_array,
                    sparse.coo_array):
            A = cls((0, 5))
            cupy.testing.assert_array_equal(
                A._getnnz(axis=0), cupy.zeros(5, dtype=cupy.intp))
            assert A._getnnz(axis=1).size == 0
            B = cls((5, 0))
            assert B._getnnz(axis=0).size == 0
            cupy.testing.assert_array_equal(
                B._getnnz(axis=1), cupy.zeros(5, dtype=cupy.intp))

    # M11 (extension) -- COO bool * int promotes to float64 too (the
    # existing parametrized test only covered csr_array and dia_array).
    def test_coo_bool_times_int_promotes_to_supported_dtype(self):
        A = sparse.coo_array(cupy.array([[True, False], [False, True]]))
        B = A * 2
        assert B.dtype == cupy.float64
        # Round-trip through other sparse ops to confirm the result
        # is usable.
        _ = (B + B)

    # _spbase.__truediv__ scalar short-circuit preserves format for
    # CSC / COO / DIA (the C4 fix originally targeted DIA but the same
    # base-class change benefits CSC and COO too -- they previously
    # collapsed to CSR via ``self.tocsr().__truediv__``).
    @pytest.mark.parametrize('cls', [
        sparse.csc_array, sparse.coo_array, sparse.dia_array])
    def test_scalar_truediv_preserves_format(self, cls):
        if cls is sparse.dia_array:
            A = cls((cupy.array([[4., 4., 4.]]), cupy.array([0])),
                    shape=(3, 3))
        else:
            A = cls(cupy.eye(3) * 4.0)
        B = A / 2.0
        assert type(B) is cls, (
            f'scalar __truediv__ must preserve format: '
            f'expected {cls.__name__}, got {type(B).__name__}')
        cupy.testing.assert_array_equal(
            B.toarray(), 2 * cupy.eye(3))

    # F2 (round-2 port from 1a) -- ``_indptr_to_coo`` rebuilt around
    # ``searchsorted`` to drop the O(major_axis) ``arange`` allocation.
    # The behaviour must still match the legacy ``repeat`` recipe.
    def test_indptr_to_coo_searchsorted_matches_legacy(self):
        from cupyx.cusparse import _indptr_to_coo
        # Exercise a few common indptr shapes (empty, 1-per-row,
        # variable, all-on-row-0, int32 vs int64).
        for indptr_list, expected in [
            ([0],                   []),
            ([0, 1, 2, 3],          [0, 1, 2]),
            ([0, 2, 5, 5, 7],       [0, 0, 1, 1, 1, 3, 3]),
            ([0, 5, 5, 5],          [0, 0, 0, 0, 0]),
        ]:
            for dtype in (cupy.int32, cupy.int64):
                indptr = cupy.array(indptr_list, dtype=dtype)
                got = _indptr_to_coo(indptr)
                assert got.dtype == dtype
                cupy.testing.assert_array_equal(
                    got, cupy.array(expected, dtype=dtype))

    # DIA + DIA preserves DIA format (completes C4: previously only
    # the scalar-mul path was format-preserving; ``A + A`` collapsed
    # to CSR).
    def test_dia_add_dia_preserves_format(self):
        A = sparse.dia_array(cupy.eye(5))
        B = A + A
        assert isinstance(B, sparse.dia_array)
        cupy.testing.assert_array_equal(B.toarray(), 2 * cupy.eye(5))

    def test_dia_sub_dia_preserves_format(self):
        A = sparse.dia_array(cupy.eye(5) * 3)
        B = sparse.dia_array(cupy.eye(5))
        C = A - B
        assert isinstance(C, sparse.dia_array)
        cupy.testing.assert_array_equal(
            C.toarray(), 2 * cupy.eye(5))

    def test_dia_add_different_offsets(self):
        # Offsets differ -- exercises the union path.
        A1 = sparse.dia_array(
            (cupy.array([[1., 2., 3., 4., 5.]]), cupy.array([0])),
            shape=(5, 5))
        A2 = sparse.dia_array(
            (cupy.array([[10., 20., 30., 40.]]), cupy.array([1])),
            shape=(5, 5))
        B = A1 + A2
        assert isinstance(B, sparse.dia_array)
        # Match scipy by round-tripping to dense.
        import scipy.sparse as ssp
        sp_A1 = ssp.dia_array(
            (numpy.array([[1., 2., 3., 4., 5.]]), numpy.array([0])),
            shape=(5, 5))
        sp_A2 = ssp.dia_array(
            (numpy.array([[10., 20., 30., 40.]]), numpy.array([1])),
            shape=(5, 5))
        cupy.testing.assert_array_equal(
            B.toarray(), cupy.asarray((sp_A1 + sp_A2).toarray()))

    def test_dia_add_scalar_zero_preserves_format(self):
        A = sparse.dia_array(cupy.eye(3))
        B = A + 0
        assert isinstance(B, sparse.dia_array)
        cupy.testing.assert_array_equal(B.toarray(), cupy.eye(3))

    def test_dia_add_non_dia_falls_back_to_csr(self):
        # DIA + CSR drops to the base CSR path (matches scipy).
        A_dia = sparse.dia_array(cupy.eye(3))
        A_csr = sparse.csr_array(cupy.eye(3))
        B = A_dia + A_csr
        # scipy returns csr_array here -- verify we match.
        import scipy.sparse as ssp
        sp_dia = ssp.dia_array(numpy.eye(3))
        sp_csr = ssp.csr_array(numpy.eye(3))
        sp_B = sp_dia + sp_csr
        assert type(B).__name__ == type(sp_B).__name__

    # F6 / C7 -- bmat preserves a unanimous CSR/CSC input format.
    # ``hstack(csr, csr)`` previously returned a ``coo_array`` because
    # the M=1 + all-CSC fast path didn't match.  Now the slow path
    # rebinds ``format`` to the unanimous input.
    def test_hstack_csr_returns_csr(self):
        A = sparse.csr_array(cupy.eye(3))
        B = sparse.csr_array(cupy.eye(3))
        result = sparse.hstack([A, B])
        assert result.format == 'csr', (
            f'hstack(csr, csr) must keep csr; got {result.format}')

    def test_vstack_csc_returns_csc(self):
        A = sparse.csc_array(cupy.eye(3))
        B = sparse.csc_array(cupy.eye(3))
        result = sparse.vstack([A, B])
        assert result.format == 'csc'

    def test_bmat_2x2_grid_all_csr_returns_csr(self):
        A = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        result = sparse.bmat([[A, A], [A, A]])
        assert result.format == 'csr'

    def test_bmat_mixed_formats_falls_back_to_coo(self):
        # No unanimous format -> COO (the slow path's natural output).
        A = sparse.csr_array(cupy.eye(2))
        B = sparse.csc_array(cupy.eye(2))
        result = sparse.bmat([[A, B]])
        assert result.format == 'coo'

    # ``bmat([])`` -- previously crashed with IndexError; now returns
    # an empty 0x0 sparse object (more graceful than scipy's
    # ValueError, but consistent with CuPy's "permissive empty" pattern).
    def test_bmat_empty_returns_empty_coo(self):
        result = sparse.bmat([])
        assert result.shape == (0, 0)
        assert result.nnz == 0

    # __imul__ / __itruediv__ now upcast bool * int to float64 (the
    # scalar __mul__ path already did this; the in-place ops were
    # missing the upcast and crashed with "Cannot cast" errors).
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.coo_array, sparse.dia_array])
    def test_iadd_imul_idiv_bool_upcasts_to_float64(self, cls):
        if cls is sparse.dia_array:
            A = cls((cupy.array([[True, True, True]]), cupy.array([0])),
                    shape=(3, 3))
        else:
            A = cls(cupy.array([[True, False], [False, True]]))
        old = A
        A *= 2
        assert A is old
        assert A.dtype == cupy.float64
        # And in-place divide also works on the now-float64 buffer.
        A /= 2
        assert A is old

    def test_imul_bool_div_float_preserves_identity(self):
        # ``bool /= 2`` previously crashed because the implementation
        # assumed ``self.data *= recip`` with a same-dtype scalar; the
        # upcast path now reassigns ``self.data`` when needed.
        A = sparse.csr_array(cupy.array([[True, False], [False, True]]))
        old = A
        A /= 2
        assert A is old
        assert A.dtype == cupy.float64

    # V2-9: non-in-place ``bool / int`` used to silently produce an
    # all-zero result.  ``cupy.result_type(bool, 2) == int64`` and
    # ``cupy.reciprocal(2, dtype=int64) == 0``, so the CSR override
    # multiplied by zero.  Promote out-of-set dtypes to float64.
    def test_truediv_bool_int_promotes_to_float64(self):
        A = sparse.csr_array(cupy.array([[True, False], [False, True]]))
        B = A / 2
        assert B.dtype == cupy.float64
        cupy.testing.assert_array_equal(B.data, cupy.array([0.5, 0.5]))

    def test_truediv_bool_int_csr_matrix(self):
        # Same fix applies to csr_matrix (shares the override).
        A = sparse.csr_matrix(cupy.array([[True, False], [False, True]]))
        B = A / 2
        assert B.dtype == cupy.float64
        cupy.testing.assert_array_equal(B.data, cupy.array([0.5, 0.5]))

    def test_truediv_float32_int_keeps_float64(self):
        # scipy upcasts ``float32 / scalar -> float64``; the original
        # CSR override preserved this and the V2-9 rewrite must too.
        A = sparse.csr_array(
            cupy.array([[1.0, 2.0]], dtype=cupy.float32))
        B = A / 2
        assert B.dtype == cupy.float64

    def test_truediv_complex_unchanged(self):
        # Complex dtypes are already in the cuSPARSE-supported set,
        # so they should pass through (no upcast).
        A = sparse.csr_array(cupy.array([[1+2j, 0], [0, 3+4j]]))
        B = A / 2
        assert B.dtype == cupy.complex128

    # F15 -- nanmax / nanmin (axis=None only); matches scipy.
    def test_nanmax_basic(self):
        A = sparse.csr_array(cupy.array(
            [[1.0, 2.0], [cupy.nan, 4.0]]))
        assert float(A.nanmax()) == 4.0

    def test_nanmin_basic(self):
        A = sparse.csr_array(cupy.array(
            [[1.0, 2.0], [cupy.nan, 4.0]]))
        assert float(A.nanmin()) == 1.0

    def test_nanmax_all_nan_with_implicit_zeros(self):
        # NaN entries plus implicit zeros: scipy returns 0 (the
        # implicit zero dominates the all-NaN explicit set).
        A = sparse.csr_array(cupy.array(
            [[cupy.nan, 0.0], [0.0, cupy.nan]]))
        assert float(A.nanmax()) == 0.0

    def test_nanmax_all_nan_no_implicit_zeros(self):
        # Fully-stored all-NaN: result is NaN.
        A = sparse.csr_array(cupy.array(
            [[cupy.nan, cupy.nan], [cupy.nan, cupy.nan]]))
        result = A.nanmax()
        assert cupy.isnan(result)

    def test_nanmax_explicit_ignores_implicit_zero(self):
        A = sparse.csr_array(cupy.array(
            [[1.0, cupy.nan], [3.0, 4.0]]))
        assert float(A.nanmax(explicit=True)) == 4.0

    def test_nanmax_axis_raises_notimplemented(self):
        A = sparse.csr_array(cupy.eye(3))
        with pytest.raises(NotImplementedError):
            A.nanmax(axis=0)
        with pytest.raises(NotImplementedError):
            A.nanmin(axis=1)

    def test_nanmax_out_raises_value_error(self):
        A = sparse.csr_array(cupy.eye(3))
        with pytest.raises(ValueError):
            A.nanmax(out=cupy.zeros(()))

    # DIA todia data shape now matches scipy: ``(num_diags,
    # col.max()+1)``, not ``(num_diags, max(M, N))``.  Avoids
    # storing zero-padded trailing columns when entries are
    # left-aligned.
    def test_coo_todia_width_matches_scipy(self):
        import scipy.sparse as ssp
        A_np = numpy.array([[1, 2, 0, 0], [3, 4, 0, 0]], dtype=float)
        A_cp = cupy.asarray(A_np)
        cp_dia = sparse.coo_array(A_cp).todia()
        sp_dia = ssp.coo_array(A_np).todia()
        assert cp_dia.data.shape == sp_dia.data.shape
        cupy.testing.assert_array_equal(
            cp_dia.data, cupy.asarray(sp_dia.data))

    def test_dia_array_dense_width_matches_scipy(self):
        import scipy.sparse as ssp
        A_np = numpy.array([[1, 2, 0, 0], [3, 4, 0, 0]], dtype=float)
        A_cp = cupy.asarray(A_np)
        cp_dia = sparse.dia_array(A_cp)
        sp_dia = ssp.dia_array(A_np)
        assert cp_dia.data.shape == sp_dia.data.shape

    # F11 -- ``csc @ csc`` returns CSC (matches scipy).  cuSPARSE
    # SpGEMM is CSR-only, so we compute as ``(B^T @ A^T)^T`` -- the
    # transpose of CSC is CSR, the gemm runs in CSR, and the final
    # ``.T`` flips back to CSC.
    def test_csc_array_matmul_csc_array(self):
        A = sparse.csc_array(cupy.array([[1., 0.], [0., 2.]]))
        B = sparse.csc_array(cupy.array([[1., 2.], [3., 4.]]))
        result = A @ B
        assert isinstance(result, sparse.csc_array)
        assert result.format == 'csc'
        cupy.testing.assert_array_equal(
            result.toarray(),
            cupy.array([[1., 2.], [6., 8.]]))

    def test_csc_matrix_matmul_csc_matrix_preserves_csc(self):
        A = sparse.csc_matrix(cupy.array([[1., 0.], [0., 2.]]))
        B = sparse.csc_matrix(cupy.array([[1., 2.], [3., 4.]]))
        result = A @ B
        assert isinstance(result, sparse.csc_matrix)
        assert result.format == 'csc'

    def test_csc_matmul_csr_returns_csr(self):
        # Mixed: csc @ csr stays in csr (no transpose round-trip).
        A = sparse.csc_array(cupy.array([[1., 0.], [0., 2.]]))
        B = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        result = A @ B
        assert result.format == 'csr'

    def test_csc_int64_matmul_csc_preserves_csc(self):
        # int64 indices route through spgemm; the (B^T @ A^T)^T trick
        # still works.
        A = sparse.csc_array(cupy.array([[1., 0.], [0., 2.]]))
        A.indices = A.indices.astype(cupy.int64)
        A.indptr = A.indptr.astype(cupy.int64)
        B = sparse.csc_array(cupy.array([[1., 2.], [3., 4.]]))
        B.indices = B.indices.astype(cupy.int64)
        B.indptr = B.indptr.astype(cupy.int64)
        result = A @ B
        assert isinstance(result, sparse.csc_array)
        assert result.indices.dtype == cupy.int64

    # ``spmatrix.__mul__`` (matrix kind, where ``*`` is matmul) now
    # short-circuits scalars to ``_mul_scalar``.  Before this fix,
    # ``dia_matrix * 2`` collapsed to ``csr_matrix`` because the
    # base ``_matmul_dispatch`` routed through ``self.tocsr()``.
    def test_dia_matrix_scalar_mul_preserves_format(self):
        A = sparse.dia_matrix(
            (cupy.array([[1., 2., 3.]]), cupy.array([0])),
            shape=(3, 3))
        B = A * 2
        assert isinstance(B, sparse.dia_matrix)
        cupy.testing.assert_array_equal(
            B.toarray(), 2 * cupy.diag(cupy.array([1., 2., 3.])))

    def test_coo_matrix_scalar_mul_preserves_format(self):
        A = sparse.coo_matrix(cupy.eye(3))
        B = A * 2
        assert isinstance(B, sparse.coo_matrix)

    def test_csc_matrix_scalar_mul_preserves_format(self):
        A = sparse.csc_matrix(cupy.eye(3))
        B = A * 2
        assert isinstance(B, sparse.csc_matrix)

    def test_csr_matrix_matmul_still_works(self):
        # Sanity: scalar short-circuit doesn't break matrix-matmul.
        A = sparse.csr_matrix(cupy.eye(3))
        B = sparse.csr_matrix(cupy.eye(3))
        result = A * B  # matrix kind: ``*`` is matmul
        assert isinstance(result, sparse.csr_matrix)
        cupy.testing.assert_array_equal(result.toarray(), cupy.eye(3))

    # M5 -- ``__class_getitem__`` for typing aliases (scipy 1.16+).
    def test_class_getitem_creates_alias(self):
        import types
        alias = sparse.coo_array[int, tuple[int]]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is sparse.coo_array

    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array,
        sparse.coo_array, sparse.dia_array,
        sparse.csr_matrix, sparse.csc_matrix,
        sparse.coo_matrix, sparse.dia_matrix,
    ])
    def test_class_getitem_works_for_all_types(self, cls):
        alias = cls[int, int]
        assert alias.__origin__ is cls

    # H2 -- setitem with column-shape RHS (1-D ndarray into a column
    # slice).  The previous ``cupy.broadcast_arrays(x, i)`` always
    # produced the union shape, which couldn't be reshaped back to
    # the index shape.  Now uses scipy's squeeze-comparison to skip
    # the union when the squeezed shapes already match.
    def test_setitem_column_with_1d_rhs(self):
        A = sparse.csr_array(cupy.zeros((3, 3)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            A[:, 0] = cupy.array([1., 2., 3.])
        cupy.testing.assert_array_equal(
            A.toarray()[:, 0], cupy.array([1., 2., 3.]))

    def test_setitem_row_with_1d_rhs(self):
        A = sparse.csr_array(cupy.zeros((3, 3)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            A[0, :] = cupy.array([1., 2., 3.])
        cupy.testing.assert_array_equal(
            A.toarray()[0, :], cupy.array([1., 2., 3.]))

    def test_setitem_column_with_scalar(self):
        # Sanity: scalar broadcast still works.
        A = sparse.csr_array(cupy.zeros((3, 3)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            A[:, 0] = 5.0
        cupy.testing.assert_array_equal(
            A.toarray()[:, 0], cupy.array([5., 5., 5.]))

    # F15 -- check_format
    def test_check_format_valid(self):
        A = sparse.csr_array(cupy.eye(3))
        A.check_format()  # no raise
        A.check_format(full_check=False)

    def test_check_format_detects_bad_indptr_start(self):
        A = sparse.csr_array(cupy.eye(3))
        A.indptr = cupy.array([5, 1, 2, 3], dtype=cupy.int32)
        with pytest.raises(
                ValueError, match='index pointer should start with 0'):
            A.check_format()

    def test_check_format_detects_out_of_bounds_indices(self):
        A = sparse.csr_array(cupy.eye(3))
        A.indices = cupy.array([0, 5, 2], dtype=cupy.int32)
        with pytest.raises(ValueError, match='indices must be < 3'):
            A.check_format()

    def test_check_format_detects_indptr_size_mismatch(self):
        A = sparse.csr_array(cupy.eye(3))
        A.indptr = cupy.array([0, 1, 2, 3, 4, 5], dtype=cupy.int32)
        with pytest.raises(ValueError, match='index pointer size'):
            A.check_format()

    def test_check_format_warns_non_integer_dtype(self):
        # Non-integer dtype warns, doesn't raise (matches scipy).
        A = sparse.csr_array(cupy.eye(3))
        A.indptr = A.indptr.astype(cupy.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                A.check_format()
            except ValueError:
                pass  # Subsequent error from non-int indptr is fine
            assert any('non-integer dtype' in str(x.message) for x in w)

    def test_check_format_full_check_false_skips_gpu_scans(self):
        # full_check=False keeps the cheap structural checks but
        # doesn't raise on out-of-range indices (the GPU scan is
        # skipped).  Useful for hot-path validation.
        A = sparse.csr_array(cupy.eye(3))
        A.indices = cupy.array([0, 5, 2], dtype=cupy.int32)
        # Cheap path passes.
        A.check_format(full_check=False)
        # Full path raises.
        with pytest.raises(ValueError):
            A.check_format(full_check=True)

    # F15 -- resize (CSR / CSC)
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array,
        sparse.csr_matrix, sparse.csc_matrix,
    ])
    def test_compressed_resize_smaller_truncates(self, cls):
        A = cls(cupy.eye(5))
        A.resize(3, 3)
        assert A.shape == (3, 3)
        cupy.testing.assert_array_equal(A.toarray(), cupy.eye(3))

    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array])
    def test_compressed_resize_larger_keeps_entries(self, cls):
        A = cls(cupy.eye(3))
        A.resize(5, 5)
        assert A.shape == (5, 5)
        assert A.nnz == 3
        expected = cupy.zeros((5, 5))
        expected[:3, :3] = cupy.eye(3)
        cupy.testing.assert_array_equal(A.toarray(), expected)

    def test_compressed_resize_drops_minor_axis_entries(self):
        A = sparse.csr_array(cupy.array(
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
        A.resize((3, 2))
        # Columns 0-1 kept; column 2 dropped.
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.array([[1., 2.], [4., 5.], [7., 8.]]))

    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array, sparse.coo_array])
    def test_resize_preserves_canonical_format(self, cls):
        # Resize is a structural truncation/pad; it does not reorder
        # or introduce duplicates, so canonical-ness is preserved.
        # Verified empirically against scipy 1.17.
        A = cls(cupy.eye(5))
        # Make sure canonical is cached True.
        assert A.has_canonical_format
        A.resize(3, 3)
        # Should still report canonical (without re-running the kernel).
        assert A.has_canonical_format

    def test_resize_preserves_non_canonical(self):
        # Sanity: if input was non-canonical, output stays
        # non-canonical (we don't accidentally claim canonical).
        A = sparse.coo_array._from_parts(
            cupy.array([1., 1., 2.]),
            cupy.array([0, 0, 1], dtype='i'),
            cupy.array([0, 0, 1], dtype='i'),
            (3, 3),
            has_canonical_format=False)
        A.resize(2, 2)
        assert A.has_canonical_format is False

    # F15 -- COO resize
    def test_coo_resize_smaller(self):
        A = sparse.coo_array(cupy.eye(5))
        A.resize(3, 3)
        assert A.shape == (3, 3)
        cupy.testing.assert_array_equal(A.toarray(), cupy.eye(3))

    def test_coo_resize_larger(self):
        A = sparse.coo_array(cupy.eye(3))
        A.resize(5, 5)
        assert A.shape == (5, 5)
        assert A.nnz == 3

    def test_coo_resize_matches_scipy(self):
        import scipy.sparse as ssp
        A_np = numpy.array(
            [[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]])
        sA = ssp.coo_array(A_np)
        sA.resize(2, 3)
        A = sparse.coo_array(cupy.asarray(A_np))
        A.resize(2, 3)
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.asarray(sA.toarray()))

    # CSR _mul_scalar no longer mutates self via sum_duplicates side
    # effect (matches scipy).  Pre-fix, ``A * 2`` would canonicalize
    # ``A`` even when the caller didn't read it back.
    def test_csr_scalar_mul_does_not_canonicalize_self(self):
        # Build a CSR with explicit duplicates, scale, verify ``A``
        # is left non-canonical.
        data = cupy.array([1.0, 1.0, 2.0])
        indices = cupy.array([0, 0, 1], dtype='i')
        indptr = cupy.array([0, 2, 3], dtype='i')
        A = sparse.csr_array._from_parts(
            data, indices, indptr, (2, 2),
            has_canonical_format=False,
            _skip_buffer_check=True)
        A_data_before = A.data.copy()
        _ = A * 2
        # A.data unchanged; canonical cache untouched.
        cupy.testing.assert_array_equal(A.data, A_data_before)
        assert A.has_canonical_format is False

    # ----- Red-team v2 regression tests (V2-1 through V2-25) -----

    # V2-1 -- ``A.power(0)`` raises (was silently producing wrong
    # sparse result; ``__pow__(0)`` already raised but the method
    # itself didn't, so users calling ``A.power(0)`` directly got
    # silent wrong results).  Use ``csr_array`` since
    # ``csr_matrix.__pow__`` is matmul-power and rejects non-square
    # before ever calling ``power``.
    def test_power_zero_raises(self):
        A = sparse.csr_array(cupy.array([[1.0, 0, 2.0], [0, 3.0, 0]]))
        # Method path raises directly.
        with pytest.raises(NotImplementedError, match='zero power'):
            A.power(0)
        # Operator path on sparse_array goes through ``_spbase.__pow__``
        # which is element-wise; same error.
        with pytest.raises(NotImplementedError, match='zero power'):
            A ** 0

    # V2-20 -- ``A.power(array)`` and ``A ** array`` both raise
    # ``NotImplementedError('input is not scalar')``.  Without the
    # scalar guard inside ``power``, the array exponent triggered
    # ``cupy.array(...) == 0`` which raises a confusing
    # "truth value of an array ... is ambiguous" message.
    def test_power_array_exponent_raises(self):
        A = sparse.csr_array(cupy.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(NotImplementedError, match='not scalar'):
            A.power(cupy.array([1, 2]))
        with pytest.raises(NotImplementedError, match='not scalar'):
            A ** cupy.array([1, 2])

    # V2-2 -- ``kron(int64_sparray, int64_sparray)`` preserves int64
    # (was downcasting to int32 because the dtype choice was based
    # on output shape, ignoring sparray input dtypes).
    def test_kron_sparray_preserves_int64(self):
        A = sparse.eye_array(8, format='csr', dtype=cupy.float64)
        A_int64 = sparse.csr_array._from_parts(
            A.data, A.indices.astype(cupy.int64),
            A.indptr.astype(cupy.int64), A.shape)
        K = sparse.kron(A_int64, A_int64)
        assert K.row.dtype == cupy.int64
        # Matrix path keeps the legacy int32-when-shape-fits behaviour.
        M = sparse.eye(8, format='csr', dtype=cupy.float64)
        M_int64 = sparse.csr_matrix._from_parts(
            M.data, M.indices.astype(cupy.int64),
            M.indptr.astype(cupy.int64), M.shape)
        K_mat = sparse.kron(M_int64, M_int64)
        assert K_mat.row.dtype == cupy.int32

    # V2-3 -- ``lsqr`` raises on int64 indices (cuSOLVER's csrlsvqr
    # is int32-only; previously silently corrupted with NaN/inf).
    def test_lsqr_rejects_int64(self):
        from cupyx.scipy.sparse.linalg import lsqr
        A = sparse.eye(4, format='csr', dtype=cupy.float64) * 2.0
        A_int64 = sparse.csr_matrix._from_parts(
            A.data, A.indices.astype(cupy.int64),
            A.indptr.astype(cupy.int64), A.shape)
        b = cupy.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match='int64'):
            lsqr(A_int64, b)

    # V2-4 -- ``has_canonical_format = True`` setter cascade.  Direct
    # ``c._has_canonical_format = True`` in cusparse.py wraps would
    # leave ``_has_sorted_indices`` uncached, forcing a kernel
    # launch on later reads.  Switched to the property setter which
    # propagates the implication.
    def test_has_canonical_propagates_to_sorted(self):
        A = sparse.csr_matrix(cupy.eye(3))
        # Force the cache via the property setter.
        for attr in ('_has_canonical_format', '_has_sorted_indices'):
            if hasattr(A, attr):
                delattr(A, attr)
        A.has_canonical_format = True
        assert A._has_canonical_format is True
        assert A._has_sorted_indices is True
        # Read property: no kernel run since the cache was set.
        assert A.has_sorted_indices is True

    # V2-6 -- ``setdiag(2D)`` raises a clear scipy-aligned error
    # (was producing a confusing concat / broadcast error from
    # downstream code).
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array, sparse.coo_array])
    def test_setdiag_rejects_2d_values(self, cls):
        A = cls(cupy.zeros((3, 3)))
        with pytest.raises(ValueError, match='0-d or 1-d'):
            A.setdiag(cupy.array([[1.0, 2.0], [3.0, 4.0]]))

    # V2-8 -- ``_from_parts`` rejects 2-D buffer (was sliding
    # through and breaking downstream ops with confusing errors).
    def test_from_parts_rejects_2d_buffers(self):
        with pytest.raises(ValueError, match='must be 1-D'):
            sparse.csr_matrix._from_parts(
                cupy.zeros((2, 2)),  # 2-D !
                cupy.array([0, 1, 0, 1], dtype='i'),
                cupy.array([0, 2, 4], dtype='i'),
                (2, 2),
                _skip_buffer_check=True)
        with pytest.raises(ValueError, match='must be 1-D'):
            sparse.coo_matrix._from_parts(
                cupy.zeros((2, 2)),  # 2-D !
                cupy.array([0, 1], dtype='i'),
                cupy.array([0, 1], dtype='i'),
                (2, 2))

    # V2-20 -- ``A ** array_exponent`` raises clear NotImplementedError
    # (was raising "truth value ambiguous" via array equality).
    def test_pow_with_array_exponent_raises_clear_error(self):
        A = sparse.csr_array(cupy.array([[2.0, 0, 3.0]]))
        with pytest.raises(NotImplementedError, match='not scalar'):
            A ** cupy.array([2, 3, 4])

    # V2-22 -- ``csr_int32.multiply(csr_int64)`` works (was raising
    # a kernel-template-mismatch TypeError because dtypes weren't
    # harmonised).
    def test_multiply_mixed_int32_int64(self):
        A_int32 = sparse.csr_array(cupy.array([[1.0, 2.0]]))
        B_int64 = sparse.csr_array._from_parts(
            cupy.array([2.0, 5.0]),
            cupy.array([0, 1], dtype=cupy.int64),
            cupy.array([0, 1, 2], dtype=cupy.int64),
            (2, 2))
        result = A_int32.multiply(B_int64)
        # Result dtype is the wider of the inputs.
        assert result.indices.dtype == cupy.int64

    # V2-23 -- ``csr.maximum(coo)`` / ``csr == coo`` etc. work via
    # ``other.tocsr()`` fallback (was raising NotImplementedError).
    def test_csr_maximum_with_non_csr(self):
        A_csr = sparse.csr_matrix(cupy.array([[1.0, 2.0]]))
        for other in (sparse.coo_matrix(cupy.array([[3.0, 4.0]])),
                      sparse.csc_matrix(cupy.array([[3.0, 4.0]]))):
            r = A_csr.maximum(other)
            cupy.testing.assert_array_equal(
                r.toarray(), cupy.array([[3.0, 4.0]]))

    def test_csr_comparison_with_non_csr(self):
        A_csr = sparse.csr_matrix(cupy.array([[1.0, 2.0]]))
        A_coo = sparse.coo_matrix(cupy.array([[1.0, 4.0]]))
        # Non-csr second operand falls through to ``self._comparison(
        # other.tocsr(), ...)`` instead of raising NotImplementedError.
        r = A_csr != A_coo
        # Only column 1 differs (2 vs 4).
        cupy.testing.assert_array_equal(
            r.toarray(), cupy.array([[False, True]]))

    # V2-24 -- public CSR/CSC constructor rejects ``indptr[0] != 0``
    # (was silently producing wrong results, sometimes crashing the
    # CUDA context downstream).
    def test_csr_constructor_rejects_bad_indptr_start(self):
        data = cupy.array([1.0, 2.0])
        ind = cupy.array([0, 1], dtype=cupy.int32)
        indptr = cupy.array([5, 6, 7], dtype=cupy.int32)
        with pytest.raises(ValueError, match='start with 0'):
            sparse.csr_matrix((data, ind, indptr), shape=(2, 2))

    # V2-25 -- complex ``sign()`` matches scipy 1.16+ (numpy 2.x
    # ``z / abs(z)``); cupy.sign's ``0+0j -> nan`` divergence is
    # masked with a zero check.
    def test_complex_sign_matches_scipy_modern(self):
        A = sparse.csr_matrix(cupy.array([[1+2j]]))
        # 1+2j / sqrt(5) ≈ 0.4472 + 0.8944j
        result = A.sign().data
        expected = cupy.array([1+2j], dtype=cupy.complex128) / cupy.sqrt(
            cupy.array(5, dtype=cupy.complex128))
        cupy.testing.assert_allclose(
            result, expected, rtol=1e-6)

    def test_complex_sign_zero_does_not_nan(self):
        # Explicit-zero entries must not surface NaN (cupy.sign(0+0j)
        # returns nan+nanj literally; we mask via ``where``).
        A = sparse.csr_matrix._from_parts(
            cupy.array([0+0j, 1+2j], dtype=cupy.complex128),
            cupy.array([0, 1], dtype='i'),
            cupy.array([0, 2], dtype='i'),
            (1, 2),
            _skip_buffer_check=True)
        result = A.sign().data
        assert not cupy.isnan(result).any()
        assert result[0] == 0+0j

    # F5 -- _from_parts checks that index dtype is wide enough for
    # the matrix shape.
    def test_from_parts_rejects_narrow_dtype(self):
        # 2 rows but 2**31+5 cols -- int32 not wide enough.
        with pytest.raises(ValueError, match='too large for index dtype'):
            sparse.csr_matrix._from_parts(
                cupy.array([1.0]),
                cupy.array([0], dtype=cupy.int32),
                cupy.array([0, 1, 1], dtype=cupy.int32),
                (2, 2**31 + 5))
        with pytest.raises(ValueError, match='too large for index dtype'):
            sparse.coo_matrix._from_parts(
                cupy.array([1.0]),
                cupy.array([0], dtype=cupy.int32),
                cupy.array([0], dtype=cupy.int32),
                (2, 2**31 + 5))

    # F4 -- bmat error message says "column" not "row" for column
    # mismatches.  The bmat fast paths short-circuit single-row /
    # single-column grids; trigger the slow path with a 2x2 grid
    # whose cells in column 1 disagree on shape[1].
    def test_bmat_column_error_message(self):
        from cupyx.scipy.sparse import csr_array as _ca
        # row 0: (1, 2) | (1, 3)
        # row 1: (1, 2) | (1, 4)   <-- column 1 mismatch (3 vs 4)
        A = _ca(cupy.zeros((1, 2)))
        B = _ca(cupy.zeros((1, 3)))
        C = _ca(cupy.zeros((1, 2)))
        D = _ca(cupy.zeros((1, 4)))
        with pytest.raises(
                ValueError, match='incompatible column dimensions'):
            sparse.bmat([[A, B], [C, D]])

    # F7 -- in-place scalar ops preserve object identity.
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array, sparse.dia_array])
    def test_iadd_imul_idiv_scalar_preserves_identity(self, cls):
        if cls is sparse.dia_array:
            A = cls((cupy.array([[1., 2., 3.]]), cupy.array([0])),
                    shape=(3, 3))
        else:
            A = cls(cupy.eye(3))
        old = A
        A *= 2
        assert A is old
        A /= 2
        assert A is old

    # F13 -- has_canonical_format getter propagates _has_sorted_indices.
    def test_canonical_propagates_sorted(self):
        A = sparse.csr_matrix(cupy.eye(3))
        # Strip cached flags so the kernel actually runs.
        for attr in ('_has_canonical_format', '_has_sorted_indices'):
            if hasattr(A, attr):
                delattr(A, attr)
        canonical = A.has_canonical_format
        assert canonical is True
        assert getattr(A, '_has_sorted_indices', False) is True

    # F14 -- setdiag accepts list, scalar, numpy.ndarray, cupy.ndarray.
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array, sparse.coo_array])
    def test_setdiag_accepts_python_types(self, cls):
        # list
        A = cls(cupy.zeros((4, 4)))
        A.setdiag([1, 2, 3, 4])
        cupy.testing.assert_array_equal(
            A.diagonal(), cupy.array([1., 2., 3., 4.]))
        # scalar
        A = cls(cupy.zeros((4, 4)))
        A.setdiag(5.0)
        cupy.testing.assert_array_equal(
            A.diagonal(), cupy.array([5., 5., 5., 5.]))
        # numpy ndarray
        A = cls(cupy.zeros((4, 4)))
        A.setdiag(numpy.array([10., 20., 30., 40.]))
        cupy.testing.assert_array_equal(
            A.diagonal(), cupy.array([10., 20., 30., 40.]))
        # cupy ndarray
        A = cls(cupy.zeros((4, 4)))
        A.setdiag(cupy.array([100., 200., 300., 400.]))
        cupy.testing.assert_array_equal(
            A.diagonal(), cupy.array([100., 200., 300., 400.]))

    def test_setdiag_does_not_mutate_input(self):
        # F14 fix flipped the in-place ``-=`` to ``=`` because
        # ``cupy.asarray`` no-ops when dtype already matches, leaving
        # a slice view that aliased the caller's buffer.
        A = sparse.csr_array(cupy.eye(4))
        values = cupy.array([1., 2., 3., 4.])
        original = values.copy()
        A.setdiag(values)
        cupy.testing.assert_array_equal(values, original)

    # F17 -- count_nonzero(axis=) on empty / all-explicit-zero arrays.
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array, sparse.coo_array])
    def test_count_nonzero_axis_empty(self, cls):
        A = cls((3, 5))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=0), cupy.zeros(5, dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1), cupy.zeros(3, dtype=cupy.intp))

    def test_count_nonzero_all_explicit_zeros(self):
        # CSR with stored-but-zero data.
        A = sparse.csr_matrix._from_parts(
            cupy.zeros(3),
            cupy.array([0, 1, 2], dtype='i'),
            cupy.array([0, 1, 2, 3], dtype='i'),
            (3, 3),
            has_canonical_format=True)
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=0), cupy.zeros(3, dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1), cupy.zeros(3, dtype=cupy.intp))

    # M11 -- bool * Python-int promotes to a sparse-supported dtype.
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.dia_array])
    def test_bool_times_int_promotes_to_supported_dtype(self, cls):
        if cls is sparse.dia_array:
            A = cls((cupy.array([[True, True, True]]), cupy.array([0])),
                    shape=(3, 3))
        else:
            A = cls(cupy.array([[True, False], [False, True]]))
        B = A * 2
        # int64 isn't a sparse-supported dtype; promote to float64.
        assert B.dtype == cupy.float64
        # Confirm the promoted result still flows through other
        # sparse ops without raising "dtype not supported".
        if cls is sparse.csr_array:
            _ = (B + B)
            _ = (B @ B)

    # Note: CSC bool is unsupported throughout (denseToSparse and
    # csr2cscEx2 both reject bool dtype as of cuSPARSE 12.7.9), so
    # there is no path to test ``csc_array(bool) * 2`` directly --
    # the ``_mul_scalar`` dtype upcast still applies if a CSC bool
    # were ever constructible, and the CSR / DIA cases above cover
    # the same code path.

    # M14 -- csr2coo preserves cached has_canonical_format.
    # Design: ``csr2coo`` reads the *cached* internal flag only -- it
    # does not trigger lazy canonical computation on the source CSR.
    # This avoids an extra D2H sync in the common case where the
    # caller hasn't asked about canonical-ness.  Users who want the
    # flag preserved should ensure it's cached first (either via
    # ``_from_parts(has_canonical_format=...)`` or by reading the
    # property before ``.tocoo()``).
    def test_csr_tocoo_preserves_cached_canonical(self):
        A = sparse.csr_array(cupy.eye(3))
        # Read the property to trigger lazy compute and cache.
        assert A.has_canonical_format
        B = A.tocoo()
        assert B.has_canonical_format

    def test_csr_tocoo_propagates_explicit_canonical(self):
        # When the source was built via ``_from_parts(has_canonical=True)``,
        # the flag is cached without any lazy compute.
        A = sparse.csr_array._from_parts(
            cupy.array([1., 2., 3.]),
            cupy.array([0, 1, 2], dtype='i'),
            cupy.array([0, 1, 2, 3], dtype='i'),
            (3, 3),
            has_canonical_format=True)
        B = A.tocoo()
        assert B.has_canonical_format

    def test_coo_tocsr_preserves_canonical(self):
        # Symmetric case.  Build a canonical COO directly.
        coo = sparse.coo_array._from_parts(
            cupy.array([1., 2., 3.]),
            cupy.array([0, 1, 2], dtype='i'),
            cupy.array([0, 1, 2], dtype='i'),
            (3, 3),
            has_canonical_format=True)
        csr = coo.tocsr()
        assert csr.has_canonical_format

    def test_csc_tocoo_does_not_claim_canonical(self):
        # CSC -> COO: the column-major layout of canonical CSC produces
        # column-major COO, which is *not* canonical (canonical means
        # row-major lex sort).  Verify we do not propagate the flag in
        # this direction -- otherwise downstream ops that trust
        # ``has_canonical_format`` would skip a needed
        # ``sum_duplicates`` / re-sort and produce wrong results.
        A = sparse.csc_array(cupy.eye(3))
        # Trigger lazy compute of canonical on the source.
        assert A.has_canonical_format
        coo = A.tocoo()
        assert not coo.has_canonical_format

    def test_coo_tocsc_does_not_claim_canonical(self):
        # Same direction: row-major canonical COO -> CSC is not
        # automatically canonical (CSC needs col-major).
        coo = sparse.coo_array._from_parts(
            cupy.array([1., 2., 3.]),
            cupy.array([0, 1, 2], dtype='i'),
            cupy.array([0, 1, 2], dtype='i'),
            (3, 3),
            has_canonical_format=True)
        csc = coo.tocsc()
        # CSC ``has_canonical_format`` may be lazily computed -- the
        # key thing is that we don't pre-cache True from the source.
        assert getattr(csc, '_has_canonical_format', None) is not True

    # New methods: csr.todia / coo.todia.  Indirect coverage exists
    # via dia_array(csr) / dia_array(coo) but the methods themselves
    # are public API now and worth exercising directly.
    def test_csr_todia_method(self):
        A = sparse.csr_array(cupy.array([[1., 0., 5.],
                                         [0., 2., 0.],
                                         [0., 0., 3.]]))
        D = A.todia()
        assert D.format == 'dia'
        assert isinstance(D, sparse.dia_array)
        cupy.testing.assert_array_equal(D.toarray(), A.toarray())
        # Offset layout: data[i, j] is value at row j-offset[i], col j.
        # For this matrix offsets should be {0, 2}.
        offsets = sorted(D.offsets.tolist())
        assert offsets == [0, 2]

    def test_csr_todia_sums_duplicates(self):
        # Duplicates on the same (row, col) must be summed before DIA
        # packing -- otherwise the last-write wins on the same DIA
        # slot and the value is wrong.
        A = sparse.csr_array._from_parts(
            cupy.array([1., 1., 2.]),
            cupy.array([0, 0, 1], dtype='i'),
            cupy.array([0, 2, 3], dtype='i'),
            (2, 2),
            has_canonical_format=False,
            _skip_buffer_check=True)
        D = A.todia()
        cupy.testing.assert_array_equal(
            D.toarray(), cupy.array([[2., 0.], [0., 2.]]))

    def test_coo_todia_method(self):
        A = sparse.coo_array(cupy.array([[1., 0., 5.],
                                         [0., 2., 0.],
                                         [0., 0., 3.]]))
        D = A.todia()
        assert D.format == 'dia'
        assert isinstance(D, sparse.dia_array)
        cupy.testing.assert_array_equal(D.toarray(), A.toarray())

    def test_coo_todia_empty(self):
        # Empty COO must round-trip to empty DIA cleanly.
        A = sparse.coo_array((3, 4))
        D = A.todia()
        assert D.format == 'dia'
        assert D.shape == (3, 4)
        assert D.nnz == 0

    def test_csr_todia_does_not_mutate_source(self):
        # ``coo.todia`` rebinds a local ``src`` rather than ``self`` --
        # the original CSR / COO must remain untouched even when it
        # had to be sum-deduped first.
        A = sparse.csr_array._from_parts(
            cupy.array([1., 1., 2.]),
            cupy.array([0, 0, 1], dtype='i'),
            cupy.array([0, 2, 3], dtype='i'),
            (2, 2),
            has_canonical_format=False,
            _skip_buffer_check=True)
        before_data = A.data.copy()
        before_indices = A.indices.copy()
        before_indptr = A.indptr.copy()
        _ = A.todia()
        cupy.testing.assert_array_equal(A.data, before_data)
        cupy.testing.assert_array_equal(A.indices, before_indices)
        cupy.testing.assert_array_equal(A.indptr, before_indptr)
        # has_canonical_format itself is allowed to be cached as True
        # if the lazy property fires during the conversion -- that is
        # not "mutation" of structural data.

    # eye_array with explicit format='dia'.  ``test_eye_array_returns_dia``
    # covers the default; this one verifies the explicit format hint
    # round-trips through the C2-fixed DIA constructor.
    def test_eye_array_explicit_dia_format(self):
        A = sparse.eye_array(5, format='dia')
        assert A.format == 'dia'
        assert isinstance(A, sparse.dia_array)
        cupy.testing.assert_array_equal(A.toarray(), cupy.eye(5))

    # _to_array regression: every container constructor must accept
    # any sparse object now that the NotImplementedError fallback is
    # gone.  Cycle through all kind/format combinations.
    def test_to_array_cross_format_construction(self):
        from cupyx.scipy.sparse._construct import _to_array
        m_csr = sparse.csr_matrix(cupy.eye(3))
        for fmt, expected_cls in [
                ('csr', sparse.csr_array),
                ('csc', sparse.csc_array),
                ('coo', sparse.coo_array),
                ('dia', sparse.dia_array),
        ]:
            out = _to_array(m_csr, format=fmt)
            assert isinstance(out, expected_cls), (
                f'{fmt}: expected {expected_cls.__name__}, '
                f'got {type(out).__name__}')

    # _from_parts with major axis = 0 (empty matrix) edge case.
    def test_from_parts_zero_major_axis(self):
        A = sparse.csr_array._from_parts(
            cupy.empty(0),
            cupy.empty(0, dtype='i'),
            cupy.array([0], dtype='i'),
            (0, 3))
        assert A.shape == (0, 3)
        assert A.nnz == 0
        # Round-trip through arithmetic.
        B = A + A
        assert B.shape == (0, 3)

    # _from_parts default validation does not regress on tight-buffer
    # construction (the most common case).
    def test_from_parts_tight_buffer_succeeds(self):
        # Default behaviour: data.size == int(indptr[-1]).
        A = sparse.csr_array._from_parts(
            cupy.array([1., 2., 3.]),
            cupy.array([0, 1, 2], dtype='i'),
            cupy.array([0, 1, 2, 3], dtype='i'),
            (3, 3))
        assert A.nnz == 3
        assert int(A.indptr[-1]) == 3

    # _with_data preserves the inductive tight-buffer invariant
    # without triggering re-validation: the new validation path must
    # not break existing arithmetic that lands here.
    def test_with_data_preserves_invariant(self):
        # Build a tight CSR, run a chain of arithmetic ops, and
        # confirm the buffer stays tight throughout.  Any silent
        # regression that drops _skip_buffer_check=True from
        # _with_data would still pass this test (validation passes
        # inductively), but a regression that breaks the invariant
        # would surface here as a ValueError from the next op.
        A = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        for _ in range(5):
            A = A * 2.0
            A = A.copy()
            A = -A
            A = A.astype(cupy.float32)
        # Each op landed back through ``_with_data`` -> ``_from_parts``;
        # the chain succeeding is the assertion.
        assert A.shape == (2, 2)

    # L4e (pyproject.toml dedup): regression test that we don't have
    # duplicate FutureWarning suppressions.  Trivial but pins the
    # config in case of future merge conflicts.  We use ``tomllib``
    # so the parsing is structural -- no flaky regex over the raw
    # file contents.
    def test_pyproject_no_duplicate_filterwarnings(self):
        try:
            import tomllib  # py>=3.11
        except ImportError:
            import tomli as tomllib  # type: ignore
        import os.path
        # Resolve relative to this test file's location so the
        # assertion isn't sensitive to the cwd pytest is invoked from.
        repo_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..'))
        path = os.path.join(repo_root, 'pyproject.toml')
        with open(path, 'rb') as f:
            cfg = tomllib.load(f)
        filterwarnings = (cfg.get('tool', {})
                          .get('pytest', {})
                          .get('ini_options', {})
                          .get('filterwarnings', []))
        assert filterwarnings, 'no filterwarnings entries in pyproject.toml'
        # FutureWarning ignore entries must not have exact-text dups.
        fw_ignores = [e for e in filterwarnings
                      if 'FutureWarning' in e and e.startswith('ignore')]
        assert len(fw_ignores) == len(set(fw_ignores)), (
            f'Duplicate FutureWarning filterwarnings: {fw_ignores}')
        # And the global pattern in question (the L4e bug) is present
        # exactly once.
        l4e_pattern = (
            'ignore:Input has data type.*output has been cast.*'
            ':FutureWarning')
        assert fw_ignores.count(l4e_pattern) == 1, (
            f'L4e FutureWarning entry should appear exactly once: '
            f'count={fw_ignores.count(l4e_pattern)}')

    # H5 / T2 -- negative shapes are rejected at construction time
    # rather than building a phantom sparse object that crashes on
    # later use.  Covers all four formats and both array/matrix kinds.
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array, sparse.coo_array,
        sparse.dia_array,
        sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix,
        sparse.dia_matrix,
    ])
    @pytest.mark.parametrize('shape', [(-3, 5), (3, -5), (-3, -5)])
    def test_negative_shape_rejected(self, cls, shape):
        with pytest.raises(ValueError, match='non-negative'):
            cls(shape)

    def test_negative_shape_kwarg_rejected_csr(self):
        # ``shape=`` kwarg path on the tuple-3 CSR/CSC constructor.
        with pytest.raises(ValueError, match='non-negative'):
            sparse.csr_array(
                (cupy.array([1.]), cupy.array([0]),
                 cupy.array([0, 1])),
                shape=(-3, 5))

    def test_negative_shape_kwarg_rejected_coo(self):
        with pytest.raises(ValueError, match='non-negative'):
            sparse.coo_array(
                (cupy.array([1.]),
                 (cupy.array([0]), cupy.array([0]))),
                shape=(-3, 5))

    def test_negative_shape_kwarg_rejected_dia(self):
        with pytest.raises(ValueError, match='non-negative'):
            sparse.dia_array(
                (cupy.zeros((1, 3)), cupy.array([0])),
                shape=(-3, 3))

    def test_zero_shape_still_accepted(self):
        # ``(0, 0)``, ``(0, n)``, ``(n, 0)`` are valid empty shapes
        # for every sparse format.
        for cls in (sparse.csr_array, sparse.csc_array,
                    sparse.coo_array, sparse.dia_array):
            for shape in ((0, 0), (0, 5), (5, 0)):
                A = cls(shape)
                assert A.shape == shape
                assert A.nnz == 0

    # F12 / M13 -- toarray(out=) now writes into the user-provided
    # buffer in-place and returns it (was silently ignored).
    @pytest.mark.parametrize('cls', [
        sparse.csr_array, sparse.csc_array,
        sparse.coo_array, sparse.dia_array])
    def test_toarray_out_is_honored(self, cls):
        A = cls(cupy.eye(3))
        out = cupy.zeros((3, 3))
        result = A.toarray(out=out)
        assert result is out, 'toarray must return the user buffer'
        cupy.testing.assert_array_equal(out, cupy.eye(3))

    def test_toarray_out_shape_mismatch(self):
        A = sparse.csr_array(cupy.eye(3))
        with pytest.raises(ValueError, match='shape'):
            A.toarray(out=cupy.zeros((4, 4)))

    def test_toarray_out_dtype_mismatch(self):
        A = sparse.csr_array(cupy.eye(3))
        with pytest.raises(ValueError, match='dtype'):
            A.toarray(out=cupy.zeros((3, 3), dtype=cupy.float32))

    def test_todense_out_is_honored(self):
        # ``todense`` delegates to ``toarray`` so the same plumbing
        # applies.
        A = sparse.csr_array(cupy.eye(3))
        out = cupy.zeros((3, 3))
        result = A.todense(out=out)
        assert result is out
        cupy.testing.assert_array_equal(out, cupy.eye(3))

    def test_toarray_order_with_out_rejected(self):
        # scipy raises ``ValueError("order cannot be specified if out
        # is not None")`` because ``out``'s memory layout is already
        # fixed and accepting ``order=`` would either be silently
        # ignored or override the user.  Match that behaviour for
        # porting parity.
        A = sparse.csr_array(cupy.array([[1., 2.], [3., 4.]]))
        out = cupy.zeros((2, 2))
        with pytest.raises(ValueError, match='order cannot be specified'):
            A.toarray(order='F', out=out)
        with pytest.raises(ValueError, match='order cannot be specified'):
            A.toarray(order='C', out=out)
        # Sanity: ``order=None`` (default) plus ``out=`` works.
        result = A.toarray(out=out)
        assert result is out

    # F9 -- getnnz(axis=) now returns per-axis counts (was raising).
    # Counts stored entries (including explicit zeros), unlike
    # ``count_nonzero`` which excludes them.
    def test_getnnz_axis_csr(self):
        A_np = numpy.array([[1., 2., 0.], [0., 3., 4.]])
        A = sparse.csr_matrix(cupy.asarray(A_np))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=0), cupy.array([1, 2, 1], dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=1), cupy.array([2, 2], dtype=cupy.intp))
        # Negative-axis aliases.
        cupy.testing.assert_array_equal(
            A.getnnz(axis=-1), A.getnnz(axis=1))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=-2), A.getnnz(axis=0))

    def test_getnnz_axis_csc(self):
        A_np = numpy.array([[1., 2., 0.], [0., 3., 4.]])
        A = sparse.csc_matrix(cupy.asarray(A_np))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=0), cupy.array([1, 2, 1], dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=1), cupy.array([2, 2], dtype=cupy.intp))

    def test_getnnz_axis_coo(self):
        A_np = numpy.array([[1., 2., 0.], [0., 3., 4.]])
        A = sparse.coo_matrix(cupy.asarray(A_np))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=0), cupy.array([1, 2, 1], dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=1), cupy.array([2, 2], dtype=cupy.intp))

    @pytest.mark.parametrize('cls', [
        sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix])
    def test_getnnz_axis_empty(self, cls):
        # An empty sparse matrix returns a zero vector of the right
        # length on either axis.
        A = cls((3, 5))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=0), cupy.zeros(5, dtype=cupy.intp))
        cupy.testing.assert_array_equal(
            A.getnnz(axis=1), cupy.zeros(3, dtype=cupy.intp))

    def test_getnnz_counts_explicit_zeros(self):
        # ``getnnz`` counts buffer entries, including explicit zeros
        # (unlike ``count_nonzero`` which masks them).
        A = sparse.csr_matrix._from_parts(
            cupy.array([1., 0., 2.]),
            cupy.array([0, 1, 2], dtype='i'),
            cupy.array([0, 1, 2, 3], dtype='i'),
            (3, 3),
            has_canonical_format=True)
        # Stored entries: 3 (one per row).  ``getnnz(axis=1)``: 1, 1, 1.
        cupy.testing.assert_array_equal(
            A.getnnz(axis=1), cupy.ones(3, dtype=cupy.intp))
        # ``count_nonzero(axis=1)`` excludes the explicit zero in row 1.
        cupy.testing.assert_array_equal(
            A.count_nonzero(axis=1),
            cupy.array([1, 0, 1], dtype=cupy.intp))

    # dia_array constructor: dtype kwarg honored across every input
    # path.  Pre-fix, the scipy-sparse branch unconditionally rebound
    # ``dtype = x.dtype``, dropping the user's choice; the cupy-sparse
    # branch then crashed with "Unable to avoid copy" when the
    # downstream ``cupy.array(..., copy=False)`` had to cast.
    def test_dia_array_dtype_kwarg_scipy_sparse_input(self):
        import scipy.sparse as ssp
        s = ssp.dia_array(numpy.eye(3))
        A = sparse.dia_array(s, dtype=cupy.float32)
        assert A.dtype == cupy.float32

    def test_dia_array_dtype_kwarg_cupy_sparse_input(self):
        # Round-trip through todia + dtype cast in one call.
        A = sparse.dia_array(
            sparse.csr_array(cupy.eye(3)), dtype=cupy.complex64)
        assert A.dtype == cupy.complex64
        cupy.testing.assert_array_equal(
            A.toarray(), cupy.eye(3, dtype=cupy.complex64))

    def test_dia_array_dtype_kwarg_dense_input(self):
        A = sparse.dia_array(cupy.eye(3), dtype=cupy.float32)
        assert A.dtype == cupy.float32

    def test_dia_array_dtype_kwarg_data_offsets_input(self):
        A = sparse.dia_array(
            (cupy.array([[1., 2., 3.]]), cupy.array([0])),
            shape=(3, 3), dtype=cupy.complex64)
        assert A.dtype == cupy.complex64

    def test_dia_array_dtype_kwarg_shape_input(self):
        A = sparse.dia_array((3, 3), dtype=cupy.complex64)
        assert A.dtype == cupy.complex64

    def test_dia_array_dtype_kwarg_default(self):
        # No dtype kwarg: source dtype is preserved.
        A = sparse.dia_array(cupy.eye(3, dtype=cupy.float32))
        assert A.dtype == cupy.float32

    # M9 -- coo dense input now picks shape-derived index dtype (int32
    # when shape fits, int64 otherwise) instead of inheriting the int64
    # default of ``cupy.nonzero``.  Matches scipy 1.17 sparse_array.
    def test_coo_array_dense_default_index_dtype_int32(self):
        A = sparse.coo_array(cupy.eye(3))
        assert A.coords[0].dtype == cupy.int32
        assert A.coords[1].dtype == cupy.int32
        # Same for matrix kind.
        M = sparse.coo_matrix(cupy.eye(3))
        assert M.row.dtype == cupy.int32
        assert M.col.dtype == cupy.int32

    def test_coo_array_dense_int64_when_shape_requires(self):
        # If shape exceeds int32, dense path would still produce int64.
        # We can't construct such a dense matrix, but verify the
        # downcast doesn't fire by construction: the dispatch uses
        # ``_sputils.get_index_dtype(maxval=max(shape))``, which is
        # int32 for shape <= 2^31-1, int64 above.  Sanity check the
        # underlying helper directly.
        from cupyx.scipy.sparse._sputils import get_index_dtype
        assert get_index_dtype(maxval=2**31 - 1) == cupy.int32
        assert get_index_dtype(maxval=2**31) == cupy.int64

    def test_coo_array_user_dtype_preserved(self):
        # The dense-branch downcast does NOT touch user-supplied
        # tuple-2 inputs; sparray ``check_contents=False`` is preserved.
        A = sparse.coo_array(
            (cupy.array([1.0]),
             (cupy.array([0], dtype='i8'),
              cupy.array([0], dtype='i8'))),
            shape=(3, 3))
        assert A.coords[0].dtype == cupy.int64
        assert A.coords[1].dtype == cupy.int64

    # M4 -- error message clarity: scalar input produces an scipy-
    # compatible "sparse array classes do not support instantiation
    # from a scalar" error for sparray; matrix kind raises with a
    # format-naming message instead of the generic catch-all.
    def test_csr_array_scalar_error_message(self):
        with pytest.raises(
                ValueError, match='do not support instantiation'):
            sparse.csr_array(5)
        with pytest.raises(
                ValueError, match='do not support instantiation'):
            sparse.csc_array(5.0)

    def test_csr_matrix_scalar_error_format_specific(self):
        # CuPy doesn't yet implement the scipy 'csr_matrix(5) -> (1,1)'
        # convenience.  Until that's added (M4 has it as a feature
        # request), the matrix kind raises a format-named error rather
        # than the generic "Unsupported initializer format".
        with pytest.raises(ValueError, match='csr_matrix'):
            sparse.csr_matrix(5)

    # L4f -- numpy ndarray input is rejected with a message that
    # tells the user how to fix it (cupy.asarray first), not the
    # generic catch-all.
    def test_numpy_ndarray_rejected_with_helpful_message(self):
        with pytest.raises(ValueError, match='cupy.asarray'):
            sparse.csr_array(numpy.eye(3))
        with pytest.raises(ValueError, match='cupy.asarray'):
            sparse.csc_matrix(numpy.eye(3))

    # _asindices: OverflowError on dtype-too-narrow Python int now
    # surfaces as IndexError instead of leaking the OverflowError.
    def test_fancy_index_overflow_becomes_indexerror(self):
        # CSR with int32 indices (default for shape (3, 3)).
        A = sparse.csr_array(cupy.eye(3))
        assert A.indices.dtype == cupy.int32
        # 2**32 doesn't fit int32; the asarray cast raises
        # OverflowError which the IndexMixin remaps to IndexError.
        with pytest.raises(IndexError):
            A[[2**32], :]
