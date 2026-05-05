from __future__ import annotations

import cupy
import numpy as np
from cupy._core import internal
from cupy import _util
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _sputils


_ufuncs = [
    'arcsin', 'arcsinh', 'arctan', 'arctanh', 'ceil', 'deg2rad', 'expm1',
    'floor', 'log1p', 'rad2deg', 'rint', 'sign', 'sin', 'sinh', 'sqrt', 'tan',
    'tanh', 'trunc',
]


class _data_matrix(_base._spbase):

    def __init__(self, data):
        self.data = data

    @property
    def dtype(self):
        """Data type of the matrix."""
        return self.data.dtype

    def _with_data(self, data, copy=True):
        raise NotImplementedError

    def _mul_scalar(self, other):
        """Scale all stored entries by a scalar.  Preserves format
        (matches SciPy's ``_data._data_matrix._mul_scalar``).

        Promotes the result to a sparse-supported dtype if
        ``numpy.result_type(self.dtype, other)`` would land outside
        the ``{bool, float32, float64, complex64, complex128}`` set --
        otherwise ``bool_csr * 2`` produces an int64 sparse object
        that no other sparse op can consume.  ``copy=False`` makes
        the ``astype`` a no-op when ``new_dtype == self.dtype``.
        """
        new_dtype = np.result_type(self.dtype, other)
        if new_dtype.char not in '?fdFD':
            new_dtype = np.float64
        return self._with_data(
            self.data.astype(new_dtype, copy=False) * other)

    def __imul__(self, other):
        """In-place scalar multiply preserving object identity.

        Overrides :meth:`_spbase.__imul__` to add the ``bool * int ->
        float64`` dtype upcast (numpy promotion would naturally land
        on int64, which cuSPARSE doesn't accept for ``data``).  The
        upcast rebinds ``self.data`` to a new array with the supported
        dtype but ``self`` itself is unchanged from the caller's
        perspective.
        """
        if _sputils.isscalarlike(other):
            new_dtype = np.result_type(self.dtype, other)
            if new_dtype.char not in '?fdFD':
                new_dtype = np.dtype(np.float64)
            if new_dtype != self.dtype:
                # ``self.data *= other`` would fail when the result
                # dtype differs from ``self.dtype`` (e.g. ``bool *
                # 2.0``); reassign instead.
                self.data = self.data.astype(new_dtype) * other
            else:
                self.data *= other
            return self
        return NotImplemented

    def __itruediv__(self, other):
        """In-place scalar divide preserving object identity.

        See :meth:`__imul__` for the dtype-upcast rationale.  Uses the
        ``self.data *= 1/other`` reciprocal trick to keep the in-place
        semantics (mirrors scipy's ``_data_matrix.__itruediv__``).
        """
        if _sputils.isscalarlike(other):
            recip = 1.0 / other
            new_dtype = np.result_type(self.dtype, recip)
            if new_dtype.char not in '?fdFD':
                new_dtype = np.dtype(np.float64)
            if new_dtype != self.dtype:
                self.data = self.data.astype(new_dtype) * recip
            else:
                self.data *= recip
            return self
        return NotImplemented

    def __abs__(self):
        """Elementwise absolute."""
        return self._with_data(abs(self.data))

    def __round__(self, ndigits=0):
        """Elementwise rounding (matches :func:`numpy.around`)."""
        return self._with_data(cupy.around(self.data, decimals=ndigits))

    def __neg__(self):
        """Elementwise negative."""
        if self.dtype.kind == 'b':
            # Match scipy 1.17: raise NotImplementedError instead of letting
            # the underlying cupy error surface.
            raise NotImplementedError(
                'negating a boolean sparse array is not supported')
        return self._with_data(-self.data)

    def astype(self, dtype, copy=True):
        """Cast the array elements to a specified type.

        Args:
            dtype: Target dtype.
            copy (bool): If ``True`` (default), the returned array does
                not share memory with ``self``.  If ``False``, ``self``
                is returned unchanged when the dtype already matches.

        Returns:
            Sparse object with the requested dtype and the same format.

        Note:
            cuSPARSE-backed arithmetic only accepts the
            ``{bool, float32, float64, complex64, complex128}`` set;
            ``astype`` to a dtype outside that set (e.g. ``int32``)
            succeeds but the result can only be consumed by the
            pure-CuPy ``toarray`` path.  Subsequent ``+``, ``@``,
            ``tocsc``, etc. raise ``TypeError`` from the cuSPARSE
            layer.  Use ``A.toarray().astype(...)`` if a dense int
            array is the goal.
        """
        dtype = np.dtype(dtype)
        if self.dtype != dtype:
            return self._with_data(self.data.astype(dtype, copy=copy))
        if copy:
            return self.copy()
        return self

    def conj(self, copy=True):
        if cupy.issubdtype(self.dtype, cupy.complexfloating):
            return self._with_data(self.data.conj(), copy=copy)
        elif copy:
            return self.copy()
        else:
            return self

    conj.__doc__ = _base._spbase.conj.__doc__

    @property
    def real(self):
        return self._with_data(self.data.real)

    @property
    def imag(self):
        return self._with_data(self.data.imag)

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)

    copy.__doc__ = _base._spbase.copy.__doc__

    def count_nonzero(self, axis=None):
        """Number of non-zero entries.

        Unlike :attr:`nnz` (length of ``data``), this counts only true
        non-zero values; explicit-zero stored entries are excluded.

        Args:
            axis ({None, 0, 1, -1, -2}, optional):
                Count nonzeros for the whole array, or along the
                specified axis.  Per-axis support requires a
                ``count_nonzero`` override on the format class — the
                generic implementation here only handles ``axis=None``.

        Returns:
            int or cupy.ndarray: Scalar count when ``axis`` is
            ``None``; otherwise a 1-D array (from format override).
        """
        # Match scipy: ensure deduped data before counting.  CuPy's
        # other read-style ops (e.g. ``_matmul_dispatch``) already
        # mutate via ``sum_duplicates``, so this is consistent.
        if hasattr(self, 'sum_duplicates'):
            self.sum_duplicates()
        if axis is None:
            return int(cupy.count_nonzero(self.data))
        # Format-specific overrides (CSR/CSC/COO) handle axis cases.
        # DIA doesn't natively, matching scipy's choice to raise.
        raise NotImplementedError(
            'axis-aware count_nonzero is not implemented for '
            f'{type(self).__name__}')

    def mean(self, axis=None, dtype=None, out=None):
        """Compute the arithmetic mean along the specified axis.

        Args:
            axis (int or ``None``): Axis along which the sum is computed.
                If it is ``None``, it computes the average of all the elements.
                Select from ``{None, 0, 1, -2, -1}``.

        Returns:
            cupy.ndarray: Summed array.

        .. seealso::
           :meth:`scipy.sparse.spmatrix.mean`

        """
        _sputils.validateaxis(axis)
        nRow, nCol = self.shape
        data = self.data.copy()

        if axis is None:
            n = nRow * nCol
        elif axis in (0, -2):
            n = nRow
        else:
            n = nCol

        return self._with_data(data / n).sum(axis, dtype, out)

    def power(self, n, dtype=None):
        """Elementwise power function.

        Args:
            n: Exponent.
            dtype: Type specifier.

        """
        # V2-20: ``A.power(array_exponent)`` would otherwise trigger
        # ``cupy.array(...) == 0`` below, which raises the confusing
        # "truth value of an array with more than one element is
        # ambiguous" error.  scipy raises ``NotImplementedError`` for
        # non-scalar exponents.
        if not _sputils.isscalarlike(n):
            raise NotImplementedError('input is not scalar')
        # V2-1: ``n == 0`` would densify the matrix (every implicit
        # zero becomes ``0**0 == 1``).  scipy raises here; the
        # equivalent guard in ``_spbase.__pow__`` only catches the
        # ``A ** 0`` operator path, so users calling ``.power(0)``
        # directly would silently get a sparse object whose stored
        # entries are 1 and implicit zeros remain 0 -- mathematically
        # wrong vs the dense ones-matrix scipy produces.
        if n == 0:
            raise NotImplementedError(
                'zero power is not supported as it would densify the '
                'matrix; use ``cupy.ones(A.shape, dtype=A.dtype)`` '
                'instead.')
        if dtype is None:
            data = self.data.copy()
        else:
            data = self.data.astype(dtype, copy=True)
        data **= n
        return self._with_data(data)


def _find_missing_index(ind, n):
    positions = cupy.arange(ind.size)
    diff = ind != positions
    return cupy.where(
        diff.any(),
        diff.argmax(),
        cupy.asarray(ind.size if ind.size < n else -1))


def _non_zero_cmp(mat, am, zero, m):
    size = np.prod(mat.shape)
    if size == mat.nnz:
        return am
    else:
        ind = mat.row * mat.shape[1] + mat.col
        zero_ind = _find_missing_index(ind, size)
        return cupy.where(
            m == zero,
            cupy.minimum(zero_ind, am),
            zero_ind)


class _minmax_mixin:
    """Mixin for min and max methods.
    These are not implemented for dia_matrix, hence the separate class.

    """

    def _min_or_max_axis(self, axis, min_or_max, explicit):
        N = self.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = self.shape[1 - axis]

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        # Do the reduction
        value = mat._minor_reduce(min_or_max, axis, explicit)
        idx_dtype = _sputils.get_index_dtype(maxval=M)
        major_index = cupy.arange(M, dtype=idx_dtype)

        mask = value != 0
        major_index = cupy.compress(mask, major_index)
        value = cupy.compress(mask, value)

        n = len(value)
        zeros = cupy.zeros(n, dtype=idx_dtype)
        value = value.astype(self.dtype, copy=False)
        # Use the appropriate container so the result inherits the
        # array vs matrix type from ``self``.  CuPy COO is currently
        # 2D-only, so reductions still produce (1, M) / (M, 1) shapes
        # even for sparse arrays (SciPy sparse arrays return shape
        # (M,) here, but that requires 1D sparse array support).
        coo_cls = self._coo_container
        if axis == 0:
            return coo_cls._from_parts(
                value, zeros, major_index, shape=(1, M))
        else:
            return coo_cls._from_parts(
                value, major_index, zeros, shape=(M, 1))

    def _min_or_max(self, axis, out, min_or_max, explicit):
        if out is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'out' parameter.")

        _sputils.validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("zero-size array to reduction operation")

            zero = cupy.zeros((), dtype=self.dtype)
            if self.nnz == 0:
                return zero
            self.sum_duplicates()
            m = min_or_max(self.data)
            if explicit:
                return m
            if self.nnz != internal.prod(self.shape):
                if min_or_max is cupy.min:
                    m = cupy.minimum(zero, m)
                elif min_or_max is cupy.max:
                    m = cupy.maximum(zero, m)
                else:
                    assert False
            return m

        if axis < 0:
            axis += 2

        return self._min_or_max_axis(axis, min_or_max, explicit)

    def _arg_min_or_max_axis(self, axis, op):
        if self.shape[axis] == 0:
            raise ValueError("Can't apply the operation along a zero-sized "
                             "dimension.")

        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        # Do the reduction
        value = mat._arg_minor_reduce(op, axis)

        # Sparse arrays return a 1-D ndarray; sparse matrices keep
        # the legacy 2-D shape (matching scipy).
        if isinstance(self, _base.sparray):
            return value
        if axis == 0:
            return value[None, :]
        else:
            return value[:, None]

    def _arg_min_or_max(self, axis, out, op, compare):
        if out is not None:
            raise ValueError("Sparse matrices do not support "
                             "an 'out' parameter.")

        _sputils.validateaxis(axis)

        if axis is None:
            if 0 in self.shape:
                raise ValueError("Can't apply the operation to "
                                 "an empty matrix.")

            if self.nnz == 0:
                return 0
            else:
                zero = cupy.asarray(self.dtype.type(0))
                mat = self.tocoo()

                mat.sum_duplicates()

                am = op(mat.data)
                m = mat.data[am]

                return cupy.where(
                    compare(m, zero), mat.row[am] * mat.shape[1] + mat.col[am],
                    _non_zero_cmp(mat, am, zero, m))

        if axis < 0:
            axis += 2

        return self._arg_min_or_max_axis(axis, op)

    def max(self, axis=None, out=None, *, explicit=False):
        """Returns the maximum of the matrix or maximum along an axis.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the sum is computed. The default is to
                compute the maximum over all the matrix elements, returning
                a scalar (i.e. ``axis`` = ``None``).
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except
                for the default value, as this argument is not used.
            explicit (bool): Return the maximum value explicitly specified and
                ignore all implicit zero entries. If the dimension has no
                explicit values, a zero is then returned to indicate that it is
                the only implicit value. This parameter is experimental and may
                change in the future.

        Returns:
            (cupy.ndarray or float): Maximum of ``a``. If ``axis`` is
            ``None``, the result is a scalar value. If ``axis`` is given,
            the result is an array of dimension ``a.ndim - 1``. This
            differs from numpy for computational efficiency.

        .. seealso:: min : The minimum value of a sparse matrix along a given
          axis.
        .. seealso:: numpy.matrix.max : NumPy's implementation of ``max`` for
          matrices

        """
        if explicit:
            api_name = 'explicit of cupyx.scipy.sparse.{}.max'.format(
                self.__class__.__name__)
            _util.experimental(api_name)
        return self._min_or_max(axis, out, cupy.max, explicit)

    def min(self, axis=None, out=None, *, explicit=False):
        """Returns the minimum of the matrix or maximum along an axis.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the sum is computed. The default is to
                compute the minimum over all the matrix elements, returning
                a scalar (i.e. ``axis`` = ``None``).
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.
            explicit (bool): Return the minimum value explicitly specified and
                ignore all implicit zero entries. If the dimension has no
                explicit values, a zero is then returned to indicate that it is
                the only implicit value. This parameter is experimental and may
                change in the future.

        Returns:
            (cupy.ndarray or float): Minimum of ``a``. If ``axis`` is
            None, the result is a scalar value. If ``axis`` is given, the
            result is an array of dimension ``a.ndim - 1``. This differs
            from numpy for computational efficiency.

        .. seealso:: max : The maximum value of a sparse matrix along a given
          axis.
        .. seealso:: numpy.matrix.min : NumPy's implementation of 'min' for
          matrices

        """
        if explicit:
            api_name = 'explicit of cupyx.scipy.sparse.{}.min'.format(
                self.__class__.__name__)
            _util.experimental(api_name)
        return self._min_or_max(axis, out, cupy.min, explicit)

    def nanmax(self, axis=None, out=None, *, explicit=False):
        """Like :meth:`max` but ignore NaN entries (matches scipy 1.17).

        Currently supported for ``axis=None`` only.  For per-axis
        reductions, drop the NaN entries explicitly before reducing
        (e.g. via ``A._with_data(A.data[~cupy.isnan(A.data)])``).

        .. seealso:: :meth:`scipy.sparse.csr_array.nanmax`
        """
        if axis is not None:
            raise NotImplementedError(
                'nanmax with axis is not yet implemented for cupy '
                'sparse')
        if out is not None:
            raise ValueError(
                "Sparse matrices do not support an 'out' parameter.")
        return self._nan_min_or_max(cupy.maximum, explicit, op_max=True)

    def nanmin(self, axis=None, out=None, *, explicit=False):
        """Like :meth:`min` but ignore NaN entries.  See :meth:`nanmax`."""
        if axis is not None:
            raise NotImplementedError(
                'nanmin with axis is not yet implemented for cupy '
                'sparse')
        if out is not None:
            raise ValueError(
                "Sparse matrices do not support an 'out' parameter.")
        return self._nan_min_or_max(cupy.minimum, explicit, op_max=False)

    def _nan_min_or_max(self, scalar_op, explicit, *, op_max):
        if 0 in self.shape:
            raise ValueError('zero-size array to reduction operation')
        zero = cupy.zeros((), dtype=self.dtype)
        if self.nnz == 0:
            return zero
        self.sum_duplicates()
        # Filter explicit NaN entries before reducing.  scipy uses
        # ``np.fmax`` / ``np.fmin`` whose ``reduce`` already ignores
        # NaN, but cupy doesn't expose ``fmax.reduce``.  Filtering
        # before ``max``/``min`` is equivalent.
        data = self.data
        if data.dtype.kind in 'fc':
            data = data[~cupy.isnan(data)]
        has_implicit_zero = self.nnz != internal.prod(self.shape)
        if data.size == 0:
            # All explicit entries were NaN.  scipy: NaN unless an
            # implicit zero is present and not ``explicit``.
            if explicit or not has_implicit_zero:
                return cupy.full((), float('nan'), dtype=self.dtype)
            return zero
        m = (cupy.max(data) if op_max else cupy.min(data))
        if explicit:
            return m
        if has_implicit_zero:
            m = scalar_op(zero, m)
        return m

    def argmax(self, axis=None, out=None):
        """Returns indices of maximum elements along an axis.

        Implicit zero elements are taken into account. If there are several
        maximum values, the index of the first occurrence is returned. If
        ``NaN`` values occur in the matrix, the output defaults to a zero entry
        for the row/column in which the NaN occurs.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the argmax is computed. If ``None`` (default),
                index of the maximum element in the flatten data is returned.
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.

        Returns:
            (cupy.narray or int): Indices of maximum elements. If array,
            its size along ``axis`` is 1.

        """
        return self._arg_min_or_max(axis, out, cupy.argmax, cupy.greater)

    def argmin(self, axis=None, out=None):
        """
        Returns indices of minimum elements along an axis.

        Implicit zero elements are taken into account. If there are several
        minimum values, the index of the first occurrence is returned. If
        ``NaN`` values occur in the matrix, the output defaults to a zero entry
        for the row/column in which the NaN occurs.

        Args:
            axis (int): {-2, -1, 0, 1, ``None``} (optional)
                Axis along which the argmin is computed. If ``None`` (default),
                index of the minimum element in the flatten data is returned.
            out (None): (optional)
                This argument is in the signature *solely* for NumPy
                compatibility reasons. Do not pass in anything except for
                the default value, as this argument is not used.

        Returns:
            (cupy.narray or int): Indices of minimum elements. If matrix,
            its size along ``axis`` is 1.

        """
        return self._arg_min_or_max(axis, out, cupy.argmin, cupy.less)


def _install_ufunc(func_name):

    def f(self):
        # V2-25: scipy 1.16+ uses numpy 2.x ``sign`` semantics for
        # complex values (``z / abs(z)``); scipy 1.15- used the
        # numpy 1.x semantics (``1+0j``).  CuPy's old code hardcoded
        # the legacy path, which now diverges from current scipy.
        #
        # ``cupy.sign`` follows numpy 2.x semantics for non-zero
        # complex inputs, but has a known divergence for ``0+0j``:
        # numpy returns ``0+0j``, cupy returns ``nan+nanj`` (it
        # computes ``z/abs(z)`` literally, which is ``0/0`` for
        # ``0+0j``).  Mask explicit zeros so sparse matrices with
        # zero-valued stored entries don't surface NaN.  Cost: one
        # extra elementwise pass on the data array, only on the
        # complex-sign code path.
        if func_name == 'sign' and self.data.dtype.kind == 'c':
            zero = self.data.dtype.type(0)
            result = cupy.where(self.data == zero, zero,
                                cupy.sign(self.data))
        else:
            ufunc = getattr(cupy, func_name)
            result = ufunc(self.data)
        return self._with_data(result)

    f.__doc__ = 'Elementwise %s.' % func_name
    f.__name__ = func_name

    setattr(_data_matrix, func_name, f)


def _install_ufuncs():
    for func_name in _ufuncs:
        _install_ufunc(func_name)


_install_ufuncs()
