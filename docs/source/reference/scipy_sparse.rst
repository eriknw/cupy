.. module:: cupyx.scipy.sparse

Sparse matrices (:mod:`cupyx.scipy.sparse`)
===========================================

.. Hint:: `SciPy API Reference: Sparse matrices (scipy.sparse) <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_

CuPy supports sparse matrices using `cuSPARSE <https://developer.nvidia.com/cusparse>`_.
These matrices have the same interfaces of `SciPy's sparse matrices <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.

Matrix vs array semantics
-------------------------

CuPy mirrors SciPy's split between **sparse matrix** classes
(``csr_matrix``, ``csc_matrix``, ``coo_matrix``, ``dia_matrix``) with
legacy ``numpy.matrix`` semantics, and **sparse array** classes
(``csr_array``, ``csc_array``, ``coo_array``, ``dia_array``) with
``numpy.ndarray`` semantics:

* ``*`` is matrix multiplication on matrices, element-wise on arrays.
* ``**`` is matrix power on matrices, element-wise on arrays.
* ``@`` is matrix multiplication on both kinds.

The sparse-array surface preserves the array-vs-matrix kind through
arithmetic, indexing, and conversions.  Construction helpers ending
in ``_array`` (``eye_array``, ``diags_array``, ``random_array``,
``block_array``) return sparse arrays; the older helpers (``eye``,
``diags``, ``random``, ``bmat``) return sparse matrices.  See
`SciPy's migration guide
<https://docs.scipy.org/doc/scipy/reference/sparse.migration_to_sparray.html>`_
for porting guidance.

2-D-only constraint (vs. SciPy 1.17+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CuPy sparse arrays are **2-D only**.  Operations that would yield a
1-D result in SciPy 1.17 instead return a 2-D ``(1, N)`` or
``(N, 1)`` shape in CuPy:

* Row indexing: ``A[0]`` and ``A[0, :]`` return a ``(1, N)`` sparse
  array (SciPy: ``(N,)``).
* Column indexing: ``A[:, 0]`` returns ``(M, 1)`` (SciPy: ``(M,)``).
* Reductions: ``A.sum(axis=0)`` returns ``(1, N)`` ``ndarray``
  (SciPy: ``(N,)``).
* Iteration: ``for row in A:`` yields ``(1, N)`` rows (SciPy: ``(N,)``).

Code that relies on the 1-D shape from SciPy 1.17 needs to call
``.ravel()`` / ``.squeeze()`` on the CuPy result.

Data dtype divergence: bool / int promotion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cuSPARSE-backed operations only accept ``bool``, ``float32``,
``float64``, ``complex64`` or ``complex128`` for the ``data`` buffer.
NumPy promotion rules can land outside that set -- e.g.,
``bool_sparse * 2`` would naturally produce ``int64`` data.  CuPy
upcasts the result to ``float64`` so subsequent ``+`` / ``@`` /
``toarray`` keep working::

    >>> A = csp.csr_array(cp.array([[True, False], [False, True]]))
    >>> (A * 2).dtype
    dtype('float64')      # SciPy: int64

If you need integer-typed data, cast explicitly via
``A.astype(numpy.float64) * 2`` (no-op when already float) or work
on a dense copy.

Index dtype (int32 / int64)
---------------------------

Like SciPy, CuPy sparse matrices automatically choose the index
dtype (``indices``, ``indptr``, ``row``, ``col``) based on the
matrix dimensions and index values:

* **int32** when all index values and dimensions fit in a 32-bit
  integer (the common case).
* **int64** when any dimension or index value exceeds
  ``2**31 - 1``.

The dtype is chosen by ``get_index_dtype`` (mirroring SciPy's logic)
and is preserved through format conversions, arithmetic, and indexing.

Operations that delegate to cuSPARSE use the native Generic API
(``SpMatDescr``) for int64 where available, with pure-CuPy
fallbacks for legacy int32-only APIs (e.g., ``csr2cscEx2``,
``xcoo2csr``, ``csrgeam2``).

Known limitations
~~~~~~~~~~~~~~~~~

The following operations are int32-only and raise ``ValueError``
when called on a matrix with int64 indices:

* :func:`cupyx.scipy.sparse.linalg.spsolve` -- the underlying
  ``cusolverSp<t>csrlsvqr`` routine has no int64 overload.
* :func:`cupyx.scipy.sparse.linalg.spilu` -- the underlying
  ``cusparse<t>csrilu02`` routine has no int64 overload.
* :func:`cupyx.scipy.sparse.linalg.spsolve_triangular` on CUDA
  builds older than 12.0, where the dispatch falls back to
  ``cusparse<t>csrsm2``.  On CUDA 12.0+ it uses ``cusparseSpSM``
  (Generic API), which supports int64.

Conversion to/from SciPy sparse matrices
----------------------------------------

``cupyx.scipy.sparse.*_matrix`` and ``scipy.sparse.*_matrix`` are not implicitly convertible to each other.
That means, SciPy functions cannot take ``cupyx.scipy.sparse.*_matrix`` objects as inputs, and vice versa.

- To convert SciPy sparse matrices to CuPy, pass it to the constructor of each CuPy sparse matrix class.
- To convert CuPy sparse matrices to SciPy, use :func:`get <cupyx.scipy.sparse.spmatrix.get>` method of each CuPy sparse matrix class.

Note that converting between CuPy and SciPy incurs data transfer between
the host (CPU) device and the GPU device, which is costly in terms of performance.

Conversion to/from CuPy ndarrays
--------------------------------

- To convert CuPy ndarray to CuPy sparse matrices, pass it to the constructor of each CuPy sparse matrix class.
- To convert CuPy sparse matrices to CuPy ndarray, use ``toarray`` of each CuPy sparse matrix instance (e.g., :func:`cupyx.scipy.sparse.csr_matrix.toarray`).

Converting between CuPy ndarray and CuPy sparse matrices does not incur data transfer; it is copied inside the GPU device.

Contents
--------

Sparse matrix classes
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   coo_matrix
   csc_matrix
   csr_matrix
   dia_matrix
   spmatrix

Sparse array classes
~~~~~~~~~~~~~~~~~~~~

Mirror the matrix classes but follow NumPy semantics (``*``
element-wise, ``@`` matmul).  Recommended for new code (matches SciPy
1.17 sparse-array surface).

.. autosummary::
   :toctree: generated/

   coo_array
   csc_array
   csr_array
   dia_array
   sparray


Functions
~~~~~~~~~

Building sparse matrices:

.. autosummary::
   :toctree: generated/

   eye
   identity
   kron
   kronsum
   diags
   spdiags
   tril
   triu
   bmat
   hstack
   vstack
   rand
   random

Building sparse arrays:

.. autosummary::
   :toctree: generated/

   eye_array
   diags_array
   block_array
   random_array

Manipulating sparse arrays:

.. autosummary::
   :toctree: generated/

   matrix_transpose
   permute_dims
   swapaxes

Index dtype helpers (re-exported from :mod:`scipy.sparse`):

.. autosummary::
   :toctree: generated/

   get_index_dtype
   safely_cast_index_arrays


Sparse matrix tools:

.. autosummary::
   :toctree: generated/

   find

Identifying sparse matrices:

.. autosummary::
   :toctree: generated/

   issparse
   isspmatrix
   isspmatrix_csc
   isspmatrix_csr
   isspmatrix_coo
   isspmatrix_dia


Submodules
~~~~~~~~~~

.. autosummary::

   csgraph - Compressed sparse graph routines
   linalg - Sparse linear algebra routines

Exceptions
~~~~~~~~~~

* :class:`scipy.sparse.SparseEfficiencyWarning`
* :class:`scipy.sparse.SparseWarning`
