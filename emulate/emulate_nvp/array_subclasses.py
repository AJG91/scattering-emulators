import numpy
import numpy as np


class NuclearMatrix(numpy.ndarray):
    __array_priority__ = 20.0

    def __new__(cls, data, dtype=None, copy=True):
        if isinstance(data, numpy.ndarray):
            old_dtype = data.dtype
            new_dtype = dtype or old_dtype
            if (new_dtype == old_dtype) and (not copy):
                self = data
            else:
                self = data.astype(new_dtype, copy=copy)

            if not isinstance(self, cls):
                self = self.view(cls)
            return self

        self = numpy.array(data, dtype=dtype, copy=copy).view(cls)
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def tt(self):
        R"""The triplet-triplet 2x2 block"""
        return self.view(numpy.ndarray)[..., :2, :2]

    @tt.setter
    def tt(self, value):
        self[..., :2, :2] = value

    @property
    def st(self):
        R"""The singlet-triplet 2x2 block"""
        # return self.view(numpy.ndarray)[..., 2:, 2:]
        return self[..., 2:, 2:]

    @st.setter
    def st(self, value):
        self[..., 2:, 2:] = value

    @property
    def mm(self):
        R"""The lp,l = j-1,j-1 element"""
        return self.view(numpy.ndarray)[..., 0, 0]

    @mm.setter
    def mm(self, value):
        self[..., 0, 0] = value

    @property
    def pp(self):
        R"""The lp,l = j+1,j+1 element"""
        return self.view(numpy.ndarray)[..., 1, 1]

    @pp.setter
    def pp(self, value):
        self[..., 1, 1] = value

    @property
    def mp(self):
        R"""The lp,l = j-1,j+1 element"""
        return self.view(numpy.ndarray)[..., 0, 1]

    @mp.setter
    def mp(self, value):
        self[..., 0, 1] = value

    @property
    def pm(self):
        R"""The lp,l = j+1,j-1 element"""
        return self.view(numpy.ndarray)[..., 1, 0]

    @pm.setter
    def pm(self, value):
        self[..., 1, 0] = value

    @property
    def zz(self):
        R"""The singlet element sp,s = 0,0 and lp,l = j,j"""
        return self.view(numpy.ndarray)[..., 2, 2]

    @zz.setter
    def zz(self, value):
        self[..., 2, 2] = value

    @property
    def ll(self):
        R"""The triplet element sp,s = 1,1 and lp,l = j,j"""
        return self.view(numpy.ndarray)[..., 3, 3]

    @ll.setter
    def ll(self, value):
        self[..., 3, 3] = value

    @property
    def zl(self):
        R"""The spin flip element sp,s = 0,1 and lp,l = j,j"""
        return self.view(numpy.ndarray)[..., 2, 3]

    @zl.setter
    def zl(self, value):
        self[..., 2, 3] = value

    @property
    def lz(self):
        R"""The spin flip element sp,s = 1,0 and lp,l = j,j"""
        return self.view(numpy.ndarray)[..., 3, 2]

    @lz.setter
    def lz(self, value):
        self[..., 3, 2] = value


class FreeScatteringFunctions(numpy.ndarray):
    __array_priority__ = -1.0

    def __new__(cls, data, dtype=None, copy=True):
        if isinstance(data, numpy.ndarray):
            old_dtype = data.dtype
            new_dtype = dtype or old_dtype
            if (new_dtype == old_dtype) and (not copy):
                self = data
            else:
                self = data.astype(new_dtype, copy=copy)

            if not isinstance(self, cls):
                self = self.view(cls)
            return self

        self = numpy.array(data, dtype=dtype, copy=copy).view(cls)
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def f(self):
        return self.view(numpy.ndarray)[0]

    @property
    def g(self):
        return self.view(numpy.ndarray)[1]

    @property
    def f_p(self):
        return self.view(numpy.ndarray)[2]

    @property
    def g_p(self):
        return self.view(numpy.ndarray)[3]

    @property
    def unpacked(self):

        return tuple(self.view(numpy.ndarray)[i] for i in range(4))


class EMWaveFunctions(np.recarray):
    __array_priority__ = -1.0

    def __new__(cls, data, dtype=None, copy=True):
        if isinstance(data, numpy.ndarray):
            old_dtype = data.dtype
            new_dtype = dtype or old_dtype
            if (new_dtype == old_dtype) and (not copy):
                self = data
            else:
                self = data.astype(new_dtype, copy=copy)

            if not isinstance(self, cls):
                self = self.view(cls)
            return self

        self = numpy.array(data, dtype=dtype, copy=copy).view(cls)
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return


class NuclearPotentialArray(numpy.recarray):
    R"""A nuclear potential in k space.

    Has a coupled triplet-triplet (tt) and a singlet-triplet (st) channel, each with a shape of j, 2*k, 2*k
    for some k. The singlet and the triplet parts of the st channel can be accessed with zz, and ll, respectively.

    Must have shape j, 2*k, 2*k, ...
    """
    __array_priority__ = -1

    def __new__(cls, shape, dtype=None, **kwargs):
        typ = [('tt', dtype), ('st', dtype)]
        self = np.recarray(shape=shape, dtype=typ, **kwargs)
        self.fill(0.)
        assert shape[1] == shape[2], 'each submatrix must be 2*k, 2*k for some k'
        assert shape[1] % 2 == 0, 'each submatrix must be 2*k, 2*k for some k'

        return self.view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __add__(self, x):
        if not isinstance(x, NuclearPotentialArray):
            raise TypeError('invalid type promotion')
        val = np.zeros_like(self)
        val.tt = self.tt + x.tt
        val.st = self.st + x.st
        return val

    def nk(self):
        return self.shape[1] // 2

    @property
    def zz(self):
        R"""Uncoupled singlet (00) channel"""
        nk = self.nk()
        return self.st[:, :nk, :nk]

    @zz.setter
    def zz(self, value):
        nk = self.nk()
        self.st[:, :nk, :nk] = value

    @property
    def ll(self):
        R"""Uncoupled triplet (11) channel"""
        nk = self.nk()
        return self.st[:, nk:, nk:]

    @ll.setter
    def ll(self, value):
        nk = self.nk()
        self.st[:, nk:, nk:] = value

    @property
    def mm(self):
        R"""Coupled j-1,j-1 channel"""
        nk = self.nk()
        return self.tt[:, :nk, :nk]

    @property
    def pp(self):
        R"""Coupled j+1,j+1 channel"""
        nk = self.nk()
        return self.tt[:, nk:, nk:]

    @property
    def mp(self):
        R"""Coupled j-1,j+1 channel"""
        nk = self.nk()
        return self.tt[:, :nk, nk:]

    @property
    def pm(self):
        R"""Coupled j+1,j-1 channel"""
        nk = self.nk()
        return self.tt[:, nk:, :nk]

    @property
    def zl(self):
        R"""Uncoupled spin flip 01 channel"""
        nk = self.nk()
        return self.st[:, :nk, nk:]

    @property
    def lz(self):
        R"""Uncoupled spin flip 10 channel"""
        nk = self.nk()
        return self.st[:, nk:, :nk]
