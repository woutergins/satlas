"""
Implementation of a class that transforms the input before kicking the arguments to the higher class.
Currently only supports HFSModel.

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""
from .hfsmodel import HFSModel
import copy

__all__ = ['TransformHFSModel']

class TransformHFSModel(HFSModel):
    """Create an HFSModel that first applies a transformation to
    the input data before evaluating."""

    def __init__(self, *args, **kwargs):
        """Passes all arguments on the :class:`.HFSModel`.
        See :class:`.HFSModel` for input information."""
        super(TransformHFSModel, self).__init__(*args, **kwargs)

    @property
    def transform(self):
        """The transformation function to be applied to the input data. Wrapping
        with the *functools.lru_cache* function is attempted, and non-callable objects
        raise an error when assigned to :attr:`.transform`."""
        return self._transform

    @transform.setter
    def transform(self, func):
        if callable(func):
            try:
                from functool import lru_cache
                self._transform = lru_cache(maxsize=512, typed=True)(func)
            except:
                self._transform = func
        else:
            raise TypeError('supplied value must be a callable!')

    def plot(self, *args, **kwargs):
        """Grants access to the :meth:`.HFSModel.plot` method, passing all arguments.
        The transformation used is temporarily changed to the identity transform."""
        remember = copy.deepcopy(self._transform)
        self._transform = lambda x: x
        to_return = super(TransformHFSModel, self).plot(*args, **kwargs)
        self._transform = remember
        return to_return

    def __call__(self, *args, **kwargs):
        return super(TransformHFSModel, self).__call__(self._transform(*args, **kwargs))
