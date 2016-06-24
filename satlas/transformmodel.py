"""
Implementation of a class that transforms the input before kicking the arguments
to the higher class. Currently only supports HFSModel.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
import copy

from .hfsmodel import HFSModel


__all__ = ['TransformHFSModel']

def identity(*args):
    return args[0]

class TransformHFSModel(HFSModel):
    """Create an HFSModel that applies both a pre-processing
    transformation on the input data and a post-processing
    transformation on the output data. Mathematically:

        .. math::
            (post \circ model \circ pre)(input)

    Both transformations are initialized to the identical
    transformation for one input argument."""

    def __init__(self, *args, **kwargs):
        """Passes all arguments on the :class:`.HFSModel`.
        See :class:`.HFSModel` for input information."""
        super(TransformHFSModel, self).__init__(*args, **kwargs)
        self._pre_transform = identity
        self._post_transform = identity

    @property
    def pre_transform(self):
        """The transformation function to be applied to the input data. Wrapping
        with the *functools.lru_cache* function is attempted, and non-callable
        objects raise an error when assigned to :attr:`.pre_transform`."""
        return self._pre_transform

    @pre_transform.setter
    def pre_transform(self, func):
        if callable(func):
            try:
                from functool import lru_cache
                self._pre_transform = lru_cache(maxsize=512, typed=True)(func)
            except:
                self._pre_transform = func
        else:
            raise TypeError('supplied value must be a callable!')

    @property
    def post_transform(self):
        """The transformation function to be applied to the output data.
        Non-callable objects raise an error when assigned to
        :attr:`.post_transform`."""
        return self._post_transform

    @post_transform.setter
    def post_transform(self, func):
        if callable(func):
            self._post_transform = func
        else:
            raise TypeError('supplied value must be a callable!')

    def plot(self, *args, **kwargs):
        """Grants access to the :meth:`.HFSModel.plot` method, passing all
        arguments. The transformation used is temporarily changed to the
        identity transform."""
        remember_pre = copy.deepcopy(self._pre_transform)
        remember_post = copy.deepcopy(self._post_transform)
        self._pre_transform = identity
        to_return = super(TransformHFSModel, self).plot(*args, **kwargs)
        self._pre_transform = remember_pre
        return to_return

    def __call__(self, *args, **kwargs):
        return self._post_transform(super(TransformHFSModel, self).__call__(self._pre_transform(*args, **kwargs)), *args, **kwargs)
