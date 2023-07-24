"""Top-level package for uframe."""

__author__ = """Christian Amesoeder and Michael Hagn"""
__email__ = 'christian.amesoeder@informatik.uni-regensburg.de'
__version__ = '0.0.1'


from .uframe import uframe
from .uframe_instance import uframe_instance
from .utils import load_uframe
from.uframe_from_mice import (uframe_from_array_mice_2,
            uframe_from_array_mice,
            uframe_noisy_array)
__all__ = [
    "uframe",
    "uframe_instance",
    "uframe_from_array_mice",
    "uframe_noisy_array",
    "uframe_from_array_mice_2",
    "load_uframe"]
