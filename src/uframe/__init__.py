"""Top-level package for uframe."""

__author__ = """Christian Amesoeder and Michael Hagn"""
__email__ = 'christian.amesoeder@informatik.uni-regensburg.de'
__version__ = '0.0.1'


from .uframe import (
    uframe,
    uframe_from_array_mice,
    uframe_noisy_array)
from .uframe_instance import uframe_instance

__all__ = [
    "uframe",
    "uframe_from_array_mice",
    "uframe_noisy_array",
    "uframe_instance"
    ]
