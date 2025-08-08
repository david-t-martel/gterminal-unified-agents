"""Common utilities package providing shared helpers, decorators, and base classes.

This package consolidates common utilities including:
- File system operations with rust-fs functional coverage
- Base classes and type definitions
- Cache utilities and helpers
- Google Cloud Storage utilities
"""

# Import main classes and functions
from .base_classes import *
from .cache_utils import *
from .file_ops import *
from .filesystem import FileSystemError
from .filesystem import FileSystemUtils
from .filesystem import copy
from .filesystem import create
from .filesystem import delete
from .filesystem import execute
from .filesystem import find
from .filesystem import fs
from .filesystem import move
from .filesystem import read
from .filesystem import replace
from .filesystem import replace_block
from .filesystem import search
from .filesystem import stat
from .filesystem import write
from .gcs import *
from .typing import *

__all__ = [
    "FileSystemError",
    # File system utilities (rust-fs coverage)
    "FileSystemUtils",
    "copy",
    # rust-fs compatible functions
    "create",
    "delete",
    "execute",
    "find",
    "fs",
    "move",
    "read",
    "replace",
    "replace_block",
    "search",
    "stat",
    "write",
]
