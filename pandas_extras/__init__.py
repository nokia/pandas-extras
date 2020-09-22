"""
    Contains functions to operate on :class:`DataFrames <pandas.DataFrame>`. All can be chained
    with the :meth:`pipe() <pandas.DataFrame.pipe>` method,
    which is the preferred way in this project.
"""
from .conversions import clear_nan, convert_to_type, NativeDict, truncate_strings
from .hierarchy import flatten_adjacency_list, get_adjacency_list_depth
from .transformations import (
    concatenate_columns,
    expand_list,
    expand_lists,
    extract_dict_key,
    extract_dictionary,
    merge_columns
)
from .util import check_duplicated_labels

__all__ = [
    'clear_nan',
    'concatenate_columns',
    'convert_to_type',
    'expand_list',
    'expand_lists',
    'extract_dict_key',
    'extract_dictionary',
    'flatten_adjacency_list',
    'get_adjacency_list_depth',
    'merge_columns',
    'NativeDict',
    'truncate_strings',
]


def __read_version_from_env():
    """
        Attempts to read the version information from the PANDAS_EXTRAS_VERSION environment
        variable. Returns 'latest' if nothing found.
    """
    import os

    version = os.environ.get('PANDAS_EXTRAS_VERSION', 'latest')
    if version.startswith('refs/tags/'): # $GITHUB_REF in github actions
        return version[10:]

    return version


__version__ = __read_version_from_env()
