"""
    Contains functions to operate on :class:`DataFrames <pandas.DataFrame>`. All can be chained with the
    :meth:`pipe() <pandas.DataFrame.pipe>` method, which is the preferred way in this project.
"""

name = 'pandas-extras'
__version__ = '0.0.1'


from .conversions import clear_nan, convert_to_type, NativeDict, truncate_strings
from .hierarchy import flatten_adjacency_list, get_adjacency_list_depth
from .transformations import (
    concatenate_columns, expand_list, expand_lists, extract_dict_key, extract_dictionary, merge_columns
)
from .util import check_duplicated_labels
