"""
    Contains functions to help transform columns data containing complex types,
    like lists or dictionaries.
"""
from functools import reduce
from itertools import zip_longest

import numpy as np
import pandas as pd


def extract_dictionary(dataframe, column, key_list=None, prefix=None, separator='.'):
    """
        Extract values of keys in ``key_list`` into separate columns.

        .. code-block:: python

            >>> df = DataFrame({
            ...    'trial_num': [1, 2, 1, 2],
            ...    'subject': [1, 1, 2, 2],
            ...    'samples': [
            ...        {'A': 1, 'B': 2, 'C': None},
            ...        {'A': 3, 'B': 4, 'C': 5},
            ...        {'A': 6, 'B': 7, 'C': None},
            ...        None,
            ...    ]
            ...})
            >>>df.pipe(extract_dictionary, 'samples', key_list=('A', 'B'))
                trial_num  subject  samples.A  samples.B
            0           1        1          1          2
            1           2        1          3          4
            2           1        2          6          7
            3           2        2        NaN        NaN

        .. warning::
            ``column`` will be dropped from the DataFrame.

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param str column: The name of the column which should be extracted.
        :param list key_list: Collection of keys that should be extracted. The new column names
                              will be created from the key names.
        :param str prefix: Prefix for new column names. By default, ``column`` will be applied
                           as prefix.
        :param str separator: The separator between the prefix and the key name for new column
                              names.

        :returns: The extracted DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    if key_list is None:
        try:
            key_list = next(val for val in dataframe[column] if isinstance(val, dict)).keys()
        except StopIteration:
            key_list = []
    for key in key_list:
        new_column = '{}{}{}'.format(prefix, separator, key) if prefix else prefix
        dataframe = extract_dict_key(
            dataframe, column, key, new_column=new_column, separator=separator
        )
    return dataframe.drop(column, axis=1)


def extract_dict_key(dataframe, column, key, new_column=None, separator='.'):
    """
        Extract values of ``key`` into ``new_column``. If key is missing, ``None`` is added to
        the column.

        .. code-block:: python

            >>> df = DataFrame({
            ...    'trial_num': [1, 2, 1, 2],
            ...    'subject': [1, 1, 2, 2],
            ...    'samples': [
            ...        {'A': 1, 'B': 2, 'C': None},
            ...        {'A': 3, 'B': 4, 'C': 5},
            ...        {'A': 6, 'B': 7, 'C': None},
            ...        None,
            ...    ]
            ...})
            >>>df.pipe(extract_dict_key, 'samples', key='A')
                trial_num  subject  samples.A                      samples
            0           1        1          1  {'A': 1, 'B': 2, 'C': None}
            1           2        1          3     {'A': 3, 'B': 4, 'C': 5}
            2           1        2          6  {'A': 6, 'B': 7, 'C': None}
            3           2        2        NaN                          NaN

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param str column: The name of the column which should be extracted.
        :param str key: Key that should be extracted.
        :param str new_column: Name of the new column. By default, ``column`` will be applied as
                               prefix to ``key``.
        :param str separator: The separator between ``column`` and ``key`` if ``new_column`` is
                              not specified.

        :returns: The extracted DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    new_column = new_column or '{}{}{}'.format(column, separator, key) if new_column != "" else key
    dataframe.loc[:, new_column] = dataframe[column].apply(
        lambda x: x.get(key) if isinstance(x, dict) else x
    ).rename(new_column)
    return dataframe


def expand_list(dataframe, column, new_column=None):
    """
        Expands lists to new rows.

        .. code-block:: python

            >>> df = DataFrame({
            ...     'trial_num': [1, 2, 3, 1, 2, 3],
            ...     'subject': [1, 1, 1, 2, 2, 2],
            ...     'samples': [
            ...         [1, 2, 3, 4],
            ...         [1, 2, 3],
            ...         [1, 2],
            ...         [1],
            ...         [],
            ...         None,
            ...     ]
            ... })
            >>> df.pipe(expand_list, 'samples', new_column='sample_id').head(7)
                trial_num  subject  sample_id
            0           1        1          1
            0           1        1          2
            0           1        1          3
            0           1        1          4
            1           2        1          1
            1           2        1          2
            1           2        1          3

        .. warning::
            Between calls of ``expand_list`` and/or ``expand_lists``, the dataframe index
            duplications must be removed, otherwise plenty of duplications will occur.

        .. warning::
            Calling ``expand_list`` on multiple columns might cause data duplications,
            that shall be handled.

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param column: The name of the column which should be extracted.
        :type column: :class: str
        :param new_column: Name of the new columns. If not defined, columns will not be renamed.
        :type new_column: :class: str

        :returns: The expanded DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    new_column = new_column or column
    values, indices = [], []
    for index, value in dataframe[column].items():
        if value and not isinstance(value, float):
            values.extend(value)
            indices.extend([index, ] * len(value))
    if indices and isinstance(indices[0], tuple):
        indices = pd.MultiIndex.from_tuples(indices, names=dataframe.index.names)
    else:
        indices = pd.Series(indices, name=dataframe.index.name)
    return pd.DataFrame({new_column: values}, index=indices).\
        merge(dataframe.drop(column, axis=1), left_index=True, right_index=True, how='outer')


def expand_lists(dataframe, columns, new_columns=None):
    """
        Expands multiple lists to new rows. Pairs elements of lists respective to their index.
        Pads with ``None`` to the longest list.

        .. code-block:: python

            >>> df = DataFrame({
            ...     'trial_num': [1, 2, 3, 1, 2, 3],
            ...     'subject': [1, 1, 1, 2, 2, 2],
            ...     'samples': [
            ...         [1, 2, 3, 4],
            ...         [1, 2, 3],
            ...         [1, 2],
            ...         [1],
            ...         [],
            ...         None,
            ...     ],
            ...     'samples2': [
            ...         [1, 2],
            ...         [1, 2, 3],
            ...         [1, 2],
            ...         [1],
            ...         [],
            ...         None,
            ...     ]
            ... })
            >>> df.pipe(
            ...     expand_lists, ['samples', 'samples'], new_column=['sample_id', 'sample_id2']
            ... ).head(7)
                trial_num  subject  sample_id  sample_id2
            0           1        1          1           1
            0           1        1          2           2
            0           1        1          3         Nan
            0           1        1          4         Nan
            1           2        1          1           1
            1           2        1          2           2
            1           2        1          3           3

        .. warning::
            Between calls of ``expand_list`` and/or ``expand_lists``, the dataframe index
            duplications must be removed, otherwise plenty of duplications will occur.

        .. warning::
            Calling ``expand_lists`` on multiple columns might cause data duplications,
            that shall be handled.

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param columns: The name of the columns which should be extracted.
        :type columns: :class: list or :class: tuple of :class: str
        :param new_columns: Name of the new columns. If not defined, columns will not be renamed.
        :type new_columns: :class: list or :class: tuple of :class: str

        :returns: The expanded DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    new_columns = new_columns or columns
    if not len(columns) == len(new_columns):
        raise ValueError('new_columns must contain the same amount of items as columns')
    if len(columns) == 1:
        return expand_list(dataframe, *columns, *new_columns)
    if not len(columns) > 1:
        raise ValueError('columns argument must contain at least two items.')
    values, indices = [], []
    for index, row in dataframe[columns].iterrows():
        if not row.empty and all(row.notna()):
            values.extend(zip_longest(*row))
            indices.extend([index, ] * max(map(len, row)))
    if indices and isinstance(indices[0], tuple):
        indices = pd.MultiIndex.from_tuples(indices, names=dataframe.index.names)
    else:
        indices = pd.Series(indices, name=dataframe.index.name)
    return pd.DataFrame(values, columns=new_columns, index=indices).fillna(np.nan).\
        merge(dataframe.drop(columns, axis=1), left_index=True, right_index=True, how='outer')


def merge_columns(dataframe, col_header_list, new_column_name, keep=None, aggr=None):
    """
        Add a new column or modify an existing one in *dataframe* called *new_column_name* by
        iterating over the rows and select the proper notnull element from the values of
        *col_header_list* columns in the given row if *keep* is filled OR call the *aggr*
        function with the values of *col_header_list*. Only one of (*keep*, *aggr*) can be filled.

        :param dataframe: the pandas.DataFrame object to modify
        :param col_header_list: list of the names of the headers to merge
        :param str new_column_name: the name of the new column, if it already exists the operation
                                    will overwrite it
        :param str keep: Specify whether the first or the last proper value is needed.
                         values: *first* and *last* as string.
        :param aggr: Callable function which will get the values of *col_header_list* as parameter.
                     The return value of this function will be the value in *new_column_name*

        :returns: The merged DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    if keep and aggr:
        raise ValueError(
            'Parameter keep and aggr can not be handled at the same time. Use only one.'
        )

    old_columns = [x for x in col_header_list if x in list(dataframe)]

    if not old_columns:
        raise ValueError(
            f'None of the following columns were found: {", ".join(col_header_list)}'
        )

    if keep:
        if keep not in ('first', 'last'):
            raise ValueError('Improper value for parameter keep. Possible values: first, last.')

        first_valid = lambda x, y: y if pd.isnull(x) else x
        if keep.startswith('f'):
            aggr = lambda x: reduce(first_valid, x.tolist())
        else:
            aggr = lambda x: reduce(first_valid, x.tolist()[::-1])

    if not callable(aggr):
        raise ValueError('Improper value for parameter aggr. It should be a function.')

    dataframe[new_column_name] = dataframe[old_columns].apply(aggr, axis=1)
    return dataframe


def concatenate_columns(dataframe, columns, new_column, descriptor=None, mapper=None):
    """
        Concatenates `columns` together along the indeces and adds a `descriptor` column,
        if specified, with the column name where the data originates from.

        .. code-block:: python

            >>> df = pd.DataFrame([
            ...     {'key': 'TICKET-1', 'assignee': 'Bob', 'reporter': 'Alice'},
            ...     {'key': 'TICKET-2', 'assignee': 'Bob', 'reporter': 'Alice'},
            ...     {'key': 'TICKET-3', 'assignee': 'Bob', 'reporter': 'Alice'},
            ... ])
            >>> df.pipe(concatenate_columns, ['assignee', 'reporter'], 'user')
                key           user        descriptor
            0   'TICKET-1'    'Alice'     'reporter'
            0   'TICKET-1'    'Bob'       'assignee'
            1   'TICKET-2'    'Alice'     'reporter'
            1   'TICKET-2'    'Bob'       'assignee'
            2   'TICKET-3'    'Alice'     'reporter'
            2   'TICKET-3'    'Bob'       'assignee'

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param columns: The name of the columns which should be concatenated.
        :type columns: :class: list
        :param new_column: Name of the new column.
        :type new_column: :class: str
        :param descriptor: Name of the new descriptor column.
        :type descriptor: :class: str
        :param mapper: A map to apply to `descriptor` values
        :type mapper: :class: dict

        :returns: The concatenated DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    if mapper is None:
        mapper = {}
    descriptor = descriptor or '_desc'
    parts = (
        pd.DataFrame(
            data={
                new_column: dataframe[col],
                descriptor: [mapper.get(col, col) for _ in range(len(dataframe.index))]
            },
            index=dataframe.index
        ) for col in columns if col in dataframe
    )

    return pd.concat(list(parts)).drop('_desc', axis=1, errors='ignore').sort_index()
