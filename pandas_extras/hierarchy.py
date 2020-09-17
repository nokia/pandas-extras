"""
    Contains functions to help manage hierarchical data in pandas.
"""
import re

from .transformations import merge_columns


def flatten_adjacency_list(dataframe, parent, right_on=None):
    """
        Creates the flattened hierarchy out of an adjancecy list.

        .. code-block:: python

            >>> df = pd.DataFrame([
            ...     {'employee': 0, 'manager': None},
            ...     {'employee': 1, 'manager': 0},
            ...     {'employee': 2, 'manager': 0},
            ...     {'employee': 3, 'manager': 0},
            ...     {'employee': 4, 'manager': 1},
            ...     {'employee': 5, 'manager': 1},
            ...     {'employee': 6, 'manager': 2},
            ...     {'employee': 7, 'manager': 6},
            ... ])
            >>> df.pipe(flatten_adjacency_list, 'manager', right_on='employee')
                employee    manager     manager_1   manager_2
            0   0           NaN         NaN         NaN
            1   1           0           NaN         NaN
            2   2           0           NaN         NaN
            3   3           0           NaN         NaN
            4   4           1           0           NaN
            5   5           1           0           NaN
            6   6           2           0           NaN
            7   7           6           2           0

            >>> df.set_index('employee').pipe(flatten_adjacency_list, 'manager')
                        manager     manager_1   manager_2
            employee
            0           NaN         NaN         NaN
            1           0           NaN         NaN
            2           0           NaN         NaN
            3           0           NaN         NaN
            4           1           0           NaN
            5           1           0           NaN
            6           2           0           NaN
            7           6           2           0

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param str parent: The name of the column that contains the parent id.
        :param str right_on: Name of the primary key column. If not given, the indices will be used.

        :returns: The flattened DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    set_index = False
    if right_on is None:
        set_index = True
        right_on = dataframe.index.name
        dataframe = dataframe.reset_index()
    rename_map = {right_on: right_on + '_y', parent: parent + '_y'}
    counter = 1
    dataframe = dataframe.merge(
        dataframe.rename(columns=rename_map)[list(rename_map.values())],
        left_on=parent, right_on=rename_map[right_on], how='left'
    ).drop(rename_map[right_on], axis=1)
    while dataframe[rename_map[parent]].notna().any():
        dataframe.rename(columns={rename_map[parent]: parent + '_' + str(counter)}, inplace=True)
        dataframe = dataframe.merge(
            dataframe.rename(columns=rename_map)[list(rename_map.values())],
            left_on=parent + '_' + str(counter), right_on=rename_map[right_on], how='left'
        ).drop(rename_map[right_on], axis=1)
        counter += 1
    if set_index:
        dataframe.set_index(right_on, inplace=True)
    return dataframe.drop(rename_map[parent], axis=1)


def get_adjacency_list_depth(dataframe, parent, right_on=None, new_column='depth'):
    """
        Calculates node depth in the adjancecy list hierarchy.

        .. code-block:: python

            >>> df = pd.DataFrame([
            ...     {'employee': 0, 'manager': None},
            ...     {'employee': 1, 'manager': 0},
            ...     {'employee': 2, 'manager': 0},
            ...     {'employee': 3, 'manager': 0},
            ...     {'employee': 4, 'manager': 1},
            ...     {'employee': 5, 'manager': 1},
            ...     {'employee': 6, 'manager': 2},
            ...     {'employee': 7, 'manager': 6},
            ... ])
            >>> df.pipe(get_adjacency_list_depth, 'manager', right_on='employee')
                employee    manager     depth
            0   0           NaN         0
            1   1           0           1
            2   2           0           1
            3   3           0           1
            4   4           1           2
            5   5           1           2
            6   6           2           2
            7   7           6           3

            >>> df.set_index('employee').pipe(
            ...     get_adjacency_list_depth, 'manager', new_column='level'
            ... )
                        manager     level
            employee
            0           NaN         0
            1           0           1
            2           0           1
            3           0           1
            4           1           2
            5           1           2
            6           2           2
            7           6           3

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param str parent: The name of the column that contains the parent id.
        :param str right_on: Name of the primary key column. If not given, the indices will be used.
        :param str new_column: Name of the new column to be created. By default `depth` will be
                               used.

        :returns: The flattened DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    dataframe = dataframe.pipe(flatten_adjacency_list, parent, right_on=right_on)
    columns = [col for col in dataframe.columns.tolist() if re.match(parent + r'(_\d)?', col)]
    dataframe = dataframe.pipe(merge_columns, columns, new_column, aggr=lambda x: x.notna().sum())
    return dataframe.drop([col for col in columns if col != parent], axis=1)
