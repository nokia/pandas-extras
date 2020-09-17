"""
    Contains utility functions.
"""


def check_duplicated_labels(dataframe):
    r"""
        Checks if there are duplications on column labels. Raises `ValueError` if there is any
        duplicated label.

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`

        :returns: The original DataFrame
        :rtype: :class:`DataFrame <pandas.DataFrame>`

        :raises: :exc:`ValueError`
    """
    if not dataframe.columns.is_unique:
        raise ValueError('Duplicated columns: {}'.format(
            ', '.join(dataframe.columns[~dataframe.columns.duplicated()].tolist())
        ))
    return dataframe
