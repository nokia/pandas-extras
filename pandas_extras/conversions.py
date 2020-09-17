"""
    Contains function that help in converting between types
"""
import pandas as pd


class NativeDict(dict):
    """
        Helper class to ensure that only native types are in the dicts produced by
        :func:`to_dict() <pandas.DataFrame.to_dict>`

        .. code-block:: python

            >>> df.to_dict(orient='records', into=NativeDict)

        .. note::

            Needed until `#21256 <https://github.com/pandas-dev/pandas/issues/21256>`_ is resolved.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(((k, self.convert_if_needed(v)) for row in args for k, v in row), **kwargs)

    @staticmethod
    def convert_if_needed(value):
        """
            Converts `value` to native python type.

            .. warning::

                Only :class:`Timestamp <pandas.Timestamp>` and numpy :class:`dtypes <numpy.dtype>`
                are converted.
        """
        if pd.isnull(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if hasattr(value, 'dtype'):
            mapper = {'i': int, 'u': int, 'f': float}
            _type = mapper.get(value.dtype.kind, lambda x: x)
            return _type(value)
        return value


def clear_nan(dataframe):
    """
        Change the pandas.NaT and the pandas.nan elements to None.

        :param dataframe: The pandas.DataFrame object which should be transformed
        :return: The modified *dataframe*
    """
    dataframe = dataframe.replace([pd.NaT], [None])
    return dataframe.where(pd.notnull(dataframe), None)


def convert_to_type(dataframe, mapper, *types, kwargs_map=None):
    r"""
        Converts columns to types specified by the ``mapper``. In case of ``integer``, ``float``,
        ``signed`` and ``unsigned`` typecasting, the smallest possible type will be chosen. See
        more details at :func:`to_numeric() <pandas.to_numeric>`.

        .. code-block:: python

            >>> df = pd.DataFrame({
            ...     'date': ['05/06/2018', '05/04/2018'],
            ...     'datetime': [156879000, 156879650],
            ...     'number': ['1', '2.34'],
            ...     'int': [4, 8103],
            ...     'float': [4.0, 8103.0],
            ...     'object': ['just some', 'strings']
            ... })
            >>> mapper = {
            ...     'number': 'number', 'integer': 'int', 'float': 'float',
            ...     'date': ['date', 'datetime']
            ... }
            >>> kwargs_map = {'datetime': {'unit': 'ms'}}
            >>> df.pipe(
            ...    convert_to_type, mapper, 'integer', 'date',
            ...    'number', 'float', kwargs_map=kwargs_map
            ... ).dtypes
            date        datetime64[ns]
            datetime    datetime64[ns]
            number             float64
            int                  int64
            float              float32
            object              object
            dtype: object

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param dict mapper: Dict with column names as values and any of the following keys:
                            ``number``, ``integer``, ``float``, ``signed``, ``unsigned``, ``date``
                            and ``datetime``.
        :param str \*types: any number of keys from the mapper. If omitted, all keys from
                            ``mapper`` will be used.
        :param dict kwargs_map: Dict of keyword arguments to apply to
                                :func:`to_datetime() <pandas.to_datetime>` or
                                :func:`to_numeric() <pandas.to_numeric>`.
                                Keys must be the column names, values are the kwargs dict.

        :returns: The converted dataframe
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    types = types or mapper.keys()
    kwargs_map = kwargs_map or {}
    for _type in types:
        if isinstance(mapper[_type], list):
            type_list = mapper[_type]
        else:
            type_list = [mapper[_type]]
        for column in type_list:
            if column in list(dataframe):
                kwargs = kwargs_map.get(column, {})
                if _type == 'number':
                    dataframe[column] = dataframe[column].apply(
                        pd.to_numeric, errors='coerce', **kwargs
                    )
                elif _type in ('date', 'datetime'):
                    dataframe[column] = dataframe[column].apply(
                        pd.to_datetime, errors='coerce', utc=True, **kwargs
                    )
                elif _type in ('integer', 'float', 'signed', 'unsigned'):
                    dataframe[column] = dataframe[column].apply(
                        pd.to_numeric, errors='coerce', downcast=_type
                    )
    return dataframe


def truncate_strings(dataframe, length_mapping):
    r"""
        Truncates strings in columns to defined length.

        .. code-block:: python

            >>> df = pd.DataFrame({
            ...    'strings': [
            ...        'foo',
            ...        'baz',
            ...    ],
            ...    'long_strings': [
            ...        'foofoofoofoofoo',
            ...        'bazbazbazbazbaz',
            ...    ],
            ...    'even_longer_strings': [
            ...        'foofoofoofoofoofoofoofoo',
            ...        'bazbazbazbazbazbazbazbaz',
            ...    ]
            ...})
            >>> df.pipe(truncate_strings, {'long_strings': 6, 'even_longer_strings': 9})
                strings  long_strings  even_longer_strings
            0       foo        foofoo            foofoofoo
            1       baz        bazbaz            bazbazbaz

        :param dataframe: The DataFrame object to work on.
        :type dataframe: :class:`DataFrame <pandas.DataFrame>`
        :param dict length_mapping: Dict of column names and desired length

        :returns: The converted dataframe
        :rtype: :class:`DataFrame <pandas.DataFrame>`
    """
    for colname, length in length_mapping.items():
        if colname in list(dataframe):
            dataframe[colname] = dataframe[colname].apply(
                lambda x, max_len=length: x[:max_len] if isinstance(x, str) else x
            )
    return dataframe
