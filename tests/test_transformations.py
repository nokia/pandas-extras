import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from pandas_extras import (
    concatenate_columns, expand_list, expand_lists,
    extract_dict_key, extract_dictionary, merge_columns,
)


class TransformationsTestCase(unittest.TestCase):
    def test_expand_list_pos_01(self):
        df = pd.DataFrame(
            {
                'test_index': [1, 2, 3, 4, 5, 6],
                'trial_num': [1, 2, 3, 1, 2, 3],
                'subject': [1, 1, 1, 2, 2, 2],
                'samples': [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1, 2],
                    [1],
                    [],
                    None,
                ]
            }
        ).set_index('test_index')
        expected = pd.DataFrame(
            {
                'newcol': [1, 2, 3, 4, 1, 2, 3, 1, 2, 1, None, None],
                'subject': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
                'trial_num': [1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 2, 3],
                'test_index': [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
            }
        )
        assert_frame_equal(expand_list(df, 'samples', 'newcol').reset_index(),
                           expected, check_like=True, check_dtype=False)


    def test_expand_list_pos_02(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 3, 1, 2, 3],
                'subject': [1, 1, 1, 2, 2, 2],
                'samples': [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1, 2],
                    [1],
                    [],
                    None,
                ]
            }
        ).set_index(['trial_num', 'subject'])
        expected = pd.DataFrame(
            {
                'samples': [1, 2, 3, 4, 1, 1, 2, 3, None, 1, 2, None],
                'subject': [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2],
                'trial_num': [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
            }
        )
        assert_frame_equal(expand_list(df, 'samples').reset_index(), expected, check_like=True)

    def test_expand_list_pos_03(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 3, 1, 2, 3],
                'subject': [1, 1, 1, 2, 2, 2],
                'samples': [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1, 2],
                    [1],
                    [],
                    pd.np.NaN,
                ]
            }
        ).set_index(['trial_num', 'subject'])
        expected = pd.DataFrame(
            {
                'samples': [1, 2, 3, 4, 1, 1, 2, 3, None, 1, 2, None],
                'subject': [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2],
                'trial_num': [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
            }
        )
        assert_frame_equal(expand_list(df, 'samples').reset_index(), expected, check_like=True)

    def test_expand_lists_pos_01(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 3, 1, 2, 3],
                'subject': [1, 1, 1, 2, 2, 2],
                'samples': [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1, 2],
                    [1],
                    [],
                    None,
                ],
                'samples2': [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1, 2],
                    [1],
                    [],
                    None,
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'newcol': [1, 2, 3, 4, 1, 2, 3, 1, 2, 1, None, None],
                'newcol2': [1, 2, 3, 4, 1, 2, 3, 1, 2, 1, None, None],
                'subject': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
                'trial_num': [1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 2, 3]
            }
        )
        assert_frame_equal(
            expand_lists(df, ['samples', 'samples2'], ['newcol', 'newcol2']).reset_index().drop('index', axis=1),
            expected,
            check_like=True
        )

    def test_expand_lists_pos_02(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 3, 1, 2, 3],
                'subject': [1, 1, 1, 2, 2, 2],
                'samples': [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1],
                    [1],
                    [],
                    None,
                ],
                'samples2': [
                    [1, 2],
                    [3],
                    [1, 2],
                    [1],
                    [],
                    None,
                ]
            }
        ).set_index(['trial_num', 'subject'])
        expected = pd.DataFrame(
            {
                'samples': [1, 2, 3, 4, 1, 1, 2, 3, None, 1, None, None],
                'samples2': [1, 2, None, None, 1, 3, None, None, None, 1, 2, None],
                'subject': [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2],
                'trial_num': [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
            }
        )
        assert_frame_equal(expand_lists(df, ['samples', 'samples2']).reset_index(), expected, check_like=True)

    def test_expand_lists_pos_03(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 3, 1, 2, 3],
                'subject': [1, 1, 1, 2, 2, 2],
                'samples': [
                    [{'testkey': 1}, {'testkey': 2}, {'testkey': 3}, {'testkey': 4}],
                    [{'testkey': 1}, {'testkey': 2}, {'testkey': 3}],
                    [{'testkey': 1}, {'testkey': 2}],
                    [{'testkey': 1}],
                    [],
                    None,
                ],
                'other_samples': [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1, 2],
                    [1],
                    [],
                    None,
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'newcol': [{'testkey': 1.0}, {'testkey': 2.0}, {'testkey': 3.0}, {'testkey': 4.0},
                           {'testkey': 1.0}, {'testkey': 2.0}, {'testkey': 3.0}, {'testkey': 1.0},
                           {'testkey': 2.0}, {'testkey': 1.0}, None, None],
                'newcol2': [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0, None, None],
                'subject': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
                'trial_num': [1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 2, 3]
            }
        )
        assert_frame_equal(
            expand_lists(df, ['samples', 'other_samples'], ['newcol', 'newcol2']).reset_index(drop=True),
            expected,
            check_like=True
        )

    def test_expand_lists_pos_04(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 3, 1, 2, 3],
                'subject': [1, 1, 1, 2, 2, 2],
                'samples': [
                    [{'testkey': 1}, {'testkey': 2}, {'testkey': 3}, {'testkey': 4}],
                    [{'testkey': 1}, {'testkey': 2}, {'testkey': 3}],
                    [{'testkey': 1}, {'testkey': 2}],
                    [{'testkey': 1}],
                    [],
                    ['this will be NaN, as None is not iterable'],
                ],
                'other_samples': [
                    [1, 2, 3, 4],
                    [1, 2, 3],
                    [1, 2],
                    [],
                    [1],
                    None,
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'newcol': [{'testkey': 1.0}, {'testkey': 2.0}, {'testkey': 3.0}, {'testkey': 4.0},
                           {'testkey': 1.0}, {'testkey': 2.0}, {'testkey': 3.0}, {'testkey': 1.0},
                           {'testkey': 2.0}, {'testkey': 1.0}, None, None],
                'newcol2': [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, None, 1.0, None],
                'subject': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
                'trial_num': [1, 1, 1, 1, 2, 2, 2, 3, 3, 1, 2, 3]
            }
        )
        assert_frame_equal(
            expand_lists(df, ['samples', 'other_samples'], ['newcol', 'newcol2']).reset_index(drop=True),
            expected,
            check_like=True
        )

    def test_extract_dict_key_pos_01(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [
                    {'A': 1, 'B': 2, 'C': None},
                    {'A': 3, 'B': 4, 'C': 5},
                    {'A': 6, 'B': 7, 'C': None},
                    None,
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [
                    {'A': 1, 'B': 2, 'C': None},
                    {'A': 3, 'B': 4, 'C': 5},
                    {'A': 6, 'B': 7, 'C': None},
                    None,
                ],
                'samples.A': [1, 3, 6, None]
            }
        )
        assert_frame_equal(extract_dict_key(df, 'samples', 'A').reset_index(drop=True), expected, check_like=True)

    def test_extract_dict_key_pos_02(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [
                    {'A': 1, 'B': 2, 'C': None},
                    {'A': 3, 'B': 4, 'C': 5},
                    {'A': 6, 'B': 7, 'C': None},
                    {'B': 8, 'C': None},
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [
                    {'A': 1, 'B': 2, 'C': None},
                    {'A': 3, 'B': 4, 'C': 5},
                    {'A': 6, 'B': 7, 'C': None},
                    {'B': 8, 'C': None},
                ],
                'newcol': [1, 3, 6, None]
            }
        )
        assert_frame_equal(
            extract_dict_key(df, 'samples', 'A', 'newcol').reset_index(drop=True),
            expected, check_like=True
        )

    def test_extract_dict_key_pos_03(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [pd.np.NaN, pd.np.NaN, pd.np.NaN, pd.np.NaN]
            }
        )
        expected = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [pd.np.NaN, pd.np.NaN, pd.np.NaN, pd.np.NaN],
                'newcol': [pd.np.NaN, pd.np.NaN, pd.np.NaN, pd.np.NaN]
            }
        )
        assert_frame_equal(
            extract_dict_key(df, 'samples', 'A', 'newcol').reset_index(drop=True),
            expected, check_like=True
        )

    def test_extract_dict_key_pos_04(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
            }
        )
        with self.assertRaises(KeyError):
            extract_dict_key(df, 'samples', 'A', 'newcol')

    def test_extract_dict_key_pos_05(self):
        df = pd.DataFrame(
            columns=('trial_num', 'subject', 'samples')
        )
        self.assertIn('newcol', extract_dict_key(df, 'samples', 'A', 'newcol').columns.to_list())

    def test_extract_dictionary_pos_01(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [
                    {'A': 1, 'B': 2, 'C': None},
                    {'A': 3, 'B': 4, 'C': 5},
                    {'A': 6, 'B': 7, 'C': None},
                    None,
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples.A': [1, 3, 6, None],
                'samples.B': [2, 4, 7, None],
            }
        )
        assert_frame_equal(
            extract_dictionary(df, 'samples', ['A', 'B']).reset_index(drop=True),
            expected, check_like=True
        )

    def test_extract_dictionary_pos_02(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [
                    {'A': 1, 'B': 2, 'C': None},
                    {'A': 3, 'B': 4, 'C': 5},
                    {'A': 6, 'B': 7, 'C': None},
                    None,
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'newcol.A': [1, 3, 6, None],
                'newcol.B': [2, 4, 7, None],
            }
        )
        assert_frame_equal(
            extract_dictionary(df, 'samples', ['A', 'B'], 'newcol').reset_index(drop=True),
            expected, check_like=True
        )

    def test_extract_dictionary_pos_03(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [
                    {'A': 1, 'B': 2, 'C': None},
                    {'A': 3, 'B': 4, 'C': 5},
                    {'A': 6, 'B': 7, 'C': None},
                    None,
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples.A': [1, 3, 6, None],
                'samples.B': [2, 4, 7, None],
                'samples.C': [None, 5, None, None]
            }
        )
        assert_frame_equal(extract_dictionary(df, 'samples').reset_index(drop=True), expected, check_like=True)

    def test_extract_dictionary_pos_04(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [
                    {'A': 1, 'B': 2, 'C': None},
                    {'A': 3, 'B': 4, 'C': 5},
                    {'A': 6, 'B': 7, 'C': None},
                    None,
                ]
            }
        )
        expected = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'A': [1, 3, 6, None],
                'B': [2, 4, 7, None],
                'C': [None, 5, None, None]
            }
        )
        assert_frame_equal(
            extract_dictionary(df, 'samples', prefix='').reset_index(drop=True),
            expected, check_like=True
        )

    def test_extract_dictionary_pos_05(self):
        df = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2],
                'samples': [None, None, None, None]
            }
        )
        expected = pd.DataFrame(
            {
                'trial_num': [1, 2, 1, 2],
                'subject': [1, 1, 2, 2]
            }
        )
        assert_frame_equal(
            extract_dictionary(df, 'samples', prefix='').reset_index(drop=True),
            expected, check_like=True
        )

    def test_extract_dictionary_pos_06(self):
        df = pd.DataFrame({
            'trial_num': [1, 2, 1, 2],
            'subject': [1, 1, 2, 2],
            'samples': [
                None,
                {'A': 1, 'B': 2, 'C': None},
                {'A': 3, 'B': 4, 'C': 5},
                {'A': 6, 'B': 7, 'C': None},
            ]
        })
        expected = pd.DataFrame({
            'trial_num': [1, 2, 1, 2],
            'subject': [1, 1, 2, 2],
            'A': [None, 1, 3, 6],
            'B': [None, 2, 4, 7],
            'C': [None, None, 5, None]
        })
        assert_frame_equal(
            extract_dictionary(df, 'samples', prefix='').reset_index(drop=True),
            expected, check_like=True
        )

    def test_merge_columns(self):
        dataframe = pd.DataFrame([
            {
                'test_1': pd.NaT,
                'test_2': [],
                'test_3': 'TEST',
                'test_4': 'TEST2'
            },
            {
                'test_1': 'TEST3',
                'test_2': ['TEST'],
                'test_3': 'TEST',
                'test_4': 'TEST2'
            },
            {
                'test_1': pd.np.NaN,
                'test_2': None,
                'test_3': 'TEST5',
                'test_4': 'TEST6'
            }
        ])
        expected_result_first = pd.DataFrame([
            {
                'test_1': None,
                'test_2': [],
                'test_3': 'TEST',
                'test_4': 'TEST2',
                'new_col_name': 'TEST'
            },
            {
                'test_1': 'TEST3',
                'test_2': ['TEST'],
                'test_3': 'TEST',
                'test_4': 'TEST2',
                'new_col_name': 'TEST3'
            },
            {
                'test_1': None,
                'test_2': None,
                'test_3': 'TEST5',
                'test_4': 'TEST6',
                'new_col_name': 'TEST5'
            }
        ])
        expected_result_last = pd.DataFrame([
            {
                'test_1': None,
                'test_2': [],
                'test_3': 'TEST',
                'test_4': 'TEST2',
                'new_col_name': 'TEST2'
            },
            {
                'test_1': 'TEST3',
                'test_2': ['TEST'],
                'test_3': 'TEST',
                'test_4': 'TEST2',
                'new_col_name': 'TEST2'
            },
            {
                'test_1': None,
                'test_2': None,
                'test_3': 'TEST5',
                'test_4': 'TEST6',
                'new_col_name': 'TEST6'
            }
        ])
        merge_columns(dataframe, ['test_1', 'test_3', 'test_4'], 'new_col_name', keep='first')
        assert_frame_equal(dataframe, expected_result_first, check_like=True)
        merge_columns(dataframe, ['test_1', 'test_3', 'test_4'], 'new_col_name', keep='last')
        assert_frame_equal(dataframe, expected_result_last, check_like=True)
        with self.assertRaises(ValueError):
            merge_columns(dataframe, ['test_1', 'test_3', 'test_4'], 'new_col_name', keep='something_wrong')
        with self.assertRaises(ValueError):
            merge_columns(dataframe, ['test_1', 'test_3', 'test_4'], 'new_col_name', aggr=sum, keep='first')

    def test_merge_columns_aggr(self):
        dataframe = pd.DataFrame([
            {
                'test_1': 1,
                'test_2': [],
                'test_3': 5,
                'test_4': 9
            },
            {
                'test_1': 0,
                'test_2': ['TEST'],
                'test_3': 9,
                'test_4': 7
            },
            {
                'test_1': 1,
                'test_2': None,
                'test_3': 8,
                'test_4': 1
            }
        ])
        expected_result = pd.DataFrame([
            {
                'test_1': 1,
                'test_2': [],
                'test_3': 5,
                'test_4': 9,
                'new_col_name': 15
            },
            {
                'test_1': 0,
                'test_2': ['TEST'],
                'test_3': 9,
                'test_4': 7,
                'new_col_name': 16
            },
            {
                'test_1': 1,
                'test_2': None,
                'test_3': 8,
                'test_4': 1,
                'new_col_name': 10
            }
        ])
        merge_columns(dataframe, ['test_1', 'test_3', 'test_4'], 'new_col_name', aggr=sum)
        assert_frame_equal(dataframe, expected_result, check_like=True, check_dtype=False)
        with self.assertRaises(ValueError):
            merge_columns(dataframe, ['test_1', 'test_3', 'test_4'], 'new_col_name', aggr='sum')

    def test_concatenate_columns_pos_01(self):
        dataframe = pd.DataFrame([
            {'key': 'TICKET-1', 'assignee': 'Bob', 'reporter': 'Alice'},
            {'key': 'TICKET-2', 'assignee': 'Bob', 'reporter': 'Alice'},
            {'key': 'TICKET-3', 'assignee': 'Bob', 'reporter': 'Alice'},
        ]).set_index('key')
        expected = pd.DataFrame([
            {'key': 'TICKET-1', 'user': 'Bob'},
            {'key': 'TICKET-1', 'user': 'Alice'},
            {'key': 'TICKET-2', 'user': 'Bob'},
            {'key': 'TICKET-2', 'user': 'Alice'},
            {'key': 'TICKET-3', 'user': 'Bob'},
            {'key': 'TICKET-3', 'user': 'Alice'},
        ]).set_index('key')
        assert_frame_equal(concatenate_columns(dataframe, ['assignee', 'reporter'], 'user'), expected)

    def test_concatenate_columns_pos_02(self):
        dataframe = pd.DataFrame([
            {'key': 'TICKET-1', 'assignee': 'Bob', 'reporter': 'Alice'},
            {'key': 'TICKET-2', 'assignee': 'Bob', 'reporter': 'Alice'},
            {'key': 'TICKET-3', 'assignee': 'Bob', 'reporter': 'Alice'},
        ]).set_index('key')
        expected = pd.DataFrame([
            {'key': 'TICKET-1', 'user': 'Bob', 'role': 'assignee'},
            {'key': 'TICKET-1', 'user': 'Alice', 'role': 'reporter'},
            {'key': 'TICKET-2', 'user': 'Bob', 'role': 'assignee'},
            {'key': 'TICKET-2', 'user': 'Alice', 'role': 'reporter'},
            {'key': 'TICKET-3', 'user': 'Bob', 'role': 'assignee'},
            {'key': 'TICKET-3', 'user': 'Alice', 'role': 'reporter'},
        ]).set_index('key')[['user', 'role']]
        assert_frame_equal(
            concatenate_columns(dataframe, ['assignee', 'reporter'], 'user', descriptor='role'),
            expected
        )

    def test_concatenate_columns_pos_03(self):
        dataframe = pd.DataFrame([
            {'key': 'TICKET-1', 'assignee': 'Bob', 'reporter': 'Alice'},
            {'key': 'TICKET-2', 'assignee': 'Bob', 'reporter': 'Alice'},
            {'key': 'TICKET-3', 'assignee': 'Bob', 'reporter': 'Alice'},
        ]).set_index('key')
        expected = pd.DataFrame([
            {'key': 'TICKET-1', 'user': 'Bob', 'role': 'a'},
            {'key': 'TICKET-1', 'user': 'Alice', 'role': 'r'},
            {'key': 'TICKET-2', 'user': 'Bob', 'role': 'a'},
            {'key': 'TICKET-2', 'user': 'Alice', 'role': 'r'},
            {'key': 'TICKET-3', 'user': 'Bob', 'role': 'a'},
            {'key': 'TICKET-3', 'user': 'Alice', 'role': 'r'},
        ]).set_index('key')[['user', 'role']]
        mapper = {'assignee': 'a', 'reporter': 'r'}
        assert_frame_equal(
            concatenate_columns(dataframe, ['assignee', 'reporter'], 'user', descriptor='role', mapper=mapper),
            expected
        )

    def test_concatenate_columns_non_existent_col(self):
        dataframe = pd.DataFrame([
            {'key': 'TICKET-1', 'assignee': 'Bob', 'reporter': 'Alice'},
            {'key': 'TICKET-2', 'assignee': 'Bob', 'reporter': 'Alice'},
            {'key': 'TICKET-3', 'assignee': 'Bob', 'reporter': 'Alice'},
        ]).set_index('key')
        expected = pd.DataFrame([
            {'key': 'TICKET-1', 'user': 'Bob', 'role': 'a'},
            {'key': 'TICKET-1', 'user': 'Alice', 'role': 'r'},
            {'key': 'TICKET-2', 'user': 'Bob', 'role': 'a'},
            {'key': 'TICKET-2', 'user': 'Alice', 'role': 'r'},
            {'key': 'TICKET-3', 'user': 'Bob', 'role': 'a'},
            {'key': 'TICKET-3', 'user': 'Alice', 'role': 'r'},
        ]).set_index('key')[['user', 'role']]
        mapper = {'assignee': 'a', 'reporter': 'r'}
        assert_frame_equal(
            concatenate_columns(dataframe, ['assignee', 'reporter', 'creator'], 'user', descriptor='role', mapper=mapper),
            expected
        )


if __name__ == '__main__':
    unittest.main()
