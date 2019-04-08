import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from pandas_extras import flatten_adjacency_list, get_adjacency_list_depth


class HierarchyTestCase(unittest.TestCase):
    def test_flatten_adjacency_list_pos_01(self):
        dataframe = pd.DataFrame([
            {'employee': 0, 'manager': None},
            {'employee': 1, 'manager': 0},
            {'employee': 2, 'manager': 0},
            {'employee': 3, 'manager': 0},
            {'employee': 4, 'manager': 1},
            {'employee': 5, 'manager': 1},
            {'employee': 6, 'manager': 2},
            {'employee': 7, 'manager': 6},
        ])
        expected = pd.DataFrame([
            {'employee': 0, 'manager': None, 'manager_1': None, 'manager_2': None},
            {'employee': 1, 'manager': 0, 'manager_1': None, 'manager_2': None},
            {'employee': 2, 'manager': 0, 'manager_1': None, 'manager_2': None},
            {'employee': 3, 'manager': 0, 'manager_1': None, 'manager_2': None},
            {'employee': 4, 'manager': 1, 'manager_1': 0, 'manager_2': None},
            {'employee': 5, 'manager': 1, 'manager_1': 0, 'manager_2': None},
            {'employee': 6, 'manager': 2, 'manager_1': 0, 'manager_2': None},
            {'employee': 7, 'manager': 6, 'manager_1': 2, 'manager_2': 0},
        ])
        assert_frame_equal(
            flatten_adjacency_list(dataframe, 'manager', right_on='employee'),
            expected
        )

    def test_flatten_adjacency_list_pos_02(self):
        dataframe = pd.DataFrame([
            {'employee': 0, 'manager': None, 'foo': 'bar'},
            {'employee': 1, 'manager': 0, 'foo': 'bar'},
            {'employee': 2, 'manager': 0, 'foo': 'bar'},
            {'employee': 3, 'manager': 0, 'foo': 'bar'},
            {'employee': 4, 'manager': 1, 'foo': 'bar'},
            {'employee': 5, 'manager': 1, 'foo': 'bar'},
            {'employee': 6, 'manager': 2, 'foo': 'bar'},
            {'employee': 7, 'manager': 6, 'foo': 'bar'},
        ]).set_index('employee')
        expected = pd.DataFrame([
            {'employee': 0, 'manager': None, 'foo': 'bar', 'manager_1': None, 'manager_2': None},
            {'employee': 1, 'manager': 0, 'foo': 'bar', 'manager_1': None, 'manager_2': None},
            {'employee': 2, 'manager': 0, 'foo': 'bar', 'manager_1': None, 'manager_2': None},
            {'employee': 3, 'manager': 0, 'foo': 'bar', 'manager_1': None, 'manager_2': None},
            {'employee': 4, 'manager': 1, 'foo': 'bar', 'manager_1': 0, 'manager_2': None},
            {'employee': 5, 'manager': 1, 'foo': 'bar', 'manager_1': 0, 'manager_2': None},
            {'employee': 6, 'manager': 2, 'foo': 'bar', 'manager_1': 0, 'manager_2': None},
            {'employee': 7, 'manager': 6, 'foo': 'bar', 'manager_1': 2, 'manager_2': 0},
        ]).set_index('employee')
        assert_frame_equal(
            flatten_adjacency_list(dataframe, 'manager'),
            expected
        )

    def test_get_adjacency_list_depth_pos_01(self):
        dataframe = pd.DataFrame([
            {'employee': 0, 'manager': None},
            {'employee': 1, 'manager': 0},
            {'employee': 2, 'manager': 0},
            {'employee': 3, 'manager': 0},
            {'employee': 4, 'manager': 1},
            {'employee': 5, 'manager': 1},
            {'employee': 6, 'manager': 2},
            {'employee': 7, 'manager': 6},
        ])
        expected = pd.DataFrame([
            {'employee': 0, 'manager': None, 'depth': 0},
            {'employee': 1, 'manager': 0, 'depth': 1},
            {'employee': 2, 'manager': 0, 'depth': 1},
            {'employee': 3, 'manager': 0, 'depth': 1},
            {'employee': 4, 'manager': 1, 'depth': 2},
            {'employee': 5, 'manager': 1, 'depth': 2},
            {'employee': 6, 'manager': 2, 'depth': 2},
            {'employee': 7, 'manager': 6, 'depth': 3},
        ])
        assert_frame_equal(
            get_adjacency_list_depth(dataframe, 'manager', right_on='employee'),
            expected[['employee', 'manager', 'depth']]
        )

    def test_get_adjacency_list_depth_pos_02(self):
        dataframe = pd.DataFrame([
            {'employee': 0, 'manager': None},
            {'employee': 1, 'manager': 0},
            {'employee': 2, 'manager': 0},
            {'employee': 3, 'manager': 0},
            {'employee': 4, 'manager': 1},
            {'employee': 5, 'manager': 1},
            {'employee': 6, 'manager': 2},
            {'employee': 7, 'manager': 6},
        ]).set_index('employee')
        expected = pd.DataFrame([
            {'employee': 0, 'manager': None, 'level': 0},
            {'employee': 1, 'manager': 0, 'level': 1},
            {'employee': 2, 'manager': 0, 'level': 1},
            {'employee': 3, 'manager': 0, 'level': 1},
            {'employee': 4, 'manager': 1, 'level': 2},
            {'employee': 5, 'manager': 1, 'level': 2},
            {'employee': 6, 'manager': 2, 'level': 2},
            {'employee': 7, 'manager': 6, 'level': 3},
        ]).set_index('employee')
        assert_frame_equal(
            get_adjacency_list_depth(dataframe, 'manager', new_column='level'),
            expected[['manager', 'level']]
        )


if __name__ == '__main__':
    unittest.main()
