import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from pandas.testing import assert_frame_equal

from pandas_extras import NativeDict, clear_nan, convert_to_type, truncate_strings

class ConversionsTestCase(unittest.TestCase):

    def test_convert_if_needed_pos_01(self):
        self.assertEqual(NativeDict.convert_if_needed(np.NaN), None)
        self.assertEqual(NativeDict.convert_if_needed(np.int32(2)), 2)
        self.assertEqual(NativeDict.convert_if_needed(np.int32(2)), 2)
        self.assertEqual(NativeDict.convert_if_needed(np.uint32(2)), 2)
        self.assertEqual(NativeDict.convert_if_needed(np.uint64(2)), 2)
        self.assertEqual(NativeDict.convert_if_needed(np.float32(1.0)), 1.0)
        self.assertEqual(NativeDict.convert_if_needed(np.float64(1.0)), 1.0)

    def test_to_dict_with_cls_pos_01(self):
        orig_dict_list = [
            {'int': 1, 'float': 2.0, 'nan': 1, 'nat': datetime.today(), 'other': 'value'},
            {'int': 1, 'float': 2.0, 'nan': None, 'nat': datetime.now(), 'other': 'value'},
            {'int': 1, 'float': 2.0, 'nan': None, 'nat': None, 'other': 'value'},
        ]
        self.assertListEqual(pd.DataFrame(orig_dict_list).to_dict(orient='records', into=NativeDict), orig_dict_list)

    def test_convert_to_type_pos_01(self):
        df = pd.DataFrame({
            'date': ['05/06/2018', '05/04/2018'],
            'datetime': ['2018-06-05T10:07:31', '2018-04-05T21:56:14'],
            'number': ['1', '2.34'],
            'int': [4, 8103],
            'float': [4.0, 8103.0],
            'object': ['just some', 'strings']
        })
        mapper = {
            'number': ['number'],
            'date': 'date',
            'datetime': ['datetime'],
            'integer': 'int',
            'float': ['float']
        }
        res = convert_to_type(df, mapper, *mapper.keys())
        assert_frame_equal(res, convert_to_type(df, mapper), check_like=True)
        self.assertTrue(ptypes.is_datetime64_ns_dtype(res['date'].dtype))
        self.assertTrue(ptypes.is_datetime64_ns_dtype(res['datetime'].dtype))
        self.assertTrue(ptypes.is_float_dtype(res['number'].dtype))
        self.assertTrue(ptypes.is_integer_dtype(res['int'].dtype))
        self.assertTrue(ptypes.is_float_dtype(res['float'].dtype))
        self.assertTrue(ptypes.is_object_dtype(res['object'].dtype))

    def test_convert_to_type_pos_02(self):
        df = pd.DataFrame({
            'date': ['05/06/2018', '05/04/2018'],
            'datetime': [1543844249621, 1543844249621],
            'number': ['1', '2.34'],
            'int': [4, 8103],
            'float': [4.0, 8103.0],
            'object': ['just some', 'strings']
        })
        mapper = {
            'number': ['number'],
            'date': 'date',
            'datetime': ['datetime'],
            'integer': 'int',
            'float': ['float']
        }
        kwargs_map = {'datetime': {'unit': 'ms'}}
        res = convert_to_type(df, mapper, *mapper.keys(), kwargs_map=kwargs_map)
        self.assertTrue(ptypes.is_datetime64_ns_dtype(res['datetime'].dtype))
        self.assertListEqual(res['datetime'].dt.year.tolist(), [2018, 2018])

    def test_clear_nan_pos_01(self):
        df = pd.DataFrame({
            'testcol1': [1, 2, np.NaN],
            'testcol2': [np.NaN, np.NaN, 3],
            'testcol3': [1, 2, 3]
        })

        result = clear_nan(df)

        expected_result = pd.DataFrame({
            'testcol1': [1, 2, None],
            'testcol2': [None, None, 3],
            'testcol3': [1, 2, 3]
        })
        assert_frame_equal(result, expected_result, check_like=True, check_dtype=False)

    def test_truncate_strings(self):
        df = pd.DataFrame({
            'strings': [
                'foo',
                'baz',
            ],
            'long_strings': [
                'foofoofoofoofoo',
                'bazbazbazbazbaz',
            ],
            'even_longer_strings': [
                'foofoofoofoofoofoofoofoo',
                'bazbazbazbazbazbazbazbaz',
            ]
        })
        expected = pd.DataFrame({
            'strings': [
                'foo',
                'baz',
            ],
            'long_strings': [
                'foofoo',
                'bazbaz',
            ],
            'even_longer_strings': [
                'foofoofoo',
                'bazbazbaz',
            ]
        })
        assert_frame_equal(df.pipe(truncate_strings, {'long_strings': 6, 'even_longer_strings': 9}), expected)


if __name__ == '__main__':
    unittest.main()
