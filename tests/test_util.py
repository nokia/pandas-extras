import unittest
from datetime import datetime

import pandas as pd
import pandas.api.types as ptypes
from pandas.testing import assert_frame_equal

from pandas_extras import check_duplicated_labels


class UtilTestCase(unittest.TestCase):
    def test_check_duplicated_labels_pos_01(self):
        df = pd.DataFrame(
            {
                'test_index': [1, 2, 3, 4, 5, 6],
                'trial_num': [1, 2, 3, 1, 2, 3],
                'subject': [1, 1, 1, 2, 2, 2],
            }
        ).set_index('test_index')
        self.assertIsNotNone(check_duplicated_labels(df))

    def test_check_duplicated_labels_neg_01(self):
        df = pd.DataFrame(
            {
                'test_index': [1, 2, 3, 4, 5, 6],
                'trial_num': [1, 2, 3, 1, 2, 3],
                'trial_num_1': [1, 1, 1, 2, 2, 2],
            }
        ).set_index('test_index').rename({'trial_num_1': 'trial_num'}, axis=1)
        with self.assertRaises(ValueError):
            check_duplicated_labels(df)


if __name__ == '__main__':
    unittest.main()
