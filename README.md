# pandas-extras
![Lint](https://github.com/nokia/pandas-extras/workflows/Pylint/badge.svg)
![Tests](https://github.com/nokia/pandas-extras/workflows/Coverage/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pandas-extras/badge/?version=latest)](https://pandas-extras.readthedocs.io/en/latest/?badge=latest)

Extension package for the popular Pandas library

# Installation
The package can be installed through pip:
```console
$ pip install pandas-extras
```

# Usage
The package contains helper function that are best called with the `pipe` operator of `DataFrame`.

```python
from pandas_extras import expand_lists

df = DataFrame(...)
df.pipe(expand_lists, *args, **kwargs)
```

## License

This project is licensed under the BSD-3-Clause license - see the [LICENSE](https://github.com/nokia/pandas-extras/blob/master/LICENSE).
