# pandas-extras
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
