SumOfSquares.py
---------------
| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![][docs-latest-img]][docs-latest-url] | [![Build Status][build-img]][build-url] |


Sum of squares optimization built on top of
[picos](https://picos-api.gitlab.io/picos/). Easy access to pseudoexpectation
operators for both formulating problems and extracting solutions via rounding
algorithms.


### Installation

To install from [pypi](https://pypi.org/project/SumOfSquares/):

```
pip install SumOfSquares
```

### Examples

To compute the sum of squares decomposition of a polynomial:
```python
>>> import sympy as sp
>>> x, y = sp.symbols('x y')
>>> p = 2*x**4 + 2*x**3*y - x**2*y**2 + 5*y**4
>>> prob = SOSProblem()
>>> c = prob.add_sos_constraint(p, [x, y])
>>> prob.solve()
>>> c.get_sos_decomp()
Matrix([
[5.0*(-0.306*x**2 + y**2)**2],
[2.057*(0.486*x**2 + x*y)**2],
[                 1.047*x**4]])
```

[More Examples](https://sums-of-squares.github.io/sos/index.html#python)

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://sums-of-squares.github.io/sos/index.html#python
[build-img]: https://github.com/yuanchenyang/SumOfSquares.py/workflows/CI/badge.svg?branch=master
[build-url]: https://github.com/yuanchenyang/SumOfSquares.py/actions?query=workflow%3ACI
