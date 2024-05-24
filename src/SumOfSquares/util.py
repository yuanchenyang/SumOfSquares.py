import sympy as sp
import numpy as np
import math
from operator import mul
from functools import reduce
from itertools import combinations
from typing import Iterable, List, Optional, Tuple

def prod(seq):
    return reduce(mul, seq, 1)

def factorial(n: int) -> int:
    return prod(range(1, n+1))

def binom(n: int, d: int) -> int:
    assert n >= d, f'invalid binom({n}, {d})!'
    return factorial(n)//factorial(n-d)//factorial(d)

def sum_tuple(t1: tuple, t2: tuple) -> tuple:
    return tuple(a1+a2 for a1, a2 in zip(t1, t2))

def is_hom(poly: sp.Poly, deg: int) -> bool:
    '''Determines if a polynomial POLY is homogeneous of degree DEG'''
    if deg == 0:
        return poly == 0
    return sum(sum(m) != deg for m in poly.monoms()) == 0

def round_sympy_expr(expr: sp.Expr, precision: int=3) -> sp.Expr:
    '''Rounds all numbers in a sympy expression to stated precision'''
    return expr.xreplace({n : round(n, precision) for n in expr.atoms(sp.Number)})

def poly_degree(p: sp.Expr, variables: List[sp.Symbol]) -> int:
    '''Returns the max degree of P when treated as a polynomial in VARIABLES'''
    return sp.poly(p, variables).total_degree()

def orth(M: np.array) -> Tuple[np.array, np.array]:
    _, D, V = np.linalg.svd(M)
    return V[D >= 1e-9], V[D < 1e-9]

def get_poly_degree(vars, polys: List[sp.Poly], deg: Optional[int]=None) -> int:
    '''Given a vector of polynomials POLY, return minimum degree to run sum of
    squares, or check if DEG is above such minimum degree if provided'''
    max_deg = max(map(lambda p: poly_degree(p, vars), polys))
    if deg is None:
        deg = math.ceil(max_deg/2)
    if 2*deg < max_deg:
        raise ValueError(f'Degree of relaxation 2*{deg} less than maximum degree {max_deg}')
    return deg
