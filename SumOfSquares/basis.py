import sympy as sp
import numpy as np
import math
from collections import defaultdict
from scipy.spatial import ConvexHull

from .util import *

def basis_hom(n, d):
    '''Generator for a homogeneous polynomial basis for n variables of degree d,
    represented as a list of tuples (same as sympy), so that:
    len(list(basis_hom(n,d))) == binom(n+d-1, d)
    '''
    if n == 1:
        yield (d,)
    elif d == 0:
        yield (0,)*n
    else:
        for di in range(d+1):
            for b in basis_hom(n-1, di):
                yield b + (d-di,)

def basis_inhom(n, d):
    '''Generator for an inhomogeneous polynomial basis for n variables of
    degree d, represented as a list of tuples (same as sympy), so that:
    len(list(basis(n,d))) == binom(n+d, d)
    '''
    return (b[:-1] for b in basis_hom(n+1, d))

class Basis():
    def __init__(self, monoms):
        '''Initializes a basis using a sequence of tuples representing monomials
        '''
        self.monoms = monoms
        self.deg = max(sum(m) for m in self.monoms)
        self.nvars = len(monoms[0])
        self.is_hom = sum(sum(m) != self.deg for m in self.monoms) == 0

        # A map from a monomial m (represented as tuple) to list of pairs (i, j)
        # for all such pairs where m = basis[i]*basis[j].
        self.sos_sym_entries = defaultdict(list)
        for i, bi in enumerate(self):
            for j, bj in enumerate(self):
                 self.sos_sym_entries[sum_tuple(bi, bj)].append((i, j))

    def __len__(self):
        return len(self.monoms)

    def __iter__(self):
        return iter(self.monoms)

    def from_degree(nvars, deg, hom=False):
        '''Constructs a basis by specifying the number of variables and degree'''
        return Basis(list((basis_hom if hom else basis_inhom)(nvars, deg)))

    def from_poly_lex(poly, sparse=True):
        '''Returns a basis from a polynomial compatible with SoS,
        ordering monomials in lexicographic order'''
        poly_deg = poly.total_degree()
        monoms = np.array(poly.monoms())/2
        full_basis = Basis.from_degree(len(poly.gens), math.ceil(poly_deg / 2),
                                       is_hom(poly, poly_deg))
        if sparse and len(monoms) >= 3: # Newton polytope sparsity reduction
            a = np.mean(monoms, axis=0)
            U, U_ = orth(monoms - a) # U orthogonal to U_
            proj, proj_ = lambda m: U.dot(m - a), lambda m: U_.dot(m - a)
            hull = ConvexHull(np.apply_along_axis(proj, 1, monoms))
            def in_hull(pt): # Point lies in affine subspace and convex hull
                return np.linalg.norm(proj_(pt)) < 1e-9 and \
                    sum(hull.equations.dot(np.append(proj(pt), 1)) > 1e-9) == 0
            return Basis(list(filter(in_hull, full_basis.monoms)))
        return full_basis

    def to_sym(self, syms):
        '''Convert basis to a list of symbolic monomials
        '''
        if self.nvars != len(syms):
            raise ValueError('Mismatched basis size!')
        return [prod(s**m for s,m in zip(syms, mono)) for mono in self]

    def check_can_represent(self, poly):
        '''Check if sympy polynomial POLY can be represented by this basis.
        Raises an error otherwise.'''
        if poly.total_degree() > self.deg * 2:
           raise ValueError(f'Polynomial degree must be at most {self.deg*2}!')

        if self.is_hom and not is_hom(poly, self.deg * 2):
            raise ValueError(f'Polynomial must be homogeneous of degree {self.deg * 2}!')

        extra_monoms = len(set(poly.monoms()) - set(self.sos_sym_entries.keys()))
        if extra_monoms > 0:
            raise ValueError(f'{extra_monoms} monomials in polynomial'\
                             ' are not in basis!')


    def sos_sym_poly_repr(self, poly):
        '''Given a polynomial p, returns a SoS-symmetric representation
        Qp where p(x) = b(x)^T Qp b(x), in terms of this basis.
        Qp is returned as a sympy matrix.
        '''
        Qp = sp.zeros(len(self), len(self))
        for monom, coeff in zip(poly.monoms(), poly.coeffs()):
            entries = self.sos_sym_entries[monom]
            for i, j in entries:
                Qp[i,j] = coeff/len(entries)
        return Qp


def poly_variable(name, variables, deg, hom=False):
    '''Returns a (possibly homogeneous) degree DEG polynomial in VARIABLES,
    with a variable (a sympy symbol) named using NAME for each coefficient. Used
    in Sum-of-Squares relaxations for polynomial optimization.
    '''
    variables = sorted(variables, key=str) # use lex order
    basis = Basis.from_degree(len(variables), deg, hom=hom)
    coeffs = sp.symbols(f'{name}_:{len(basis)}')
    return sum(coeff * prod(var**power for var, power in zip(variables, monom))
               for monom, coeff in zip(basis, coeffs))