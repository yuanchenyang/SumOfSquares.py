import unittest
import sympy as sp
from SumOfSquares import *
from SumOfSquares.util import binom, poly_degree, is_hom

class TestBasis(unittest.TestCase):
    def test_basis_generator(self):
        for n, d in ((9, 4), (10, 3), (3,3), (2, 3), (1, 5), (5, 1)):
            self.assertEqual(len(list(basis_inhom(n, d))), binom(n+d,d))
            self.assertEqual(len(list(basis_hom(n, d))), binom(n+d-1,d))

    def test_basis(self):
        x, y = sp.symbols('x y')
        b = Basis.from_degree(2, 3, hom=True)

        # Should not return error
        b.check_can_represent(sp.poly(x**2*y**4))

        with self.assertRaises(ValueError):
            # Degree too high
            b.check_can_represent(sp.poly(x**7))

        with self.assertRaises(ValueError):
            # Not homogeneous
            b.check_can_represent(sp.poly(x**2*y**3))

    def test_poly_variable(self):
        vs = sp.symbols('x y')
        for hom in (True, False):
            for deg in range(5):
                p = poly_variable('c', vs, deg, hom=hom)
                self.assertEqual(poly_degree(p, vs), deg)
                self.assertEqual(is_hom(sp.Poly(p, vs), deg), hom)

    def test_sym_matrix_variable(self):
        x, y = sp.symbols('x y')
        n = 4
        deg = 2
        M = matrix_variable('M', [x, y], deg, n, hom=False, sym=True)
        for i in range(n):
            for j in range(n):
                self.assertEqual(M[i,j], M[j,i])
