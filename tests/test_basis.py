import unittest
import sympy as sp
from SumOfSquares import *
from SumOfSquares.util import binom

class TestBasis(unittest.TestCase):
    def test_basis_generator(self):
        for n, d in ((9, 4), (10, 3), (3,3), (2, 3), (1, 5), (5, 1)):
            assert len(list(basis_inhom(n, d))) == binom(n+d,d)
            assert len(list(basis_hom(n, d))) == binom(n+d-1,d)

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
