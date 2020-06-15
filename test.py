import unittest
import sympy as sp
import numpy as np
import picos as pic

from util import *
from basis import basis_hom, basis_inhom, Basis
from SoS import SOSProblem

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


class TestSoS(unittest.TestCase):

    def test_sos_simple(self):
        x, y = sp.symbols('x y')
        p1 = 2*x**4 + 2*x**3*y - x**2*y**2 + 5*y**4        # SoS
        p2 = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1       # Not SoS
        p3 = p2*(x**2 + y**2 + 1)                          # SoS

        prob = SOSProblem()
        c = prob.add_sos_constraint(p1, [x, y])
        self.assertTrue(c.basis.is_hom)
        prob.solve()

        # Extract SoS decomposition
        S = c.get_sos_decomp()
        self.assertTrue(round_sympy_expr(sp.expand(sum(S))-p1, precision=2) == 0)

        prob = SOSProblem()
        c = prob.add_sos_constraint(p2, [x, y])
        self.assertFalse(c.basis.is_hom)
        with self.assertRaises(pic.SolutionFailure):
            prob.solve()

        prob = SOSProblem()
        c = prob.add_sos_constraint(p3, [x, y])
        prob.solve()

    def test_sos_opt(self):
        x, y, s, t = sp.symbols('x y s t')
        p = s*x**6 + t*y**6 - x**4*y**2 - x**2*y**4 - x**4 + 3*x**2*y**2 \
            - y**4 - x**2 - y**2 + 1
        prob = SOSProblem()
        prob.add_sos_constraint(p, [x, y])
        sv, tv = prob.sym_to_var(s), prob.sym_to_var(t)
        prob.set_objective('min', 2*sv+tv)
        prob.solve(solver='mosek')
        self.assertAlmostEqual(sv.value, 1.0991855923132654)
        self.assertAlmostEqual(tv.value, 1.349190696633841)

    def test_multiple_sos(self):
        x, y, t = sp.symbols('x y t')
        p1 = t*(1 + x*y)**2 - x*y + (1 - y)**2
        p2 = (1 - x*y)**2 + x*y + t*(1 + y)**2
        prob = SOSProblem()
        prob.add_sos_constraint(p1, [x, y])
        prob.add_sos_constraint(p2, [x, y])
        tv = prob.sym_to_var(t)
        prob.set_objective('min', tv)
        prob.solve(solver='mosek')
        self.assertAlmostEqual(tv.value, 0.24999999503912618)

    def test_unconstrained_poly_opt(self):
        x, y, t = sp.symbols('x y t')
        p = x**4 + x**2 - 3*x**2*y**2 + y**6
        prob = SOSProblem()
        prob.add_sos_constraint(p-t, [x, y])
        tv = prob.sym_to_var(t)
        prob.set_objective('max', tv)
        prob.solve(solver='cvxopt', verbosity=0)
        self.assertAlmostEqual(tv.value, -0.1725688562256166)

    def test_isocahedral_form(self):
        x, y, z, t = sp.symbols('x y z t')
        phi = (1+np.sqrt(5))/2
        f = (x+phi*y)*(x-phi*y)*(y+phi*z)*(y-phi*z)*(z+phi*x)*(z-phi*x)
        p = t * (x**2 + y**2 + z**2)**3 - f
        prob = SOSProblem()
        c = prob.add_sos_constraint(p, [x, y, z])
        tv = prob.sym_to_var(t)
        prob.set_objective('min', tv)
        prob.solve(solver='mosek')
        self.assertAlmostEqual(tv.value, 0.8472135957347698)

        # Test pseudoexpectations
        self.assertAlmostEqual(c.pexpect((x**2 + y**2 + z**2)**3), 1)
        self.assertAlmostEqual(c.pexpect(f), 0.8472135957347698)

    def test_pexpect(self):
        x, y, z = sp.symbols('x y z')
        phi = (1+np.sqrt(5))/2
        f = (x+phi*y)*(x-phi*y)*(y+phi*z)*(y-phi*z)*(z+phi*x)*(z-phi*x)
        x2 = (x**2 + y**2 + z**2)

        prob = SOSProblem()
        PEx = prob.get_pexpect([x, y, z], 6, hom=True)
        prob.add_constraint(PEx(x2**3) == 1)
        prob.set_objective('max', PEx(f))
        prob.solve(solver='mosek')
        self.assertAlmostEqual(PEx(f), 0.8472136281136226)

if __name__ == '__main__':
    unittest.main()
