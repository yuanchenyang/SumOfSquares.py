import unittest
import sympy as sp
import numpy as np
import picos as pic

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
        sv = prob.sym_to_var(s)
        tv = prob.sym_to_var(t)
        prob.set_objective('min', 2*sv + tv)
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
        prob.set_objective('min', prob.sym_to_var(t))
        prob.solve(solver='mosek')
        self.assertAlmostEqual(prob.value, 0.24999999503912618)

    def test_unconstrained_poly_opt(self):
        x, y, t = sp.symbols('x y t')
        p = x**4 + x**2 - 3*x**2*y**2 + y**6
        prob = SOSProblem()
        c = prob.add_sos_constraint(p-t, [x, y], sparse=True) # Newton polytope
        prob.set_objective('max', prob.sym_to_var(t))
        prob.solve(solver='mosek')
        self.assertAlmostEqual(prob.value, -0.17797853649283987)

    def test_unconstrained_poly_opt_sparse(self):
        x, y, z, t = sp.symbols('x y z t')
        p = x**4*z**2 + x**2*z**4 - 3*x**2*y**2*z**2 + y**6
        prob = SOSProblem()
        # Newton polytope reduction
        c = prob.add_sos_constraint(p-t*z**6, [x, y, z], sparse=True)
        prob.set_objective('max', prob.sym_to_var(t))
        prob.solve(solver='mosek')
        self.assertAlmostEqual(prob.value, -0.17797853649283987)
        monoms = set([(0, 0, 3), (0, 1, 2), (1, 0, 2), (0, 2, 1),
                      (1, 1, 1), (2, 0, 1), (0, 3, 0)])
        self.assertEqual(monoms, set(c.basis.monoms))

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

    def test_chebyshev(self):
        # Compute leading coefficient of Chebyshev polynomials
        deg = 8
        x, gam = sp.symbols('x gam')
        p = gam * x**deg + poly_variable('p1', [x], deg-1)
        prob = SOSProblem()

        # -1 <= p <= 1 on interval [-1, 1]
        t1 = poly_variable('t1', [x], deg-2)
        t2 = poly_variable('t2', [x], deg-2)
        prob.add_sos_constraint(t1, [x])
        prob.add_sos_constraint(t2, [x])
        prob.add_sos_constraint(1-p + (x+1)*(x-1)*t1, [x])
        prob.add_sos_constraint(p+1 + (x+1)*(x-1)*t2, [x])

        prob.set_objective('max', prob.sym_to_var(gam))
        prob.solve(solver='mosek')
        self.assertAlmostEqual(prob.value, 127.99999971712037)

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

    def test_pexpect_cert(self):
        # Pseudoexpectation certificate
        x, y = sp.symbols('x y')
        p = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1 # Motzkin polynomial
        prob = SOSProblem()
        pEx = prob.get_pexpect([x, y], 6)
        prob.add_constraint(pEx(p) == -1)
        prob.solve()
        self.assertAlmostEqual(pEx(p), -1)

    def test_eq_constrained_poly_opt(self):
        x, y = sp.symbols('x y')
        prob = poly_opt_prob([x, y], x - y, [x**2-x, y**2-y], [], 1)
        prob.solve(solver='mosek')
        self.assertAlmostEqual(prob.value, -1.0000000022528606)

    def test_ineq_constrained_poly_opt(self):
        x, y = sp.symbols('x y')
        prob1 = poly_opt_prob([x, y], x + y, [x**2+y**2-1, y-x**2-0.5], [x, y-0.5], 1)
        prob1.solve(solver='mosek')
        self.assertAlmostEqual(prob1.value, 0.49999999888338087)

        prob2 = poly_opt_prob([x, y], x + y, [x**2+y**2-1, y-x**2-0.5], [x, y-0.5], 2)
        prob2.solve(solver='mosek')
        self.assertAlmostEqual(prob2.value, 1.3910970905754896)


if __name__ == '__main__':
    unittest.main()
