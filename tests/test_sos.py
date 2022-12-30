import unittest
import sympy as sp
import numpy as np
import picos as pic

from SumOfSquares import *

class TestSoS(unittest.TestCase):

    def setUp(self):
        self.solver = 'cvxopt'

    def test_sos_simple(self):
        x, y = sp.symbols('x y')
        p1 = 2*x**4 + 2*x**3*y - x**2*y**2 + 5*y**4        # SoS
        p2 = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1       # Not SoS
        p3 = p2*(x**2 + y**2 + 1)                          # SoS

        prob = SOSProblem()
        c = prob.add_sos_constraint(p1, [x, y])
        self.assertTrue(c.basis.is_hom)
        prob.solve(solver=self.solver)

        # Extract SoS decomposition
        S = c.get_sos_decomp()
        self.assertTrue(round_sympy_expr(sp.expand(sum(S))-p1, precision=2) == 0)

        prob = SOSProblem()
        c = prob.add_sos_constraint(p2, [x, y])
        self.assertFalse(c.basis.is_hom)
        with self.assertRaises(pic.SolutionFailure):
            prob.solve(solver=self.solver)

        prob = SOSProblem()
        c = prob.add_sos_constraint(p3, [x, y])
        prob.solve(solver=self.solver)

    def test_sos_opt(self):
        x, y, s, t = sp.symbols('x y s t')
        p = s*x**6 + t*y**6 - x**4*y**2 - x**2*y**4 - x**4 + 3*x**2*y**2 \
            - y**4 - x**2 - y**2 + 1
        prob = SOSProblem()
        prob.add_sos_constraint(p, [x, y])
        prob.set_objective('min', 2*prob[s] + prob[t])
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(prob[s].value, 1.0991922234972025, 4)
        self.assertAlmostEqual(prob[t].value, 1.3491774310708642, 4)

    def test_multiple_sos(self):
        x, y, t = sp.symbols('x y t')
        p1 = t*(1 + x*y)**2 - x*y + (1 - y)**2
        p2 = (1 - x*y)**2 + x*y + t*(1 + y)**2
        prob = SOSProblem()
        prob.add_sos_constraint(p1, [x, y])
        prob.add_sos_constraint(p2, [x, y])
        prob.set_objective('min', prob[t])
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(prob.value, 0.24999999503912618, 5)

    def test_unconstrained_poly_opt(self):
        x, y, t = sp.symbols('x y t')
        p = x**4 + x**2 - 3*x**2*y**2 + y**6
        prob = SOSProblem()
        c = prob.add_sos_constraint(p-t, [x, y], sparse=True) # Newton polytope
        prob.set_objective('max', prob[t])
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(prob.value, -0.17797853649283987, 5)

    def test_unconstrained_poly_opt_sparse(self):
        x, y, z, t = sp.symbols('x y z t')
        p = x**4*z**2 + x**2*z**4 - 3*x**2*y**2*z**2 + y**6
        prob = SOSProblem()
        # Newton polytope reduction
        c = prob.add_sos_constraint(p-t*z**6, [x, y, z], sparse=True)
        prob.set_objective('max', prob[t])
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(prob.value, -0.17797853649283987, 5)
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
        prob.set_objective('min', prob[t])
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(prob[t].value, 0.8472135957347698, 5)

        # Test pseudoexpectations
        self.assertAlmostEqual(c.pexpect((1.0*x**2 + y**2 + z**2)**3), 1)
        self.assertAlmostEqual(c.pexpect(f), 0.8472135957347698, 5)

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

        prob.set_objective('max', prob[gam])
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(prob.value, 127.99999971712037, 5)

    def test_pexpect(self):
        x, y, z = sp.symbols('x y z')
        phi = (1+np.sqrt(5))/2
        f = (x+phi*y)*(x-phi*y)*(y+phi*z)*(y-phi*z)*(z+phi*x)*(z-phi*x)
        x2 = (x**2 + y**2 + z**2)

        prob = SOSProblem()
        PEx = prob.get_pexpect([x, y, z], 6, hom=True)
        prob.add_constraint(PEx(x2**3) == 1)
        prob.set_objective('max', PEx(f))
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(PEx(f), 0.8472136281136226, 5)

    def test_pexpect_cert(self):
        # Pseudoexpectation certificate
        x, y = sp.symbols('x y')
        p = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1 # Motzkin polynomial
        prob = SOSProblem()
        pEx = prob.get_pexpect([x, y], 6)
        prob.add_constraint(pEx(p) == -1)
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(pEx(p), -1, 7)

    def test_eq_constrained_poly_opt(self):
        x, y = sp.symbols('x y')
        prob = poly_opt_prob([x, y], x - y,
                             eqs=[x**2-x, y**2-y],
                             ineqs=[],
                             deg=1)
        prob.solve(solver=self.solver)
        self.assertAlmostEqual(prob.value, -1.0000000022528606, 5)

    def test_ineq_constrained_poly_opt(self):
        x, y = sp.symbols('x y')
        prob1 = poly_opt_prob([x, y], x + y,
                              eqs=[x**2+y**2-1, y-x**2-0.5],
                              ineqs=[x, y-0.5],
                              deg=1)
        prob1.solve(solver=self.solver)
        self.assertAlmostEqual(prob1.value, 0.49999999888338087, 5)

        prob2 = poly_opt_prob([x, y], x + y,
                              eqs=[x**2+y**2-1, y-x**2-0.5],
                              ineqs=[x, y-0.5],
                              deg=2)
        prob2.solve(solver=self.solver)
        self.assertAlmostEqual(prob2.value, 1.3910970905754896, 5)

    def test_poly_cert_set_eq(self):
        x, y = sp.symbols('x y')
        # Psatz for infeasibility of {x^2+y^2=1, y^3+x=2}
        prob1 = poly_cert_prob([x,y], -1, eqs=[x**2, y**2, x+y-1], deg=2)
        prob1.solve(solver=self.solver)

        # Psatz for infeasibility of {x^2+y^2=1, y^3+x=2}
        prob2 = poly_cert_prob([x,y], -1, eqs=[x**2+y**2-1, y**3+x-2], deg=2)
        prob2.solve(solver=self.solver)

    def test_poly_cert_set_ineq(self):
        x, y = sp.symbols('x y')

        # Certify x*y nonnegative on set {x>=0, y>=0, x+y<=1}
        prob1 = poly_cert_prob([x,y], x*y, ineqs=[x, y, 1-x-y], ineq_prods=True, deg=1)
        prob1.solve(solver=self.solver)

        # Infeasible when products of inequalities are not considered
        prob2 = poly_cert_prob([x,y], x*y, ineqs=[x, y, 1-x-y], ineq_prods=False, deg=1)
        with self.assertRaises(pic.SolutionFailure):
            prob2.solve(solver=self.solver)

if __name__ == '__main__':
    unittest.main()
