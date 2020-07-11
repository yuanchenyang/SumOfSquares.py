import math
import picos as pic
import sympy as sp
import numpy as np
from picos import Problem
from collections import defaultdict
from operator import floordiv, and_

from .util import *
from .basis import Basis, poly_variable

class SOSProblem(Problem):
    '''Defines an Sum of Squares problem, a subclass of picos.Problem.
    (also see: https://gitlab.com/picos-api/picos/-/issues/138 for an
    implementation that also extends picos.Problem)
    '''
    def __init__(self, *args, **kwargs):
        '''Takes same arguments as picos.Problem
        '''
        Problem.__init__(self, *args, **kwargs)
        self._sym_var_map = {}
        self._sos_constraints = {}
        self._sos_const_count = 0
        self._pexpect_count = 0

    def sym_to_var(self, sym):
        '''Map between a sympy symbol to a unique picos variable. As sympy
        symbols are hashable, each symbol is assigned a unique picos variable.
        A new picos variable is created if it previously doesn't exist.
        '''
        if sym not in self._sym_var_map:
            self._sym_var_map[sym] = pic.RealVariable(repr(sym))
        return self._sym_var_map[sym]

    def sp_to_picos(self, expr):
        '''Converts a sympy affine expression to a picos expression, converting
        numeric values to floats, and sympy symbols to picos variables.
        '''
        if expr.func == sp.Symbol:
            return self.sym_to_var(expr)
        elif expr.func == sp.Add:
            return sum(map(self.sp_to_picos, expr.args))
        elif expr.func == sp.Mul:
            return prod(map(self.sp_to_picos, expr.args))
        else:
            return pic.Constant(float(expr))

    def sp_mat_to_picos(self, mat):
        '''Converts a sympy matrix a picos affine expression, converting
        numeric values to floats, and sympy symbols to picos variables.
        '''
        num_rows, num_cols = mat.shape
        # Use picos operator overloading
        return reduce(floordiv, [reduce(and_, map(self.sp_to_picos, mat.row(r)))
                                 for r in range(num_rows)])

    def add_sos_constraint(self, expr, variables, name='', sparse=False):
        '''Adds a constraint that the polynomial EXPR is a Sum-of-Squares. EXPR
        is a sympy expression treated as a polynomial in VARIABLES. Any symbols
        in EXPR not in VARIABLES are converted to picos variables
        (see SOSProblem.sym_to_var). Can optionally name the constraint with
        NAME. SPARSE uses Newton polytope reduction to do computations in a
        reduced-size basis. Returns a SOSConstraint object.
        '''
        self._sos_const_count += 1
        name = name or f'_Q{self._sos_const_count}'
        variables = sorted(variables, key=str) # To lex order
        poly = sp.poly(expr, variables)
        deg = poly.total_degree()
        assert deg % 2 == 0, 'Polynomial degree must be even!'

        hom = is_hom(poly, deg)
        mono_to_coeffs = dict(zip(poly.monoms(), map(self.sp_to_picos, poly.coeffs())))
        basis = Basis.from_poly_lex(poly, sparse=sparse)

        Q = pic.SymmetricVariable(name, len(basis))
        for mono, pairs in basis.sos_sym_entries.items():
            coeff = mono_to_coeffs.get(mono, 0)
            self.add_constraint(sum(Q[i,j] for i,j in pairs) == coeff)

        pic_const = self.add_constraint(Q >> 0)
        return SOSConstraint(pic_const, Q, basis, variables, deg)

    def get_pexpect(self, variables, deg, hom=False, name=''):
        '''Returns a degree DEG pseudoexpectation operator. This operator is a
        function that takes in a polynomial of at most degree DEG in VARIABLES,
        and returns a picos affine expression. If HOM=True, this polynomial must
        also be homogeneous. This operator has the property that
        pexpect(p(x)^2) >= 0 for any suitable polynomial p(x).

        Since the return value of pexpect(p(x)) is a picos expression, it can be
        used in other constraints/objectives in the current problem.

        The constraints associated with this operator are registered with the
        SOSProblem instance, and can be optionally named as NAME.

        '''
        self._pexpect_count += 1
        name = name or f'_X{self._pexpect_count}'
        variables = sorted(variables, key=str) # To lex order
        basis = Basis.from_degree(len(variables), deg//2)

        X = pic.SymmetricVariable(name, len(basis))
        for monom, indices in basis.sos_sym_entries.items():
            if len(indices) > 1:
                ip, jp = indices[0]
                for i,j in indices[1:]:
                    self.add_constraint(X[ip, jp] == X[i, j])
                    ip, jp = i, j

        self.add_constraint(X >> 0)

        def pexpect(p):
            poly = sp.poly(p, variables)
            basis.check_can_represent(poly)
            return self.sp_mat_to_picos(basis.sos_sym_poly_repr(poly)) | X
        return pexpect

def poly_opt_prob(vars, obj, eqs=None, ineqs=None, deg=None, sparse=True):
    '''Formulates and returns a degree DEG Sum-of-Squares relaxation of a
    polynomial optimization problem in variables VARS that mininizes OBJ
    subject to equality constraints EQS (g(x) = 0) and inequality constraints
    INEQS (h(x) >= 0). SPARSE uses Newton polytope reduction to do computations
    in a reduced-size basis. Returns an instance of SOSProblem.
    '''
    prob = SOSProblem()
    gamma = sp.symbols('gamma')
    gamma_p = prob.sym_to_var(gamma)
    eqs, ineqs = eqs or [], ineqs or []

    max_deg = max(map(lambda p: sp.poly(p, vars).total_degree(), [obj] + eqs + ineqs))
    if deg is None:
        deg = math.ceil(max_deg/2)
    if 2*deg < max_deg:
        raise ValueError(f'Degree of relaxation 2*{deg} less than maximum degree {max_deg}')


    f = 0 # obviously non-negative polynomial for (in)equalities constraints
    for i, eq in enumerate(eqs):
        p = poly_variable(f'c{i}', vars, 2*deg - poly_degree(eq, vars))
        f += p * eq
    for i, ineq in enumerate(ineqs):
        s = poly_variable(f'd{i}', vars, 2*deg - poly_degree(eq, vars))
        prob.add_sos_constraint(s, vars, name=f'd{i}', sparse=sparse)
        f += s * ineq

    prob.add_sos_constraint(obj - gamma - f, vars, sparse=sparse)
    prob.set_objective('max', gamma_p)
    return prob

class SOSConstraint:
    '''Defines a Sum-of-Squares constraint, returned by
    SOSProblem.add_sos_constraint. Holds information about the SoS constraint
    and its dual, and allows one to compute the pseudoexpectation of any
    polynomial.
    '''
    def __init__(self, pic_const, Q, basis, symbols, deg):
        self.pic_const = pic_const
        self.Q = Q
        self.basis = basis
        self.symbols = symbols
        self.b_sym = basis.to_sym(symbols)
        self.deg = deg

    @property
    def Qval(self):
        '''Optimization variable Q where p(x) = b^T Q b, where p(x) is polynomial
        constrained to be SoS, and b is the basis.'''
        if self.Q.value is None:
            raise ValueError('Missing value for sos constraint variable!'
                             ' (is the problem solved?)')
        return self.Q.value

    def get_chol_factor(self):
        '''Returns L, the Cholesky factorization of Q = LL^T. Adds a small
        multiple of identity to Q if it has small negative eigenvalues.
        '''
        mineig = min(min(np.linalg.eig(self.Qval)[0]), 0)
        return np.linalg.cholesky(self.Qval - np.eye(len(self.basis))*mineig*1.1)

    def get_sos_decomp(self, precision=3):
        '''Returns a vector containing the sum of squares decompositon of this
        constraint'''
        L = sp.Matrix(self.get_chol_factor())
        S = (L.T @ sp.Matrix(self.b_sym)).applyfunc(lambda x: x**2)
        return round_sympy_expr(S, precision)

    def pexpect(self, expr):
        '''Computes the pseudoexpectation of a given polynomial EXPR'''
        poly = sp.poly(expr, self.symbols)
        self.basis.check_can_represent(poly)
        Qp = self.basis.sos_sym_poly_repr(poly)
        X = sp.Matrix(len(self.basis), len(self.basis), self.pic_const.dual)
        return sum(sp.matrix_multiply_elementwise(X, Qp))
