import numpy as np
import scipy
import scipy.sparse
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 1 / 2 * np.dot(np.dot(x.T, self.A), x) - np.dot(self.b.T, x)

    # def grad(self, x):
    #     return 1/2 * np.dot((self.A + self.A.T), x) - self.b
    def grad(self, x):
        return self.A.dot(x) - self.b


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()
    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        in_log = (1 + np.exp(-self.b * self.matvec_Ax(x)))
        logreg = np.mean(np.log(in_log)) + (self.regcoef / 2) * np.linalg.norm(x) ** 2
        return logreg

    def grad(self, x):
        m = len(self.b)
        b_ax = self.b * self.matvec_Ax(x)
        coeff = - np.exp(-b_ax) / (1 + np.exp(-b_ax))
        logreg_grad = np.array(self.matvec_ATx(coeff * self.b)) / m + self.regcoef * x
        return logreg_grad


def create_log_reg_oracle(A, b, regcoef):
    matvec_Ax = lambda x: A.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr() * x
    matvec_ATx = lambda x: A.T.dot(x) if isinstance(A, np.ndarray) \
        else A.tocsr().T * x

    def matmat_ATsA(s, mat=A):
        if isinstance(mat, np.ndarray):
            return mat.T.dot(np.multiply(mat, s.reshape(len(s), 1)))
        A = mat.tocsr()
        sA = A.multiply(s.reshape(len(s), 1))
        return A.T * sA

    return LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
