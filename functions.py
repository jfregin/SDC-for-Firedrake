#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:19:24 2020

@author: jf
TODO:
    -extract DC-implicit and SDC-implicit methods and add them to the fcs class
    -remove unnecessary functions (DC2 or DC) best would be to unify DC-EE and DC-IE into one function
"""

class fcs:
    import numpy as np

    def subinterval(a, b, num_int):
        """
        Parameters
        ----------
        a : real
            lower boundary
        b : real
            upper boundary
        num_int : integer
            number of grid points between a and b
        Returns
        -------
        sub_int : new grid with num_int subintervals in between two points
        in the grid. Type:(array)
        """
        import numpy as np
        n = num_int + 1
        step = (b - a) / n
        sub_int = np.arange(a, b, step)
        return sub_int

    def refine_grid(grid1d, subintervals=4, subfunc=subinterval):
        """
        Parameters
        ----------
        grid1d : array
            one dimensional grid array
        subintervals : integer
            number of intervals in between grid points
        subfunc : fuction
            function that creates a grid with subintervals gridpoints between two
            integers a and b. b is not included in the new grid
        Returns
        -------
        new_grid : array
            grid with equally spaced subintervals that includes the starting and
            end points a and b.
        """
        import numpy as np
        si = subintervals
        g = grid1d
        new_grid = subfunc(g[0], g[1], si)
        for i in range(2, len(g)):
            new_grid = np.append(new_grid, subfunc(g[(i - 1)], g[i], si))
        new_grid = np.append(new_grid, g[(-1)])
        return new_grid

    def coord_trans(cin, a, b, A=-1, B=1):
        """
        transforms coordinates cin within the interval [A,B]
        to coordinates cout within a new interval [a,b]

        Parameters
        ----------
        cin : coordinates within interval [A,B]
        a : lower boundary of new interval
        b : upper boundary of new interval
        A : initial interval lower boundary, default is A=-1
        B : initial interval upper boundary, default is B=1

        Returns
        -------
        cout : coordinates within new interval [a,b]
        """
        cout = ((b - a) * cin + a * B - b * A) / (B - A)
        return cout

    def NewtonVM(t):
        """
        t: array or list containing nodes.
        returns: array Newton Vandermode Matrix. Entries are in the lower
        triangle
        Polynomial can be created with
        scipy.linalg.solve_triangular(NewtonVM(t),y,lower=True) where y
        contains the points the polynomial need to pass through
        """
        import numpy as np
        t = np.asarray(t)
        dim = len(t)
        VM = np.zeros([dim, dim])
        VM[:, 0] = 1
        for i in range(1, dim):
            VM[:, i] = (t[:] - t[(i - 1)]) * VM[:, i - 1]

        return VM

    def Horner_newton(weights, xi, x):
        """
        Horner scheme to evaluate polynomials based on newton basis
        """
        import numpy as np
        y = np.zeros_like(x)
        for i in range(len(weights)):
            y = y * (x - xi[(-i - 1)]) + weights[(-i - 1)]

        return y

    def GL(M, a, b, ct=coord_trans, A=-1, B=1):
        """
        calculates Gauss-Legendre nodes for Gaussian quadrature at int(M)
        Points between float(a) and float(b).
        A and B shouldn't be changed
        """
        # is a lot faster than GaussLegendre but slower than _getNodes
        import numpy as np
        # calculate nodes and weights on [-1,1]
        nodes, weights = np.polynomial.legendre.leggauss(M)
        nodes = ct(nodes, a, b)  # transform back to interval [a,b]
        weights = (b-a)/(B-A)*weights  # also transform weights to [a,b]
        return nodes, weights

    def eval(f, nodes, weights, pw=False):
        # performes gaussian quadrature given the function f
        import types
        import numpy as np

        if pw is False:
            if type(f) == types.FunctionType:
                return np.dot(weights, f(nodes))
            else:
                return np.dot(weights, f)
        else:
            l = len(weights)
            summation = np.zeros(l)
            if type(f) == types.FunctionType:
                for i in range(0, l):
                    summation[i] = np.dot(weights[0:i+1], f(nodes[0:i+1]))
                return summation
            else:
                for i in range(0, l):
                    summation[i] = np.dot(weights[0:i+1], f[0:i+1])
                return summation

    def GQ(f, a, b, M=10, coord_trans=coord_trans, GL=GL, eval=eval, pw=False):
        """
        performs gaussian quadrature on function f based on the interval [a,b]
        """
        nodes, weights = GL(M, a, b)
        # print('weights: ' +str(weights))
        return eval(f, nodes, weights, pw=pw)

    def Newton_polynomial_specific(x, t='', y='', NVM=NewtonVM, HN=Horner_newton):
        """
        input: see NewtonVM
        returns two arrays
        x:
        yy: Lagrange Polynomial passing through points(t_i,y_i)
        Note: maybe use *args to prevent t='' and y=''
        """
        from scipy.linalg import solve_triangular
        VM = NVM(t)
        weights = solve_triangular(VM, y, lower=True)
        yy = HN(weights, t, x)
        return x, yy

    def Newton_polynomial(t, y, N_linspace=100, NVM=NewtonVM, HN=Horner_newton):
        """
        input: see NewtonVM
        returns two arrays
        x: resolves t in N_linspace steps
        yy: Lagrange Polynomial passing through points(t_i,y_i)
        """
        from scipy.linalg import solve_triangular
        import numpy as np
        VM = NVM(t)
        weights = solve_triangular(VM, y, lower=True)
        x = np.linspace(t[0], t[(-1)], N_linspace)
        yy = HN(weights, t, x)
        return x, yy

    def nodes_weights(n, a, b, A=-1, B=1):
        # nodes and weights for gauss legendre quadrature
        from scipy.special import legendre
        import numpy as np
        poly = legendre(n)
        polyd = poly.deriv()
        nodes= poly.roots
        nodes = np.sort(nodes)
        weights = 2/((1-nodes**2)*(np.polyval(polyd,nodes))**2)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights=(b-a)/(B-A)*weights
        return nodes, weights

    def my_get_weights(n, a, b, nodes, nodes_weights=nodes_weights, NewtonVM=NewtonVM, Horner_newton=Horner_newton, eval=eval):
        # integrates lagrange polynomials to the points [nodes]
        import scipy
        import numpy as np
        nodes_m, weights_m=nodes_weights(np.ceil(n/2), a, b)  # use gauss-legendre quadrature to integrate polynomials
        weights = np.zeros(n)
        for j in np.arange(n):
            coeff = np.zeros(n)
            coeff[j] = 1.0  # is unity because it needs to be scaled with y_j for interpolation we have  sum y_j*l_j
            poly_coeffs = scipy.linalg.solve_triangular(NewtonVM(nodes), coeff, lower=True)
            eval_newt_poly = Horner_newton(poly_coeffs, nodes, nodes_m)
            weights[j] = eval(eval_newt_poly, nodes_m, weights_m)
        return weights

    def lnw(n, a, b, A=-1, B=1):
        import numpy as np
        from scipy.special import legendre
        # nodes and weights for gauss - lobatto quadrature
        # See Abramowitz & Stegun p 888
        nodes = np.zeros(n)
        nodes[0] = A
        nodes[-1] = B
        poly = legendre(n-1)
        polyd = poly.deriv()
        subnodes = np.sort(polyd.roots)
        nodes[1:-1] = subnodes
        weights = 2/(n*(n-1)*(np.polyval(poly, nodes))**2)
        weights[0] = 2/(n*(n-1))
        weights[-1] = 2/(n*(n-1))
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights = (b-a)/(B-A)*weights
        return nodes, weights

    def rnw(n, a, b, A=-1, B=1):
        import numpy as np
        from scipy.special import legendre
        # nodes and weights for gauss - radau quadrature # See Abramowitz & Stegun p 888
        nodes = np.zeros(n)
        nodes[0] = A
        p = np.poly1d([1, 1])
        pn = legendre(n)
        pn1 = legendre(n-1)
        poly, remainder = (pn + pn1)/p  # [1] returns remainder from polynomial division
        nodes[1:] = np.sort(poly.roots)
        weights = 1/n**2 * (1-nodes[1:])/(pn1(nodes[1:]))**2
        weights = np.append(2/n**2,weights)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights = (b-a)/(B-A)*weights
        return nodes, weights

    def rnw_r(n, a, b, A=-1, B=1):
        import numpy as np
        from scipy.special import legendre
        # nodes and weights for gauss - radau IIA quadrature
        # See Abramowitz & Stegun p 888
        nodes = np.zeros(n)
        nodes[0] = A
        p = np.poly1d([1, 1])
        pn = legendre(n)
        pn1 = legendre(n-1)
        poly, remainder = (pn + pn1)/p  # [1] returns remainder from polynomial division
        nodes[1:] = np.sort(poly.roots)
        weights = 1/n**2 * (1-nodes[1:])/(pn1(nodes[1:]))**2
        weights = np.append(2/n**2, weights)
        nodes = ((b - a) * nodes + a * B - b * A) / (B - A)
        weights = (b - a)/(B - A)*weights
        nodes = ((b + a) - nodes)[::-1]  # reverse nodes
        weights = weights[::-1]  # reverse weights
        return nodes, weights

    def Qmatrix(nodes, a, my_get_weights=my_get_weights):
        """
        Integration Matrix 
        """
        import numpy as np
        M = len(nodes)
        Q = np.zeros([M, M])

        # for all nodes, get weights for the interval [tleft,node]
        for m in np.arange(M):
            w = my_get_weights(M, a, nodes[m],nodes)
            Q[m, 0:] = w

        return Q

    def Smatrix(Q):
        """
        Integration matrix based on Q: sum(S@vector) returns integration
        """
        from copy import deepcopy
        import numpy as np
        M = len(Q)
        S = np.zeros([M, M])

        S[0, :] = deepcopy(Q[0, :])
        for m in np.arange(1, M):
            S[m, :] = Q[m, :] - Q[m - 1, :]

        return S

    def Q_fast(t):
        """
        EQ. 3.6 SISDC Paper
        """
        import numpy as np
        dim = len(t)-2
        diff = np.diff(t[0:-1])
        Q_f = np.zeros([dim,dim])
        for i in range(0,dim):
            Q_f[i,0:i+1] = diff[0:i+1]
        delta_t = t[-1]-t[0]
        Q_f = 1/delta_t * Q_f
        return Q_f

    def Q_slow(t, Qf=Q_fast):
        # this one should be correct
        import numpy as np
        Q_f = Qf(t)
        Q_s = np.zeros(Q_f.shape)
        Q_s[1:, :-1] = Q_f[1:, 1:]
        return Q_s

    def IMEX_matrix(lf, ls, t):
        import numpy as np
        dim = len(t)-2
        diff = np.diff(t[0:-1])
        R = np.zeros([dim, dim], dtype=np.complex)
        terms = (1+1j*diff*ls)/(1-1j*diff*lf)
        diag = np.cumprod(terms)
        np.fill_diagonal(R, diag)
        return R

    def stabl(lf, ls, t, k=3, gw=my_get_weights, IMEX_matrix=IMEX_matrix):
        from numpy.linalg import matrix_power as mp
        import numpy as np
        dim = len(t)-2
        a = t[0]
        b = t[-1]
        delta_t = b-a
        nodes = t[1:-1]
        Q_f = fcs.Q_fast(t)
        Q_s = fcs.Q_slow_alt(t)
        Q = fcs.Qmatrix(nodes, a, gw)
        Q_delta = 1/(b-a)*Q
        Id = np.identity(dim, dtype=np.complex)
        L = Id - delta_t*(1j*lf*Q_f + 1j*ls*Q_s)
        L_inv = np.linalg.inv(L)
        R = delta_t*((1j*(lf+ls)*Q_delta)-(1j*lf*Q_f + 1j*ls*Q_s))
        garb, q = fcs.rnw(dim, a, b)
        q = q[::-1]
        R2 = IMEX_matrix(lf, ls, t)
        term = mp(L_inv@R, k)
        term2 = mp(L_inv@R, k)@R2
        for j in range(0, k):
            term += mp(L_inv@R, j)@L_inv
            term2 += mp(L_inv@R, j)@L_inv

        ones = np.ones(len(Q))
        return 1 + 1j*(lf+ls)*q.dot(term.dot(ones)), 1 + 1j*(lf+ls)*q.dot(term2.dot(ones))
