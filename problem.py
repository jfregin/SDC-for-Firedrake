class PROBLEM():
    from firedrake import dx, TrialFunctions, TestFunctions
    
    def __init__(self, y0, ff1, ff2, fs1, fs2, V, c_s, U):
        from firedrake import dx, TrialFunctions, TestFunctions, Function, split
        self._y0 = y0      # U_n
        self.ff1 = ff1    # basic
        self.ff2 = ff2    # baisc
        self.fs1 = fs1    # basic
        self.fs2 = fs2    # basic
        self.V = V
        self.u_, self.p_ = TrialFunctions(V)
        self.v1, self.v2 = TestFunctions(V)
        self.c_s = c_s
        self.U = U
        self._U1 = Function(V)
        self.u1_, self.p1_ = split(self._U1)
        self._U_n = y0
        self.u_n, self.p_n = split(self._U_n)
        
    @property
    def U_n(self):
        return self._U_n
    
    @U_n.setter
    def U_n(self, new_U_n):
        from firedrake import split

        self._U_n = new_U_n
        self.u_n, self.p_n = split(self._U_n)

    @property
    def U1(self):
        return self._U1
    
    @U1.setter
    def U1(self, new_U1):
        from firedrake import split
        self._U1 = new_U1
        self.u1_, self.p1_ = split(self._U1)

    @property
    def y0(self):
        return self._y0
    
    @y0.setter
    def y0(self, new_y0):
        from firedrake import split
        self._y0 = new_y0
        self.y1_, self.y2_ = split(self._y0)
        
    def lhs(self,u_,p_):
        from firedrake import inner, dx
        l1 = inner(u_, self.v1)*dx
        l2 = inner(p_, self.v2)*dx
        l = l1 + l2
        return l
        
    def ff(self, u_, p_):
        """
        construct weak form of fast problem (works only for linear problems)
        """
        from firedrake import inner, dx
        r1 = inner(self.ff1(p_, self.c_s), self.v1)*dx 
        r2 = inner(self.ff2(u_, self.c_s), self.v2)*dx 
        rhs = r1 + r2
        return rhs

    def fs(self, u_n, p_n):
        """
        construct weak form of slow problem (works only for linear problems)
        """
        from firedrake import inner, dx
        r1 = inner(self.fs1(u_n, self.U), self.v1)*dx
        r2 = inner(self.fs2(p_n, self.U), self.v2)*dx
        rhs =  r1 + r2
        return rhs
    
    def f(self, u_n, p_n):
        return self.ff(u_n,p_n) + self.fs(u_n, p_n)