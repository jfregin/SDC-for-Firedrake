class SOLVER(object):
    from firedrake import dx, solve, Function, assemble
    
    def __init__(self, object):
        import numpy as np
        self.p = object             # problem
        self.baseIntegrator = 'IMEX'    # base integrator for SDC
        self.quadrature = 'radau_r'
        self.t = np.nan     #current time
        self.a = np.nan     # time at start
        self.b = np.nan     # time at end
        self.ts= np.nan     # time stamps (points in time where you want to know your solution)
        self.dts = np.nan   # delta_ts
        self.dt = np.nan    # current delta_t
        self.i = 0          # current index
        self.M = np.nan
        self.iterations = np.nan
        self.nodes = np.nan
        self.weights= np.nan
        self.sol = np.nan
        self.feval = np.nan
        self.Q = np.nan
        self.S = np.nan
        self.tau = np.nan
        self.solution = [self.p.y0,self.p.y0]
        self.quad = np.nan
        self.testing = np.nan
        self.residual = np.nan
        self.linear = np.nan
        self.nonlinear= np.nan
        self.isnonlinear = True
        self.kfast = np.nan
        self.CFL = 0
        self.delta_x = 0
        
        
    def dot(a,b):
        """
        a = array containing numbers, b array containing n-dimensional ufl functions
        """
        n = len(a)
        out = []
        for i in range(0,n):
            out.append(float(a[i])*b[i]) # float is important... for whatever reason UFL does not like numpy float
        return sum(out)

    
    def matmul_UFL(self, a, b):
        import numpy as np
        from firedrake import assemble
        # b is nx1 array!
        n = np.shape(a)[0]
        result = [float(0)]*n
        for j in range(n):
            for k in range(n):
                result[j] += float(a[j,k])*b[k]
            result[j] = assemble(result[j])
        return result

    def invert_mass_matrix(self,u, M, V):
        from firedrake import Function, solve
        """
        Helper routine to invert mass matrix
        Args:
            u (dtype_u): current values
        Returns:
            dtype_u: inv(M)*u
        """

        inverse = Function(V)

        A = M

        solve(A, inverse, u, solver_parameters={'ksp_rtol': 1e-9})

        return inverse

    
    def dot(self, a,b):
        n = len(a)
        out = []
        for i in range(0,n):
            out.append(float(a[i])*b[i]) # float is important... for whatever reason...
        return sum(out)

    def eval_f(self,M, F):
        from firedrake import Function, assemble
        """
        M = "Mass matrix" but is infact the matrix given by assemble(U - dt * ff(U1))
        F = rhs variational problem/linear form
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
             dtype_f: the RHS divided into two parts
        """

        f = Function(self.p.V)
        #self.w.assign(u.values) #update u in F (does not happen here) u.assign before function call should suffice
        f = Function(self.p.V, assemble(F))

        f = self.invert_mass_matrix(f, M, self.p.V)

        return f
    
    def eval_f_nonlinear(self, L,R,U):
        from firedrake import TrialFunction, derivative, solve
        F = L - R
        #du = TrialFunction(self.p.V)
        #J = derivative(F, U, du) # maybe worth to directly provide jacobian
        #solve(F == 0, U, J=J, solver_parameters={'ksp_rtol': 1e-10, 'ksp_atol': 1e-10, 'snes_rtol': 1e-10, 'snes_atol': 1e-10, 'assembled_py_type': 'hypre', 'snes_monitor': None})
        solve(F == 0, U, solver_parameters={'ksp_rtol': 1e-10, 'ksp_atol': 1e-10, 'snes_rtol': 1e-10, 'snes_atol': 1e-10, 'assembled_py_type': 'hypre', 'snes_monitor': None})
        return U

                
    # setup for sdc  
    def setup(self, ts, SDC_nodes = 8, SDC_iterations = 5, baseIntegrator = 'IMEX', isnonlinear = True, delta_x = 20):
        import numpy as np 
        self.isnonlinear = isnonlinear
        self.ts = ts
        self.a = ts[0]
        self.t = ts[0]
        self.b = ts[-1]
        self.dts = np.diff(self.ts)
        self.dt = self.dts[0]
        self.M = SDC_nodes
        self.iterations = SDC_iterations
        self.delta_x = delta_x
        
    # calculate nodes and weights, based on choosen quadrature. TODO: implement further rules    
    def nodes_weights(self):
        from functions import fcs as fcs
        if self.quadrature == 'radau_r':           
            self.nodes, self.weights = fcs.rnw_r(self.M ,self.t , self.t + self.dt,A=-1,B=1)    

    def midpoint(self, t):
        import numpy as np 
        from firedrake import Constant, Function, split, assemble, solve

        n = len(t) #number of steps
        dt_v = np.diff(t)
        dt = Constant(0) # create UFL dt to update form

        # create list to store solutions from IMEX
        y = [Function(self.p.y0)]
        
        #prepare rhs for quadrature in final update 
        self.feval= ['-']*self.M

        for i in range(1,n):
            dt.assign(dt_v[i-1])
            u_n, p_n = split(y[i-1])
            self.p.U1.assign(y[i-1])
            u_, p_ = split(self.p.U1)
            #L = self.p.lhs(u_, p_) - dt* self.p.ff(u_, p_) # bilinear form
            #R = self.p.lhs(u_n, p_n) + dt* self.p.fs(u_n, p_n) # linear form
            
            if self.isnonlinear:
                L = self.p.lhs(self.p.u_, self.p.p_)
                M = assemble(L)
                y_eval = self.eval_f(M, self.p.fs(u_n, p_n))
                # calculate k1 for explicit butcher table
                ks1_1, ks1_2 = split(y_eval)
                
                # set up variables for calculating k1  (k_f = k_fast) for implicit butcher table
                k_f = Function(self.p.V)
                k_f1, k_f2 = split(k_f)
                L = self.p.lhs(k_f1, k_f2)
                
                #R = self.p.lhs(u_n, p_n) + dt* self.p.fs(u_n, p_n) # linear form
                R = self.p.ff(u_n + 0.5*dt*(k_f1 + ks1_1), p_n + 0.5*dt*(k_f2 + ks1_2))
                
                # calculate k1 for implicit butcher table
                self.eval_f_nonlinear(L,R, k_f)
                k_f1, k_f2 = split(k_f)
                #print('NONLINEAR')
                self.kfast = k_f
            
                # calculate k2 for explicit butcher table
                L = self.p.lhs(self.p.u_, self.p.p_)
                M = assemble(L)      
                y_eval = self.eval_f(M, self.p.fs(u_n + 0.5*dt*(k_f1 + ks1_1), p_n + 0.5*dt*(k_f2 + ks1_2)))
                ks2_1, ks2_2 = split(y_eval)
                
                # set up and calculate solution at next time step
                R = self.p.lhs(u_n, p_n) + dt*(self.p.lhs(k_f1, k_f2) + self.p.lhs(ks2_1, ks2_2))
                rhs = self.eval_f(M,R) 
                self.p.U1.assign(rhs) # don't think this is necessary but lets see if it changes anyrhing...
                y.append(Function(rhs))

            else:
                # set up and calculate k1 for explicit butcher table
                L = self.p.lhs(self.p.u_, self.p.p_)
                M = assemble(L)
                y_eval = self.eval_f(M, self.p.fs(u_n, p_n))
                ks1_1, ks1_2 = split(y_eval)
                
                # set up and calculate k1 for implicit butcher table
                k_f = Function(self.p.V)
                k_f1, k_f2 = split(k_f)
                F =  self.p.lhs(k_f1, k_f2)  - self.p.ff(u_n + 0.5*dt*(k_f1 + ks1_1), p_n + 0.5*dt*(k_f2 + ks1_2))
                
                #F =  self.p.lhs(k_f1, k_f2)  - self.p.ff(u_n, p_n)
                solve(F == 0, k_f , solver_parameters={'ksp_rtol': 1e-9})
                self.kfast = k_f
                y_eval = self.eval_f(M, self.p.fs(u_n + 0.5*dt*(k_f1 + ks1_1), p_n + 0.5*dt*(k_f2 + ks1_2)))
                ks2_1, ks2_2 = split(y_eval)
                
                # set up and calculate solution at next timestep
                R = self.p.lhs(u_n, p_n) + dt*(self.p.lhs(k_f1, k_f2) + self.p.lhs(ks2_1, ks2_2))
                rhs = self.eval_f(M,R) 
                y.append(Function(rhs))
            # calculate rhs for quadrature
            # no need to use nonlinear solve, since problem is linear in u_. this is just Ax = b
            self.feval[i-1] = self.eval_f(assemble(self.p.lhs(self.p.u_, self.p.p_)),self.p.f(rhs.sub(0),rhs.sub(1)))

        self.sol = y
        #needed later for SDC, is not used here
        self.tau = dt_v

    # IMEX for initial trajectory
    def backwards_e(self, t):
        # TODO: need to verify method
        import numpy as np
        from firedrake import Constant, Function, split, assemble 
    
        n = len(t) #number of steps
        dt_v = np.diff(t)
        dt = Constant(0) # create UFL dt to update form
        
        # create list to store solutions from IMEX
        y = [Function(self.p.y0)]
        
        #prepare rhs for quadrature in final update 
        self.feval= ['-']*self.M
            
        for i in range(1,n):
            dt.assign(dt_v[i-1])
            u_n, p_n = split(y[i-1])
            self.p.U1.assign(y[i-1])
            u_, p_ = split(self.p.U1)
            L = self.p.lhs(u_, p_) - dt* (self.p.ff(u_, p_) +self.p.fs(u_, p_))# bilinear form
            R = self.p.lhs(u_n, p_n) # linear form

            if self.isnonlinear:
                rhs = self.eval_f_nonlinear(L, R, self.p.U1) 
                y.append(Function(rhs))

            else:
                L = self.p.lhs(self.p.u_, self.p.p_) - dt*( self.p.ff(self.p.u_, self.p.p_) + self.p.fs(self.p.u_, self.p.p_) )# bilinear form
                R = self.p.lhs(u_n, p_n)# linear form
                M = assemble(L)
                rhs = self.eval_f(M,R) 
                y.append(Function(rhs))
            # calculate rhs for quadrature
            # no need to use nonlinear solve, since problem is linear in u_. this is just Ax = b
            self.feval[i-1] = self.eval_f(assemble(self.p.lhs(self.p.u_, self.p.p_)),self.p.f(rhs.sub(0),rhs.sub(1)))

        self.sol = y
        #needed later for SDC, is not used here
        self.tau = dt_v
        
        
    # IMEX for initial trajectory
    def IMEX_c(self, t):
        # TODO: need to verify method
        import numpy as np
        from firedrake import Constant, Function, split, assemble 
    
        n = len(t) #number of steps
        dt_v = np.diff(t)
        dt = Constant(0) # create UFL dt to update form
        
        # create list to store solutions from IMEX
        y = [Function(self.p.y0)]
        
        #prepare rhs for quadrature in final update 
        self.feval= ['-']*self.M
            
        for i in range(1,n):
            dt.assign(dt_v[i-1])
            u_n, p_n = split(y[i-1])
            self.p.U1.assign(y[i-1])
            u_, p_ = split(self.p.U1)
            L = self.p.lhs(u_, p_) - dt* self.p.ff(u_, p_) # bilinear form
            R = self.p.lhs(u_n, p_n) + dt* self.p.fs(u_n, p_n) # linear form

            if self.isnonlinear:
                rhs = self.eval_f_nonlinear(L, R, self.p.U1) 
                y.append(Function(rhs))

            else:
                L = self.p.lhs(self.p.u_, self.p.p_) - dt* self.p.ff(self.p.u_, self.p.p_) # bilinear form
                R = self.p.lhs(u_n, p_n) + dt* self.p.fs(u_n, p_n) # linear form
                M = assemble(L)
                rhs = self.eval_f(M,R) 
                y.append(Function(rhs))
            # calculate rhs for quadrature
            # no need to use nonlinear solve, since problem is linear in u_. this is just Ax = b
            self.feval[i-1] = self.eval_f(assemble(self.p.lhs(self.p.u_, self.p.p_)),self.p.f(rhs.sub(0),rhs.sub(1)))

        self.sol = y
        #needed later for SDC, is not used here
        self.tau = dt_v
        
    # one SDC iteration
    def SDC_sweep(self):
        import numpy as np 
        from copy import deepcopy
        from firedrake import Constant, split, assemble, inner, dx
        Q = self.Q
        S = self.S
        d_tau = self.tau
        dt = Constant(0)
        # append initial 
        y = self.sol
        y_n = deepcopy(y) 
        
        quad = self.matmul_UFL(S,self.feval)
        self.quad = quad

        for i in range(len(y)-1):
            dt.assign(d_tau[i])
            
            #self.p.U1.assign(y_n[i]) <- why does this cause trouble?
            self.p.U1 = deepcopy(y_n[i])
            u_, p_ = split(self.p.U1)
            u_n, p_n = split(y_n[i])
            u_o, p_o = split(y[i])
            u_o1, p_o1 = split(y[i+1])
            #quad[i].assign(Constant(0))
            q1, q2 = split(quad[i])
            if self.isnonlinear:
                L = self.p.lhs(u_, p_) - dt* self.p.ff(u_, p_) # bilinear form
                R = self.p.lhs(u_n, p_n) + dt*( - self.p.ff(u_o1, p_o1) \
                                                                  + self.p.fs(u_n, p_n) \
                                                                  - self.p.fs(u_o, p_o)) \
                                                                  + inner(q1, self.p.v1)*dx \
                                                                  + inner(q2, self.p.v2)*dx

                y_n[i+1] = self.eval_f_nonlinear(L, R, self.p.U1)
                

        
            else:
                L = self.p.lhs(self.p.u_, self.p.p_) - dt* self.p.ff(self.p.u_, self.p.p_) # bilinear form
                R = self.p.lhs(u_n, p_n) + dt*( - self.p.ff(u_o1, p_o1) \
                                                                  + self.p.fs(u_n, p_n) \
                                                                  - self.p.fs(u_o, p_o)) \
                                                                  + inner(q1, self.p.v1)*dx \
                                                                  + inner(q2, self.p.v2)*dx
                M = assemble(L)
                y_n[i+1] = self.eval_f(M, R)

            # calculate with new y for  new SDC iteration
            self.feval[i] = self.eval_f(assemble(self.p.lhs(self.p.u_, self.p.p_)),self.p.f(y_n[i+1].sub(0),y_n[i+1].sub(1)))
            
        self.sol = y_n
        
        #residual
        quad = self.matmul_UFL(Q,self.feval)
        self.residual = ['']*self.M
        for i in range(0,self.M):
            self.residual[i] = self.p.y0 + quad[i] - self.sol[1:][i]

    def step(self):
        from functions import fcs as fcs
        import numpy as np 
        from firedrake import assemble
        """
        step forward one timestep
        """
        a = self.t
        b = self.ts[self.i+1]        
        self.nodes_weights()

        # initial trajectory
        self.IMEX_c(np.append(a,self.nodes))

        self.Q = fcs.Qmatrix(self.nodes, a)
        self.S = fcs.Smatrix(self.Q)       
        i = 0
 
        # SDC below here
        while i < self.iterations:  
            self.SDC_sweep()
            i += 1
        
        # final update    
        dot_product = self.p.y0 + self.dot(self.weights, self.feval)
        self.testing = assemble(self.dot(self.weights, self.feval))
        self.y = assemble(dot_product)

        self.i += 1
        self.t = b

        #check CFL
        #min_dt = max(np.diff(self.nodes))
        #max_U = max(self.y.sub(0).dat.data)
        #CFL = min_dt*max_U/self.delta_x

        #if self.CFL < CFL:
        #    self.CFL = CFL

        #if CFL > 1:
        #    print('CFL_Warning, CFL: ' + str(CFL))
        
        #update y0 in problem for next timestep
         # REMOVE THIS LATER JUST FOR IMEX. TEST         #self.y = self.sol[-1]
        #self.y = self.sol[-1]
        self.p.y0 = self.y
        try:
            self.dt = self.dts[self.i]
        except:
            self.dt = self.dts

    def step_test(self):
        from functions import fcs as fcs
        import numpy as np 
        from firedrake import assemble
        """
        step forward one timestep
        """
        a = self.t
        b = self.ts[self.i+1]
        self.nodes_weights()

        self.midpoint(np.append(a,self.nodes))
        
        self.i += 1
        self.t = b
        self.y = self.sol[-1]
        self.p.y0 = self.y
        
        try:
            self.dt = self.dts[self.i]
        except:
            self.dt = self.dts
            
            
    def solve_test(self):
        if solve.M != 1:
            print('Warning M is not 1, you may get unwanted results')
            
        while self.t != self.b:
            self.step_test()
            self.solution.append(self.y)
        self.solution = self.solution[1:]
    
    # step through all timesteps  
    def solve(self):
        while self.t != self.b:
            self.step()
            self.solution.append(self.y)
        self.solution = self.solution[1:]
