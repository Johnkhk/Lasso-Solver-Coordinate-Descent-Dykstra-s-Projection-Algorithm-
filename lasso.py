from ast import arg
import numpy as np
import cvxpy as cp
import copy
import random
import time
import pandas as pd


class lasso_solver:
    def __init__(self, lambdu=1):
        self.lambdu = lambdu
        self.maxiters = 100
        self.termination_cond = 0.0001
        
    
    def generate_X_y_b(self, m=20,n=20):
        # X = np.random.rand(m,n).astype(np.longdouble)
        # y = np.random.rand(m).astype(np.longdouble)
        # b = np.random.rand(n).astype(np.longdouble)
        X = np.random.rand(m,n)
        y = np.random.rand(m)
        b = np.random.rand(n)
        self.m, self.n = X.shape
        # self.n = X.size

        self.select = list(range(n))
        return X,y,b

    def soft_thresh(self, rouj,zj):
        if rouj < -self.lambdu:
            return (rouj+self.lambdu)/zj

        elif rouj > self.lambdu:
            return (rouj-self.lambdu)/zj

        else:
            return 0

    def min_betaj(self,X,y,b,j):
        selector = [i for i in range(X.shape[1]) if i != j]
        rouj = X[:,j].T @ (y-X[:,selector] @ b[selector])
        zj = np.linalg.norm(X[:,j])**2
        thresh = self.soft_thresh(rouj,zj)
        return thresh

    def coord_desc(self,X,y,beta,b=False):
        """
        Cyclic Coordinate Descent on Lasso
        """
        if not b:
            beta = np.zeros(self.n)
        it=0
        while it < self.maxiters:
            for j in range(self.n):
                min_b = self.min_betaj(X,y,beta,j)
                beta[j]=min_b
            it+=1
        return beta
    
    def obj_func(self,y,X,b): # 2
        return np.linalg.norm(y-X@b)**2 + self.lambdu*np.linalg.norm(b, ord=1)

    def cvx_min(self, X,y,b):
        # b = cp.Variable(self.n)
        b = cp.Variable(len(b))
        objective = cp.Minimize(cp.norm2(y - X @ b)**2 + self.lambdu*cp.norm1(b)) #+ np.max(np.abs(b))
        prob = cp.Problem(objective)
        sol = prob.solve()
        # print("cvx beta: ",b.value)
        return sol

    def Pci(self,uz,Xcol):
        b = self.lambdu
        if Xcol.T @ uz > b:
            return uz - ((Xcol.T@uz - b)/np.linalg.norm(Xcol)**2)*Xcol
        elif - Xcol.T @ uz > b:
            Xcol=-Xcol
            return uz - ((Xcol.T@uz - b)/np.linalg.norm(Xcol)**2)*Xcol
        else:
            return uz

    def dykstras(self,X,y,b):
        """
        Dykstra's Projection Algorithm on Dual of Lasso
        """
        it = 0
        d = self.n
        u = [0]*(d+1)
        u[-1] = copy.deepcopy(y)
        z = [np.zeros(self.m) for i in range(self.n)]
        while it < self.maxiters:
            u[0] = u[-1].copy()
            for i in range(1,d+1):
                u[i] = copy.deepcopy(self.Pci(u[i-1].copy() + z[i-1].copy(), X[:,i-1].copy()))
                z[i-1] = u[i-1].copy() + z[i-1] - u[i].copy()
            
            it+=1
        return u[-1]

    def together(self,X,y,b, verbose=True):
        """
        Coordinate Descent & Dykstra's Algorithm
        Together, same iterations for comparison
        """
        ### dykstra init ###
        it = 0
        d = self.n
        u = [0]*(d+1)
        u[-1] = copy.deepcopy(y)
        z = [np.zeros(self.m) for i in range(self.m)]

        ### coord desc init ###
        beta = np.zeros(self.n).astype(np.longdouble)
        it=0

        ### Main Loop ###
        # while it < self.maxiters:
        while True:
            betaprev = beta.copy()
            u[0] = u[-1].copy()
            for i in range(1,d+1):

                ### coord_desc ###
                min_b = self.min_betaj(X,y,beta,i-1)
                beta[i-1]=min_b

                ### dykstra ###
                u[i] = copy.deepcopy(self.Pci(u[i-1].copy() + z[i-1].copy(), X[:,i-1].copy()))
                z[i-1] = u[i-1].copy() + z[i-1] - u[i].copy()

                ### equal check ###
                if verbose:
                    print("###"*10," \n")
                    print(z[i-1])
                    print("\n")
                    print(X[:,i-1]*beta[i-1])
                    print("%%%"*10," \n")
                
            ### termination condition ###
            if np.linalg.norm(beta-betaprev)<self.termination_cond:
                print("iterations: ",it, "dist: ", np.linalg.norm(beta-betaprev))
                break

            it+=1
    def coord_desc_2(self,X,y,betaf,b=False):
        """
        Cyclic Coordinate Descent on Lasso
        Has termination condition rather than termination iteration
        """
        beta = np.zeros(betaf.shape)

        if not b:
            beta = np.zeros(self.n)
        it=0
        # while it < self.maxiters:
        while True:
            betaprev = beta.copy()
            for j in range(len(beta)):
                min_b = self.min_betaj(X,y,beta,j)
                beta[j]=min_b
            ## terminal condition ##
            if np.linalg.norm(beta-betaprev)<self.termination_cond:
                print("iterations: ",it, "dist: ", np.linalg.norm(beta-betaprev))
                break
            it+=1
        return beta, it
    def dykstras2(self,X,y,b):
        """
        Dykstra's Projection Algorithm on Dual of 
        Has termination condition rather than termination iteration
        """
        it = 0
        # d = self.n
        d = X.shape[1]
        u = [0]*(d+1)
        u[-1] = copy.deepcopy(y)
        z = [np.zeros(X.shape[0]) for i in range(X.shape[1])]
        # compare = y-X@cbeta
        while True:
            u[0] = u[-1].copy()
            betaprev = np.linalg.pinv(X) @ (y-u[0].copy())

            for i in range(1,d+1):
                u[i] = copy.deepcopy(self.Pci(u[i-1].copy() + z[i-1].copy(), X[:,i-1].copy()))
                z[i-1] = u[i-1].copy() + z[i-1] - u[i].copy()

            beta = np.linalg.pinv(X) @ (y-u[-1].copy())
            if np.linalg.norm(beta-betaprev)<self.termination_cond:
                print("iterations: ",it, "dist: ", np.linalg.norm(beta-betaprev))
                break
            it+=1
        return u[-1]

    def gen_exp(self,m,n,s,l):
        X = np.random.rand(m,n) # random X
        idxs = random.sample(list(range(n)),n-s)
        b = np.random.rand(n)
        b[idxs] = 0 # sparse beta
        w = np.random.rand(m) # random noise
        y = X@b + w
        return X,y,b

    def performance(self,b,beta,y,X):
        """
        Compute Performance
            b: true beta
            beta: calculated beta
        """
        l2error = np.linalg.norm(beta-b)
        srp_tolerance = 0.1
        rec_suc, N = 0, len(beta)
        for j in range(N):
            if abs(b[j]-beta[j])<srp_tolerance*min(b[j],beta[j]):
                rec_suc+=1
        srp = (N-rec_suc)/N
        obj_val = self.obj_func(y,X,beta)

        return l2error,srp,obj_val

    def experiment(self,m_rows,n_cols,sparse, lambd):
        L = len(m_rows)

        ### exp ###
        results,speed_results=[],[]
        s,lam=sparse,lambd
        for i in range(L):
            m,n = m_rows[i],n_cols[i]
            X,y,b = self.gen_exp(m,n,s,lam)

            # cord desc #
            t1=time.time()
            beta,it_c = self.coord_desc_2(X,y,b,True)
            runtime_c = time.time() - t1
            l2error,srp,objval = self.performance(b,beta,y,X)
            results.append([m,n,s,lam,objval,l2error,srp,runtime_c,it_c])

            t1=time.time()
            cvx_sol = ls.cvx_min(X,y,b)
            runtime_s = time.time() - t1

            t1=time.time()
            d_sol = ls.dykstras2(X,y,b)
            runtime_d = time.time() - t1

            speed_results.append([m,n,s,lam,cvx_sol,runtime_s,objval,runtime_c,objval,runtime_d])
            
        return results, speed_results




if __name__=="__main__":
    ls = lasso_solver()
    X,y,b = ls.generate_X_y_b()

    ### coordinate descent ###
    cbeta=ls.coord_desc(X,y,b)
    csol=ls.obj_func(y,X,cbeta)
    compare = y-X@cbeta

    ### dykstra projection ###
    du = ls.dykstras(X,y,b)

    ### cvx solver ###
    cvxsol = ls.cvx_min(X,y,b)

    ### compare results ###
    print(compare,"\n\n",du,"\n\n") # compare uhat of Dykstra and Coord_desc
    print("cvx: ",cvxsol,"\ncoord_desc & dykstra: ",csol) # compare (coord_desc, Dykstra) and CVX_solver

    ### together ###
    ls.together(X,y,b,verbose=False) # change verbose to True to check equal conditions

    ### Experiments ###
    # m_rows = [20,20,20,40,40,40,60,60,60]
    # n_cols = [20,40,60,20,40,60,20,40,60]
    # m_rows = [30,30,30,100,100,100,60,200,200]
    # n_cols = [20,50,90,20,50,90,20,50,90]
    # m_rows = [60,120,180,60,120,180,60,120,180]
    # n_cols = [10,20,30,10,20,30,10,20,30]
    m_rows = [20,50,90,20,50,90,20,50,90]
    n_cols = [30,30,30,100,100,100,200,200,200]

    res1,sres1 = ls.experiment(m_rows,n_cols,sparse=8, lambd=0.01)
    res2,sres2 = ls.experiment(m_rows,n_cols,sparse=3, lambd=300)
    res3,sres3 = ls.experiment(m_rows,n_cols,sparse=8, lambd=300)
    res4,sres4 = ls.experiment(m_rows,n_cols,sparse=3, lambd=0.01)
    resf = res1+res2+res3+res4
    sresf = sres1+sres2+sres3+sres4

    ### get TEX file (Speed) ###
    headers=["m","n","s","lambda","CVX val","CVX time", "BCD val","BCD time", "Dyk val","Dyk time"]
    dfs = pd.DataFrame(sresf, columns=headers)
    dfs.to_latex(buf="table_s.tex",escape=False,index=False)

    ### Get TEX file ###
    headers=["m","n","s","lambda","obj val","l2 Error","SRP","Runtime","Iters"]
    df1 = pd.DataFrame(res1, columns=headers)
    df1.to_latex(buf="table1.tex",escape=False,index=False)
    df2 = pd.DataFrame(res2, columns=headers)
    df2.to_latex(buf="table2.tex",escape=False,index=False)
    df3 = pd.DataFrame(res3, columns=headers)
    df3.to_latex(buf="table3.tex",escape=False,index=False)
    df4 = pd.DataFrame(res4, columns=headers)
    df4.to_latex(buf="table4.tex",escape=False,index=False)
    dff = pd.DataFrame(resf, columns=headers)
    dff.to_latex(buf="tablef.tex",escape=False,index=False)


    


    




        