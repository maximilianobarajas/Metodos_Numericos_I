import numpy as np
import pandas as pd
class jacobi: 
    def __init__(self,A: np.array,b: np.array,x_0: np.array,tol: np.float64):
        self.A=A;self.b=b
        self.x_0=x_0;self.tol=tol
        self.M=self.construir_M()
        self.J=self.construir_J()
        self.c=self.construir_c()
    def construir_M(self):
        M=np.zeros((len(self.A),len(self.A)))
        for i in range(len(self.A)):
            for j in range(len(self.A)):
                if(i==j):
                    M[i][j]=self.A[i][j]
        return M
    def construir_J(self):
        return np.identity(len(self.A)) - np.dot(np.linalg.inv(self.M),self.A)
    def construir_c(self):
        return np.dot(np.linalg.inv(self.M),self.b)
    def aplicar_metodo(self):
        x_1=np.zeros(len(self.A))
        X:list=[]
        X.append(self.x_0)
        while True:
            x_1=np.dot(self.J,self.x_0) + self.c
            if(np.linalg.norm(x_1-self.x_0,ord=np.inf)<=self.tol):
                break
            self.x_0=x_1
            X.append(self.x_0)
        print(self.x_0)
        return pd.DataFrame(X,columns=['x' + str(i+1) for i in range(len(self.x_0))])
