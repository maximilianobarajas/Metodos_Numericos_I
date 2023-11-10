import pandas as pd
from sympy import Matrix
class Newton():
    def __init__(self,ecuaciones,variables,x0,tol) -> None:
        self.ecuaciones=ecuaciones
        self.variables=variables
        self.Jacobiana=self.calcular_jacobiana()
        self.x_0=Matrix(x0)
        self.n=len(x0)
        self.tol=tol
    def calcular_jacobiana(self):
        return self.ecuaciones.jacobian(self.variables)
    def aplicar_metodo(self):
        matriz_jacobiana=self.Jacobiana.subs([(self.variables[i],self.x_0[i]) for i in range(len(self.variables))])
        X=[self.x_0]
        diferencias=["-"]
        while True:
            y_0=matriz_jacobiana.solve(-1*self.ecuaciones.subs([(self.variables[i],self.x_0[i]) for i in range(len(self.variables))]))
            x_1=self.x_0+y_0
            X.append(x_1)
            diferencias.append((x_1-self.x_0).dot(x_1-self.x_0))
            if((x_1-self.x_0).dot(x_1-self.x_0)<=self.tol):
                break
            else:
                self.x_0=x_1
        return pd.DataFrame([[R[i] for i in range(self.n)] for R in X],columns=['x' + str(i+1) for i in range(self.n)])
