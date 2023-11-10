import numpy as np
import pandas as pd

class gauss_seidel:
    def __init__(self, A: np.array, b: np.array, x_0: np.array, tol: np.float64):
        self.A = A
        self.b = b
        self.x_0 = x_0
        self.tol = tol
        self.L, self.D, self.U = self.descomposicion_L_D_U()
        self.c = self.buscar_c()
        self.G=(np.eye(len(self.x_0)) -np.dot(np.linalg.inv(self.D-self.L),self.A))

    def descomposicion_L_D_U(self):
        n = len(self.A)
        L = -1*np.tril(self.A,k=-1)
        D = np.diag(np.diag(self.A))
        U = -1*np.triu(self.A,k=1)
        return L, D, U

    def buscar_c(self):
        return np.dot(np.linalg.inv((self.D-self.L)),self.b)

    def aplicar_metodo(self):
        x_1 = np.zeros(len(self.A))
        X = [self.x_0]
        while True:
            x_1 = np.dot(self.G,self.x_0) + self.c
            if np.linalg.norm(x_1 - self.x_0,ord=np.inf) <= self.tol:
                break
            self.x_0 = x_1
            X.append(self.x_0)
        print(self.x_0)
        return pd.DataFrame(X,columns=['x' + str(i+1) for i in range(len(self.x_0))])
