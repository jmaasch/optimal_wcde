import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from utils import Utils


class DataGenerator():


    def __init__(self):
        self.u = Utils()
    
    
    def get_data_lemma_5(self,
                         n: int = 1000, 
                         x_causes_y: bool = True,
                         xy_coeff: float = 1.0,
                         coefficient_range: tuple = (1.1, 1.25),
                         exp: int = 1,
                         scale: bool = False):

        '''
        '''

        # Sample noise terms for structural equations.
        noise = [np.random.normal(loc = 0.0, scale = 1.0, size = n) for i in range(5)]

        # Define coefficient generator.
        coeff = lambda : np.random.uniform(low = coefficient_range[0],
                                           high = coefficient_range[1],
                                           size = 1)

        fun = lambda x: coeff()*x**exp
        
        # Define variables.
        if not x_causes_y:
            xy_coeff = 0
        B2 = noise[0]
        A = fun(B2) + noise[1]
        G1 = fun(A) + fun(B2) + noise[2]
        G2 = fun(B2) + noise[3]
        Y = xy_coeff*A**exp + fun(G1) + fun(G2) + noise[4]
            
        df = pd.DataFrame({"A": A.reshape(-1), 
                           "Y": Y.reshape(-1), 
                           "B2": B2.reshape(-1),
                           "G1": G1.reshape(-1), 
                           "G2": G2.reshape(-1)})
        if scale:
            df = self.u.scale_dataframe(df)

        # Define graph.
        g = nx.DiGraph({"A": ["G1", "Y"], 
                        "Y": [], 
                        "B2": ["A", "G1", "G2"],
                        "G1": ["Y"], 
                        "G2": ["Y"]})

        return df, g


    def get_data_lemma_8(self,
                         n: int = 1000, 
                         x_causes_y: bool = True,
                         xy_coeff: float = 1.0,
                         coefficient_range: tuple = (1.1, 1.25),
                         exp: int = 1,
                         scale: bool = False):

        '''
        '''

        # Sample noise terms for structural equations.
        noise = [np.random.normal(loc = 0.0, scale = 1.0, size = n) for i in range(5)]

        # Define coefficient generator.
        coeff = lambda : np.random.uniform(low = coefficient_range[0],
                                           high = coefficient_range[1],
                                           size = 1)

        fun = lambda x: coeff()*x**exp
        
        # Define variables.
        if not x_causes_y:
            xy_coeff = 0
        G2 = noise[0]
        A = fun(G2) + noise[1]
        B1 = fun(G2) + fun(A) + noise[2]
        G1 = fun(B1) + fun(G2) + noise[3]
        Y = xy_coeff*A**exp + fun(G1) + fun(G2) + noise[4]
            
        df = pd.DataFrame({"A": A.reshape(-1), 
                           "Y": Y.reshape(-1), 
                           "B1": B1.reshape(-1),
                           "G1": G1.reshape(-1), 
                           "G2": G2.reshape(-1)})
        if scale:
            df = self.u.scale_dataframe(df)

        # Define graph.
        g = nx.DiGraph({"A": ["B1", "Y"], 
                        "Y": [], 
                        "B1": ["G1"],
                        "G1": ["Y"], 
                        "G2": ["A", "B1", "G1", "Y"]})

        return df, g
    