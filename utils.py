import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations
from sklearn.preprocessing import StandardScaler


class Utils():

    
    def plot_from_adj(self,
                      adjacency_matrix,
                      labels,
                      figsize = (10,10),
                      dpi = 200,
                      node_size = 800,
                      arrow_size = 10):
        
        g = nx.from_numpy_array(adjacency_matrix, create_using = nx.DiGraph)
        plt.figure(figsize = figsize, dpi = dpi)  
        nx.draw_shell(g, 
                      node_size = node_size, 
                      labels = dict(zip(list(range(len(labels))), labels)), 
                      arrowsize = arrow_size,
                      node_color = "pink",
                      with_labels = True)
        plt.show()
        plt.close()
        

    def plot_from_graph(self,
                        g,
                        figsize = (10,10),
                        dpi = 200,
                        node_size = 800,
                        arrow_size = 10):
        
        plt.figure(figsize = figsize, dpi = dpi)  
        nx.draw_shell(g, 
                      node_size = node_size, 
                      arrowsize = arrow_size,
                      node_color = "pink",
                      with_labels = True)
        plt.show()
        plt.close()


    def get_powerset(self, 
                     iterable):
        
        s = list(iterable)
        powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        return list(powerset)


    def scale_dataframe(self, 
                        df: pd.DataFrame) -> pd.DataFrame:
        
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df)
        return pd.DataFrame(scaled_values, columns = df.columns)
                