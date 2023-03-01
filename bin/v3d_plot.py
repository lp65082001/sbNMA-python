import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random 
import pickle

class plot_3d_network:

    def __init__(self,adjancy_matrix,position,path):
        self.adj = adjancy_matrix
        self.pos = position
        self.name = path

    def figure(self,graph,angle=30):
        #pos = nx.get_node_attributes(graph,'pos')
        with plt.style.context("bmh"):
            fig = plt.figure(figsize=(10,7))
            ax = Axes3D(fig)
            for xi,yi,zi in self.pos:      
                ax.scatter(xi,yi,zi,edgecolor='b',alpha=0.9,s=100)
                for i,j in enumerate(graph.edges()):
                    x = np.array((self.pos[j[0]-1][0],self.pos[j[1]-1][0]))
                    y = np.array((self.pos[j[0]-1][1],self.pos[j[1]-1][1]))
                    z = np.array((self.pos[j[0]-1][2],self.pos[j[1]-1][2]))
                    ax.plot(x,y,z,c='black',alpha=0.9, linewidth=0.5)
        ax.view_init(30,angle)
 
        pickle.dump(fig,open('FigureObject.fig.pickle','wb'))
        #ax.set_facecolor('white')
        plt.savefig(self.name+'.png',bbox_inches='tight')
        #plt.show()


    def graph_gen(self,cutoff=0.5):
        node_index = []
        for i in range(self.adj.shape[0]):
            node_index.append(i+1)
        
        edge_list = []
        correlation_list = np.where((np.triu(self.adj,1) < -cutoff) | (np.triu(self.adj,1)>cutoff))
        for k in range(0,correlation_list[0].shape[0]):
            edge_list.append((correlation_list[0][k]+1,correlation_list[1][k]+1))

        G = nx.DiGraph()
        G.add_edges_from(edge_list)
        G.add_nodes_from(node_index)
        return G
    
    def plot_network(self):
        graph = self.graph_gen()
        self.figure(graph)