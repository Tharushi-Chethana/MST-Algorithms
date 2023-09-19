class Graph:
    def __init__(self, totalNodes, type):
        self.totalNodes = totalNodes
        self.nodes = range(self.totalNodes)
        
        self.type = type
        
        self.adj_list = {node: set() for node in self.nodes}
        
    def add_edge(self, start, end, weight):
        edge = (end, weight)
        
        # for directed graph
        if self.type==0:
            self.adj_list[start].add(edge)
            
        # for undirected graph
        else:
            self.adj_list[start].add(edge)
            reverse_edge = (start, weight)
            self.adj_list[end].add(reverse_edge)
            
    def printAdjList(self):
        for key in self.adj_list.keys():
            print("node", key, ": ", self.adj_list[key])


# type = int(input("Please enter 0 or 1 (directed graph:0 and undirected graph:1): "))
# print(type)
type=1
graph = Graph(8,type)

graph.add_edge(0, 1, 10)
graph.add_edge(0, 2, 1)
graph.add_edge(0, 3, 4)
graph.add_edge(1, 2, 3)
graph.add_edge(1, 4, 0)
graph.add_edge(2, 5, 8)
graph.add_edge(2, 3, 2)
graph.add_edge(3, 5, 2)
graph.add_edge(3, 6, 7)
graph.add_edge(4, 7, 8)
graph.add_edge(4, 5, 1)
graph.add_edge(5, 7, 9)
graph.add_edge(5, 6, 6)
graph.add_edge(6, 7, 12)

# graph.printAdjList()