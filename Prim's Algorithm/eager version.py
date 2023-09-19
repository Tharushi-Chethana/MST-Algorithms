
from graph import Graph 
import heapq 

class eagerVersion:
    def __init__(self, graph):
        self.graph = graph
        self.totalNodes = self.graph.totalNodes
        self.visited = [False] * self.totalNodes

    def relaxEdgesAtNode(self, currentNodeIndex, ipq):
        self.visited[currentNodeIndex] = True  
        edges = self.graph.adj_list[currentNodeIndex]
        
        for end, weight in edges:
            endNodeIndex = end  # Get the index of the neighboring node
            
            if self.visited[endNodeIndex]:
                continue  
            
            # Use heapq.heappush to insert the edge into the priority queue 'ipq'
            heapq.heappush(ipq, (weight, currentNodeIndex, endNodeIndex))


    def eagerPrimes(self, s=0):
        totalEdges = self.totalNodes - 1
        edgeCount = 0
        mstEdges = [] 
        ipq = []
        
        self.relaxEdgesAtNode(s, ipq)
        
        while ipq and edgeCount < totalEdges:         
            weight, start, end = heapq.heappop(ipq)  # Get the edge with the minimum weight
            
            if self.visited[end]:
                continue  # Skip if the destination node is already visited
            
            mstEdges.append((start, end, weight))  # Add the edge to the MST
            edgeCount += 1
            
            self.relaxEdgesAtNode(end, ipq)
            
        if edgeCount < totalEdges:
            return None 
        
        return mstEdges
        

graph = Graph(7, type=1)

graph.add_edge(0, 2, 0)
graph.add_edge(0, 5, 7)
graph.add_edge(0, 3, 5)
graph.add_edge(0, 1, 9)
graph.add_edge(1, 3, -2) 
graph.add_edge(1, 6, 4)
graph.add_edge(1, 4, 3)
graph.add_edge(2, 5, 6)
graph.add_edge(3, 5, 2)
graph.add_edge(3, 6, 3)
graph.add_edge(5, 6, 1)
graph.add_edge(6, 4, 6)


eager_prims = eagerVersion(graph)

mst = eager_prims.eagerPrimes()
print(mst)
