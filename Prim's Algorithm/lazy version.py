from graph import Graph
import heapq

class lazyVersion:
    def __init__(self,graph):
        self.graph=graph
        self.totalNodes = self.graph.totalNodes
        self.visited = [False] * self.totalNodes
      
    def addEdge(self, nodeIndex, pq):
        self.visited[nodeIndex] = True
        edges = self.graph.adj_list[nodeIndex]
        # print(edges)
        
        for end, weight in edges:
            # print((end,weight))
            if not self.visited[end]:
                heapq.heappush(pq, (weight, nodeIndex, end))
        
            
        
    def lazyPrimes(self, s=0):
        totalEdges = self.totalNodes - 1
        edgeCount = 0
        mstEdges = []
        pq=[]
        self.addEdge(s,pq)
        
        while pq and edgeCount < totalEdges:         
            weight, nodeIndex, end = heapq.heappop(pq)
            # print(nodeIndex, end, weight)
            # startIndex = nodeIndex
            # nodeIndex = end
            
            if self.visited[end]:
                continue
            
            mstEdges.append((nodeIndex, end, weight))
            edgeCount += 1
            
            self.addEdge(end, pq)
        
        if edgeCount != totalEdges:
            return (None, None)
        
        return (mstEdges)
        
graph = Graph(8, type=1)

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

lazy_prims = lazyVersion(graph)
mst = lazy_prims.lazyPrimes()
print(mst)