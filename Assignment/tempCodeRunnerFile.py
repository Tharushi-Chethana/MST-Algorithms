dense_mst_kruskal = kruskal(dense_graph_matrix)
sparse_mst_kruskal = kruskal(sparse_graph_list)

dense_mst_lazy = prim_lazy(dense_graph_matrix)
sparse_mst_lazy = prim_lazy(sparse_graph_list)

dense_mst_eager = prim_eager(dense_graph_matrix)
sparse_mst_eager = prim_eager(sparse_graph_list)