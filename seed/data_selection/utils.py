import faiss
import torch
import numpy as np
import heapq
from scipy.sparse import lil_matrix


def build_sparse_knn_graph_gpu(embeddings: torch.Tensor, knn_k=20, sim_threshold=0.9, gpu_id=0, batch_size=8192):
    """
    Efficient GPU FAISS sparse graph construction.
    Returns scipy.sparse.lil_matrix adjacency matrix.
    """
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    print(f"Embeddings shape: {embeddings.shape}")
    N, D = embeddings.shape
    xb = embeddings.cpu().numpy().astype(np.float32)

    # FAISS GPU index
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatIP(D)
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    gpu_index.add(xb)
    print("GPU Index built")

    # Initialize sparse graph
    graph = lil_matrix((N, N), dtype=np.bool_)
    print("Sparse graph initialized")

    # Batch query top-k nearest neighbors
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        print(f"Processing batch {start} to {end}")
        sims_batch, idxs_batch = gpu_index.search(xb[start:end], knn_k + 1)  # Include self

        for i in range(end - start):
            row_idx = start + i
            for j, sim in zip(idxs_batch[i][1:], sims_batch[i][1:]):  # skip self
                if sim >= sim_threshold:
                    graph[row_idx, j] = True
                    graph[j, row_idx] = True  # Symmetric

    return graph


def build_sparse_knn_graph_gpu_local_scaling(
    embeddings: torch.Tensor,
    knn_k=20,
    sim_threshold=0.9,
    alpha=0.95,
    gpu_id=0,
    batch_size=8192,
):
    """
    GPU FAISS sparse kNN graph with conservative local scaling.

    Edge condition:
        sim >= max(sim_threshold, alpha * sigma_i)

    Returns:
        scipy.sparse.lil_matrix
    """
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    print(f"Embeddings shape: {embeddings.shape}")
    N, D = embeddings.shape
    xb = embeddings.cpu().numpy().astype(np.float32)

    # FAISS GPU index
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatIP(D)
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    gpu_index.add(xb)
    print("GPU Index built")

    # Compute local scaling sigma_i
    print("Computing local scaling sigmas...")
    sigmas = np.zeros(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sims_batch, _ = gpu_index.search(xb[start:end], knn_k + 1)
        sigmas[start:end] = sims_batch[:, -1]  # k-th nearest neighbor similarity

    # Initialize sparse graph
    graph = lil_matrix((N, N), dtype=np.bool_)
    print("Sparse graph initialized")

    # Build sparse graph
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        print(f"Processing batch {start} to {end}")
        sims_batch, idxs_batch = gpu_index.search(xb[start:end], knn_k + 1)

        for i in range(end - start):
            row_idx = start + i
            sigma_i = sigmas[row_idx]
            adaptive_threshold = max(sim_threshold, alpha * sigma_i)

            for j, sim in zip(idxs_batch[i][1:], sims_batch[i][1:]):
                if sim >= adaptive_threshold:
                    graph[row_idx, j] = True
                    graph[j, row_idx] = True

    return graph


def greedy_mwis_sparse(weights: torch.Tensor, adj_matrix, max_select=None):
    """
    Greedy MWIS selection, input sparse matrix
    """
    N = len(weights)
    selected = []
    removed = np.zeros(N, dtype=bool)

    # Build max-heap
    heap = [(-weights[i].item(), i) for i in range(N)]
    heapq.heapify(heap)

    while heap and (max_select is None or len(selected) < max_select):
        neg_w, u = heapq.heappop(heap)
        if removed[u]:
            continue
        selected.append(u)
        removed[u] = True
        # Remove neighbors
        neighbors = adj_matrix.rows[u]  # lil_matrix property
        removed[neighbors] = True

    return selected

def select_mwis_subset(training_info: torch.Tensor,
                           influence_score: torch.Tensor,
                           k_select=5000,
                           knn_k=20,
                           sim_threshold=0.9,
                           gpu_id=0):
    """
    MWIS pipeline high-performance GPU version
    """
    print("Building sparse graph on GPU with batches...")
    graph = build_sparse_knn_graph_gpu_local_scaling(training_info, knn_k=knn_k, sim_threshold=sim_threshold, gpu_id=gpu_id)
    print("Running greedy MWIS selection...")
    selected_indices = greedy_mwis_sparse(influence_score, graph, max_select=k_select)
    return selected_indices