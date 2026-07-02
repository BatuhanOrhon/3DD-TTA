import torch
import torch.nn as nn

# Projedeki gibi KNN_CUDA kullaniyoruz. Eger ileride bir hata (error) alirsak,
# PyTorch'un standart torch.cdist ve torch.topk fonksiyonlarina gecebiliriz.
# Ornek saf PyTorch k-NN cozumu:
# dists = torch.cdist(x, x)
# topk_dists, topk_idx = torch.topk(dists, k, dim=-1, largest=False)

try:
    from knn_cuda import KNN
except ImportError:
    print("Warning: knn_cuda not installed, graph_spectral.py will fail to initialize KNN_CUDA.")


class GraphSpectralDNA(nn.Module):
    def __init__(self, k=10, delta=0.1, gamma=0.6, M=100, use_4d_gft=False, device='cuda'):
        """
        Graph Spectral DNA Module for 3DD-TTA.
        
        Args:
            k (int): Number of nearest neighbors for KNN graph.
            delta (float): Hyperparameter for RBF kernel.
            gamma (float): Multiplier for outlier filtering threshold.
            M (int): Number of low-frequency components to keep.
            use_4d_gft (bool): If True, computes GFT on all 4 dimensions of h_0 (XYZ + feature). 
                               If False, computes only on 3 spatial dimensions (XYZ).
            device (str): Device to run computations on.
        """
        super().__init__()
        self.k = k
        self.delta = delta
        self.gamma = gamma
        self.M = M
        self.use_4d_gft = use_4d_gft
        self.device = device
        
        self.knn = KNN(k=self.k, transpose_mode=True)
        
    def forward(self, h_0):
        """
        Computes the Graph Fourier Transform (GFT) and extracts the low-frequency components.
        
        Args:
            h_0 (torch.Tensor): Latent points tensor of shape (B, N, C) where C is usually 4.
            
        Returns:
            H_orig_low (torch.Tensor): Low frequency spectral components, shape (B, M, C_out).
            U_o (torch.Tensor): Eigenvectors of the graph Laplacian, shape (B, N, N).
        """
        B, N, C = h_0.shape
        
        # 1. Extract the first 3 spatial dimensions
        h_0_spatial = h_0[:, :, :3].contiguous()
        
        # 2. Construct k-NN graph
        # KNN_CUDA returns L2 distances and indices of shape (B, N, k)
        dist, idx = self.knn(h_0_spatial, h_0_spatial)
        
        # Compute RBF weights: w_ij = exp(-d(x_i, x_j)^2 / (2 * delta^2))
        dist_sq = dist ** 2
        w = torch.exp(-dist_sq / (2 * self.delta ** 2))
        
        # Construct dense adjacency matrix A of shape (B, N, N)
        A = torch.zeros((B, N, N), device=h_0.device)
        batch_idx = torch.arange(B, device=h_0.device).view(B, 1, 1).expand(B, N, self.k)
        row_idx = torch.arange(N, device=h_0.device).view(1, N, 1).expand(B, N, self.k)
        A[batch_idx, row_idx, idx] = w
        
        # Symmetrize A (since KNN graph is directed, we make it undirected for graph Laplacian)
        A = torch.max(A, A.transpose(1, 2))
        
        # 3. Outlier Filtering
        degrees = A.sum(dim=-1)  # (B, N)
        sum_all_degrees = degrees.sum(dim=-1)  # (B,)
        
        # Calculate threshold tau = gamma * (average degree)
        tau = self.gamma * (sum_all_degrees / N)
        tau = tau.view(B, 1)  # Broadcast over nodes
        
        # Keep edges only if node's degree > tau, otherwise set to 0.
        valid_nodes = (degrees > tau).float()  # (B, N)
        A_o = A * valid_nodes.unsqueeze(1) * valid_nodes.unsqueeze(2)  # (B, N, N)
        
        # 4. Compute Symmetric Normalized Graph Laplacian: L^o = I - D^{-1/2} A^o D^{-1/2}
        D_o_diag = A_o.sum(dim=-1)  # (B, N)
        
        # Calculate D^{-1/2}, setting 0 to 0 to avoid division by zero for outliers
        D_inv_sqrt_diag = torch.zeros_like(D_o_diag)
        mask = D_o_diag > 0
        D_inv_sqrt_diag[mask] = 1.0 / torch.sqrt(D_o_diag[mask])
        
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt_diag)  # (B, N, N)
        
        # Compute D^{-1/2} A^o D^{-1/2}
        normalized_A = torch.bmm(torch.bmm(D_inv_sqrt, A_o), D_inv_sqrt)
        
        # Compute L^o = I - normalized_A
        I = torch.eye(N, device=h_0.device).unsqueeze(0).expand(B, N, N)
        L_o = I - normalized_A  # (B, N, N)
        
        # 5. Perform Eigen-decomposition on L^o
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, U_o = torch.linalg.eigh(L_o)
        
        # 6. Perform Graph Fourier Transform (GFT)
        signal = h_0 if self.use_4d_gft else h_0_spatial
        
        # GFT: H_orig = (U^o)^T @ h_0
        H_orig = torch.bmm(U_o.transpose(1, 2), signal)  # (B, N, C_out)
        
        # 7. Extract low-frequency components
        H_orig_low = H_orig[:, :self.M, :]  # (B, M, C_out)
        
        # Detach to ensure these are constants during the diffusion guidance phase
        return H_orig_low.detach(), U_o.detach()
