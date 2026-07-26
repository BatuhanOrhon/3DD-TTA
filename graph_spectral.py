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
    def __init__(self, k=10, delta=0.1, gamma=0.6, M=100, M_mid=1024, M_high=2048, use_4d_gft=False, device='cuda'):
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
        self.M_mid = M_mid
        self.M_high = M_high
        self.use_4d_gft = use_4d_gft
        self.device = device
        
        self.knn = KNN(k=self.k, transpose_mode=True)
        
    def forward(self, h_0):
        """
        Computes the Graph Fourier Transform (GFT) and extracts the low-frequency components.
        
        Args:
            h_0 (torch.Tensor): Latent points tensor of shape (B, N, C) where C is usually 4.
            
        Returns:
            H_orig (torch.Tensor): Full frequency spectral components, shape (B, N, C_out).
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
        
        # 4. Compute Unnormalized Graph Laplacian: L^o = D^o - A^o
        D_o_diag = A_o.sum(dim=-1)  # (B, N)
        D_o = torch.diag_embed(D_o_diag)  # (B, N, N)
        L_o = D_o - A_o  # (B, N, N)
        
        # FIX: İzole edilmiş düğümlerin (outliers) özdeğerini 0'dan 1000'e çıkararak
        # onları en yüksek frekanslara itiyoruz. Böylece düşük frekanslar sadece
        # gerçek şekli yansıtacak ve geometri (translation invariance) korunacak.
        isolated_nodes = (D_o_diag == 0).float()  # (B, N)
        L_o = L_o + torch.diag_embed(isolated_nodes * 1000.0)
        
        # Add numerical stability jitter (epsilon) to the diagonal to prevent 
        # convergence failures on ill-conditioned matrices (e.g. density_inc noise with overlapping points)
        jitter = torch.eye(N, device=L_o.device).unsqueeze(0) * 1e-5
        L_o_stable = L_o + jitter
        
        # 5. Perform Eigen-decomposition on L^o
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, U_o = torch.linalg.eigh(L_o_stable)
        
        # 6. Perform Graph Fourier Transform (GFT)
        signal = h_0 if self.use_4d_gft else h_0_spatial
        
        # GFT: H_orig = (U^o)^T @ h_0
        H_orig = torch.bmm(U_o.transpose(1, 2), signal)  # (B, N, C_out)
        
        # 7. (Removed) We no longer extract only low-frequency components here, 
        # so tta_gsd.py can handle multi-band slicing (Low, Mid, High).
        
        # Detach to ensure these are constants during the diffusion guidance phase
        return H_orig.detach(), U_o.detach()
