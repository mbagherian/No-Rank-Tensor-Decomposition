# no-rank-tensor-decomposition/metric_learning/decomposition.py
import numpy as np
import warnings

# TensorLy imports with auto-install
try:
    import tensorly as tl
    from tensorly.decomposition import parafac, tucker
    TENSORLY_AVAILABLE = True
except ImportError:
    print("⚠️  TensorLy not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorly"])
    import tensorly as tl
    from tensorly.decomposition import parafac, tucker
    TENSORLY_AVAILABLE = True

warnings.filterwarnings('ignore')

class ReconstructionAnalyzer:
    def __init__(self):
        tl.set_backend('numpy')
    
    def calculate_reconstruction_metrics(self, original_data, reconstructed_data):
        """Calculate reconstruction error and explained variance"""
        original_tensor = tl.tensor(original_data)
        reconstructed_tensor = tl.tensor(reconstructed_data)
        
        # Reconstruction error (Frobenius norm)
        reconstruction_error = tl.norm(original_tensor - reconstructed_tensor) / tl.norm(original_tensor)
        
        # Explained variance
        total_variance = tl.norm(original_tensor) ** 2
        residual_variance = tl.norm(original_tensor - reconstructed_tensor) ** 2
        explained_variance = 1 - (residual_variance / total_variance)
        
        return {
            'reconstruction_error': float(reconstruction_error),
            'explained_variance': float(explained_variance)
        }
    
    def reconstruct_from_cp(self, weights, factors, original_shape):
        """Reconstruct tensor from CP decomposition"""
        return tl.cp_to_tensor((weights, factors))
    
    def reconstruct_from_tucker(self, core, factors, original_shape):
        """Reconstruct tensor from Tucker decomposition"""
        return tl.tucker_to_tensor((core, factors))
    
    def reconstruct_from_metric_learning(self, embeddings, original_data):
        """Reconstruct images from metric learning embeddings using linear regression"""
        from sklearn.linear_model import LinearRegression
        
        embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        original_flat = original_data.reshape(original_data.shape[0], -1)
        
        # Learn reconstruction weights using linear regression
        regressor = LinearRegression()
        regressor.fit(embeddings_np, original_flat)
        reconstructed_flat = regressor.predict(embeddings_np)
        
        # Reshape back to original image dimensions
        reconstructed = reconstructed_flat.reshape(original_data.shape)
        return reconstructed

def cp_decomposition_embeddings(images, ranks=[5, 10, 20]):
    """Perform CP decomposition and return embeddings for different ranks"""
    print("🔄 Performing CP Decomposition...")
    
    # Reshape images to tensor: (n_samples, height, width)
    tensor_data = np.array(images)  # Shape: (n_samples, height, width)
    
    cp_results = {}
    reconstructor = ReconstructionAnalyzer()
    
    for rank in ranks:
        try:
            # Perform CP decomposition
            weights, factors = parafac(tensor_data, rank=rank, init='random', tol=1e-6)
            
            # Use the first factor matrix as embeddings (sample embeddings)
            sample_embeddings = factors[0]  # Shape: (n_samples, rank)
            
            # Reconstruct and calculate metrics
            reconstructed = reconstructor.reconstruct_from_cp(weights, factors, tensor_data.shape)
            metrics = reconstructor.calculate_reconstruction_metrics(tensor_data, reconstructed)
            
            cp_results[f'CP-R{rank}'] = {
                'embeddings': sample_embeddings,
                'reconstruction_metrics': metrics,
                'reconstructed': reconstructed
            }
            print(f"   CP Rank {rank}: embeddings shape {sample_embeddings.shape}, "
                  f"Recon Error: {metrics['reconstruction_error']:.4f}, "
                  f"Explained Var: {metrics['explained_variance']:.4f}")
            
        except Exception as e:
            print(f"   CP Rank {rank} failed: {e}")
            cp_results[f'CP-R{rank}'] = None
    
    return cp_results

def tucker_decomposition_embeddings(images, ranks=[5, 10, 20]):
    """Perform Tucker decomposition and return embeddings for different ranks"""
    print("🔄 Performing Tucker Decomposition...")
    
    # Reshape images to tensor: (n_samples, height, width)
    tensor_data = np.array(images)  # Shape: (n_samples, height, width)
    
    tucker_results = {}
    reconstructor = ReconstructionAnalyzer()
    
    for rank in ranks:
        try:
            # Perform Tucker decomposition
            core, factors = tucker(tensor_data, rank=[rank, tensor_data.shape[1], tensor_data.shape[2]], 
                                 init='random', tol=1e-6)
            
            # Use the first factor matrix as embeddings (sample embeddings)
            sample_embeddings = factors[0]  # Shape: (n_samples, rank)
            
            # Reconstruct and calculate metrics
            reconstructed = reconstructor.reconstruct_from_tucker(core, factors, tensor_data.shape)
            metrics = reconstructor.calculate_reconstruction_metrics(tensor_data, reconstructed)
            
            tucker_results[f'Tucker-R{rank}'] = {
                'embeddings': sample_embeddings,
                'reconstruction_metrics': metrics,
                'reconstructed': reconstructed
            }
            print(f"   Tucker Rank {rank}: embeddings shape {sample_embeddings.shape}, "
                  f"Recon Error: {metrics['reconstruction_error']:.4f}, "
                  f"Explained Var: {metrics['explained_variance']:.4f}")
            
        except Exception as e:
            print(f"   Tucker Rank {rank} failed: {e}")
            tucker_results[f'Tucker-R{rank}'] = None
    
    return tucker_results