# L7 matmul: ours vc5e98d run1 (results/final-c5e98d5a75/ours-L236710-seat-246.jsonl); organizers' checker: 10/10 cases
import numpy as np

def kernel(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Dimensions do not match for matrix multiplication"
    
    tile_rows = 128
    tile_cols = 512
    
    # Initialize result
    c = np.zeros((M, N), dtype=np.float32)
    
    # Process rows in tiles
    for i in range(0, M, tile_rows):
        end_i = min(i + tile_rows, M)
        # Process columns in tiles
        for j in range(0, N, tile_cols):
            end_j = min(j + tile_cols, N)
            
            # Initialize accumulator for this tile in float64
            acc = np.zeros((end_i - i, end_j - j), dtype=np.float64)
            
            # Process k in tiles
            for k in range(0, K, tile_rows):
                end_k = min(k + tile_rows, K)
                
                # Extract tile from a and b
                tile_a = a[i:end_i, k:end_k].astype(np.float64)
                tile_b = b[k:end_k, j:end_j].astype(np.float64)
                
                # Perform tile-wise matrix multiplication
                for r in range(end_i - i):
                    for c_col in range(end_j - j):
                        val = 0.0
                        for k_col in range(end_k - k):
                            val += tile_a[r, k_col] * tile_b[k_col, c_col]
                        acc[r, c_col] += val
            
            # Write the accumulated result back to c (convert to float32)
            c[i:end_i, j:end_j] = acc.astype(np.float32)
    
    return c
