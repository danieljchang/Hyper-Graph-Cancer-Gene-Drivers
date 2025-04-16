import pandas as pd
import numpy as np

def load_mutation_data(filepath):
    """
    Load mutation data from a tab-delimited file.
    Assumes the first column contains gene names.
    Filters out genes or samples with no mutations.
    Returns:
        mut_df: a DataFrame with genes as index and samples as columns.
    """
    df = pd.read_csv(filepath, sep="\t", header=0, dtype=str)
    # Remove rows with missing or empty gene names
    df = df.dropna(subset=[df.columns[0]])
    df = df[df.iloc[:, 0] != ""]
    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    # Convert values to numeric (0/1); non-numeric become 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    # Filter out genes with no mutations and samples with no mutations
    df = df.loc[df.sum(axis=1) != 0, df.sum(axis=0) != 0]
    return df

# Example usage:
# mut_df = load_mutation_data("GBM_mc3_gene_level.txt")
def construct_hypergraph(mut_df, threshold=5):
    """
    Constructs a hypergraph incidence matrix H.
    For each gene, find other genes with co-mutation count above threshold.
    
    Args:
        mut_df: DataFrame of mutation data (genes x samples)
        threshold: minimum co-mutation count to consider a connection
        
    Returns:
        H: numpy array of shape (num_genes, num_hyperedges)
        hyperedges: list of hyperedges, each a list of gene names
        genes: list of gene names (from mut_df.index)
    """
    genes = mut_df.index.tolist()
    mut_data = mut_df.values
    # Compute gene-gene co-mutation matrix:
    co_mut_counts = np.dot(mut_data, mut_data.T)
    hyperedges = []
    for i, gene in enumerate(genes):
        # Find genes with co-mutation counts above threshold for gene i
        co_mutated_idx = np.where(co_mut_counts[i] >= threshold)[0]
        # Only create a hyperedge if more than one gene meets criteria (i.e. a module)
        if len(co_mutated_idx) > 1:
            # Get gene names
            edge = [genes[j] for j in co_mutated_idx]
            hyperedges.append(edge)
    
    # Now build an incidence matrix H (genes x hyperedges)
    H = np.zeros((len(genes), len(hyperedges)))
    for j, edge in enumerate(hyperedges):
        for gene in edge:
            i = genes.index(gene)
            H[i, j] = 1
    return H, hyperedges, genes

# Example usage:
# H, hyperedges, gene_list = construct_hypergraph(mut_df, threshold=5)
def compute_transition_matrix(H, epsilon=1e-10):
    """
    Compute the hypergraph transition matrix P.
    
    Args:
        H: incidence matrix (genes x hyperedges)
        epsilon: a small value added to degree entries to avoid division by zero
        
    Returns:
        P: transition probability matrix (genes x genes)
    """
    gene_degrees = np.sum(H, axis=1) + epsilon
    hyperedge_degrees = np.sum(H, axis=0) + epsilon
    Dv_inv = np.diag(1 / gene_degrees)
    De_inv = np.diag(1 / hyperedge_degrees)
    P = Dv_inv.dot(H).dot(De_inv).dot(H.T)
    return P

def hypergraph_random_walk(P, seed_indices, restart_prob=0.15, max_iter=100, tol=1e-6):
    """
    Perform a random walk with restart on the hypergraph.
    
    Args:
        P: transition probability matrix (genes x genes)
        seed_indices: indices of seed genes (e.g. genes known to be mutated in a sample)
        restart_prob: probability of restart (1 - restart_prob is the teleport probability)
        max_iter: maximum iterations
        tol: convergence tolerance
        
    Returns:
        s: importance score vector (genes-length, normalized to sum to 1)
    """
    n = P.shape[0]
    # Initialize seed vector v0: only the seed genes get nonzero values.
    v0 = np.zeros(n)
    if len(seed_indices) == 0:
        raise ValueError("No seed indices provided for random walk.")
    v0[seed_indices] = 1 / len(seed_indices)
    s = v0.copy()
    for i in range(max_iter):
        s_new = (1 - restart_prob) * v0 + restart_prob * P.T.dot(s)
        if np.linalg.norm(s_new - s, 1) < tol:
            break
        s = s_new
    # Normalize:
    s = s / s.sum() if s.sum() != 0 else s
    return s

# Example usage:
# P = compute_transition_matrix(H)
# Here, choose seed indices based on a sample’s mutated genes.
# For instance, if you want to start from all genes mutated in a sample:
# seed_indices = [gene_list.index(gene) for gene in mutated_genes if gene in gene_list]
# s = hypergraph_random_walk(P, seed_indices, restart_prob=0.15)
def aggregate_module_scores(s, hyperedges, gene_list):
    """
    Aggregate gene importance scores over hyperedges to get module scores.
    
    Args:
        s: gene importance score vector (numpy array of length = number of genes)
        hyperedges: list of hyperedges, each is a list of gene names (module)
        gene_list: list of gene names corresponding to the indices in s
        
    Returns:
        module_scores: a dictionary mapping a hyperedge (module) index to its aggregated score
    """
    module_scores = {}
    for idx, edge in enumerate(hyperedges):
        # Get indices for the genes in this hyperedge
        indices = [gene_list.index(g) for g in edge if g in gene_list]
        if indices:
            # Example: average the gene scores
            module_score = np.mean(s[indices])
            module_scores[idx] = module_score
    return module_scores

# Example usage:
# module_scores = aggregate_module_scores(s, hyperedges, gene_list)
# Then, you can rank modules by module_scores.
def prioritize_modules(mutation_filepath, threshold=5, restart_prob=0.15, sample_mutated_genes=None):
    """
    End-to-end pipeline:
      - Load mutation data
      - Construct hypergraph (incidence matrix H)
      - Compute transition matrix P
      - Run hypergraph random walk with restart using seed genes from a sample
      - Aggregate gene scores to module (hyperedge) scores
    
    Args:
        mutation_filepath: path to the mutation file
        threshold: co-mutation count threshold for hyperedge construction
        restart_prob: random walk restart probability
        sample_mutated_genes: list of genes mutated in the sample (if None, all genes are used as seeds)
        
    Returns:
        gene_scores: dictionary mapping gene names to importance scores
        module_scores: dictionary mapping hyperedge (module) indices to aggregated scores
        hyperedges: list of hyperedges (each hyperedge is a list of gene names)
    """
    # Step 1: Load mutation data
    mut_df = load_mutation_data(mutation_filepath)
    
    # Step 2: Construct hypergraph incidence matrix H and hyperedges
    H, hyperedges, gene_list = construct_hypergraph(mut_df, threshold=threshold)
    
    # Step 3: Compute transition matrix P
    P = compute_transition_matrix(H)
    
    # Step 4: Define seed genes. If not provided, use all genes with at least one mutation in the sample.
    if sample_mutated_genes is None:
        sample_mutated_genes = mut_df.index[mut_df.sum(axis=1) > 0].tolist()
    seed_indices = [gene_list.index(gene) for gene in sample_mutated_genes if gene in gene_list]
    
    # Step 5: Run hypergraph random walk to get gene-level importance scores
    s = hypergraph_random_walk(P, seed_indices, restart_prob=restart_prob)
    
    # Map the scores back to gene names:
    gene_scores = {gene_list[i]: s[i] for i in range(len(gene_list))}
    
    # Step 6: Aggregate gene scores over hyperedges to get module-level scores
    module_scores = aggregate_module_scores(s, hyperedges, gene_list)
    
    return gene_scores, module_scores, hyperedges


if __name__ == "__main__":
    # Set the file path to your mutation file.
    mutation_filepath = "./data/GBM/GBM_mc3_gene_level.txt"
    
    # For a particular sample, you could provide its mutated genes; here we use all genes that are mutated.
    gene_scores, module_scores, hyperedges = prioritize_modules(mutation_filepath, threshold=5, restart_prob=0.15)
    
    # Sort and display gene scores:
    sorted_gene_scores = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)
    print("Top gene scores:")
    for gene, score in sorted_gene_scores[:10]:
        print(f"{gene}: {score:.5f}")
    
    # Write all gene scores to a text file
    with open("gene_scores.txt", "w") as f:
        for gene, score in sorted_gene_scores:
            f.write(f"{gene}\t{score:.5f}\n")
    # Sort and display module (hyperedge) scores:
    sorted_module_scores = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)
    print("\nTop module (hyperedge) scores:")
    for module_idx, score in sorted_module_scores[:10]:
        print(f"Module {module_idx} (genes: {hyperedges[module_idx]}): {score:.5f}")
    # Write all module scores to a text file
    with open("hyperedge_scores.txt", "w") as f:
        for module_idx, score in sorted_module_scores:
            genes_in_module = ", ".join(hyperedges[module_idx])
            f.write(f"Module {module_idx} (genes: {genes_in_module}): {score:.5f}\n")