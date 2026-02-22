use sha2::{Sha256, Digest};
use rayon::prelude::*;

/// merkle_tree provides high-performance verification of model update hashes.
pub struct MerkleTree {
    pub root: String,
    pub leaves: Vec<String>,
}

impl MerkleTree {
    /// Constructs a new Merkle Tree from a list of model update hashes.
    pub fn new(leaves: Vec<String>) -> Self {
        if leaves.is_empty() {
            return MerkleTree {
                root: String::new(),
                leaves: Vec::new(),
            };
        }

        let mut current_level = leaves.clone();
        
        while current_level.len() > 1 {
            let mut next_level = Vec::with_capacity((current_level.len() + 1) / 2);
            
            // Process pairs of hashes in parallel using Rayon
            let chunks: Vec<Vec<String>> = current_level.chunks(2).map(|c| c.to_vec()).collect();
            
            next_level = chunks.par_iter()
                .map(|pair| {
                    let mut hasher = Sha256::new();
                    if pair.len() == 2 {
                        hasher.update(format!("{}{}", pair[0], pair[1]));
                    } else {
                        // For odd number of leaves, promote the last one or pair it with itself
                        hasher.update(format!("{}{}", pair[0], pair[0]));
                    }
                    format!("{:x}", hasher.finalize())
                })
                .collect();
            
            current_level = next_level;
        }

        MerkleTree {
            root: current_level[0].clone(),
            leaves,
        }
    }

    /// Verifies if a given hash is part of the tree.
    /// In a real system, we'd use Merkle Proofs here.
    pub fn verify(&self, hash: &str) -> bool {
        self.leaves.contains(&hash.to_string())
    }
}

impl MerkleTree {
    pub fn build(leaves: Vec<String>) -> String {
        let tree = Self::new(leaves);
        tree.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_root_single_leaf() {
        let leaves = vec!["hash1".to_string()];
        let tree = MerkleTree::new(leaves);
        assert!(!tree.root.is_empty());
    }

    #[test]
    fn test_merkle_root_multi_leaves() {
        let leaves = vec![
            "hash1".to_string(),
            "hash2".to_string(),
            "hash3".to_string(),
            "hash4".to_string(),
        ];
        let root = MerkleTree::build(leaves);
        assert!(!root.is_empty());
    }

    #[test]
    fn test_merkle_verification() {
        let leaves = vec!["hash1".to_string(), "hash2".to_string()];
        let tree = MerkleTree::new(leaves);
        assert!(tree.verify("hash1"));
        assert!(!tree.verify("hash3"));
    }
}
