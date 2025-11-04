import json
import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from process import PythonCodeProcessor


class CodeGraphDataset(Dataset):
    """Dataset class for processing Python code pairs into graph representations."""
    
    def __init__(self, 
                 jsonl_path, 
                 processor=None,
                 cache_dir=None,
                 force_reprocess=False,
                 max_nodes=1000,
                 embedding_size=128):
        """
        Args:
            jsonl_path: Path to JSONL file with format: {index, code, contrast, label}
            processor: PythonCodeProcessor instance (creates new one if None)
            cache_dir: Directory to cache processed graphs (default: same dir as jsonl)
            force_reprocess: If True, reprocess even if cache exists
            max_nodes: Maximum nodes in adjacency matrix
            embedding_size: Dimension of token embeddings
        """
        self.jsonl_path = jsonl_path
        self.max_nodes = max_nodes
        self.embedding_size = embedding_size
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(jsonl_path), 'cached_graphs')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache file path
        cache_file = os.path.join(
            cache_dir, 
            f"{os.path.basename(jsonl_path).replace('.jsonl', '')}_graphs.pkl"
        )
        self.cache_file = cache_file
        
        # Initialize or use provided processor
        if processor is None:
            self.processor = PythonCodeProcessor(
                embedding_size=embedding_size,
                max_nodes=max_nodes
            )
            self.needs_embedding = True
        else:
            self.processor = processor
            self.needs_embedding = False
        
        # Load or process data
        if os.path.exists(cache_file) and not force_reprocess:
            print(f"Loading cached graphs from {cache_file}")
            self._load_from_cache()
        else:
            print(f"Processing graphs from {jsonl_path}")
            self._process_and_cache()
    
    def _load_raw_data(self):
        """Load raw JSONL data."""
        with open(self.jsonl_path, 'r') as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        print(f"Loaded {len(data)} examples from {self.jsonl_path}")
        return data
    
    def _process_and_cache(self):
        """Process all code samples and cache results."""
        raw_data = self._load_raw_data()
        
        # Train embedding if needed
        if self.needs_embedding:
            print("Extracting AST node sequences from corpus...")
            token_sequences = []
            for item in tqdm(raw_data, desc="Extracting tokens", unit="sample"):
                # Extract token sequences from code
                code_seq = self.processor.code_to_sequence(str(item['code']))
                if code_seq:
                    token_sequences.append(code_seq)
                
                # Extract token sequences from contrast code
                contrast_seq = self.processor.code_to_sequence(str(item['contrast']))
                if contrast_seq:
                    token_sequences.append(contrast_seq)
            
            print(f"Total token sequences for embedding: {len(token_sequences)}")
            
            # Train Word2Vec on the token sequences
            print("Training Word2Vec embeddings on AST tokens...")
            self.processor.train_embedding_from_sequences(token_sequences)
            print(f"Vocabulary size: {self.processor.max_token}")
        
        # Process each example
        self.data = []
        print("Processing code into graphs...")
        
        for item in tqdm(raw_data, desc="Processing samples", unit="sample"):
            try:
                # Process original code
                code_results = self.processor.process_pipeline(str(item['code']), return_all=True)
                
                # Process contrast code
                contrast_results = self.processor.process_pipeline(str(item['contrast']), return_all=True)
                
                # Skip if either failed to process
                if code_results is None or contrast_results is None:
                    print(f"Skipping index {item['index']} due to parsing error")
                    continue
                
                # Create data entry
                entry = {
                    'index': item['index'],
                    'label': item['label'],
                    
                    # Code graphs
                    'code_graph': code_results['graph'],
                    'code_num_nodes': code_results['num_nodes'],
                    'code_sequence': code_results['sequence'],
                    'code_indexed_blocks': code_results.get('indexed_blocks', None),
                    
                    # Contrast graphs
                    'contrast_graph': contrast_results['graph'],
                    'contrast_num_nodes': contrast_results['num_nodes'],
                    'contrast_sequence': contrast_results['sequence'],
                    'contrast_indexed_blocks': contrast_results.get('indexed_blocks', None),
                }
                
                self.data.append(entry)
                
            except Exception as e:
                print(f"Error processing index {item['index']}: {e}")
                continue
        
        print(f"Successfully processed {len(self.data)} examples")
        
        # Save to cache
        print(f"Saving to cache: {self.cache_file}")
        cache_data = {
            'data': self.data,
            'vocab': self.processor.vocab,
            'max_token': self.processor.max_token,
            'embedding_size': self.embedding_size,
            'max_nodes': self.max_nodes
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Cache saved successfully")
    
    def _load_from_cache(self):
        """Load processed data from cache."""
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.data = cache_data['data']
        self.processor.vocab = cache_data['vocab']
        self.processor.max_token = cache_data['max_token']
        print(f"Loaded {len(self.data)} cached examples")
        print(f"Vocabulary size: {self.processor.max_token}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                - code_graph: adjacency matrix (max_nodes, max_nodes)
                - contrast_graph: adjacency matrix (max_nodes, max_nodes)
                - code_features: node feature indices or embeddings
                - contrast_features: node feature indices or embeddings
                - label: classification label
                - index: original index
        """
        item = self.data[idx]
        
        # Convert to tensors
        return {
            'index': item['index'],
            'label': torch.tensor(item['label'], dtype=torch.long),
            
            # Code graph data
            'code_graph': torch.from_numpy(item['code_graph']).float(),
            'code_num_nodes': item['code_num_nodes'],
            'code_sequence': item['code_sequence'],
            
            # Contrast graph data
            'contrast_graph': torch.from_numpy(item['contrast_graph']).float(),
            'contrast_num_nodes': item['contrast_num_nodes'],
            'contrast_sequence': item['contrast_sequence'],
        }
    
    def get_embedding_matrix(self):
        """Get the Word2Vec embedding matrix for use in GNN."""
        if self.processor.vocab is None:
            return None
        
        # Create embedding matrix: (vocab_size + 1, embedding_dim)
        # +1 for unknown tokens
        vocab_size = len(self.processor.vocab)
        embedding_dim = self.processor.embedding_size
        
        embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
        
        for word, idx in self.processor.vocab.key_to_index.items():
            embedding_matrix[idx] = self.processor.vocab[word]
        
        # Unknown token gets zero vector (already initialized)
        return torch.from_numpy(embedding_matrix).float()
    
    def get_stats(self):
        """Get dataset statistics."""
        if len(self.data) == 0:
            return {}
        
        code_nodes = [item['code_num_nodes'] for item in self.data]
        contrast_nodes = [item['contrast_num_nodes'] for item in self.data]
        labels = [item['label'] for item in self.data]
        
        stats = {
            'total_samples': len(self.data),
            'avg_code_nodes': np.mean(code_nodes),
            'max_code_nodes': np.max(code_nodes),
            'min_code_nodes': np.min(code_nodes),
            'avg_contrast_nodes': np.mean(contrast_nodes),
            'max_contrast_nodes': np.max(contrast_nodes),
            'min_contrast_nodes': np.min(contrast_nodes),
            'label_distribution': {
                'label_0': labels.count(0),
                'label_1': labels.count(1) if 1 in labels else 0
            }
        }
        return stats


# Example usage
if __name__ == '__main__':
    # Load training data
    train_dataset = CodeGraphDataset(
        jsonl_path='dataset/python/train.jsonl',
        max_nodes=1000,
        embedding_size=128,
        force_reprocess=False  # Set to True to reprocess
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    stats = train_dataset.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Get a sample
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    sample = train_dataset[0]
    print(f"Index: {sample['index']}")
    print(f"Label: {sample['label']}")
    print(f"Code graph shape: {sample['code_graph'].shape}")
    print(f"Code num nodes: {sample['code_num_nodes']}")
    print(f"Code sequence (first 10): {sample['code_sequence'][:10]}")
    print(f"Contrast graph shape: {sample['contrast_graph'].shape}")
    print(f"Contrast num nodes: {sample['contrast_num_nodes']}")
    
    # Get embedding matrix
    embedding_matrix = train_dataset.get_embedding_matrix()
    print(f"\nEmbedding matrix shape: {embedding_matrix.shape}")
    
    # Load dev/test data using same processor (shares vocabulary)
    print("\n" + "="*60)
    print("Loading dev dataset with shared vocabulary...")
    print("="*60)
    dev_dataset = CodeGraphDataset(
        jsonl_path='dataset/python/dev.jsonl',
        processor=train_dataset.processor,  # Share processor and vocabulary
        max_nodes=1000
    )
    print(f"Dev dataset size: {len(dev_dataset)}")
    
    # Example: Create DataLoader for GNN training
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid pickling issues
    )
    
    print("\n" + "="*60)
    print("Testing DataLoader...")
    print("="*60)
    for batch in train_loader:
        print(f"Batch size: {len(batch['label'])}")
        print(f"Code graphs shape: {batch['code_graph'].shape}")
        print(f"Contrast graphs shape: {batch['contrast_graph'].shape}")
        print(f"Labels: {batch['label']}")
        break  # Just test first batch