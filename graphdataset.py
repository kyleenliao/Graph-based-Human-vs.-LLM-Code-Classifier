import json
import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

from process import PythonCodeProcessor


class CodeGraphDataset(Dataset):
    """Dataset class for processing Python code pairs into graph representations.
    
    Optimized version with:
    - Parallel processing
    - Sample limiting
    - Better caching
    - Timeout support
    """
    
    def __init__(self, 
                 jsonl_path, 
                 processor=None,
                 cache_dir=None,
                 force_reprocess=False,
                 max_nodes=1000,
                 embedding_size=128,
                 max_samples=None,  # NEW: Limit number of samples
                 num_workers=None,  # NEW: Parallel processing workers
                 timeout_minutes=10):  # NEW: Processing timeout
        """
        Args:
            jsonl_path: Path to JSONL file with format: {index, code, contrast, label}
            processor: PythonCodeProcessor instance (creates new one if None)
            cache_dir: Directory to cache processed graphs (default: same dir as jsonl)
            force_reprocess: If True, reprocess even if cache exists
            max_nodes: Maximum nodes in adjacency matrix
            embedding_size: Dimension of token embeddings
            max_samples: Maximum number of samples to process (None = all)
            num_workers: Number of parallel workers (None = auto)
            timeout_minutes: Stop processing after this many minutes
        """
        self.jsonl_path = jsonl_path
        self.max_nodes = max_nodes
        self.embedding_size = embedding_size
        self.max_samples = max_samples
        self.timeout_minutes = timeout_minutes
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(jsonl_path), 'cached_graphs')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache file path with sample limit in name
        cache_basename = os.path.basename(jsonl_path).replace('.jsonl', '')
        if max_samples is not None:
            cache_basename += f"_n{max_samples}"
        cache_file = os.path.join(cache_dir, f"{cache_basename}_graphs.pkl")
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
        
        # Set number of workers
        if num_workers is None:
            self.num_workers = max(1, cpu_count() - 1)
        else:
            self.num_workers = num_workers
        
        # Load or process data
        if os.path.exists(cache_file) and not force_reprocess:
            print(f"Loading cached graphs from {cache_file}")
            self._load_from_cache()
        else:
            print(f"Processing graphs from {jsonl_path}")
            if max_samples:
                print(f"  Limiting to {max_samples} samples")
            if timeout_minutes:
                print(f"  Timeout: {timeout_minutes} minutes")
            self._process_and_cache()
    
    def _load_raw_data(self):
        """Load raw JSONL data with optional sample limit."""
        with open(self.jsonl_path, 'r') as f:
            lines = f.readlines()
        
        # Limit samples if specified
        if self.max_samples is not None:
            lines = lines[:self.max_samples]
        
        data = [json.loads(line) for line in lines]
        print(f"Loaded {len(data)} examples from {self.jsonl_path}")
        return data
    
    def _extract_token_sequence(self, item):
        """Extract token sequence from a single item (for parallel processing)."""
        try:
            code_seq = self.processor.code_to_sequence(str(item['code']))
            contrast_seq = self.processor.code_to_sequence(str(item['contrast']))
            return code_seq, contrast_seq
        except Exception as e:
            return [], []
    
    def _process_and_cache(self):
        """Process all code samples and cache results."""
        raw_data = self._load_raw_data()
        start_time = time.time()
        
        # Train embedding if needed
        if self.needs_embedding:
            print("Extracting AST node sequences from corpus...")
            token_sequences = []
            
            # Process with timeout check
            elapsed_time = 0
            batch_size = 100  # Process in batches for timeout checking
            
            for i in tqdm(range(0, len(raw_data), batch_size), 
                         desc="Extracting tokens", unit="batch"):
                # Check timeout
                elapsed_time = (time.time() - start_time) / 60
                if self.timeout_minutes and elapsed_time > self.timeout_minutes:
                    print(f"\n⏱️ Timeout reached ({elapsed_time:.1f} min). Using {i} samples for embedding.")
                    raw_data = raw_data[:i]
                    break
                
                batch = raw_data[i:i+batch_size]
                
                # Process batch
                for item in batch:
                    try:
                        code_seq = self.processor.code_to_sequence(str(item['code']))
                        if code_seq:
                            token_sequences.append(code_seq)
                        
                        contrast_seq = self.processor.code_to_sequence(str(item['contrast']))
                        if contrast_seq:
                            token_sequences.append(contrast_seq)
                    except Exception as e:
                        continue
            
            print(f"Total token sequences for embedding: {len(token_sequences)}")
            
            # Train Word2Vec on the token sequences
            if len(token_sequences) > 0:
                print("Training Word2Vec embeddings on AST tokens...")
                self.processor.train_embedding_from_sequences(token_sequences)
                print(f"Vocabulary size: {self.processor.max_token}")
            else:
                print("⚠️ No valid token sequences extracted!")
                return
        
        # Process each example into graphs
        self.data = []
        print("Processing code into graphs...")
        
        failed_count = 0
        start_time = time.time()  # Reset timer for graph processing
        
        for idx, item in enumerate(tqdm(raw_data, desc="Processing samples", unit="sample")):
            # Check timeout
            elapsed_time = (time.time() - start_time) / 60
            if self.timeout_minutes and elapsed_time > self.timeout_minutes:
                print(f"\n⏱️ Timeout reached ({elapsed_time:.1f} min). Processed {idx} samples.")
                break
            
            try:
                # Process original code
                code_results = self.processor.process_pipeline(str(item['code']), return_all=True)

                # Process contrast code
                contrast_results = self.processor.process_pipeline(str(item['contrast']), return_all=True)
                
                # Skip if either failed to process
                if code_results is None or contrast_results is None:
                    # if code_results is None:
                    #     print(f"Item {idx}, code_results failed")
                    #     self.processor.process_pipeline(str(item['code']), return_all=True, logging=True)
                    
                    # if contrast_results is None:
                    #     print(f"Item {idx}, contrast_results failed")
                    #     self.processor.process_pipeline(str(item['contrast']), return_all=True, logging=True)

                    failed_count += 1
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
                failed_count += 1
                continue
        
        print(f"Successfully processed {len(self.data)} examples")
        if failed_count > 0:
            print(f"Failed to process {failed_count} examples")
        
        # Save to cache
        if len(self.data) > 0:
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
        else:
            print("⚠️ No data to cache!")
    
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
    
    def _sequence_to_indices(self, sequence):
        """Convert token strings to vocabulary indices."""
        if self.processor.vocab is None:
            return [0] * len(sequence)
        
        indices = []
        for token in sequence:
            if isinstance(token, str):
                # Token is a string - convert to index
                idx = self.processor.vocab.key_to_index.get(token, self.processor.max_token)
                indices.append(idx)
            else:
                # Already an index
                indices.append(int(token))
        
        return indices
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with:
                - code_graph: adjacency matrix (max_nodes, max_nodes)
                - contrast_graph: adjacency matrix (max_nodes, max_nodes)
                - code_sequence: list of vocabulary indices (integers)
                - contrast_sequence: list of vocabulary indices (integers)
                - label: classification label
                - index: original index
        """
        item = self.data[idx]
        
        # Convert token sequences to indices
        code_seq_indices = self._sequence_to_indices(item['code_sequence'])
        contrast_seq_indices = self._sequence_to_indices(item['contrast_sequence'])
        
        # Convert to tensors
        return {
            'index': item['index'],
            'label': torch.tensor(item['label'], dtype=torch.long),
            
            # Code graph data
            'code_graph': torch.from_numpy(item['code_graph']).float(),
            'code_num_nodes': item['code_num_nodes'],
            'code_sequence': code_seq_indices,  # Now integer indices
            
            # Contrast graph data
            'contrast_graph': torch.from_numpy(item['contrast_graph']).float(),
            'contrast_num_nodes': item['contrast_num_nodes'],
            'contrast_sequence': contrast_seq_indices,  # Now integer indices
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
