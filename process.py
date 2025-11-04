import ast
import json
from gensim.models.word2vec import Word2Vec
import numpy as np
import networkx as nx
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
import textwrap

from ast_tree_python import ASTNode, BlockNode, SingleNode

class PythonCodeProcessor:
    """Process Python code strings into various AST representations."""
    
    def __init__(self, embedding_size=128, max_nodes=1000):
        self.embedding_size = embedding_size
        self.w2v_model = None
        self.vocab = None
        self.max_token = None
        self.max_nodes = max_nodes
        
    def convert_code_snippet(self, code_str):
        try:
            fixers = get_fixers_from_package("lib2to3.fixes")
            tool = RefactoringTool(fixers)
            cleaned = textwrap.dedent(code_str).replace("\r", "").rstrip() + "\n"
            refactored = tool.refactor_string(cleaned, name="snippet")
            return str(refactored)
        except Exception as e:
            print("Conversion failed for snippet:", e)
            #print(code_str)
            return code_str
        
    def parse_code(self, code_string):
        """Parse a Python code string into an AST tree."""
        try:
            code_string = self.convert_code_snippet(code_string)
            tree = ast.parse(code_string)
            return tree
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
            return None
    
    def get_ast_node(self, code_string):
        """Convert code string to ASTNode representation."""
        tree = self.parse_code(code_string)
        if tree is None:
            return None
        return ASTNode(tree)
    
    def get_block_node(self, code_string):
        """Convert code string to BlockNode representation."""
        tree = self.parse_code(code_string)
        if tree is None:
            return None
        return BlockNode(tree)
    
    def get_sequence(self, node, sequence=None):
        """Extract token sequence from AST node (depth-first traversal)."""
        if sequence is None:
            sequence = []
        
        sequence.append(node.token)
        for child in node.children:
            self.get_sequence(child, sequence)
        
        return sequence
    
    def get_blocks(self, node, blocks=None):
        """Extract block-level structures from AST."""
        if blocks is None:
            blocks = []
        
        blocks.append(node)
        for child in node.children:
            self.get_blocks(child, blocks)
        
        return blocks
    
    def code_to_sequence(self, code_string):
        """Convert code string directly to token sequence."""
        ast_node = self.get_ast_node(code_string)
        if ast_node is None:
            return []
        return self.get_sequence(ast_node)
    
    def code_to_blocks(self, code_string):
        """Convert code string to block sequences."""
        block_node = self.get_block_node(code_string)
        if block_node is None:
            return []
        return self.get_blocks(block_node)
    
    def build_connection_list(self, node, connections=None, node_id=0, parent_id=None):
        """Build parent-child connection list from AST.
        
        Returns:
            connections: List where connections[i] contains the parent node ID of node i
            node_count: Total number of nodes
        """
        if connections is None:
            connections = []
        
        # Add connection from current node to parent
        if parent_id is not None:
            connections.append(parent_id)
        else:
            connections.append(node_id)  # Root connects to itself
        
        current_id = node_id
        next_id = node_id + 1
        
        # Process children
        for child in node.children:
            connections, next_id = self.build_connection_list(
                child, connections, next_id, current_id
            )
        
        return connections, next_id
    
    def create_adjacency_matrix(self, connections, max_size=None):
        """Create adjacency matrix from connection list.
        
        Args:
            connections: List where connections[i] is the parent of node i
            max_size: Maximum size of matrix (default: self.max_nodes)
            
        Returns:
            Adjacency matrix as boolean numpy array
        """
        if max_size is None:
            max_size = self.max_nodes
        
        num_nodes = len(connections)
        adj_matrix = np.zeros((max_size, max_size), dtype='bool_')
        
        for i in range(num_nodes):
            # Self-loop
            adj_matrix[i][i] = True
            # Bidirectional edge with parent
            parent = connections[i]
            adj_matrix[i][parent] = True
            adj_matrix[parent][i] = True
        
        return adj_matrix
    
    def code_to_graph(self, code_string):
        """Convert code to graph adjacency matrix.
        
        Returns:
            adjacency_matrix: Boolean numpy array of shape (max_nodes, max_nodes)
            num_nodes: Actual number of nodes in the graph
        """
        ast_node = self.get_ast_node(code_string)
        if ast_node is None:
            return None, 0
        
        # Build connection list
        connections, num_nodes = self.build_connection_list(ast_node)
        
        # Create adjacency matrix
        adj_matrix = self.create_adjacency_matrix(connections)
        
        return adj_matrix, num_nodes
    
    def train_embedding_from_sequences(self, token_sequences, size=None):
        """Train Word2Vec embedding on pre-extracted token sequences.
        
        Args:
            token_sequences: List of token sequences (already extracted from AST)
            size: Embedding dimension (default: self.embedding_size)
        """
        if size is None:
            size = self.embedding_size
        
        if not token_sequences:
            print("No valid token sequences provided")
            return None
        
        # Train Word2Vec directly on token sequences
        self.w2v_model = Word2Vec(
            sentences=token_sequences,
            vector_size=size,
            sg=1,  # Skip-gram
            min_count=1,  # Keep rare tokens for code
            workers=4
        )
        
        self.vocab = self.w2v_model.wv
        self.max_token = len(self.vocab)
        
        return self.w2v_model
    
    def train_embedding(self, code_samples, size=None):
        """Train Word2Vec embedding on multiple code samples.
        
        Args:
            code_samples: List of code strings
            size: Embedding dimension (default: self.embedding_size)
        """
        if size is None:
            size = self.embedding_size
        
        # Convert all code samples to sequences
        corpus = []
        for code in code_samples:
            seq = self.code_to_sequence(code)
            if seq:
                corpus.append(seq)
        
        if not corpus:
            print("No valid sequences generated from code samples")
            return None
        
        # Use the new method
        return self.train_embedding_from_sequences(corpus, size)
    
    def tree_to_index(self, node):
        """Convert AST node tree to index representation using trained embeddings."""
        if self.vocab is None:
            raise ValueError("Need to train embedding first using train_embedding()")
        
        token = node.token
        # Use max_token as unknown token index
        result = [self.vocab.key_to_index.get(token, self.max_token)]
        
        for child in node.children:
            result.append(self.tree_to_index(child))
        
        return result
    
    def code_to_indexed_blocks(self, code_string):
        """Convert code to indexed block sequences (ready for neural network input)."""
        if self.vocab is None:
            raise ValueError("Need to train embedding first using train_embedding()")
        
        blocks = self.code_to_blocks(code_string)
        indexed_blocks = []
        
        for block in blocks:
            indexed_block = self.tree_to_index(block)
            indexed_blocks.append(indexed_block)
        
        return indexed_blocks
    
    def build_node_list(self, node, nodes=None, node_id=0, parent_id=None):
        """Build list of nodes with their tokens for visualization.
        
        Returns:
            nodes: List of (node_id, token, parent_id) tuples
            next_id: Next available node ID
        """
        if nodes is None:
            nodes = []
        
        # Add current node
        nodes.append((node_id, node.token, parent_id))
        
        current_id = node_id
        next_id = node_id + 1
        
        # Process children
        for child in node.children:
            nodes, next_id = self.build_node_list(child, nodes, next_id, current_id)
        
        return nodes, next_id
    
    def visualize_graph(self, code_string, output_file='ast_graph.png', 
                       max_nodes_display=50, layout='tree'):
        """Visualize the AST graph structure.
        
        Args:
            code_string: Python code to visualize
            output_file: Output image file path
            max_nodes_display: Maximum nodes to display (for readability)
            layout: Layout algorithm ('tree', 'spring', 'circular', 'kamada_kawai')
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("Please install: pip install matplotlib networkx")
            return
        
        # Get AST and build node list
        ast_node = self.get_ast_node(code_string)
        if ast_node is None:
            return
        
        nodes, num_nodes = self.build_node_list(ast_node)
        
        # Limit display for readability
        if num_nodes > max_nodes_display:
            print(f"Graph has {num_nodes} nodes, displaying first {max_nodes_display}")
            nodes = nodes[:max_nodes_display]
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with labels
        for node_id, token, parent_id in nodes:
            G.add_node(node_id, label=token)
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Choose layout
        if layout == 'tree':
            # Hierarchical tree layout
            pos = self._hierarchy_pos(G, root=0)
        elif layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Get node labels
        labels = nx.get_node_attributes(G, 'label')
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=800, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=15, 
                              arrowstyle='->', width=2, alpha=0.6)
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        plt.title(f"AST Graph Visualization ({len(nodes)} nodes)", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Graph visualization saved to {output_file}")
    
    def _hierarchy_pos(self, G, root=0, width=1., vert_gap=0.2, vert_loc=0, 
                      xcenter=0.5, pos=None, parent=None, parsed=[]):
        """Create hierarchical tree layout (recursive helper)."""
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = self._hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                         vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                         pos=pos, parent=root, parsed=parsed)
        return pos
    
    def visualize_adjacency_matrix(self, code_string, output_file='adjacency_matrix.png'):
        """Visualize the adjacency matrix as a heatmap.
        
        Args:
            code_string: Python code to visualize
            output_file: Output image file path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Please install: pip install matplotlib")
            return
        
        # Get adjacency matrix
        adj_matrix, num_nodes = self.code_to_graph(code_string)
        
        if num_nodes == 0:
            print("No nodes to visualize")
            return
        
        # Only show the relevant portion
        adj_matrix_cropped = adj_matrix[:num_nodes, :num_nodes]
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(adj_matrix_cropped, cmap='Blues', interpolation='nearest')
        plt.colorbar(label='Connection')
        plt.title(f'Adjacency Matrix ({num_nodes}x{num_nodes} nodes)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Node Index')
        plt.ylabel('Node Index')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Adjacency matrix saved to {output_file}")
    
    def print_graph_info(self, code_string):
        """Print detailed information about the graph structure."""
        ast_node = self.get_ast_node(code_string)
        if ast_node is None:
            return
        
        nodes, num_nodes = self.build_node_list(ast_node)
        connections, _ = self.build_connection_list(ast_node)
        
        print("=" * 60)
        print("GRAPH STRUCTURE INFO")
        print("=" * 60)
        print(f"Total nodes: {num_nodes}")
        print(f"\nNode list (ID, Token, Parent ID):")
        print("-" * 60)
        
        for i, (node_id, token, parent_id) in enumerate(nodes[:20]):  # Show first 20
            parent_str = str(parent_id) if parent_id is not None else "None (root)"
            print(f"  {node_id:3d}: {token:20s} <- parent: {parent_str}")
        
        if len(nodes) > 20:
            print(f"  ... ({len(nodes) - 20} more nodes)")
        
        # Calculate graph statistics
        adj_matrix, _ = self.code_to_graph(code_string)
        adj_cropped = adj_matrix[:num_nodes, :num_nodes]
        
        num_edges = (np.sum(adj_cropped) - num_nodes) // 2  # Subtract self-loops, divide by 2 for undirected
        
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {num_edges} (excluding self-loops)")
        print(f"  Self-loops: {num_nodes}")
        print(f"  Density: {num_edges / (num_nodes * (num_nodes - 1) / 2):.4f}" if num_nodes > 1 else "  Density: N/A")
        print("=" * 60)
    
    
    def process_pipeline(self, code_string, return_all=False):
        """Complete processing pipeline for a single code string.
        
        Args:
            code_string: Python code as string
            return_all: If True, returns dict with all representations
        
        Returns:
            If return_all=True: dict with 'ast', 'sequence', 'blocks', 'indexed_blocks'
            If return_all=False: just the indexed blocks (for training)
        """
        results = {}
        
        # Parse to AST
        ast_node = self.get_ast_node(code_string)
        if ast_node is None:
            return None if not return_all else {}
        
        results['ast'] = ast_node
        
        # Get token sequence
        sequence = self.get_sequence(ast_node)
        results['sequence'] = sequence
        
        # Get blocks
        block_node = self.get_block_node(code_string)
        blocks = self.get_blocks(block_node) if block_node else []
        results['blocks'] = blocks
        
        # Get graph representation
        graph, num_nodes = self.code_to_graph(code_string)
        results['graph'] = graph
        results['num_nodes'] = num_nodes
        
        # Get indexed blocks (if embedding is trained)
        if self.vocab is not None:
            indexed_blocks = self.code_to_indexed_blocks(code_string)
            results['indexed_blocks'] = indexed_blocks
        
        return results if return_all else results.get('indexed_blocks', blocks)


# Example usage
if __name__ == '__main__':
    with open("dataset/python/train.jsonl", "r") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    sample_code = str(data[15730]['code'])
    print(sample_code)
    # Initialize processor
    processor = PythonCodeProcessor(embedding_size=128)
    
    # Process single code snippet (without embedding)
    print("=" * 50)
    print("Processing without embedding:")
    print("=" * 50)
    results = processor.process_pipeline(sample_code, return_all=True)
    print(f"Token sequence: {results['sequence'][:10]}...")  # First 10 tokens
    print(f"Number of blocks: {len(results['blocks'])}")
    
    # Train embedding on multiple samples
    print("\n" + "=" * 50)
    print("Training embedding:")
    print("=" * 50)
    code_samples = [sample_code]
    processor.train_embedding(code_samples)
    print(f"Vocabulary size: {processor.max_token}")
    
    # Process with embedding
    print("\n" + "=" * 50)
    print("Processing with embedding:")
    print("=" * 50)
    results = processor.process_pipeline(sample_code, return_all=True)
    print(f"Token sequence: {results['sequence'][:10]}...")
    print(f"Indexed blocks (first block): {results['indexed_blocks'][0]}")
    
    # Get embedding vector for a token
    if 'rotate' in processor.vocab:
        vector = processor.vocab['rotate']
        print(f"\nEmbedding vector for 'rotate': shape {vector.shape}")
        
        
    processor.visualize_graph(sample_code, layout='tree')
    processor.visualize_adjacency_matrix(sample_code)
    processor.print_graph_info(sample_code)