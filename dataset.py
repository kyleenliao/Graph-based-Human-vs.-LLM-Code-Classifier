from collections import defaultdict
import json
import os
import ast
from ast2json import ast2json

# "code" is the human written code
# "contrast" is the model generated code
# "label" 

class AstGraphGenerator(object):

    def __init__(self, source):
        self.graph = defaultdict(lambda: [])
        self.source = source  # lines of the source code

    def __str__(self):
        return str(self.graph)

    def _getid(self, node):
        try:
            lineno = node.lineno - 1
            return "%s: %s" % (type(node), self.source[lineno].strip())

        except AttributeError:
            return type(node)

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)

            elif isinstance(value, ast.AST):
                node_source = self._getid(node)
                value_source = self._getid(value)
                self.graph[node_source].append(value_source)
                # self.graph[type(node)].append(type(value))
                self.visit(value)

import ast
import networkx as nx

# class ASTGraphBuilder(ast.NodeVisitor):
#     def __init__(self):
#         self.graph = nx.DiGraph()
#         self.node_id = 0
#         self.parent_stack = []

#     def _add_node(self, node):
#         node_label = type(node).__name__
#         node_attrs = {}

#         # Include detailed attributes (identifiers, values, operators, etc.)
#         for field, value in ast.iter_fields(node):
#             if isinstance(value, (str, int, float, bool)):
#                 node_attrs[field] = value
#             elif value is None:
#                 node_attrs[field] = None

#         node_name = f"{node_label}_{self.node_id}"
#         self.graph.add_node(node_name, label=node_label, **node_attrs)
#         self.node_id += 1
#         return node_name

#     def generic_visit(self, node):
#         current = self._add_node(node)

#         if self.parent_stack:
#             parent = self.parent_stack[-1]
#             self.graph.add_edge(parent, current, type="child")

#         self.parent_stack.append(current)
#         super().generic_visit(node)
#         self.parent_stack.pop()

# def ast_to_graph(code):
#     tree = ast.parse(code)
#     builder = ASTGraphBuilder()
#     builder.visit(tree)
#     return builder.graph

class ASTGraphBuilder(ast.NodeVisitor):
    """
    Builds a semantically meaningful AST graph suitable for GNNs (e.g., GIN).
    Simplifies the Python AST by:
    - Removing low-value nodes like Load/Store/Del
    - Collapsing Attribute chains (e.g., math.cos)
    - Merging operator nodes into their symbol names (+, -, *, /)
    - Keeping identifiers and function calls
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_id = 0
        self.parent_stack = []

    # --- Utility functions ---
    def _get_op_symbol(self, op):
        """Return a string symbol for an operator node."""
        ops = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.Mod: "%", ast.Pow: "**", ast.BitAnd: "&", ast.BitOr: "|",
            ast.BitXor: "^", ast.FloorDiv: "//", ast.And: "and", ast.Or: "or",
            ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">=", ast.Not: "not", ast.USub: "-"
        }
        return ops.get(type(op), type(op).__name__)

    def _collapse_attribute(self, node):
        """Convert nested Attribute nodes (like math.cos) into a dotted name."""
        if isinstance(node, ast.Attribute):
            return f"{self._collapse_attribute(node.value)}.{node.attr}"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            # Handle attribute in calls: e.g. math.sin(x)
            return self._collapse_attribute(node.func)
        return type(node).__name__

    # --- Core graph construction ---
    def _add_node(self, label, **attrs):
        name = f"{label}_{self.node_id}"
        self.graph.add_node(name, label=label, **attrs)
        self.node_id += 1
        return name

    def generic_visit(self, node):
        # Skip unhelpful node types
        if isinstance(node, (ast.Load, ast.Store, ast.Del, ast.Pass)):
            return

        label = type(node).__name__

        # --- Simplify node types for semantic clarity ---
        if isinstance(node, ast.BinOp):
            label = self._get_op_symbol(node.op)
        elif isinstance(node, ast.BoolOp):
            label = self._get_op_symbol(node.op)
        elif isinstance(node, ast.UnaryOp):
            label = self._get_op_symbol(node.op)
        elif isinstance(node, ast.Compare):
            label = self._get_op_symbol(node.ops[0])
        elif isinstance(node, ast.Attribute):
            label = self._collapse_attribute(node)
        elif isinstance(node, ast.Name):
            label = node.id
        elif isinstance(node, ast.Constant):
            # Represent constants with their type, not raw value
            label = str(type(node.value).__name__)
        elif isinstance(node, ast.Call):
            label = f"Call:{self._collapse_attribute(node.func)}"

        # --- Add node ---
        current = self._add_node(label)

        # --- Connect to parent ---
        if self.parent_stack:
            parent = self.parent_stack[-1]
            self.graph.add_edge(parent, current, type="child")

        self.parent_stack.append(current)
        super().generic_visit(node)
        self.parent_stack.pop()


def ast_to_graph(code):
    tree = ast.parse(code)
    builder = ASTGraphBuilder()
    builder.visit(tree)
    return builder.graph


with open("dataset/python/train.jsonl", "r") as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]
print(f"Loaded {len(data)} training examples.")

ind = 45730 #0 
print(data[ind].keys())
print(data[ind]["code"])
print(ast.dump(ast.parse(data[ind]["code"], mode='exec'), indent=4))
walk = ast.walk(ast.parse(data[ind]["code"], mode='exec'))
wl = list(walk)
#print(wl)

G = ast_to_graph(data[ind]["code"])
print("AST Graph Nodes and Edges:")
print(G.nodes(data=True))
print(G.edges(data=True))

import matplotlib.pyplot as plt

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, labels={n: G.nodes[n]['label'] for n in G.nodes})
plt.show()

# print("AST Walk Nodes:")
# tree = ast.parse(data[0]["code"], mode='exec')

# source_lines = data[0]["code"].splitlines()
# generator = AstGraphGenerator(source_lines)

# generator = AstGraphGenerator(source_lines)
# generator.visit(tree) 

# for parent, children in generator.graph.items():
#     print(f"\n{parent}:")
#     for child in children:
#         print(f"  -> {child}")

# import networkx as nx
# import matplotlib.pyplot as plt

# G = nx.DiGraph()
# for parent, children in generator.graph.items():
#     for child in children:
#         G.add_edge(parent, child)

# plt.figure(figsize=(12, 8))
# nx.draw(G, with_labels=True, node_size=3000, font_size=8)
# plt.show()

# for i in range(10):
#     print("--- Example", i, "---")
#     print(data[i]["code"])
#     print(data[i]["contrast"])
#     print(data[i]["label"])


# results = []
# for i, item in enumerate(data):
#     # Check all values in each JSON object
#     for key, value in item.items():
#         if isinstance(value, str) and "return (newx, newy)" in value:
#             results.append((i, key, value))

# # Print results
# for idx, key, val in results:
#     print(f"Found in entry {idx}, key '{key}':")
#     print(val)
#     print(data[idx]["code"])
#     print("-" * 50)
    
# print("-----")

# results = []
# for i, item in enumerate(data):
#     # Check all values in each JSON object
#     for key, value in item.items():
#         if isinstance(value, str) and "return (rotated_x, rotated_y)" in value:
#             results.append((i, key, value))

# # Print results
# for idx, key, val in results:
#     print(f"Found in entry {idx}, key '{key}':")
#     print(val)
#     #print(data[idx]["code"])
#     print("-" * 50)

