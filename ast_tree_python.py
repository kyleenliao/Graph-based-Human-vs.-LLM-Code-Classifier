import ast
from collections import defaultdict

class ASTNode(object):
    def __init__(self, node, token=""):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = token
        self.children = []

class BlockNode(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token(node)
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        if isinstance(self.node, ast.AST):
            return len(list(ast.iter_child_nodes(self.node))) == 0
        return True

    def get_token(self, node):
        if isinstance(node, str):
            token = node
        elif isinstance(node, set):
            token = 'Modifier'
        elif isinstance(node, ast.AST):
            token = node.__class__.__name__
        else:
            token = ''
        return token

    def ori_children(self, root):
        if isinstance(root, ast.AST):
            # Python equivalent: skip body for method/class definitions
            if self.token in ['FunctionDef', 'AsyncFunctionDef', 'ClassDef']:
                # Get all children except body
                children = []
                for field, value in ast.iter_fields(root):
                    if field != 'body':
                        if isinstance(value, list):
                            children.extend(value)
                        elif isinstance(value, ast.AST):
                            children.append(value)
            else:
                children = list(ast.iter_child_nodes(root))
        elif isinstance(root, set):
            children = list(root)
        else:
            children = []

        def expand(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    for sub_item in expand(item):
                        yield sub_item
                elif item:
                    yield item

        return list(expand(children))

    def add_children(self):
        if self.is_str:
            return []
        
        # Python control flow statements
        logic = ['If', 'For', 'While', 'With', 'AsyncWith', 'AsyncFor', 'Try']
        children = self.ori_children(self.node)
        
        if self.token in logic:
            return [BlockNode(children[0])] if children else []
        elif self.token in ['FunctionDef', 'AsyncFunctionDef', 'ClassDef']:
            return [BlockNode(child) for child in children]
        else:
            return [BlockNode(child) for child in children if self.get_token(child) not in logic]


class SingleNode(ASTNode):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = []

    def is_leaf(self):
        if self.is_str:
            return True
        return len(list(ast.iter_child_nodes(self.node))) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        
        name = self.node.__class__.__name__
        token = name
        is_name = False
        
        if self.is_leaf():
            if hasattr(self.node, 'id'):
                token = self.node.id
                is_name = True
            elif hasattr(self.node, 'name'):
                token = self.node.name
                is_name = True
            elif hasattr(self.node, 'value'):
                token = str(self.node.value) if hasattr(self.node, 'value') else name
            elif hasattr(self.node, 'arg'):
                token = self.node.arg
                is_name = True
            else:
                token = name
        else:
            if hasattr(self.node, 'op'):
                token = self.node.op.__class__.__name__
            elif hasattr(self.node, 'ops'):
                token = self.node.ops[0].__class__.__name__
            else:
                token = name
        
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token


def defn_token(ast_node, lower=True):
    if isinstance(ast_node, str):
        return ast_node

    name = ast_node.__class__.__name__
    token = name
    is_name = False

    token = str(ast_node).split('_ast.')
    if len(token) >= 2:
        token = token[1]
        token = token.split(' ')[0]
    else:
        token = name

    # print("\n Token info: ", ast_node, token, name)

    # if hasattr(ast_node, 'id'):  # Name node
    #     token = ast_node.id
    #     is_name = True
    # elif hasattr(ast_node, 'name'):  # FunctionDef, ClassDef, etc.
    #     token = ast_node.name
    #     is_name = True
    # elif hasattr(ast_node, 'value'):  # Constant/Num/Str nodes
    #     token = str(ast_node.value)
    # elif hasattr(ast_node, 'arg'):  # arguments
    #     token = ast_node.arg
    #     is_name = True
    # elif hasattr(ast_node, 'op'):
    #     op_name = ast_node.op.__class__.__name__
    #     token = op_name
    # elif hasattr(ast_node, 'ops'):  # Compare node
    #     token = ast_node.ops[0].__class__.__name__
    # else:
        
    #     else:
    #         token = name

    if token is None:
        token = name

    # print("Token info 2", token)

    return token


#unique nodes: thing talking abt with kyleen for tree vs graph structures
# def instantiate_AST_Tree(root_node, node_label_dict = True, unique_nodes = True):
#     node_label_dict = {}
#     edges = []
#     node_id_offsets = defaultdict(int)
#     variable_ids = {}

#     def get_node_id(child):
#         child_node_id = defn_token(child) + "_" + str(id(child))

#         if isinstance(child, ast.Name):
#             variable_name = child.id
#             if variable_name not in variable_ids:
#                 variable_ids[variable_name] = str(len(variable_ids))  # Unique ID per variable name
#             child_node_id = variable_ids[variable_name]

#         if unique_nodes:
#             node_id_offsets[child_node_id] += 1
#             child_node_id += "_" + str(node_id_offsets[child_node_id])
#             node_label_dict[child_node_id] = ASTNode(child, child_node_id)   

#         elif child_node_id not in node_label_dict:
#             node_label_dict[child_node_id] = ASTNode(child, child_node_id)

#         return child_node_id

#     for parent_node in ast.walk(root_node):

#         print(parent_node)
        
#         parent_node_id = get_node_id(parent_node)

#         for child_node in ast.iter_child_nodes(parent_node):
#             child_node_id = get_node_id(child_node)
            
#             node_label_dict[parent_node_id].children.append(node_label_dict[child_node_id])

#             edges.append([node_label_dict[child_node_id], node_label_dict[child_node_id].token, node_label_dict[parent_node_id]])

#     for edge in edges:
#         print(edge[1], edge[2].token)

#     return root_node, edges

def instantiate_AST_Tree(root_node, node_label_dict=None, unique_nodes=False):
    if node_label_dict is None:
        node_label_dict = {}

    edges = []
    node_id_offsets = defaultdict(int)

    # Dictionary to map each variable name to a global unique ID
    variable_ids = {}

    def get_node_id(child):
        # Ensure the ID is always a string
        child_node_id = defn_token(child) + "_" + str(id(child))
        
        # For variable nodes (e.g., Name nodes), assign global unique IDs
        if isinstance(child, ast.Name):
            variable_name = child.id
            if variable_name not in variable_ids:
                # Assign a new global unique ID for each variable name
                variable_ids[variable_name] = str(len(variable_ids))  # Can be any unique ID
            child_node_id = variable_ids[variable_name]

        # Ensure the node is added to the dictionary only once
        if unique_nodes:
            node_id_offsets[child_node_id] += 1
            child_node_id += "_" + str(node_id_offsets[child_node_id])
            node_label_dict[child_node_id] = ASTNode(child, child_node_id)    
        elif child_node_id not in node_label_dict:
            node_label_dict[child_node_id] = ASTNode(child, child_node_id)

        return child_node_id

    root_node_id = get_node_id(root_node)

    # Traverse the AST and process the nodes
    for parent_node in ast.walk(root_node):
       # print(parent_node)  # For debugging
        parent_node_id = get_node_id(parent_node)

        for child_node in ast.iter_child_nodes(parent_node):
            child_node_id = get_node_id(child_node)

            # Add child node to the parent's children list
            node_label_dict[parent_node_id].children.append(node_label_dict[child_node_id])

            # Add edge between parent and child
            edges.append([node_label_dict[child_node_id], node_label_dict[child_node_id].token, node_label_dict[parent_node_id]])

    return node_label_dict[root_node_id], edges, node_label_dict  # Return the AST root node and the edges


# code_string = """i += 1\ni+=2"""
# tree = ast.parse(code_string)
# r_n, edges, ids = instantiate_AST_Tree(tree)
# print(edges)

# r_n, edges, ids = instantiate_AST_Tree(tree, unique_nodes = False)
# print(r_n)
# print(edges)