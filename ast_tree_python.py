import ast

class ASTNode(object):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = self.add_children()

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
            # Handle leaf nodes with specific attributes
            if hasattr(self.node, 'id'):  # Name node
                token = self.node.id
                is_name = True
            elif hasattr(self.node, 'name'):  # FunctionDef, ClassDef, etc.
                token = self.node.name
                is_name = True
            elif hasattr(self.node, 'value'):  # Constant/Num/Str nodes
                token = str(self.node.value) if hasattr(self.node, 'value') else name
            elif hasattr(self.node, 'arg'):  # arguments
                token = self.node.arg
                is_name = True
            else:
                token = name
        else:
            # Handle operators
            if hasattr(self.node, 'op'):
                op_name = self.node.op.__class__.__name__
                token = op_name
            elif hasattr(self.node, 'ops'):  # Compare node
                token = self.node.ops[0].__class__.__name__
            else:
                token = name
        
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

    def add_children(self):
        # if self.is_str:
        #     return []
        
        # children = list(ast.iter_child_nodes(self.node))
        
        # Python-specific control flow handling
        # if self.token in ['FunctionDef', 'AsyncFunctionDef']:
        #     # Return only the body, skip decorators, args, returns
        #     return [ASTNode(child) for child in self.node.body]
        # elif self.token in ['If']:
        #     # Return only the test condition
        #     return [ASTNode(self.node.test)]
        # elif self.token in ['While']:
        #     return [ASTNode(self.node.test)]
        # elif self.token == 'For':
        #     # Return target and iter, skip body
        #     result = []
        #     if hasattr(self.node, 'target'):
        #         result.append(ASTNode(self.node.target))
        #     if hasattr(self.node, 'iter'):
        #         result.append(ASTNode(self.node.iter))
        #     return result
        # else:
        #     return [ASTNode(child) for child in children]
        

        if self.is_str:
            return []
        
        children = list(ast.iter_child_nodes(self.node))

        # Python-specific control flow handling
        if self.token in ['FunctionDef', 'AsyncFunctionDef']:
            # Return only the body, skip decorators, args, returns
            return [ASTNode(child) for child in self.node.body]
        elif self.token in ['If']:
            # Return test condition and bodies
            result = [ASTNode(self.node.test)]
            if hasattr(self.node, 'body'):
                result.extend([ASTNode(child) for child in self.node.body])
            if hasattr(self.node, 'orelse') and self.node.orelse:
                result.extend([ASTNode(child) for child in self.node.orelse])
            return result
        elif self.token in ['While']:
            # Return test condition and body
            result = [ASTNode(self.node.test)]
            if hasattr(self.node, 'body'):
                result.extend([ASTNode(child) for child in self.node.body])
            if hasattr(self.node, 'orelse') and self.node.orelse:
                result.extend([ASTNode(child) for child in self.node.orelse])
            return result
        elif self.token == 'For':
            # Return target, iter, and body
            result = []
            if hasattr(self.node, 'target'):
                result.append(ASTNode(self.node.target))
            if hasattr(self.node, 'iter'):
                result.append(ASTNode(self.node.iter))
            if hasattr(self.node, 'body'):
                result.extend([ASTNode(child) for child in self.node.body])
            if hasattr(self.node, 'orelse') and self.node.orelse:
                result.extend([ASTNode(child) for child in self.node.orelse])
            return result
        else:
            return [ASTNode(child) for child in children]

    def get_children_with_edge_types(self):
        """Return children as tuples of (ASTNode, edge_type) where edge_type indicates
        which AST attribute the child comes from (e.g., 'body', 'test', 'orelse', 'target', 'iter', 'child').
        
        Returns:
            List of tuples: [(ASTNode, edge_type), ...]
        """
        if self.is_str:
            return []
        
        children = list(ast.iter_child_nodes(self.node))

        # Python-specific control flow handling with edge types
        if self.token in ['FunctionDef', 'AsyncFunctionDef']:
            # Return only the body, skip decorators, args, returns
            return [(ASTNode(child), 'body') for child in self.node.body]
        elif self.token in ['If']:
            # Return test condition and bodies with edge types
            result = [(ASTNode(self.node.test), 'test')]
            if hasattr(self.node, 'body'):
                result.extend([(ASTNode(child), 'body') for child in self.node.body])
            if hasattr(self.node, 'orelse') and self.node.orelse:
                result.extend([(ASTNode(child), 'orelse') for child in self.node.orelse])
            return result
        elif self.token in ['While']:
            # Return test condition and body with edge types
            result = [(ASTNode(self.node.test), 'test')]
            if hasattr(self.node, 'body'):
                result.extend([(ASTNode(child), 'body') for child in self.node.body])
            if hasattr(self.node, 'orelse') and self.node.orelse:
                result.extend([(ASTNode(child), 'orelse') for child in self.node.orelse])
            return result
        elif self.token == 'For':
            # Return target, iter, and body with edge types
            result = []
            if hasattr(self.node, 'target'):
                result.append((ASTNode(self.node.target), 'target'))
            if hasattr(self.node, 'iter'):
                result.append((ASTNode(self.node.iter), 'iter'))
            if hasattr(self.node, 'body'):
                result.extend([(ASTNode(child), 'body') for child in self.node.body])
            if hasattr(self.node, 'orelse') and self.node.orelse:
                result.extend([(ASTNode(child), 'orelse') for child in self.node.orelse])
            return result
        else:
            # For other nodes, determine edge type from AST field name
            # We need to map children back to their field names
            result = []
            mapped_ast_nodes = set()  # Track which AST nodes we've already mapped
            
            for field, value in ast.iter_fields(self.node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST) and item in children:
                            result.append((ASTNode(item), field))
                            mapped_ast_nodes.add(id(item))  # Use id() to track AST node identity
                elif isinstance(value, ast.AST) and value in children:
                    result.append((ASTNode(value), field))
                    mapped_ast_nodes.add(id(value))
            
            # If we couldn't map all children, use 'child' as default
            for child in children:
                if id(child) not in mapped_ast_nodes:
                    result.append((ASTNode(child), 'child'))
            
            return result


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
                    #if field != 'body':
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