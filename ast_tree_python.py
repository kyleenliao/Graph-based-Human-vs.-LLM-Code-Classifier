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
        if self.is_str:
            return []
        
        children = list(ast.iter_child_nodes(self.node))
        
        # Python-specific control flow handling
        if self.token in ['FunctionDef', 'AsyncFunctionDef']:
            # Return only the body, skip decorators, args, returns
            return [ASTNode(child) for child in self.node.body]
        elif self.token in ['If']:
            # Return only the test condition
            return [ASTNode(self.node.test), ASTNode(self.node.body)]
        elif self.token in ['While']:
            return [ASTNode(self.node.test), ASTNode(self.node.body)]
        elif self.token == 'For':
            # Return target and iter, skip body
            result = []
            if hasattr(self.node, 'target'):
                result.append(ASTNode(self.node.target))
            if hasattr(self.node, 'iter'):
                result.append(ASTNode(self.node.iter))
            if hasattr(self.node, 'body'):
                result.append(ASTNode(self.node.body))
            return result
        else:
            return [ASTNode(child) for child in children]


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