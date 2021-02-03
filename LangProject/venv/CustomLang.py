import string

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS


# parse position
class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def next_tok(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


# token types
tok_int = 'integer'
tok_ID = 'idnetifier'
tok_keyword = 'keyword'
tok_plus = 'plus'
tok_minus = 'minus'
tok_mult = 'multiply'
tok_div = 'divide'

tok_equality = 'equals'
tok_equal_equals = 'equals equals'
tok_less_than = 'less than'
tok_greater_than = 'greater than'
tok_EOF = 'end of file'

KEYWORDS = [
    'variable',
    'if_this',
    'then_this'
]


# token class
class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.next_tok()

        if pos_end:
            self.pos_end = pos_end.copy()

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'


# lexer   parsing and assigning token values to valid input / syntax error checking invalid input

class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.next_tok()

    def next_tok(self):
        self.pos.next_tok(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in ' \t':
                self.next_tok()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '+':
                tokens.append(Token(tok_plus, pos_start=self.pos))
                self.next_tok()
            elif self.current_char == '-':
                tokens.append(Token(tok_minus, pos_start=self.pos))
                self.next_tok()
            elif self.current_char == '*':
                tokens.append(Token(tok_mult, pos_start=self.pos))
                self.next_tok()
            elif self.current_char == '/':
                tokens.append(Token(tok_div, pos_start=self.pos))
                self.next_tok()
            # also checks next tok for ==
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.next_tok()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(tok_EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS:
            num_str += self.current_char
            self.next_tok()
        return Token(tok_int, int(num_str), pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.next_tok()

        tok_type = tok_keyword if id_str in KEYWORDS else tok_ID
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_equals(self):
        tok_type = tok_equality
        pos_start = self.pos.copy()
        self.next_tok()
        # if the the next token is also and equals this signifies and == operator
        if self.current_char == '=':
            self.next_tok()
            tok_type = tok_equal_equals

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = tok_less_than
        pos_start = self.pos.copy()
        self.next_tok()
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        tok_type = tok_greater_than
        pos_start = self.pos.copy()
        self.next_tok()
        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)


# defining/assigning the node types
class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end


class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class IfNode:
    def __init__(self, cases):
        self.cases = cases

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (self.cases[len(self.cases) - 1][0]).pos_end


# parser

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tok_idx = -1
        self.next_tok()

    def next_tok(self, ):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]
        return self.current_tok

    def parse(self):
        res = self.expr()
        if not res.error and self.current_tok.type != tok_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '+', '-', '*', '/', '==', '<', '>'"
            ))
        return res

    def if_expr(self):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.current_tok.matches(tok_keyword, 'if_this'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'if_this'"
            ))

        res.register_next_tokment()
        self.next_tok()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_tok.matches(tok_keyword, 'then_this'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'then_this'"
            ))

        res.register_next_tokment()
        self.next_tok()

        expr = res.register(self.expr())
        if res.error: return res
        cases.append((condition, expr))

        return res.success(IfNode(cases))

    def value(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (tok_int):
            res.register_next_tokment()
            self.next_tok()
            return res.success(NumberNode(tok))

        elif tok.type == tok_ID:
            res.register_next_tokment()
            self.next_tok()
            return res.success(VarAccessNode(tok))



        elif tok.matches(tok_keyword, 'if_this'):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        return res.failure(InvalidSyntaxError(
            tok.pos_start, tok.pos_end,
            "Expected int, identifier, '+', '-'"
        ))

    def term(self):
        return self.bin_op(self.value, (tok_mult, tok_div))

    def arith_expr(self):
        return self.bin_op(self.term, (tok_plus, tok_minus))

    def comp_expr(self):
        res = ParseResult()
        node = res.register(self.bin_op(self.arith_expr, (tok_equal_equals, tok_less_than, tok_greater_than)))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected int,  identifier, '+', '-' "
            ))

        return res.success(node)

    def expr(self):
        res = ParseResult()

        if self.current_tok.matches(tok_keyword, 'variable'):
            res.register_next_tokment()
            self.next_tok()

            if self.current_tok.type != tok_ID:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier"
                ))

            var_name = self.current_tok
            res.register_next_tokment()
            self.next_tok()

            if self.current_tok.type != tok_equality:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '='"
                ))

            res.register_next_tokment()
            self.next_tok()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.bin_op(self.comp_expr, ((tok_keyword, 'if_this'), (tok_keyword, 'if_this'))))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'variable', int, identifier, '+', '-' "
            ))

        return res.success(node)

    def bin_op(self, func_a, ops, func_b=None):
        if func_b == None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            res.register_next_tokment()
            self.next_tok()
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)


# result parse
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.next_tok_count = 0

    def register_next_tokment(self):
        self.next_tok_count += 1

    def register(self, res):
        self.next_tok_count += res.next_tok_count
        if res.error: self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.next_tok_count == 0:
            self.error = error
        return self


# class for runtime error checking
class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


# value operations
class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Division by zero',
                    self.context
                )

            return Number(self.value / other.value).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)


# context

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


# symbol table to resolve varible names

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]


# interpreter
#


class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if not value:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res

        context.symbol_table.set(var_name, value)
        return res.success(value)

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: return res

        if node.op_tok.type == tok_plus:
            result, error = left.added_to(right)
        elif node.op_tok.type == tok_minus:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == tok_mult:
            result, error = left.multed_by(right)
        elif node.op_tok.type == tok_div:
            result, error = left.dived_by(right)

        elif node.op_tok.type == tok_equal_equals:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == tok_less_than:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == tok_greater_than:
            result, error = left.get_comparison_gt(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.error: return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.error: return res
                return res.success(expr_value)

        return res.success(None)


# symbol table for comparison expressions
global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number(0))
global_symbol_table.set("FALSE", Number(0))
global_symbol_table.set("TRUE", Number(1))


def run(fn, text):
    # create tokens from input
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error: return None, error

    # create abstract token tree
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # run
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error


# error messages
class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        # result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)
