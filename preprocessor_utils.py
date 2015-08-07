import re

class TypeReplacer(object):
    def __init__(self, macro_name, templated_type, base_class, internal_property, deref):
        self.pattern = re.compile(macro_name + r"\((?P<var>.+?)\)")
        self.templated_type = templated_type
        self.deref = deref
        self.base_class = base_class
        self.internal_property = internal_property

    def rephrase(self, type_name):
        def wrapped(match):
            var = match.group("var")
            if self.deref:
                return "(<%s[%s]*>((<%s>(%s)).%s))[0]" % (self.templated_type, type_name, self.base_class, var, self.internal_property)
            else:
                return "(<%s[%s]*>((<%s>(%s)).%s))" % (self.templated_type, type_name, self.base_class, var, self.internal_property)
        return wrapped

    def __call__(self, *args, **kwargs):
        return self.replace(*args, **kwargs)

    def replace(self, type_name, text):
        return self.pattern.sub(self.rephrase(type_name), text)

class WrapperReplacer(object):
    def __init__(self, pattern, wrapper_function):
        self.pattern = pattern
        self.wrapper_function = wrapper_function

    def __call__(self, *args, **kwargs):
        return self.replace(*args, **kwargs)

    def replace(self, type_name, text):
        return text.replace(self.pattern, self.wrapper_function % (type_name,))

class LambdaReplacer(object):
    def __init__(self, macro_name, lambdaf):
        self.pattern = re.compile(macro_name + r"\((?P<var>.+?)\)")
        self.lambdaf = lambdaf

    def __call__(self, *args, **kwargs):
        return self.replace(*args, **kwargs)

    def replace(self, type_name, text):
        return self.pattern.sub(self.lambdaf(type_name), text)

class TypedName(LambdaReplacer):
    def __init__(self):
        def replacer(type_name):
            def wrapped(match):
                var = match.group("var")
                return '%s_%s' % (var, type_name)
            return wrapped

        super(TypedName, self).__init__('TYPED', replacer)


REPLACERS = [
    TypeReplacer("DEREF_MAT", "CMat", "Mat", "matinternal", deref=True),
    TypeReplacer("PTR_MAT", "CMat",  "Mat", "matinternal", deref=False),
    WrapperReplacer("WRAP_MAT", 'WrapMat_%s'),

    TypeReplacer("DEREF_LAYER", "CLayer", "Layer", "layerinternal", deref=True),
    TypeReplacer("PTR_LAYER", "CLayer",  "Layer", "layerinternal", deref=False),
    WrapperReplacer("WRAP_LAYER", 'WrapLayer_%s'),

    TypeReplacer("DEREF_RNN", "CRNN", "RNN", "layerinternal", deref=True),
    TypeReplacer("PTR_RNN", "CRNN",  "RNN", "layerinternal", deref=False),
    WrapperReplacer("WRAP_RNN", 'WrapRNN_%s'),

    TypeReplacer("DEREF_GRU", "CGRU", "GRU", "layerinternal", deref=True),
    TypeReplacer("PTR_GRU", "CGRU",  "GRU", "layerinternal", deref=False),
    WrapperReplacer("WRAP_GRU", 'WrapGRU_%s'),

    TypeReplacer("DEREF_STACKEDLAYER", "CStackedInputLayer", "StackedInputLayer", "layerinternal", deref=True),
    TypeReplacer("PTR_STACKEDLAYER", "CStackedInputLayer", "StackedInputLayer", "layerinternal", deref=False),
    WrapperReplacer("WRAP_STACKEDLAYER", 'WrapStackedLayer_%s'),

    TypeReplacer("DEREF_LSTMSTATE", "CLSTMState", "LSTMState", "lstmstateinternal", deref=True),
    TypeReplacer("PTR_LSTMSTATE", "CLSTMState", "LSTMState", "lstmstateinternal", deref=False),
    WrapperReplacer("WRAP_LSTMSTATE", 'WrapLSTMState_%s'),

    TypeReplacer("DEREF_LSTM", "CLSTM", "LSTM", "layerinternal", deref=True),
    TypeReplacer("PTR_LSTM", "CLSTM", "LSTM", "layerinternal", deref=False),
    WrapperReplacer("WRAP_LSTM", 'WrapLSTM_%s'),

    TypeReplacer("DEREF_STACKEDLSTM", "CStackedLSTM", "StackedLSTM", "layerinternal", deref=True),
    TypeReplacer("PTR_STACKEDLSTM", "CStackedLSTM", "StackedLSTM", "layerinternal", deref=False),
    WrapperReplacer("WRAP_STACKEDLSTM", 'WrapStackedLSTM_%s'),

    TypedName()
]

for solver in ["SGD", "AdaGrad", "RMSProp", "AdaDelta", "Adam"]:
    REPLACERS.append(
        TypeReplacer("DEREF_" + solver.upper(), "C" + solver, solver, "solverinternal", deref=True)
    )
    REPLACERS.append(
        TypeReplacer("PTR_" + solver.upper(), "C" + solver, solver, "solverinternal", deref=False)
    )

TYPE_NPYINTERNAL_DICT = {
    'int':    'np.NPY_INT32',
    'float':  'np.NPY_FLOAT32',
    'double': 'np.NPY_FLOAT64',
}

TYPE_NUMPY_PRETTY = {
    'int':    'np.int32',
    'float':  'np.float32',
    'double': 'np.float64',
}

def modify_snippet(pyp, code, type_name):
    modified = code
    modified = modified.replace('TYPE_NAME',       type_name)
    modified = modified.replace('TYPE_NPYINTERNAL', TYPE_NPYINTERNAL_DICT.get(type_name))
    modified = modified.replace('TYPE_NPYPRETTY', TYPE_NUMPY_PRETTY.get(type_name))

    for replacer in REPLACERS:
        modified = replacer(type_name, modified)

    pyp.indent(modified)



def type_repeat_with_types(pyp, types, code):
    for typ in types:
        modify_snippet(pyp, code, typ)

def type_repeat(pyp, code):
    type_repeat_with_types(pyp, ["int", "float", "double"], code)

def type_frepeat(pyp, code):
    type_repeat_with_types(pyp, ["float", "double"], code)

def typed_expression_args_with_types(pyp, types, args, code):
    if type(args) == tuple:
        args_class
    assert len(args) > 0
    if len(args) > 1:
        check_str = []
        for arg1, arg2 in zip(args[:-1], args[1:]):
            check_str.append('(%s).dtypeinternal != (%s).dtypeinternal' % (arg1, arg2))
        check_str = 'if ' + ' or '.join(check_str) + ':'
        pyp.indent(check_str)
        pyp.indent('   raise ValueError("All arguments must be of the same type")')

    first_run = True
    for typ in types:
        if_str = 'if' if first_run else 'elif'
        first_run = False
        pyp.indent(if_str + ' (%s).dtypeinternal == %s:' % (args[0], TYPE_NPYINTERNAL_DICT[typ]))
        modify_snippet(pyp, code, typ)
    pyp.indent('else:')
    types_str = ', '.join([TYPE_NUMPY_PRETTY[typ] for typ in types])
    pyp.indent('    raise ValueError("Invalid dtype:" + str(' + args[0] + '.dtype) + " (should be one of ' + types_str+ ')")')

def typed_expression_args(pyp, args, code):
    typed_expression_args_with_types(pyp, ["int", "float", "double"], args, code)

def typed_fexpression_args(pyp, args, code):
    typed_expression_args_with_types(pyp, ["float", "double"], args, code)

def typed_expressions_with_types(pyp, lst, cast_to, types, code):
    assert len(lst) > 0
    pyp.indent('if len(%s) == 0:' % (lst,))
    pyp.indent("    raise ValueError('list cannot be empty')")
    pyp.indent('common_dtype = (<%s>(%s[0])).dtypeinternal' % (cast_to, lst,))
    pyp.indent('for el in %s:' % (lst,))
    pyp.indent('    if (<%s>el).dtypeinternal != common_dtype:' % (cast_to,))
    pyp.indent('        common_dtype = -1')
    pyp.indent('        break')
    pyp.indent('if common_dtype == -1:')
    pyp.indent('    raise ValueError("All the arguments must be of the same type")')

    first_run = True
    for typ in types:
        if_str = 'if' if first_run else 'elif'
        first_run = False
        pyp.indent(if_str + ' common_dtype == %s:' % (TYPE_NPYINTERNAL_DICT[typ],))
        modify_snippet(pyp, code, typ)
    pyp.indent('else:')
    types_str = ', '.join([TYPE_NUMPY_PRETTY[typ] for typ in types])
    pyp.indent('    raise ValueError("Invalid dtype:" + str(' + lst + '[0].dtype) + " (should be one of ' + types_str+ ')")')


def typed_expression_list(pyp, lst, cast_to, code):
    typed_expressions_with_types(pyp, lst, cast_to, ["int", "float", "double"], code)

def typed_fexpression_list(pyp, lst, cast_to, code):
    typed_expressions_with_types(pyp, lst, cast_to, ["float", "double"], code)

def typed_expression(pyp, code):
    return typed_expression_args(pyp, ["self"], code)

def typed_fexpression(pyp, code):
    return typed_fexpression_args(pyp, ["self"], code)

def rich_typed_expression(pyp, replacable_type, code):
    def modify_snippet(type_name):
        modified = code
        modified = modified.replace('TYPE_NAME', type_name)
        modified = modified.replace('TEMPLATED_TYPE', '%s[%s]' % (replacable_type, type_name))
        modified = modified.replace('TEMPLATED_CAST', '<%s[%s]>' % (replacable_type, type_name))
        pyp.indent(modified)

    pyp.indent('if self.dtypeinternal == np.NPY_INT32:')
    modify_snippet('int')
    pyp.indent('elif self.dtypeinternal == np.NPY_FLOAT32:')
    modify_snippet('float')
    pyp.indent('elif self.dtypeinternal == np.NPY_FLOAT64:')
    modify_snippet('double')
    pyp.indent('else:')
    pyp.indent('    raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of int32, float32, float64)")')

WITHOUT_INT = ["float", "double"]
