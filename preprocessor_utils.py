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


REPLACERS = [
    TypeReplacer("DEREF_MAT", "CMat", "Mat", "matinternal", deref=True),
    TypeReplacer("PTR_MAT", "CMat",  "Mat", "matinternal", deref=False),
    WrapperReplacer("WRAP_MAT", 'WrapMat_%s')
]

TYPE_NPYINTERNAL_DICT = {
    'int':    'np.NPY_INT32',
    'float':  'np.NPY_FLOAT32',
    'double': 'np.NPY_FLOAT64',
}

def typed_expression_args(pyp, args, code):
    def modify_snippet(type_name):
        modified = code
        modified = modified.replace('TYPE_NAME',       type_name)
        modified = modified.replace('TYPE_NPYINTERNAL', TYPE_NPYINTERNAL_DICT.get(type_name))

        for replacer in REPLACERS:
            modified = replacer(type_name, modified)

        pyp.indent(modified)

    assert len(args) > 0
    if len(args) > 1:
        check_str = []
        for arg1, arg2 in zip(args[:-1], args[1:]):
            check_str.append('(<Mat>%s).dtypeinternal != (<Mat>%s).dtypeinternal' % (arg1, arg2))
        check_str = 'if ' + ' or '.join(check_str) + ':'
        pyp.indent(check_str)
        pyp.indent('   raise ValueError("All arguments must be of the same type")')
    pyp.indent('if (<Mat>%s).dtypeinternal == np.NPY_INT32:' % (args[0],))
    modify_snippet('int')
    pyp.indent('elif (<Mat>%s).dtypeinternal == np.NPY_FLOAT32:' % (args[0],))
    modify_snippet('float')
    pyp.indent('elif (<Mat>%s).dtypeinternal == np.NPY_FLOAT64:' % (args[0],))
    modify_snippet('double')
    pyp.indent('else:')
    pyp.indent('    raise ValueError("Invalid dtype:" + str(' + args[0] + '.dtype) + " (should be one of int32, float32, float64)")')

def typed_expression(pyp, code):
    return typed_expression_args(pyp, ["self"], code)
    # def modify_snippet(type_name):
    #     modified = code
    #     modified = modified.replace('TYPE_NAME', type_name)

    #     for replacer in REPLACERS:
    #         modified = replacer(type_name, modified)

    #     pyp.indent(modified)

    # pyp.indent('if self.dtypeinternal == np.NPY_INT32:')
    # modify_snippet('int')
    # pyp.indent('elif self.dtypeinternal == np.NPY_FLOAT32:')
    # modify_snippet('float')
    # pyp.indent('elif self.dtypeinternal == np.NPY_FLOAT64:')
    # modify_snippet('double')
    # pyp.indent('else:')
    # pyp.indent('    raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of int32, float32, float64)")')







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
