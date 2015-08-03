import re

class TypeReplacer(object):
    def __init__(self, macro_name, templated_type, deref):
        self.pattern = re.compile(macro_name + r"\((?P<var>.+?)\)")
        self.templated_type = templated_type
        self.deref = deref

    def rephrase(self, type_name):
        def wrapped(match):
            var = match.group("var")
            if self.deref:
                return "(<%s[%s]*>(%s))[0]" % (self.templated_type, type_name, var)
            else:
                return "(<%s[%s]*>(%s))" % (self.templated_type, type_name, var)
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
    TypeReplacer("DEREF_MAT", "CMat", deref=True),
    TypeReplacer("PTR_MAT", "CMat", deref=False),
    WrapperReplacer("WRAP_MAT", 'WrapMat_%s')
]

def typed_expression(pyp, code):
    def modify_snippet(type_name):
        modified = code
        modified = modified.replace('TYPE_NAME', type_name)

        for replacer in REPLACERS:
            modified = replacer(type_name, modified)

        pyp.indent(modified)

    pyp.indent('if self.dtypeinternal == np.NPY_INT32:')
    modify_snippet('int')
    pyp.indent('elif self.dtypeinternal == np.NPY_FLOAT32:')
    modify_snippet('float')
    pyp.indent('elif self.dtypeinternal == np.NPY_FLOAT64:')
    modify_snippet('double')
    pyp.indent('else:')
    pyp.indent('    raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of int32, float32, float64)")')


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
