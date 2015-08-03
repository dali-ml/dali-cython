def typed_expression(pyp, replacable_expression, replacable_type, code):
    pyp.indent('if self.dtype == np.int32:')
    typefixed = code.replace('TYPED_EXPRESSION', '(<%s[int]>(%s))' % (replacable_type, replacable_expression))
    pyp.indent(typefixed)

    pyp.indent('elif self.dtype == np.float32:')
    typefixed = code.replace('TYPED_EXPRESSION', '(<%s[float]>(%s))' % (replacable_type, replacable_expression))
    pyp.indent(typefixed)

    pyp.indent('elif self.dtype == np.float64:')
    typefixed = code.replace('TYPED_EXPRESSION', '(<%s[double]>(%s))' % (replacable_type, replacable_expression))
    pyp.indent(typefixed)
    pyp.indent('else:')
    pyp.indent('    raise ValueError("Invalid dtype:" + str(self.dtype) + " (should be one of int32, float32, float64)")')
