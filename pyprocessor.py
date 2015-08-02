#!/usr/bin/python
#!/usr/local/bin/python

import re
import sys

from collections import namedtuple

class PyProcessorCapture(object):
    def __init__(self, default_indent=0):
        self.default_indent = default_indent
        self.reset()

    def out(self, s, end='\n'):
        self._output.append(s + end)


    def indent(self, s, end='\n', indent=None):
        if indent is None:
            indent = self.default_indent


        for line in s.split('\n'):
            self.out(' ' * indent + line)

    def msg(self, m):
        print("Message: ", m)

    def reset(self):
        self._output = []

    def get(self):
        return ''.join(self._output)


Snippet = namedtuple("Snippet", ["position", "length", "code", "indentation"])


def process_snippet(code, prefix, suffix):
    lines = code.split('\n')
    indentation = lines[0].index(prefix)
    for i in range(len(lines)):
        lines[i] = lines[i][indentation:]

    return indentation, '\n'.join(lines[1:-1])

def process_inline_snippet(code, prefix, suffix):
    lines = code.split('\n')
    indentation = lines[0].index(prefix)
    function = lines[0][indentation + len(prefix) + 1:]
    if len(lines) > 2:
        lines = [line[indentation:] for line in lines[1:-1]]

        code = function + '"""' + '\n'.join(lines) + '""")'
        return indentation, code
    else:
        return indentation, function

def process_file(pyp_filepath, prefix="pyp", suffix = "endpyp"):
    with open(pyp_filepath) as pyp_f:
        pyp_source = pyp_f.read()

    if suffix is None:
        suffix = prefix

    pyp_pattern = re.compile(r'(?!\n)\ *(?s)' + prefix + r'\s.*?' + suffix, re.MULTILINE)
    pyp_inline_pattern = re.compile(r'(^|(?!\n))\ *(?s)' + prefix + r':.*?' + suffix, re.MULTILINE)

    pyp_snippets = []
    for match in pyp_pattern.finditer(pyp_source):
        position = match.start()
        length = len(match.group())
        indentation, code = process_snippet(match.group(), prefix, suffix)
        pyp_snippets.append(Snippet(position, length, code, indentation))

    for match in pyp_inline_pattern.finditer(pyp_source):
        position = match.start()
        length = len(match.group())
        indentation, code = process_inline_snippet(match.group(), prefix, suffix)
        pyp_snippets.append(Snippet(position, length, code, indentation))

    # make sure the code snippets get executed in order
    pyp_snippets = sorted(pyp_snippets, key=lambda x: x.position)

    pyp_outputs = []

    pyp = PyProcessorCapture()
    variable_space = {'pyp': pyp}


    last_end = 0

    modified_source = ''

    for i, snippet in enumerate(pyp_snippets):
        pyp.reset()
        pyp.default_indent = snippet.indentation
        exec(snippet.code, variable_space)

        modified_source = modified_source + pyp_source[last_end:snippet.position]
        last_end = snippet.position + snippet.length

        modified_source = modified_source + pyp.get()

    modified_source = modified_source + pyp_source[last_end:]

    return modified_source

if __name__ == '__main__':
    res = process_file(sys.argv[1])
    print(res)
