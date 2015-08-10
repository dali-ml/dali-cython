import types

def apply_recursively_on_type(x, f, target_type, list_callback=None):
    if type(x) == target_type:
        return f(x)
    elif type(x) == list or isinstance(x, types.GeneratorType):
        ret = [ apply_recursively_on_type(el, f, target_type, list_callback) for el in x]
        if list_callback and all(type(el) == target_type for el in x):
            ret = list_callback(ret)
        return ret
    elif type(x) == dict:
        res = {}
        for k,v in x.items():
            res[k] = apply_recursively_on_type(v, f, target_type, list_callback)
        return res
    else:
        return x
