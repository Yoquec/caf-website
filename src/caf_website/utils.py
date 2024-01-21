from functools import reduce

def compose(*fns):
    return reduce(lambda f, g: lambda x: g(f(x)), fns)
