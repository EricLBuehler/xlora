def x(d, a=1, b=2):
    pass


import inspect

print(inspect.signature(x).parameters)

dat = {"a": 1, "b": 2, "c": 3, "d": 123}
dat = {k: v for k, v in dat.items() if k in inspect.signature(x).parameters}
print(dat)
