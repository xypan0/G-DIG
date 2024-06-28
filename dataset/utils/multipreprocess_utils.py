from itertools import repeat


def starmap_with_kwargs(pool, fn, args_iter=None, kwargs_iter=None):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    if args is None:
        return fn(**kwargs)
    else:
        return fn(*args, **kwargs)