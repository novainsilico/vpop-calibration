try:
    from tqdm.notebook import tqdm
except ImportError:

    class tqdm:  # type: ignore
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass
