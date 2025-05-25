try:
    from ._version import __version__
except ImportError:
    # _version.py is only created after running `pip install`
    __version__ = "0.0.0.dev0+placeholder"
