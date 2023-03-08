import argparse
from collections.abc import Callable


class TestAction(argparse.Action):
    """This class is for the test flag."""

    def __init__(self, option_strings, dest, test_function: Callable[[], None], **kwargs):
        self._test_function = test_function
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        # if testing flag was set, ignore everything else and just test
        self._test_function()
        parser.exit()

    @classmethod
    def build(cls, test_function: Callable[[], None]):
        """Returns test action that can be used as an actual argparse action. Basically returns partial function."""

        return lambda option_strings, dest, **kwargs: TestAction(option_strings, dest, test_function, **kwargs)
