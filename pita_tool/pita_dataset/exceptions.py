"""
Custom exceptions for the dataset generation module.
"""


class NoAnnotationsFileFound(Exception):
    """Raised when no annotations are found."""

    def __init__(self, message="No annotations found."):
        self.message = message
        super().__init__(self.message)
