class ApplicationError(Exception):
    """Base error suitable for mapping to an HTTP response."""


class NotFoundError(ApplicationError):
    pass


class ValidationError(ApplicationError):
    pass


class CostLimitError(ApplicationError):
    pass


class ProviderError(ApplicationError):
    pass

\n