class ModelNotFoundError(Exception):
    pass


class InvalidKeyError(KeyError):
    pass


class DecisionAPIError(Exception):
    """Raised when a Decision model request fails for a reason other than an
    invalid key or an unrecognized model.

    This wraps errors raised by the underlying ``langchain-typesafe``
    integration (e.g., bad requests, permission errors, rate limits exceeded
    without needing programmatic retry handling, timeouts, connection
    failures, and server errors) so that callers of
    :class:`~langchain_dartmouth.decisions.DartmouthDecisions` only ever see
    exceptions defined in this package.
    """

    pass


class DecisionRateLimitError(DecisionAPIError):
    """Raised when a Decision model request is rejected due to rate limiting.

    :ivar retry_after_ms: The server-requested delay, in milliseconds, before
        retrying, or ``None`` if the response did not include a valid retry
        delay.
    :vartype retry_after_ms: float, optional
    """

    def __init__(self, *args, retry_after_ms: float | None = None):
        super().__init__(*args)
        self.retry_after_ms = retry_after_ms
