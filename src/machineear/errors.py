"""Actionable errors. Anomaly status is omitted whenever one of these is raised for input."""


class MachineEarError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
