class DomainError(Exception):
    """Basisklasse für domänenspezifische Fehler."""


class ValidationError(DomainError):
    pass
