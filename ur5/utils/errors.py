class ur5Error(Exception):
    """Base class for exceptions in ur5."""

    pass


class XMLError(ur5Error):
    """Exception raised for errors related to xml."""

    pass


class SimulationError(ur5Error):
    """Exception raised for errors during runtime."""

    pass


class RandomizationError(ur5Error):
    """Exception raised for really really bad RNG."""

    pass
