class AnalysisError(Exception):
    """A problem the user can fix (missing variable, too few values, ...).

    The message is shown verbatim in the results panel, so write it for a
    person, not for a log file.
    """
