import os


def RepoDir() -> str:
    """Root of the fr3_trajopt project (directory containing this file)."""
    return os.path.dirname(os.path.realpath(__file__))


def PvampRootDir() -> str:
    """Root of the PVAMP monorepo (parent of fr3_trajopt/, vamp/, cricket/, ...)."""
    return os.path.dirname(RepoDir())
