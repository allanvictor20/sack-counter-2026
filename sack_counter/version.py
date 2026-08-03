"""
version.py — Single source of truth for the Sack Counter version.

Version strings used to be hardcoded per file and had drifted across five
releases simultaneously: the HUD printed v20, the display window v21, the
end-of-session report v20, ``run.py`` v23, and the output artefacts were
named ``_v21``.  Anyone reading a log or a screenshot had no reliable way
to tell which build produced it.

This is a leaf module — it imports nothing from the package — so any
module can read the version without risking a circular import.
"""

__version__ = "23.0"

# Short tag used in banners, window titles and output filenames.
VERSION_TAG = f"v{__version__.split('.')[0]}"
