"""
colors.py — BGR colour constants for the Sack Counter drawing routines.

These are semantic aliases onto the Modernist design tokens in
:mod:`sack_counter.theme`.  The names are unchanged from earlier
versions so every existing import keeps working; only the values were
retuned to the design system.

The system is deliberately near-monochrome: one accent red carries all
the meaning (a sack was counted, this is the door, this needs a look)
and everything else is a neutral.  Resist adding a fourth hue here —
the console reads as one surface because it does not have one.
"""

from .theme import (
    ACCENT, ACCENT_400, ACCENT_500,
    INK, PAPER,
    NEUTRAL_300, NEUTRAL_500, NEUTRAL_600,
)

C_PERSON   = PAPER          # tracked worker, carrying nothing yet
C_SACK     = NEUTRAL_300    # detected sack, no owner
C_SACK_ST  = NEUTRAL_600    # sack sitting still
C_OWNED    = ACCENT         # sack assigned to a worker, high confidence
C_GHOST    = NEUTRAL_500    # extrapolated position, not a detection
C_LINE_A   = ACCENT         # legacy gate lines (kept for older overlays)
C_LINE_B   = NEUTRAL_500
C_COUNTED  = ACCENT         # worker who has delivered
C_HUD_BG   = INK            # console scrim
C_ANOMALY  = ACCENT_500     # needs a look
C_REID     = ACCENT_400     # track was lost and recognised again
C_BOX      = PAPER          # box detection — secondary to sacks
C_LOW_CONF = NEUTRAL_500
C_MED_CONF = ACCENT_400
