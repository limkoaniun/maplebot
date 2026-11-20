# Octopus template source

Copied without modification from the user-provided reference checkout:
`maplestory-auto-ds/monster/zhang_yu/` (a local WSL checkout of the reference project)

Reference checkout commit: `8c1f53f`.
Copied on 2026-11-14. The reference assets were visually verified as Octopus sprites;
the two poses are move and hit (no death pose). Both PNGs contain binary alpha masks.
The source repository does not provide per-image upstream provenance here, so no
additional origin or license is asserted. Blue Mushroom templates were not copied.

These match visible Octopuses in the prepared Construction Site capture.
Detection calibration is separate from saved user settings.

Additional walking poses `move_1.png` through `move_4.png` were downloaded on
2026-11-14 from the public MapleStory.io GMS version 83 sprite API, using the
repository's `fetch_mob("Octopus")` helper. The API's decoded `move_0` pixels are
identical to the reference checkout's `move_0.png`, confirming the same sprite
set. The five stand poses duplicate the five move poses and are omitted. All
four death poses returned in the flat archive are excluded. This leaves six
distinct RGBA images (five walking poses and one hit pose), including the two
unmodified reference-checkout files above.

API source: https://maplestory.io/api/GMS/83/mob

[
  {
    "file": "move_0.png",
    "size": [
      62,
      69
    ],
    "sha256": "f422383dda8a816df07c42924d474b61f23ad1493067326c0f368ce33eb6e3fa"
  },
  {
    "file": "hit1_0.png",
    "size": [
      74,
      80
    ],
    "sha256": "291813df466f24f800b5df88d33faab4924c1711c12c5ee8303720f4aeb809ac"
  }
]
