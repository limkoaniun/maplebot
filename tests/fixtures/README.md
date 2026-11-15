Fixture minimaps copied from https://github.com/KenYu910645/MapleStoryAutoLevelUp (MIT, (c) 2025 Ken Yu).

The additional `minimaps/profile_*.png` and `map_recognition/` images are small
recorded minimap crops used to verify identity, scrolling alignment, and rejection
of the wrong map. `nametags/` contains local character/scenery crops and a HUD
lookalike; `expected.json` records their positions. They exclude chat and the rest
of the desktop. The character templates under `assets/` are required by those tests.

See `detection_edges/README.md` for the recorded monster search-boundary cases and
their sprite provenance. Full live recordings and diagnostic screenshots remain
under ignored local directories; do not add them to this fixture collection.
