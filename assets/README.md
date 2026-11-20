# Character templates

`player-nametag.png` and `player-head.png` are small crops of your own character's nametag and head, prepared from a local game capture. They show your in-game name and look, so `.gitignore` keeps them out of this public repository: each checkout supplies its own. The settings app's **Browse...** button copies a nametag crop into this folder.

The Construction Site example uses these paths. The recorded nametag regressions in `tests/test_recorded_nametags.py` also use them, together with the git-ignored frames in `tests/fixtures/nametags/`, and skip when those local files are absent. A different character or appearance needs new calibration; do not replace these assets silently. Full captures are kept locally under ignored directories.

A personal exe build (`packaging/build_exe.ps1` without `-NoExample`/`-Public`) copies `assets/player-*` from the working tree so the bundled example works; a public build never includes them.
