# Monster templates

Each monster folder contains the PNG poses used by the vision matcher. Only PNGs directly inside a selected folder are loaded; `_excluded_death/` keeps death-animation poses out of live target matching. Do not flatten those subdirectories into the live library.

The current Construction Site profile selects `blue_snail`, `octopus`, `orange_mushroom`, `stump`, and `green_mushroom`. Other included folders support previous map calibration. Per-monster thresholds and scales belong to the map profile, not these image files.

See [octopus/SOURCE.md](octopus/SOURCE.md) for the recorded reference-checkout/API provenance of the octopus assets. The small copied templates used by boundary regressions are documented in [tests/fixtures/detection_edges/README.md](../tests/fixtures/detection_edges/README.md). These are game-image assets; no additional image license is asserted by this handoff.
