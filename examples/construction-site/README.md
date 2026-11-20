# Construction Site handoff snapshot

This is the measured Double Stab character setup from the 15 November 2026 handoff. It preserves the 44-point circuit, 17 transitions, map reference, keys, monster calibration, and HP/MP/loot settings for development. These files are not automatically loaded.

The live controller uses `config/user.yaml` and `config/map_profiles/`. On the original PC those already exist: **use them as-is** unless deliberately restoring this snapshot. Changes made after this handoff may be newer than the example.

On a fresh Windows clone with no existing settings, run this from the repository root to copy the snapshot into the ignored runtime locations:

```powershell
$profileKey = 'kerning-city-construction-site-7bcaa4fc6a'
$snapshot = Join-Path 'examples' 'construction-site'
$profileDirectory = Join-Path 'config' 'map_profiles'
$destinations = @(
    'config/user.yaml',
    (Join-Path $profileDirectory ($profileKey + '.json')),
    (Join-Path $profileDirectory ($profileKey + '.png'))
)
foreach ($destination in $destinations) {
    if (Test-Path -LiteralPath $destination) {
        throw "Existing settings found: $destination. Review or back them up before restoring."
    }
}
New-Item -ItemType Directory -Path $profileDirectory -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $snapshot 'user.yaml') -Destination $destinations[0]
Copy-Item -LiteralPath (Join-Path $snapshot ($profileKey + '.json')) -Destination $destinations[1]
Copy-Item -LiteralPath (Join-Path $snapshot ($profileKey + '.png')) -Destination $destinations[2]
```

Open Settings and review the profile before running. The snapshot uses a prepared 1366 × 768 frame and a `[20, 82, 210, 128]` minimap crop. The reference PNG's SHA-256 is `b131badde8048e46b7f79abe17f4be1b6e966dc8295619fca0834e38aa964427`. The route coordinates depend on that reference; changing the display/crop requires recalibration.

Z is Double Stab, Alt normal attack, C jump, Q HP, A MP, and X pickup. HP triggers at 70% with a 75% refill target and 50% farming minimum; MP triggers at 20%. Nametag/head templates under `assets/` are specific to this character. This example is not a universal preset for other characters or maps.

No process state, signal files, potion verification evidence, screenshots, or credentials are restored. Run fresh supervised checks before claiming live success on another machine. See [the development handoff](../../docs/DEVELOPMENT_HANDOFF.md).
