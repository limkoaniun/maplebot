# Gain evidence fixtures

Small crops of the HUD from the supervised Kerning City Construction Site run recorded on
15 Nov 2026 (1366x768 prepared frame, i.e. the frame after `prepare_frame`). They hold only the
EXP digits above the EXP bar or the gain notification lines with the scenery directly behind
them: no chat, character name, account panel or desktop. The full frames stay under the ignored
local `log/` directory and must not be added here.

Sources are `log/final-circuit-review-v4/<name>.png` unless marked `log/combat-frame-NN.png`
(the controller's combat trace of the same day).

## EXP text, crop `[x=738, y=737, w=107, h=14]`

| File | Source frame | Content |
| --- | --- | --- |
| `exp-text-4975-a.png` | `z0001-before` | "4,975 [91.12%]" |
| `exp-text-4975-b.png` | `z0003-after450ms` | same value, a later capture (pixel-identical) |
| `exp-text-4990.png` | `z0003-after1200ms` | "4,990 [91.39%]" (99 changed pixels) |
| `exp-text-next.png` | `z0005-after1200ms` | the next value (78 changed pixels) |

## Gain line stacks, crop `[x=1150, y=576, w=214, h=44]`

Three line slots at 14 px pitch: the newest line is the bottom 16 rows (`[1150, 604, 214, 16]`),
the next ones are 14 and 28 rows higher.

| File | Source frame | Content (newest first) |
| --- | --- | --- |
| `lines-empty.png` | `z0007-before` | no line |
| `lines-exp.png` | `z0023-after450ms` | "You received EXP (+15)" on dark scenery |
| `lines-exp-bright-wall.png` | `z0042-before` | "You received EXP (+10)" on a white wall |
| `lines-meso-over-exp.png` | `z0037-before` | "You have gained mesos (+12)", faded EXP line above |
| `lines-item-over-meso.png` | `z0054-after1200ms` | "Tree Branch x1 earned. (Etc)", faded meso line above |
| `lines-meso-exp-item.png` | `z0058-after3000ms` | meso, EXP, faded item line |
| `lines-decay-0.png` .. `lines-decay-4.png` | `z0003-after1200ms`, `z0004-after450ms`, `z0004-after1200ms`, `z0003-after3000ms`, `z0005-after450ms` | one "You received EXP (+15)" line fading out (79, 64, 37, 19, 8 outlined-white pixels) |
| `lines-shift-before.png`, `lines-shift-after.png` | `z0037-after450ms`, `z0037-after1200ms` | a meso line, then an EXP line that pushed it one slot up |
| `lines-repeat-before.png`, `lines-repeat-after.png` | `combat-frame-27`, `combat-frame-28` | "You received EXP (+4)", then the same text again below it |
| `lines-double-before.png`, `lines-double-after.png` | `combat-frame-22`, `combat-frame-23` | two EXP lines, then a meso and a "Red Potion x1 earned. (Use)" line arrived between captures |

No recorded pair shows one fading line over scenery that changes brightness between captures.
`tests/test_gains.py::_faded` models it instead: the text and outline pixels of `lines-exp.png`
alpha-blended over flat grey 30 and grey 110. The "same text, old line moved up" test rearranges
real slots of `lines-decay-4.png`, `lines-exp.png` and `lines-empty.png`; it is not a recorded pair either.

## Templates under `assets/`

`assets/gain-line-exp-prefix.png` is `[x=1231, y=605, w=100, h=14]` of `z0023-after450ms`
("You received EXP ") and `assets/gain-line-meso-prefix.png` is `[x=1194, y=605, w=129, h=14]`
of `z0058-after3000ms` ("You have gained mesos "). `config/default.yaml` (`evidence.templates`) refers
to them; they match this client's font and UI scale only.
