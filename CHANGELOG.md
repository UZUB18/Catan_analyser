# Changelog

All notable project updates are tracked here.

## 2026-02-13

### Added
- Integrated three new Catanatron engines into the built-in CLI player registry:
  - `GT` -> `GameTheoryEngine`
  - `STAT` -> `StatsEngine`
  - `WILD` -> `WildSheepCultEngine`
- Added `catanatron/players/three_engines.py` to host those engine implementations inside this repo.

### Changed
- Updated the app menu action **Open Catanatron CLI** to launch the local repo implementation via:
  - `python -m catanatron.cli.play`
  - local `PYTHONPATH` injection for the session
  - shell helper functions (`catanatron`, `catanatron-play`) mapped to the local module
- Updated CLI startup output in the launched terminal to show `--help-players` and example matchups including `GT/STAT/WILD`.
- Improved player parsing to support both comma-separated and PowerShell space-expanded player code input.

### Fixed
- Fixed compatibility in `three_engines` leader detection for different `get_enemy_colors(...)` signatures across Catanatron variants.
- Replaced silent/implicit player parse failures with explicit validation errors:
  - unknown engine codes
  - too many players
  - bad player initialization arguments
- Converted parse errors to proper Click `--players` parameter errors for clearer CLI feedback.

### Extensibility Notes
- To add future engines:
  1. add the engine class under `catanatron/players/`
  2. register a new `CliPlayer(...)` entry in `catanatron/cli/cli_players.py`
  3. (optional) expose via `--code=...` for temporary/experimental engines.
