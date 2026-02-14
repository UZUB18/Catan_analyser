from click.testing import CliRunner

from catanatron.cli.cli_players import parse_cli_string
from catanatron.cli.play import simulate


def test_parse_cli_string_builds_players() -> None:
    players = parse_cli_string("R,W,VP,F")
    assert len(players) == 4
    assert [player.color.value for player in players] == ["RED", "BLUE", "ORANGE", "WHITE"]


def test_catanatron_cli_help_players() -> None:
    runner = CliRunner()
    result = runner.invoke(simulate, ["--help-players"])
    assert result.exit_code == 0
    assert "RandomPlayer" in result.output
