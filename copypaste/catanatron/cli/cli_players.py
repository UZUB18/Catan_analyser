from collections import namedtuple

from rich.table import Table

from catanatron.models.player import Color, HumanPlayer, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.game_theory_engine import GameTheoryEngine
from catanatron.players.three_engines import (
    StatsEngine,
    WildSheepCultEngine,
)


# Player must have a CODE, NAME, DESCRIPTION, CLASS.
CliPlayer = namedtuple("CliPlayer", ["code", "name", "description", "import_fn"])
CLI_PLAYERS = [
    CliPlayer(
        "H", "HumanPlayer", "Human player, uses input() to get action.", HumanPlayer
    ),
    CliPlayer("R", "RandomPlayer", "Chooses actions at random.", RandomPlayer),
    CliPlayer(
        "W",
        "WeightedRandomPlayer",
        "Like RandomPlayer, but favors buying cities, settlements, and dev cards when possible.",
        WeightedRandomPlayer,
    ),
    CliPlayer(
        "VP",
        "VictoryPointPlayer",
        "Chooses randomly from actions that increase victory points immediately if possible, else at random.",
        VictoryPointPlayer,
    ),
    CliPlayer(
        "G",
        "GreedyPlayoutsPlayer",
        "For each action, will play N random 'playouts'. "
        + "Takes the action that led to best winning percent. "
        + "First param is NUM_PLAYOUTS",
        GreedyPlayoutsPlayer,
    ),
    CliPlayer(
        "M",
        "MCTSPlayer",
        "Decides according to the MCTS algorithm. First param is NUM_SIMULATIONS.",
        MCTSPlayer,
    ),
    CliPlayer(
        "F",
        "ValueFunctionPlayer",
        "Chooses the action that leads to the most immediate reward, based on a hand-crafted value function.",
        ValueFunctionPlayer,
    ),
    CliPlayer(
        "AB",
        "AlphaBetaPlayer",
        "Implements alpha-beta algorithm. That is, looks ahead a couple "
        + "levels deep evaluating leafs with hand-crafted value function. "
        + "Params are DEPTH, PRUNNING",
        AlphaBetaPlayer,
    ),
    CliPlayer(
        "SAB",
        "SameTurnAlphaBetaPlayer",
        "AlphaBeta but searches only within turn",
        SameTurnAlphaBetaPlayer,
    ),
    CliPlayer(
        "GT",
        "GameTheoryEngine",
        "Opponent-aware utility engine that balances self-gain with leader denial.",
        GameTheoryEngine,
    ),
    CliPlayer(
        "STAT",
        "StatsEngine",
        "Dice-probability and expected-value driven engine with risk-aware choices.",
        StatsEngine,
    ),
    CliPlayer(
        "WILD",
        "WildSheepCultEngine",
        "Sheep-focused eccentric engine with a practical safety floor.",
        WildSheepCultEngine,
    ),
]


def parse_cli_string(player_string):
    players = []
    normalized = player_string.strip()
    if "," in normalized:
        player_keys = [key.strip() for key in normalized.split(",") if key.strip()]
    else:
        # PowerShell can expand comma-delimited values passed as
        # "--players A,B,C" into space-delimited strings.
        player_keys = [key.strip() for key in normalized.split() if key.strip()]
    if not player_keys:
        raise ValueError("No player codes provided. Use --players like R,R,R,R.")
    colors = [c for c in Color]
    if len(player_keys) > len(colors):
        raise ValueError(
            f"Too many players ({len(player_keys)}). Maximum supported is {len(colors)}."
        )

    players_by_code = {cli_player.code: cli_player for cli_player in CLI_PLAYERS}
    for i, key in enumerate(player_keys):
        parts = key.split(":")
        code = parts[0]
        cli_player = players_by_code.get(code)
        if cli_player is None:
            available = ",".join(player.code for player in CLI_PLAYERS)
            raise ValueError(
                f"Unknown player code '{code}'. Use --help-players. Available: {available}"
            )
        params = [colors[i]] + parts[1:]
        try:
            player = cli_player.import_fn(*params)
        except Exception as exc:
            raise ValueError(
                f"Failed to initialize player '{code}' ({cli_player.name}): {exc}"
            ) from exc
        players.append(player)
    return players


def register_cli_player(code, player_class):
    CLI_PLAYERS.append(
        CliPlayer(
            code,
            player_class.__name__,
            player_class.__doc__,
            player_class,
        ),
    )


CUSTOM_ACCUMULATORS = []


def register_cli_accumulator(accumulator_class):
    CUSTOM_ACCUMULATORS.append(accumulator_class)


def player_help_table():
    table = Table(title="Player Legend")
    table.add_column("CODE", justify="center", style="cyan", no_wrap=True)
    table.add_column("PLAYER")
    table.add_column("DESCRIPTION")
    for player in CLI_PLAYERS:
        table.add_row(player.code, player.name, player.description)
    return table
