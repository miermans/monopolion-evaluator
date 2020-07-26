
from monopolion_evaluator.protobuf.reader import parse_delimited_file


def test_parse_delimited_file():
    # 0 0 1
    game_outcomes = parse_delimited_file('fixtures/toy_data_2player.gz')
    for game_outcome in game_outcomes:
        assert game_outcome.winningPlayer in [0, 1]

        assert len(game_outcome.gameState.players) == 2
        for player in game_outcome.gameState.players:
            assert 0 <= player.cash <= 10 ** 4

        assert len(game_outcome.gameState.propertyStates) == 28
        for property_state in game_outcome.gameState.propertyStates:
            if property_state.isOwned:
                assert property_state.owner in [0, 1]
                assert 0 <= property_state.buildingCount <= 5

