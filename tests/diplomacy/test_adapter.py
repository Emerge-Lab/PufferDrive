"""Basic tests for adapters bridging C implementation to original API expectations."""

from tests.diplomacy.adapters import GameAdapter, MapAdapter


def test_game_adapter_basics():
    game = GameAdapter()
    assert game.get_year() == 1901
    assert game.get_phase().startswith("SPRING")


def test_map_adapter_location_and_sc():
    m = MapAdapter()
    par = m.get_location("PAR")
    assert par is not None
    assert m.is_supply_center("PAR") is True


def test_map_adapter_adjacency():
    m = MapAdapter()
    adj = m.get_adjacencies("PAR", "A")
    assert "BUR" in adj
    assert "ENG" not in adj



