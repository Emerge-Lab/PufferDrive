"""Test C game state management.

This module tests that the game state is correctly initialized,
updated, and managed throughout gameplay.
"""

import pytest
from pufferlib.ocean.diplomacy import Diplomacy


class TestCGameState:
    """Tests for C game state management."""

    def test_initial_phase(self):
        """Test that the game starts in Spring 1901 Movement phase."""
        env = Diplomacy()
        obs, info = env.reset()
        # TODO: Add API to query current phase from C
        # assert phase == "SPRING 1901 MOVEMENT"

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_initial_year(self):
        """Test that the game starts in year 1901."""
        pass

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_initial_units(self):
        """Test that each power starts with correct units."""
        # Austria: F TRI, A VIE, A BUD
        # England: F LON, F EDI, A LVP
        # France: F BRE, A PAR, A MAR
        # Germany: F KIE, A BER, A MUN
        # Italy: F NAP, A ROM, A VEN
        # Russia: F SEV, F STP(sc), A MOS, A WAR
        # Turkey: F ANK, A CON, A SMY
        pass

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_phase_progression(self):
        """Test that phases progress correctly."""
        # Spring Movement -> Spring Retreat -> Fall Movement -> Fall Retreat -> Winter Adjustment
        pass

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_year_progression(self):
        """Test that year increments after winter adjustment."""
        pass

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_unit_creation(self):
        """Test that units can be created during adjustment phase."""
        pass

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_unit_removal(self):
        """Test that units can be removed during adjustment phase."""
        pass

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_supply_center_ownership(self):
        """Test that supply center ownership is tracked correctly."""
        pass

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_game_over_max_years(self):
        """Test that game ends after max_years."""
        env = Diplomacy(max_years=1)
        # Play through to end of 1901
        # Assert game is over
        pass

    @pytest.mark.skip(reason="Game state API not yet implemented")
    def test_dislodged_units(self):
        """Test that dislodged units are tracked correctly."""
        pass
