"""Test C order parsing and validation.

This module tests that orders can be correctly parsed from strings
and validated against the current game state.
"""

import pytest
from pufferlib.ocean.diplomacy import Diplomacy, binding


class TestCOrderParsing:
    """Tests for C order parsing."""

    def test_parse_hold_order(self):
        """Test parsing HOLD orders."""
        env = Diplomacy()

        # Test "A PAR H"
        result = binding.test_parse_order(env.env_handle, "A PAR H")
        assert result is not None
        assert result["type"] == 1  # ORDER_HOLD
        assert result["unit_type"] == 1  # UNIT_ARMY
        assert result["unit_location"] == 46  # PAR

    def test_parse_move_order(self):
        """Test parsing MOVE orders."""
        env = Diplomacy()

        # Test "A PAR - BUR"
        result = binding.test_parse_order(env.env_handle, "A PAR - BUR")
        assert result is not None
        assert result["type"] == 2  # ORDER_MOVE
        assert result["unit_type"] == 1  # UNIT_ARMY
        assert result["unit_location"] == 46  # PAR
        assert result["target_location"] == 16  # BUR

    def test_parse_support_hold_order(self):
        """Test parsing SUPPORT HOLD orders."""
        env = Diplomacy()

        # Test "A PAR S A MAR"
        result = binding.test_parse_order(env.env_handle, "A PAR S A MAR")
        assert result is not None
        assert result["type"] == 3  # ORDER_SUPPORT_HOLD
        assert result["unit_type"] == 1  # UNIT_ARMY
        assert result["target_unit_location"] == 37  # MAR

    def test_parse_support_move_order(self):
        """Test parsing SUPPORT MOVE orders."""
        env = Diplomacy()

        # Test "A PAR S A MAR - BUR"
        result = binding.test_parse_order(env.env_handle, "A PAR S A MAR - BUR")
        assert result is not None
        assert result["type"] == 4  # ORDER_SUPPORT_MOVE
        assert result["unit_type"] == 1  # UNIT_ARMY
        assert result["target_unit_location"] == 37  # MAR
        assert result["dest_location"] == 16  # BUR

    def test_parse_convoy_order(self):
        """Test parsing CONVOY orders."""
        env = Diplomacy()

        # Test "F ENG C A WAL - BRE"
        result = binding.test_parse_order(env.env_handle, "F ENG C A WAL - BRE")
        assert result is not None
        assert result["type"] == 5  # ORDER_CONVOY
        assert result["unit_type"] == 2  # UNIT_FLEET

    def test_parse_fleet_order(self):
        """Test parsing fleet orders."""
        env = Diplomacy()

        # Test "F BRE - MAO"
        result = binding.test_parse_order(env.env_handle, "F BRE - MAO")
        assert result is not None
        assert result["type"] == 2  # ORDER_MOVE
        assert result["unit_type"] == 2  # UNIT_FLEET

    def test_parse_invalid_location(self):
        """Test parsing with invalid location returns None."""
        env = Diplomacy()

        result = binding.test_parse_order(env.env_handle, "A XXX H")
        assert result is None

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        env = Diplomacy()

        # Lowercase
        result1 = binding.test_parse_order(env.env_handle, "a par h")
        # Uppercase
        result2 = binding.test_parse_order(env.env_handle, "A PAR H")
        # Mixed
        result3 = binding.test_parse_order(env.env_handle, "A Par H")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1["type"] == result2["type"] == result3["type"]


class TestCOrderValidation:
    """Tests for C order validation."""

    def test_validate_legal_move(self):
        """Test that a legal move passes validation."""
        env = Diplomacy()

        # France (power_id=2) has A PAR at start
        # PAR can move to BUR (adjacent land)
        result = binding.test_validate_order(env.env_handle, 2, "A PAR - BUR")
        assert result == 0  # Valid

    def test_validate_hold_always_valid(self):
        """Test that HOLD is always valid for owned units."""
        env = Diplomacy()

        # France has A PAR
        result = binding.test_validate_order(env.env_handle, 2, "A PAR H")
        assert result == 0  # Valid

    def test_validate_unit_not_owned(self):
        """Test that ordering another power's unit fails."""
        env = Diplomacy()

        # France trying to order Germany's unit at MUN
        result = binding.test_validate_order(env.env_handle, 2, "A MUN H")
        assert result == -1  # Invalid

    def test_validate_non_adjacent_move(self):
        """Test that moving to non-adjacent location fails."""
        env = Diplomacy()

        # France A PAR trying to move to ROM (not adjacent)
        result = binding.test_validate_order(env.env_handle, 2, "A PAR - ROM")
        assert result == -1  # Invalid

    def test_validate_army_cant_move_to_water(self):
        """Test that army can't move to pure water."""
        env = Diplomacy()

        # France A PAR trying to move to ENG (water)
        result = binding.test_validate_order(env.env_handle, 2, "A PAR - ENG")
        assert result == -1  # Invalid (army can't go to water)

    def test_validate_fleet_legal_move(self):
        """Test that fleet can move to adjacent water/coast."""
        env = Diplomacy()

        # France F BRE can move to MAO (adjacent water)
        result = binding.test_validate_order(env.env_handle, 2, "F BRE - MAO")
        assert result == 0  # Valid

    def test_validate_support_adjacent(self):
        """Test that support requires adjacency."""
        env = Diplomacy()

        # France A PAR supporting A MAR - BUR
        # PAR is adjacent to BUR, so this should be valid
        result = binding.test_validate_order(env.env_handle, 2, "A PAR S A MAR - BUR")
        assert result == 0  # Valid

    def test_validate_wrong_unit_type(self):
        """Test that unit type mismatch fails validation."""
        env = Diplomacy()

        # France has A PAR (army), not fleet
        result = binding.test_validate_order(env.env_handle, 2, "F PAR H")
        assert result == -1  # Invalid (wrong unit type)
