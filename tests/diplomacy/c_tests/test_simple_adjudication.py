"""Test simple order adjudication scenarios.

These tests build up from the simplest possible scenarios to more complex ones,
allowing incremental development of the adjudication engine.
"""

import pytest
from tests.diplomacy.adapters import GameAdapter


class TestSimpleMovement:
    """Test basic movement without support or convoy."""

    def test_single_uncontested_move(self):
        """A single unit moving to an empty adjacent location should succeed."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('FRANCE', [('A', 'PAR')])
        
        # PAR is adjacent to BUR
        game.set_orders('FRANCE', ['A PAR - BUR'])
        game.process()
        
        # Unit should have moved
        units = game.get_units('FRANCE')
        assert 'A BUR' in units
        assert 'A PAR' not in units

    def test_hold_order(self):
        """A unit with hold order should stay in place."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('FRANCE', [('A', 'PAR')])
        
        game.set_orders('FRANCE', ['A PAR H'])
        game.process()
        
        # Unit should still be in Paris
        units = game.get_units('FRANCE')
        assert 'A PAR' in units

    def test_invalid_move_non_adjacent(self):
        """A unit trying to move to a non-adjacent location should fail (stay in place)."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('FRANCE', [('A', 'PAR')])
        
        # PAR is not adjacent to ROM
        game.set_orders('FRANCE', ['A PAR - ROM'])
        game.process()
        
        # Unit should still be in Paris (move failed)
        units = game.get_units('FRANCE')
        assert 'A PAR' in units
        assert 'A ROM' not in units

    def test_army_cannot_move_to_water(self):
        """An army trying to move to water should fail."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('FRANCE', [('A', 'BRE')])
        
        # BRE is coastal but MAO is pure water
        game.set_orders('FRANCE', ['A BRE - MAO'])
        game.process()
        
        # Unit should still be in Brest
        units = game.get_units('FRANCE')
        assert 'A BRE' in units
        assert 'A MAO' not in units

    def test_fleet_cannot_move_to_land(self):
        """A fleet trying to move to pure land should fail."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('GERMANY', [('F', 'KIE')])
        
        # MUN is pure land
        game.set_orders('GERMANY', ['F KIE - MUN'])
        game.process()
        
        # Unit should still be in Kiel
        units = game.get_units('GERMANY')
        assert 'F KIE' in units
        assert 'F MUN' not in units

    def test_head_to_head_bounce(self):
        """Two units moving to the same location should bounce."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('ITALY', [('A', 'VEN')])
        game.set_units('AUSTRIA', [('A', 'VIE')])
        
        # Both moving to TYR
        game.set_orders('ITALY', ['A VEN - TYR'])
        game.set_orders('AUSTRIA', ['A VIE - TYR'])
        game.process()
        
        # Both units should bounce and stay in place
        italy_units = game.get_units('ITALY')
        austria_units = game.get_units('AUSTRIA')
        assert 'A VEN' in italy_units
        assert 'A VIE' in austria_units
        assert 'A TYR' not in italy_units
        assert 'A TYR' not in austria_units

    def test_three_way_bounce(self):
        """Three units moving to the same location should all bounce."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('AUSTRIA', [('A', 'VIE')])
        game.set_units('GERMANY', [('A', 'MUN')])
        game.set_units('ITALY', [('A', 'VEN')])
        
        # All three moving to TYR
        game.set_orders('AUSTRIA', ['A VIE - TYR'])
        game.set_orders('GERMANY', ['A MUN - TYR'])
        game.set_orders('ITALY', ['A VEN - TYR'])
        game.process()
        
        # All units should bounce
        assert 'A VIE' in game.get_units('AUSTRIA')
        assert 'A MUN' in game.get_units('GERMANY')
        assert 'A VEN' in game.get_units('ITALY')
        assert 'A TYR' not in game.get_units('AUSTRIA')

    def test_unit_swap_blocked(self):
        """Two units trying to swap places without convoy should bounce."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('GERMANY', [('A', 'BER'), ('F', 'KIE')])
        
        # Trying to swap without convoy
        game.set_orders('GERMANY', ['A BER - KIE', 'F KIE - BER'])
        game.process()
        
        # Both should stay in place (head-to-head)
        units = game.get_units('GERMANY')
        assert 'A BER' in units
        assert 'F KIE' in units

    def test_following_unit(self):
        """A unit can move into a space being vacated."""
        game = GameAdapter()
        game.clear_units()
        # Use actually adjacent locations: PAR-BUR-MAR is not a valid chain
        # Use VIE-BUD-TRI instead (all adjacent)
        game.set_units('AUSTRIA', [('A', 'VIE'), ('A', 'TRI')])
        
        # VIE moves to BUD, TRI moves to VIE (chain)
        game.set_orders('AUSTRIA', ['A VIE - BUD', 'A TRI - VIE'])
        game.process()
        
        # Both moves should succeed
        units = game.get_units('AUSTRIA')
        assert 'A BUD' in units
        assert 'A VIE' in units
        assert 'A TRI' not in units


class TestSimpleSupport:
    """Test basic support mechanics (to be implemented)."""

    @pytest.mark.skip(reason="Support adjudication not yet implemented")
    def test_supported_attack(self):
        """A supported attack should dislodge a defender."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('ITALY', [('A', 'VEN'), ('A', 'TYR')])
        game.set_units('AUSTRIA', [('F', 'TRI')])
        
        # Attack with support
        game.set_orders('ITALY', ['A VEN - TRI', 'A TYR S A VEN - TRI'])
        game.set_orders('AUSTRIA', ['F TRI H'])
        game.process()
        
        # Italian army should dislodge Austrian fleet
        italy_units = game.get_units('ITALY')
        austria_units = game.get_units('AUSTRIA')
        assert 'A TRI' in italy_units
        assert 'F TRI' not in austria_units

    @pytest.mark.skip(reason="Support adjudication not yet implemented")
    def test_support_prevents_dislodgement(self):
        """Support to hold should prevent dislodgement."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('AUSTRIA', [('A', 'TRI')])
        game.set_units('ITALY', [('A', 'VEN'), ('A', 'TYR')])
        
        # Attack with support vs hold with support
        game.set_orders('AUSTRIA', ['A TRI - VEN', 'Nothing here'])  # Just moving TRI
        game.set_orders('ITALY', ['A VEN H', 'A TYR S A VEN'])
        game.process()
        
        # Austrian attack should bounce
        assert 'A TRI' in game.get_units('AUSTRIA')
        assert 'A VEN' in game.get_units('ITALY')


class TestCircularMovement:
    """Test circular movement (to be implemented)."""

    @pytest.mark.skip(reason="Circular movement not yet implemented")
    def test_simple_circular_movement(self):
        """Three units in a circle should all move."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('TURKEY', [('F', 'ANK'), ('A', 'CON'), ('A', 'SMY')])
        
        # Circular movement: ANK->CON->SMY->ANK
        game.set_orders('TURKEY', ['F ANK - CON', 'A CON - SMY', 'A SMY - ANK'])
        game.process()
        
        # All should move
        units = game.get_units('TURKEY')
        assert 'F CON' in units
        assert 'A SMY' in units
        assert 'A ANK' in units


class TestOrderValidation:
    """Test that invalid orders are caught."""

    def test_ordering_another_powers_unit(self):
        """Cannot order another power's units."""
        game = GameAdapter()
        game.clear_units()
        game.set_units('ENGLAND', [('F', 'LON')])
        
        # Germany tries to order England's unit
        game.set_orders('GERMANY', ['F LON - NTH'])
        game.process()
        
        # English unit should stay in place
        assert 'F LON' in game.get_units('ENGLAND')
        assert 'F NTH' not in game.get_units('ENGLAND')
        assert 'F NTH' not in game.get_units('GERMANY')
