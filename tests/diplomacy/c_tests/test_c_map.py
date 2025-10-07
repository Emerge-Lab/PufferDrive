"""Test C map data and adjacency calculations.

This module tests that the map data loaded from the C implementation
matches expected values and that adjacency calculations are correct.
"""

import pytest


class TestCMap:
    """Tests for C map data and adjacencies."""

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_map_has_76_locations(self):
        """Test that the map has exactly 76 locations."""
        # TODO: Add API to access map data from C
        pass

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_location_names(self):
        """Test that location names are correctly loaded."""
        # Known locations from standard Diplomacy map
        expected_locations = [
            "ADR", "AEG", "ALB", "ANK", "APU", "ARM",
            "BAL", "BAR", "BEL", "BER", "BLA", "BOH",
            # ... etc
        ]
        # TODO: Verify location names match
        pass

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_location_types(self):
        """Test that location types are correctly set."""
        # Examples:
        # - BOH (Bohemia) should be LAND
        # - BAL (Baltic Sea) should be WATER
        # - BEL (Belgium) should be COAST
        pass

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_supply_centers(self):
        """Test that supply centers are correctly marked."""
        # Standard Diplomacy has 34 supply centers
        pass

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_home_centers(self):
        """Test that home centers are correctly assigned."""
        # Austria: BUD, TRI, VIE
        # England: EDI, LON, LVP
        # France: BRE, MAR, PAR
        # Germany: BER, KIE, MUN
        # Italy: NAP, ROM, VEN
        # Russia: MOS, SEV, STP, WAR
        # Turkey: ANK, CON, SMY
        pass

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_adjacency_cache_army(self):
        """Test that army movement adjacencies are correct."""
        # Test some known adjacencies:
        # - Paris (land) should be adjacent to Burgundy (land)
        # - Paris should NOT be adjacent to English Channel (water)
        pass

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_adjacency_cache_fleet(self):
        """Test that fleet movement adjacencies are correct."""
        # Test some known adjacencies:
        # - English Channel should be adjacent to London (coast)
        # - English Channel should NOT be adjacent to Paris (land)
        pass

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_specific_adjacencies(self):
        """Test specific known adjacencies from standard map."""
        # Test cases from DATC:
        # - Belgium is adjacent to: BUR, ENG, HOL, NTH, PIC, RUH
        # - Munich is adjacent to: BER, BOH, BUR, KIE, RUH, SIL, TYR, SWI
        pass

    @pytest.mark.skip(reason="C map access API not yet implemented")
    def test_coast_adjacencies(self):
        """Test that coastal locations have both land and sea adjacencies."""
        # Constantinople should be adjacent to:
        # - Land: ANK, BUL
        # - Sea: AEG, BLA
        pass
