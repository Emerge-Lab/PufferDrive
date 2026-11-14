"""Workers of one cache domain train on a share of the maps; the shares must partition the maps without bias."""

import itertools

from pufferlib.ocean.drive.drive import cache_domain_map_files


TOWN_COUNT = 8
VARIANTS_PER_TOWN = 16
MAP_FILES = [
    f"town{town:02d}_variant{variant:02d}.bin" for town in range(TOWN_COUNT) for variant in range(VARIANTS_PER_TOWN)
]


def test_shares_partition_the_maps_in_proportion_and_cover_every_town():
    share_bounds = [0.0, 0.296875, 0.59375, 0.875, 1.0]
    domain_maps = [cache_domain_map_files(MAP_FILES, share) for share in itertools.pairwise(share_bounds)]
    assert [len(maps) for maps in domain_maps] == [38, 38, 36, 16]
    assert sorted(map_file for maps in domain_maps for map_file in maps) == MAP_FILES
    for maps in domain_maps:
        assert len({map_file.split("_")[0] for map_file in maps}) == TOWN_COUNT


def test_whole_share_keeps_every_map_and_a_tiny_map_set_leaves_no_domain_empty():
    assert cache_domain_map_files(MAP_FILES, (0.0, 1.0)) == MAP_FILES
    two_maps = MAP_FILES[:2]
    quarters = [cache_domain_map_files(two_maps, (start / 4, (start + 1) / 4)) for start in range(4)]
    assert all(len(maps) == 1 for maps in quarters)
    assert {map_file for maps in quarters for map_file in maps} == set(two_maps)
