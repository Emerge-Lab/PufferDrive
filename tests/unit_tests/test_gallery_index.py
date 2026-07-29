from pathlib import Path

import pufferlib.viz


def _write_replay_pages(folder_path: Path, filenames):
    for filename in filenames:
        (folder_path / filename).write_text("<html></html>")


def test_gallery_index_builds_inclusive_failure_filters(tmp_path):
    filenames = ["both.html", "clean.html", "collision.html", "offroad.html"]
    _write_replay_pages(tmp_path, filenames)
    file_metrics = {
        "both.html": {
            "score": 0.1,
            "offroad_rate": 1.0,
            "collision_rate": 1.0,
            "at_fault_collision_rate": 1.0,
            "red_light_violation_rate": 1.0,
        },
        "clean.html": {
            "score": 0.9,
            "offroad_rate": 0.0,
            "collision_rate": 0.0,
            "at_fault_collision_rate": 0.0,
            "red_light_violation_rate": 0.0,
        },
        "collision.html": {
            "score": 0.2,
            "offroad_rate": 0.0,
            "collision_rate": 1.0,
            "at_fault_collision_rate": 1.0,
            "red_light_violation_rate": 0.0,
        },
        "offroad.html": {
            "score": 0.3,
            "offroad_rate": 1.0,
            "collision_rate": 0.0,
            "at_fault_collision_rate": 0.0,
            "red_light_violation_rate": 0.0,
        },
    }

    pufferlib.viz.build_gallery_index(tmp_path, file_metrics=file_metrics)

    index_html = (tmp_path / "index.html").read_text()
    assert 'id="failureFilters"' not in index_html
    assert 'data-filter="all"' in index_html
    assert 'data-filter="offroad"' in index_html
    assert 'data-filter="collision"' in index_html
    assert 'data-filter="atfault"' in index_html
    assert 'data-filter="redlight"' in index_html
    assert '<span class="category-count">4</span>' in index_html
    assert index_html.count('<span class="category-count">2</span>') == 3
    assert '<span class="category-count">1</span>' in index_html
    assert '<nav class="category-bar" aria-label="Replay category">' in index_html
    assert index_html.index('class="category-bar"') < index_html.index('class="scenario-header"')
    assert "Show category" not in index_html
    assert "category-button is-active" in index_html
    assert '<span class="selection-mark" aria-hidden="true">&#10003;</span>' in index_html
    assert '<header class="top-header">' in index_html
    assert '<aside class="sidebar">' not in index_html
    assert "Global overview" not in index_html
    assert 'class="global-stats"' not in index_html
    top_header_start = index_html.index('<header class="top-header">')
    top_header_end = index_html.index("</header>", top_header_start)
    category_bar_position = index_html.index('class="category-bar"')
    browse_controls_position = index_html.index('class="browse-controls"')
    assert top_header_start < category_bar_position < top_header_end
    assert category_bar_position < browse_controls_position < top_header_end
    assert (
        '<option value="both.html" data-name="both.html" data-offroad="true" '
        'data-collision="true" data-atfault="true" data-redlight="true">'
    ) in index_html
    assert 'data-offroad="false" data-collision="false" data-atfault="false" data-redlight="false"' in index_html
    assert "const allOptions = Array.from(select.options);" in index_html
    assert "const matchingOptions = allOptions.filter(optionMatchesActiveFilter);" in index_html
    assert "compareOptions" not in index_html
    assert "const nextIndex = select.selectedIndex + direction;" in index_html
    assert '<main class="replay-stage">' in index_html
    assert 'id="currentReplayName"' in index_html
    assert "Selected replay" in index_html
    assert "function updateScenarioSummary()" in index_html
    assert 'id="sortKey"' not in index_html
    assert 'id="sortDir"' not in index_html
    assert 'id="currentMetrics"' not in index_html
    assert "SUMMARY_METRICS" not in index_html
    assert "Score 0.10" not in index_html
    assert "Replay index" in index_html
    assert "linear-gradient" not in index_html
    assert "radial-gradient" not in index_html
    assert "box-shadow:" not in index_html
    assert "backdrop-filter" not in index_html
    assert "transform: translate" not in index_html


def test_gallery_index_disables_empty_failure_filter(tmp_path):
    _write_replay_pages(tmp_path, ["offroad.html"])
    file_metrics = {
        "offroad.html": {"offroad_rate": 1.0, "collision_rate": 0.0},
    }

    pufferlib.viz.build_gallery_index(tmp_path, file_metrics=file_metrics)

    index_html = (tmp_path / "index.html").read_text()
    assert 'data-filter="offroad" aria-pressed="false">' in index_html
    assert 'data-filter="collision" aria-pressed="false" disabled>' in index_html
    assert 'data-filter="all" aria-pressed="true"' in index_html


def test_gallery_index_without_metrics_keeps_simple_navigation(tmp_path):
    _write_replay_pages(tmp_path, ["episode_000001.html", "episode_000000.html"])

    pufferlib.viz.build_gallery_index(tmp_path)

    index_html = (tmp_path / "index.html").read_text()
    assert 'id="failureFilters"' not in index_html
    assert 'id="sortKey"' not in index_html
    assert 'id="sortDir"' not in index_html
    assert "Global overview" not in index_html
    assert 'src="episode_000000.html"' in index_html
    assert '<option value="episode_000000.html"' in index_html
    assert '<option value="episode_000001.html"' in index_html
