from maplebot.config import deep_merge, load_config, parse_color_codes


def test_deep_merge_nested_override_and_new_key():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    out = deep_merge(base, {"a": {"y": 3, "z": 4}, "c": 5})
    assert out == {"a": {"x": 1, "y": 3, "z": 4}, "b": 1, "c": 5}
    assert base["a"]["y"] == 2


def test_load_config_defaults_have_required_sections():
    cfg = load_config()
    for key in ("system", "game_window", "keys", "player", "minimap", "monsters",
                "attack", "route", "health", "stuck", "search"):
        assert key in cfg, key


def test_load_config_user_override(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text("system:\n  fps_main: 5\n")
    assert load_config(str(p))["system"]["fps_main"] == 5


def test_parse_color_codes():
    assert parse_color_codes({"255,0,0": "left none none"}) == {(255, 0, 0): "left none none"}


def test_default_search_settings_do_not_jitter():
    cfg = load_config()
    assert cfg["search"]["stall_frames"] == 0
    assert cfg["search"]["attack_while_moving"] is False
