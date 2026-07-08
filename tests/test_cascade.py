import time


def _cascade(app_modules):
    return app_modules["app.state"].cascade


def test_order_returns_full_ladder_plus_local_by_default(app_modules):
    cascade = _cascade(app_modules)
    ladder = cascade.order()
    assert ladder[:-1] == cascade._serving_ladder()
    assert ladder[-1] == cascade._local_model()


def test_cool_sidelines_a_model_from_the_live_order(app_modules):
    cascade = _cascade(app_modules)
    base = cascade._serving_ladder()
    target = base[0]
    cascade.cool(target, 300)
    ladder = cascade.order()
    assert target not in ladder[:-1]  # still absent from the cloud portion
    assert ladder[-1] == cascade._local_model()  # local tail untouched


def test_cool_never_shortens_an_existing_longer_cooldown(app_modules):
    cascade = _cascade(app_modules)
    model = cascade._serving_ladder()[0]
    cascade.cool(model, 300)
    long_until = cascade.model_until[model]
    cascade.cool(model, 5)  # shorter cooldown must not override the longer one
    assert cascade.model_until[model] == long_until


def test_reset_cooldowns_clears_state_and_reports_count(app_modules):
    cascade = _cascade(app_modules)
    base = cascade._serving_ladder()
    cascade.cool(base[0], 300)
    cascade.dead.add(base[1])
    cleared = cascade.reset_cooldowns()
    assert cleared == 2
    assert cascade.model_until == {}
    assert cascade.dead == set()


def test_sticky_cursor_rotates_live_models_to_front(app_modules):
    cascade = _cascade(app_modules)
    base = cascade._serving_ladder()
    second = base[1]
    cascade.stats.note_serving(second)
    ladder = cascade.order()
    assert ladder[0] == second


def test_dead_model_is_excluded_from_order(app_modules):
    cascade = _cascade(app_modules)
    dead_model = cascade._serving_ladder()[0]
    cascade.dead.add(dead_model)
    ladder = cascade.order()
    assert dead_model not in ladder


def test_preferred_known_model_moves_to_front(app_modules):
    cascade = _cascade(app_modules)
    base = cascade._serving_ladder()
    preferred = base[2]
    ladder = cascade.order(preferred)
    assert ladder[0] == preferred


def test_local_only_mode_returns_just_the_local_model(app_modules):
    cascade = _cascade(app_modules)
    from app.config import LOCAL_ONLY

    ladder = cascade.order(LOCAL_ONLY)
    assert ladder == [cascade._local_model()]


def test_note_status_429_cools_and_records(app_modules):
    cascade = _cascade(app_modules)
    model = cascade._serving_ladder()[0]
    cascade.note_status(model, 429)
    assert model in cascade.model_until
    assert cascade.stats._m(model)["rate_limited"] == 1


def test_note_status_404_marks_dead(app_modules):
    cascade = _cascade(app_modules)
    model = cascade._serving_ladder()[0]
    cascade.note_status(model, 404)
    assert model in cascade.dead


def test_soonest_cooldown_reflects_wall_clock(app_modules):
    cascade = _cascade(app_modules)
    model = cascade._serving_ladder()[0]
    cascade.cool(model, 30)
    remaining = cascade.soonest_cooldown()
    assert remaining is not None
    assert 0 < remaining <= 30
