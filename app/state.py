"""Process-wide singletons: the ladder config, stats tracker, and cascade
router. Imported by every route module instead of each re-instantiating."""

from app.ladder import ladder_config
from app.stats import Stats
from app.cascade import Cascade

stats = Stats.load()
cascade = Cascade(stats)

_model_override = None


def get_model_override():
    return _model_override


def set_model_override(model):
    global _model_override
    _model_override = model
