import pytest

from rfdetr.util.metrics import MetricsTensorBoardSink


class DummyWriter:
    """Minimal stand-in for torch.utils.tensorboard.SummaryWriter (no real I/O)."""
    def __init__(self):
        self.scalars = []   # list[(tag, value, step)]
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


@pytest.fixture
def sink(tmp_path):
    """MetricsTensorBoardSink with a dummy writer injected."""
    s = MetricsTensorBoardSink(output_dir=str(tmp_path))
    s.writer = DummyWriter()
    return s


def test_no_per_class_when_absent(sink):
    """No per-class tags when results_json keys are missing."""
    sink.update({"epoch": 1, "train_loss": 0.5, "test_loss": 0.4})
    tags = [t for (t, _, _) in sink.writer.scalars]
    assert "Loss/Train" in tags
    assert "Loss/Test" in tags
    # Ensure nothing that looks like <Class>/<Metric>/<Variant> got logged
    assert not any(t.count("/") >= 1 and not t.startswith(("Loss/", "Metrics/")) for t in tags)


def test_class_first_layout_renames_and_filtering(sink):
    """
    Verify:
      - Tag layout: <ClassName>/<MetricName>/<Variant>
      - Renames: map@50:95->mAP50_95, map@50->mAP50, f1_score->F1
      - Non-numeric and NaN fields are ignored
    """
    values = {
        "epoch": 7,
        "test_results_json": {
            "class_map": [
                {
                    "class": "dogs",
                    "map@50:95": 0.1234,
                    "map@50": 0.2500,
                    "precision": 0.80,
                    "recall": 0.60,
                    "f1_score": 0.685,
                    "notes": "ignore-me",            # non-numeric => ignored
                    "nan_field": float("nan"),       # NaN => ignored
                },
                {
                    "class": "cats",
                    "map@50:95": 0.3456,
                    "map@50": 0.5000,
                    "precision": 0.70,
                    "recall": 0.75,
                },
                {
                    "class": "elephants",
                    "map@50:95": 0.05,
                    "map@50": 0.10,
                    "precision": 0.20,
                    "recall": 0.15,
                },
            ]
        },
        "ema_test_results_json": {
            "class_map": [
                {
                    "class": "dogs",
                    "map@50:95": 0.1300,
                    "map@50": 0.2600,
                    "precision": 0.82,
                    "recall": 0.62,
                    "f1_score": 0.705,
                }
            ]
        },
    }

    sink.update(values)

    logged = sink.writer.scalars
    tags = [t for (t, _, _) in logged]

    # Class names are already simple; slugging shouldn't change them
    cls_dogs = "dogs"
    cls_cats = "cats"
    cls_ele = "elephants"

    # Base variant checks
    for cls in (cls_dogs, cls_cats, cls_ele):
        assert f"{cls}/mAP50_95/Base" in tags
        assert f"{cls}/mAP50/Base" in tags
        assert f"{cls}/precision/Base" in tags
        assert f"{cls}/recall/Base" in tags

    # F1 (from f1_score) should exist for dogs in Base
    assert f"{cls_dogs}/F1/Base" in tags

    # EMA variant for dogs
    assert f"{cls_dogs}/mAP50_95/EMA" in tags
    assert f"{cls_dogs}/mAP50/EMA" in tags
    assert f"{cls_dogs}/precision/EMA" in tags
    assert f"{cls_dogs}/recall/EMA" in tags
    assert f"{cls_dogs}/F1/EMA" in tags

    # Non-numeric / NaN not logged
    unexpected = [t for t in tags if any(key in t for key in ("notes", "nan_field"))]
    assert not unexpected, f"Unexpected tags: {unexpected}"

    # Spot-check a couple of values
    vmap = {tag: val for (tag, val, _) in logged}
    assert vmap[f"{cls_dogs}/mAP50_95/Base"] == pytest.approx(0.1234)
    assert vmap[f"{cls_dogs}/F1/EMA"] == pytest.approx(0.705)


def test_handles_empty_or_wrong_class_map(sink):
    """Gracefully ignore when class_map is empty or not a list."""
    for bad in ([], {}, None):
        sink.update({"epoch": 2, "test_results_json": {"class_map": bad}})
    # No per-class tags should have been added
    assert not any(t[0].count("/") >= 2 and not t[0].startswith(("Loss/", "Metrics/")) for t in sink.writer.scalars)


def test_flush_and_close_do_not_crash(sink):
    """Lifecycle calls should be safe with the dummy writer."""
    sink.update({"epoch": 0})
    sink.writer.flush()
    sink.writer.close()
    assert sink.writer.flushed is True
    assert sink.writer.closed is True