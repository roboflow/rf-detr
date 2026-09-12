---
description: How RF-DETR's export pipeline is put together, and the step-by-step recipe for adding a new export format as an Exporter subclass with its own configuration and registry entry.
---

# Exporter Blueprint

!!! tip "Key Takeaways"

    - Every export format is an `Exporter` subclass built from its own configuration dataclass
    - `RFDETR.export()` is a facade: it prepares one `ExportGraph` and hands it to the exporter the registry names
    - The registry is plain data, so a format can be refused or its install hint printed before its dependency is imported
    - Adding a format touches its own package, the registry, and `pyproject.toml` — never the base class
    - This is an in-tree contribution recipe, not a plugin API: there is no public `register()` hook

This page describes how the export pipeline is assembled and what it takes to add a format to it. If you only want to export a model, read [Export RF-DETR Model](export.md) instead — everything here is internal API.

!!! warning "Internal API"

    Every module named on this page except `RFDETR.export()` is internal. Modules with a leading underscore (`rfdetr.export._onnx`, `rfdetr.export._coreml`, …) carry no stability guarantee, and `rfdetr.export.base` and `rfdetr.export.registry` are equally free to change between releases. The contract described here is a contributor's contract, not a public extension point.

## The pipeline

One export is four steps, and only the third knows which format was asked for.

```text
RFDETR.export(format=..., **kwargs)
  |
  |-- registry.normalize_format / resolve_exporter   -> ExporterClass      (no model work yet)
  |-- ExporterClass.build_config(**kwargs)           -> MyFormatConfig     (other formats' knobs dropped)
  |-- ExporterClass(config)                          -> exporter           (capabilities validated)
  |-- prepare.prepare_export_graph(model, ...)       -> ExportGraph        (format-independent)
  \-- exporter(graph)                                -> Path to artifact
```

**What the user asked for** and **what the model looks like** are kept apart on purpose. The configuration carries the first, `ExportGraph` carries the second, and an exporter is the only place they meet. That is what makes a format testable without a real model: most of the export test suite builds a throwaway `ExportGraph` and never traces anything.

### `prepare_export_graph` — the format-independent half

`rfdetr.export.prepare.prepare_export_graph` does the graph-shaping work every format needs, once:

- freezes each DINOv2 backbone's position embeddings to the export shape, so antialiased bicubic interpolation (which has no ONNX symbolic) stays out of the traced graph
- builds the example input the converter traces with
- wraps the backbone when `backbone_only=True`
- resolves input/output names and the dynamic-axes mapping
- runs one forward pass, so a broken graph fails here rather than inside a third-party converter
- moves everything to CPU, where every converter traces

It returns a frozen `ExportGraph`: `model`, `input_tensors`, `input_names`, `output_names`, `dynamic_axes`, `shape`, `backbone_only`. Nothing in it names a format.

### `registry.py` — data, not imports

Every format except ONNX sits behind an optional dependency, and none of them may be imported by `import rfdetr`. So the registry stores the *dotted path* of each exporter class rather than the class itself, and only `resolve_exporter()` imports it:

```python
REGISTRY: Mapping[str, ExporterEntry] = {
    "onnx": ExporterEntry("rfdetr.export._onnx.exporter", "OnnxExporter", "onnx", "ONNX", supports_dynamic_batch=True),
    # ...
}
```

An entry also carries the few facts that must be known *before* the import: the user-facing `label`, the `pip_extra` named in the missing-dependency message, `supports_dynamic_batch`, and `dynamic_batch_reason`. Those last three mirror class attributes of the exporter itself. The duplication is deliberate — it is what lets `reject_unsupported_dynamic_batch()` refuse a doomed request, in the format's own words, without paying tens of seconds and hundreds of megabytes for `coremltools` or TensorFlow first. `tests/export/test_registry.py` is the only thing keeping the two copies in sync, so it asserts they match.

### `base.py` — what every format gets for free

`rfdetr.export.base` names no format. It holds exactly two things: `ExportConfig`, the settings shared by all of them, and `Exporter`, the abstract class they subclass.

`Exporter.__call__` runs before and after your `_convert`:

- switches the model into its export-friendly forward — exactly once, and idempotently, so a two-stage format composing another exporter stays safe
- normalizes whatever `_convert` returned into a `Path`
- logs the success line

`Exporter.__init__` validates the configuration against the class's declared capabilities, so an unsupported request is refused at construction — before the caller pays for a full DINOv2 forward pass.

## Adding a format

Seven steps. None of them is an edit to `base.py`.

### 1. Write the configuration dataclass

It lives beside the exporter that reads it, subclasses `ExportConfig`, and declares only the knobs your format understands. A setting that does not apply should be impossible to pass, not silently ignored.

```python
# src/rfdetr/export/_myformat/exporter.py
from dataclasses import dataclass

from rfdetr.export.base import ExportConfig, Exporter


@dataclass(frozen=True, slots=True)
class MyFormatConfig(ExportConfig):
    """Settings for ``format="myformat"``.

    Attributes:
        precision: Storage precision of the written artifact.
    """

    precision: str | None = None
```

Keep it `frozen=True, slots=True` like the others: a configuration is a value, and nothing downstream should mutate it.

### 2. Write the exporter class

Subclass `Exporter[MyFormatConfig]`, declare your capabilities as class attributes, and implement `_convert`.

```python
class MyFormatExporter(Exporter[MyFormatConfig]):
    """Convert a prepared graph to MyFormat."""

    config_class = MyFormatConfig
    setting_names = {"precision": "myformat_precision"}
    format = "myformat"
    display_name = "MyFormat"
    supports_dynamic_batch = False
    dynamic_batch_reason = "(the graph bakes a fixed input shape). Export one model per batch size instead."
    supports_notes = False
    notes_reason = "MyFormat has no ONNX-style metadata slot"
    experimental = True
    experimental_note = "Upstream converter is unstable."
    pip_extra = "myformat"

    def _convert(self, graph: ExportGraph) -> Path:
        """Write the artifact and return where it landed."""
        from rfdetr.export._myformat.converter import convert  # heavy optional dependency

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self._export_name(backbone_only=graph.backbone_only)}.myf"
        convert(graph.model, graph.input_tensors, output_path, precision=self.config.precision)
        return output_path
```

`_export_name` above is your own helper, not something the base class hands you — step 3 shows what it should contain. Splitting `_convert` into small named steps the way the existing exporters do keeps each one testable on a throwaway graph.

The capability attributes:

| Attribute                | Meaning                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `config_class`           | The dataclass `build_config()` instantiates                                                     |
| `setting_names`          | Your config's format-specific fields, mapped to the `RFDETR.export()` keyword each is read from |
| `format`                 | Canonical name, matching the registry key                                                       |
| `display_name`           | How the format is spelled in messages to users                                                  |
| `supports_dynamic_batch` | Whether a dynamic batch dimension can be baked in                                               |
| `dynamic_batch_reason`   | Why not, and what to do instead — shown when it cannot                                          |
| `supports_notes`         | Whether the artifact has a metadata slot for the user's `notes`                                 |
| `notes_reason`           | Which slot is missing, named in the dropped-`notes` warning                                     |
| `experimental`           | Whether constructing the exporter warns that the format is work-in-progress                     |
| `pip_extra`              | The `rfdetr[...]` extra that installs your dependencies                                         |

Three rules for `_convert`:

- **Import the heavy dependency inside a method, never at module scope of a file the registry might import early.** Raise `ImportError` with the exact `pip install "rfdetr[myformat]"` command; the other formats route this through a small `_check_<dep>_available()` helper so tests can monkeypatch one choke point.
- **Return the path, do not log the success line.** The base class does that, and doing it twice is how log output drifted between formats before.
- **Do not switch the model into export mode.** It already is — `__call__` did it.

### 3. Name the artifact through the shared helpers

`rfdetr.export._naming` owns filename precedence so the formats cannot drift apart:

```python
from rfdetr.export._naming import append_backbone_marker, resolve_export_stem


def _export_name(self, *, backbone_only: bool) -> str:
    """Resolve the artifact's filename stem."""
    stem, is_custom = resolve_export_stem(
        self.config.variant_name,
        self.config.output_name,
        default="backbone_model" if backbone_only else "inference_model",
    )
    if not is_custom and self.config.precision:
        stem = f"{stem}_{self.config.precision}"
    return append_backbone_marker(
        stem,
        backbone_only=backbone_only,
        named=bool(self.config.variant_name or self.config.output_name),
    )
```

`output_name` wins over `variant_name` wins over the default, and both inputs are sanitized against path traversal. `is_custom` tells you to suppress any precision or backend suffix you would otherwise append — the caller asked for that exact name. `append_backbone_marker` is not optional: without it a backbone-only export silently overwrites a full-detector export written under the same name.

### 4. Register it

```python
REGISTRY: Mapping[str, ExporterEntry] = {
    # ...
    "myformat": ExporterEntry(
        "rfdetr.export._myformat.exporter",
        "MyFormatExporter",
        "myformat",
        "MyFormat",
        dynamic_batch_reason="(the graph bakes a fixed input shape). Export one model per batch size instead.",
    ),
}
```

`supports_dynamic_batch`, `label` and `dynamic_batch_reason` must match your class attributes exactly — the mirror test enforces it. Add a short spelling to `ALIASES` if one is worth having (`"trt"`, `"pte"`). Set `preimport` only if your format has an import-order hazard; TFLite is the one precedent, because TensorFlow must load before anything pulls in ONNX's C extension.

### 5. Add the dependency extra

Add `myformat = [...]` under `[project.optional-dependencies]` in `pyproject.toml`, matching the `pip_extra` you declared. The `ci-deps-resolution` workflow derives its matrix from that table, so the new extra is checked on every supported interpreter automatically.

### 6. Document the format

Add it to `RFDETR.export()`'s `format` docstring in `src/rfdetr/detr.py` and to [Export RF-DETR Model](export.md) — installation extra, a basic example, output files, and an inference snippet.

### 7. Test it

Two suites in `tests/export/test_registry.py` are parametrized over `REGISTRY`, so they pick your format up the moment you register it: the capability mirror (registry entry versus class attributes) and the dynamic-batch guard, which also asserts a fixed-batch format explains what to do instead. Everything else is a hand-written case — add your format to the configuration-type list, and write `tests/export/test_myformat_export.py` for filename resolution, the backbone marker, the missing-dependency message, and numerical parity against eager PyTorch if the runtime can be installed in CI.

Most of those need no real model: build a throwaway `ExportGraph` around a `MagicMock` the way `tests/export/test_executorch_export.py` does, and only the parity test has to trace anything.

## Patterns worth copying

### Two-stage formats

TFLite and TensorRT do not convert from PyTorch. They run an ONNX export first and convert its output, which means they must hand the ONNX stage a configuration rather than re-deriving one:

```python
def onnx_stage(self) -> Any:
    from rfdetr.export._onnx.exporter import OnnxConfig

    return OnnxConfig.derive(self, opset_version=self.opset_version)


def _convert(self, graph: ExportGraph) -> str:
    onnx_path = OnnxExporter(self.config.onnx_stage())(graph)
    return self.build_engine(str(onnx_path))
```

`OnnxConfig.derive` copies every shared setting, `notes` included, so the intermediate `.onnx` a two-stage export passes on is named and annotated exactly as a direct `format="onnx"` export would be. Composing the ONNX exporter is safe because the export-mode switch in `__call__` is idempotent.

### Validating rather than copying a setting

`setting_names` covers the common case of copying a keyword across. When a format must validate or derive one instead, override `_format_settings`. ExecuTorch does, because an unresolved backend must be an error rather than a quiet `"xnnpack"` default:

```python
@classmethod
def _format_settings(cls, settings: Mapping[str, Any]) -> dict[str, Any]:
    backend = settings.get("backend")
    if backend is None:
        raise ValueError("format='executorch' requires a backend; none was resolved.")
    return {"backend": backend, "soc": settings.get("soc")}
```

### Worked example

`src/rfdetr/export/_openvino/exporter.py` is the smallest complete exporter: config dataclass, lazy dependency import, name resolution, one setting translated into the converter's own flag, and a `_convert` that is six lines of sequencing. Read that one before writing a new format.

## Known limitations

- **There is no plugin API.** `REGISTRY` is a module-level literal with no `register()` hook, so an external package cannot add a format without editing RF-DETR. Formats live in-tree because each one carries CI matrix entries, a dependency extra, and parity tests that only make sense inside the repository.
- **The capability mirror is test-enforced, not type-enforced.** Nothing but `tests/export/test_registry.py` stops a registry entry from disagreeing with the class it names.

## Next Steps

- [Export RF-DETR Model](export.md) — the user-facing export guide
- [Deploy a Trained RF-DETR Model](deploy.md) — deployment paths for exported artifacts
- [Migration Guide](../getting-started/migration.md) — what moved when the exporter classes landed
