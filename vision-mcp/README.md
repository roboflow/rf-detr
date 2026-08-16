# RF-DETR Vision MCP — Phase 1

This package runs as two independent asyncio processes. `vision-engine` owns RF-DETR models, camera capture, tracking, SQLite, and artifacts. `vision-mcp` is a stateless stdio adapter that only calls the engine over HTTP; it never starts the engine.

## Install and configure

Python 3.11–3.13 is required. From this directory:

```bash
uv sync --all-groups
cp config.example.yaml config.yaml
```

Edit `config.yaml` as needed. Image and stream URLs require host entries in `security.allowed_url_hosts`; local image/video inputs require roots in `security.filesystem_roots`. The HTTP server always binds to `127.0.0.1`.

On macOS, allow the terminal application running the engine to use the camera in **System Settings → Privacy & Security → Camera**. Several consecutive entirely black startup frames produce a permission warning; one dark frame does not.

## Run

Start the engine manually in one terminal:

```bash
uv run vision-engine --config config.yaml
```

Start the MCP server separately:

```bash
uv run vision-mcp --engine http://127.0.0.1:8765
```

The browser endpoints are:

- Latest frame: <http://127.0.0.1:8765/streams/webcam/frame.jpg?annotate=true>
- Live JSON: <http://127.0.0.1:8765/streams/webcam/live>
- Debug JPEG: <http://127.0.0.1:8765/debug/preview> (requires `debug.preview: true`)
- Debug MJPEG: <http://127.0.0.1:8765/debug/stream> (requires `debug.preview: true`)

Inspect every registered MCP tool and resource with:

```bash
npx @modelcontextprotocol/inspector vision-mcp
```

## Metric meanings

- `current_objects`: detections in the latest processed frame.
- `frame_detections`: detections summed across processed frames.
- `active_tracks`: tracked objects visible in the latest processed frame.
- `unique_objects`: distinct temporary tracking instances during the requested period.
- `events`: entries, exits, crossings, violations, and stream state changes.

Tracking IDs are temporary motion-consistency handles, not identities. They do not identify people, persist across restarts, or match objects between streams.

## Concurrency and hardware limits

Blocking capture runs in one thread per stream; inference is serialized in one thread per loaded model; SQLite has one bounded writer task. Coroutines cross blocking boundaries with worker threads. Frame queues are bounded and drop the oldest frame to preserve recency.

Sustained inference on a fanless M1 may thermally throttle, so long-run P95 latency can drift.

### Unverified on this hardware

- NVIDIA metrics through optional `pynvml` and CUDA/multi-GPU selection.
- Sustained production-FPS multi-stream throughput.
- TensorRT and ONNX export are out of scope.
