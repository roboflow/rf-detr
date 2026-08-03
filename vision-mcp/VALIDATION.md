# Live validation transcript

- Date: 2026-08-01 (America/New_York)
- Hardware: MacBook Pro (MacBookPro17,1), Apple M1, 8 cores, 16 GB RAM
- OS: macOS 26.4.1 (25E253)
- Resolved inference device: `mps` (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- Camera source: OpenCV camera index `0`, reported by the engine as `camera:0`
- Model: `demo` / `RFDETRNano`, COCO detection, 384 x 216 processed frame
- Configured processing rate: 3 FPS
- Measured live processing rate after capture fix: 2.99-3.00 FPS
- Engine: `http://127.0.0.1:8765`
- Viewer: `http://127.0.0.1:8800`

This is an append-only account of the live session. Failures are retained alongside their fixes and
retests. Physical-action checks are not marked passed until the operator explicitly confirms the
action before the MCP call.

## Preflight and live recovery

### Engine and viewer startup — FAIL

- Action: Started `vision-engine` with `config.example.yaml`, then started the browser viewer.

- Tool calls: MCP `get_system_status`, `list_active_streams`, `get_stream_status`,
    `get_model_status`, `get_current_counts`, `get_active_tracks`, `get_queue_metrics`, and
    `get_recent_errors` through an initialized stdio `ClientSession`.

- Actual response (relevant fields):

    ```json
    {
      "registered_tools": 33,
      "system_health": "unhealthy",
      "stream": {
        "state": "connected",
        "health": "unhealthy",
        "health_reasons": [
          "no frame processed for 214s"
        ],
        "captured_frames": 20,
        "processed_frames": 3,
        "dropped_frames": 17,
        "queue_depth": 0,
        "last_frame_at": "2026-08-01T20:12:47.684Z"
      },
      "model": {
        "loaded_at": "2026-08-01T20:12:47.142Z",
        "inference_count": 3
      }
    }
    ```

- Viewer: `unhealthy`, `16.4 FPS`, zero objects/tracks, last-frame age 146.5 seconds.

- Result: **FAIL**. The MCP transport and schemas worked, but OpenCV/AVFoundation capture repeatedly
    reconnected after `VideoCapture.grab()` calls made while pacing the webcam.

### Webcam capture pacing regression test — FAIL, then PASS

- Action: Added `test_webcam_pacing_does_not_use_backend_grab`, which models a webcam backend that
    continues to return frames from `read()` but cannot tolerate pacing with `grab()`.

- Test command:

    ```text
    .venv/bin/pytest tests/test_phase1_contract.py::test_webcam_pacing_does_not_use_backend_grab -q
    ```

- Original result: **FAIL** at `capture.grab()`.

- Fix: Webcam sources now wait until the next processing deadline without calling backend `grab()`;
    URL and file draining behavior is unchanged.

- Retest command: `.venv/bin/pytest tests/test_phase1_contract.py -q`

- Retest result: **PASS**, `8 passed`.

### Engine restart and MCP recovery — PASS

- Action: Stopped and restarted only `vision-engine`; left the viewer running and used a newly
    spawned stateless `vision-mcp` stdio facade.

- Tool calls: MCP `get_stream_status` and `get_model_status`, repeated after three seconds.

- Actual response (relevant fields):

    ```json
    {
      "first": {
        "state": "connected",
        "health": "healthy",
        "processed_fps": 2.99,
        "captured_frames": 136,
        "processed_frames": 114,
        "queue_depth": 0,
        "queue_capacity": 2,
        "loaded_at": "2026-08-01T20:40:40.763Z",
        "inference_count": 114
      },
      "after_3s": {
        "state": "connected",
        "health": "healthy",
        "processed_fps": 3.0,
        "captured_frames": 146,
        "processed_frames": 124,
        "queue_depth": 0,
        "queue_capacity": 2,
        "loaded_at": "2026-08-01T20:40:40.763Z",
        "inference_count": 124
      }
    }
    ```

- Viewer: `connected/healthy`, `3.0/3.0 fps`, queue `0/2`, last-frame age 0.3 seconds.

- Result: **PASS**. Frames and inference counts climbed, the model load timestamp stayed singular and
    stable, the queue stayed within its bound, and both the viewer and MCP facade recovered without a
    viewer or MCP-server restart.

### Blank/stale viewer report — FAIL, then PASS

- Operator report: The in-app viewer appeared blank.

- Inspection: Both ports were listening and the viewer returned HTML and a JPEG, but the camera frame
    was stale by about 8,110 seconds. The dashboard classified the stream as `unhealthy`, and the
    capture log showed repeated reconnects.

- Action: Reattached the viewer tab, stopped and restarted `vision-engine`, then restarted the
    stateless viewer proxy. The MCP server itself was not restarted.

- Recovered camera: OpenCV camera index 0 reported 30 FPS after restart.

- Tool calls: MCP `get_stream_status`, `get_current_counts`, and `get_active_tracks` through an
    initialized stdio `ClientSession`.

- Actual response (relevant fields):

    ```json
    {
      "stream": {
        "state": "connected",
        "health": "healthy",
        "processed_fps": 2.98,
        "captured_frames": 1000,
        "processed_frames": 976,
        "queue_depth": 0,
        "queue_capacity": 2,
        "last_frame_at": "2026-08-02T01:15:36.669Z"
      },
      "counts": {
        "current_objects": 2,
        "counts_by_class": {
          "person": 1,
          "chair": 1
        }
      },
      "active_tracks": 2
    }
    ```

- Evidence artifact: `15efa3d649c0443d9fe1fc688647baa4`, JPEG, 15,059 bytes, created at
    `2026-08-02T01:15:09.018Z`. Visual inspection showed the annotated live camera frame with one
    person and one chair.

- Result: **PASS** for recovery and visibility. This is infrastructure evidence only; it is not used
    as the ordered presence check because the operator had not confirmed a physical action before the
    count call.

## Preliminary non-physical MCP checks

These calls verify contracts and expose issues, but they do not replace the ordered checks below.

### Latency percentile ordering — PASS

- Tool: `get_inference_latency(stream_id="webcam", time_window="15m", interval="1m")`
- Actual: 62 samples; p50 103.47 ms, p95 158.38 ms, p99 195.26 ms, max 838.38 ms.
- Result: **PASS**, `p50 <= p95 <= p99`.

### Excessive bucket request — PASS

- Tool: `get_counts_by_class(stream_id="webcam", time_window="24h", interval="1s")`
- Actual: MCP error `INVALID_TIME_WINDOW`; 86,400 requested buckets exceed the 500-bucket limit.
- Result: **PASS**.

### Annotated artifact creation — PARTIAL

- Tool: `get_latest_annotated_frame(stream_id="webcam")`
- Actual: artifact `8d2c4cb96c0b4c53a60b344cfc28b5e6`, JPEG, 9,442 bytes, created at
    `2026-08-01T20:41:18.948Z`.
- Result: **PARTIAL**. Artifact creation passed; disk/viewer pixel comparison remains to be performed
    during the ordered evidence check.

### Fifteen-minute throughput versus viewer — FAIL

- Tool: `get_processing_throughput(stream_id="webcam", time_window="15m", interval="1m")`
- Actual: tool `processed_fps` 0.0689; newest bucket 1.0333. Viewer: 3.0 FPS.
- Result: **FAIL**. The calculation averages a fresh run over the entire requested wall-clock window,
    including time before this engine instance started. This failure remains open for the metrics check.

## Ordered physical checks

### Presence: operator stepped out — FAIL

- Physical action: Operator confirmed `frame empty` after stepping completely out of the webcam
    frame.

- Tool: `get_current_counts(stream_id="webcam")` through the MCP stdio transport.

- Actual response:

    ```json
    {
      "schema_version": "1.0",
      "stream_id": "webcam",
      "current_objects": 1,
      "counts_by_class": {
        "chair": 1
      },
      "frame_at": "2026-08-02T01:17:56.475Z"
    }
    ```

- Result: **FAIL**. The operator was absent, but the unrestricted COCO stream counted a visible chair.

- Fix in progress: Restrict the validation stream to `person`, `cell phone`, and `bottle`, preserving
    both the presence check and the required second-object-class check. Restart and same-action retest
    required before advancing.

### Presence: operator stepped out — PASS after fix

- Fix: Restricted the webcam validation classes in `config.example.yaml` to `person`, `cell phone`,
    and `bottle`, so static background furniture does not invalidate the human-presence check.

- Restart verification: Camera reopened at 30 FPS, the engine processed at 3.0 FPS, stream health was
    `healthy`, queue depth was 0/2, and reconnect attempts were 0.

- Physical action: After the restart, the operator explicitly confirmed they were still completely
    out of the camera frame.

- Tool: `get_current_counts(stream_id="webcam")` through the MCP stdio transport.

- Actual response:

    ```json
    {
      "schema_version": "1.0",
      "stream_id": "webcam",
      "current_objects": 0,
      "counts_by_class": {},
      "frame_at": "2026-08-02T01:23:47.136Z"
    }
    ```

- Result: **PASS**. The same empty-frame action now reports zero objects. The previous failure remains
    above as part of the truthful fix-and-retest record.

### Presence: operator stepped into frame — PASS

- Physical action: Operator explicitly confirmed they had stepped into and remained in the webcam
    frame.

- Tool: `get_current_counts(stream_id="webcam")` through the MCP stdio transport.

- Actual response:

    ```json
    {
      "schema_version": "1.0",
      "stream_id": "webcam",
      "current_objects": 1,
      "counts_by_class": {
        "person": 1
      },
      "frame_at": "2026-08-02T01:40:23.797Z"
    }
    ```

- Result: **PASS**. The confirmed in-frame operator was reported as exactly one person.

### Presence: second object class — PASS

- Physical action: After being asked to hold either a phone or bottle clearly in view, the operator
    replied `continue`; the MCP result identified the object as a cell phone.

- Tool: `get_current_counts(stream_id="webcam")` through the MCP stdio transport.

- Actual response:

    ```json
    {
      "schema_version": "1.0",
      "stream_id": "webcam",
      "current_objects": 2,
      "counts_by_class": {
        "person": 1,
        "cell phone": 1
      },
      "frame_at": "2026-08-02T01:41:09.311Z"
    }
    ```

- Result: **PASS**. A second configured object class appeared alongside the person.

### Stream and model discovery/status — PASS

- Transport: All calls below used one initialized MCP stdio session.
- `list_active_streams()` returned one `webcam` stream with state `connected`, health `healthy`,
    model `demo`, source `camera:0`, 2,961 processed frames, 2 current objects, and 2 active tracks.
- `list_models()` returned `demo`: `RFDETRNano`, detection task, MPS device, confidence 0.4, loaded.
- `get_model_info(model="demo")` returned resolution 384 and the expected 80-class COCO model
    vocabulary.
- `get_model_status(model="demo")` returned loaded `true`, device `mps`, loaded at
    `2026-08-02T01:20:36.000Z`, 2.846-second load time, 2,961 inferences, last inference at
    `2026-08-02T01:42:10.594Z`, 85.63 ms mean latency, and queue depth 1.
- Result: **PASS**. Discovery, model metadata, and live status agreed; the load timestamp remained
    stable while inference advanced.

### Tracking: stationary person — PASS

- Physical action: Operator lowered the phone, remained fully in frame, stood still for at least five
    seconds, and explicitly confirmed `standing still`.
- Tool: Two `get_active_tracks(stream_id="webcam")` calls through one MCP stdio session, separated
    by three seconds.
- Sample 1: One person, track ID 1, confidence 0.9038, first seen
    `2026-08-02T01:39:15.927Z`, age 253.19 seconds, zone `left_half`.
- Sample 2: One person, track ID 1, confidence 0.9005, the same first-seen timestamp, age 256.19
    seconds, zone `left_half`.
- Bounding box movement across samples was sub-pixel to 0.7 pixels per coordinate.
- Result: **PASS**. The person track ID persisted, age advanced by three seconds, and the stationary
    bounding box remained stable.

### Tracking: operator left frame — PASS

- Physical action: Operator stepped completely out, waited, and explicitly confirmed `out again`.
- Tool: `get_active_tracks(stream_id="webcam")` through the MCP stdio transport.
- Actual: `active_tracks` was 0 and `tracks` was empty.
- Result: **PASS**. The prior person track cleared after the operator left the frame.

### Tracking: operator returned — PASS

- Physical action: Operator stepped back in, remained visible, and explicitly confirmed `back in`.
- Tool: `get_active_tracks(stream_id="webcam")` through the MCP stdio transport.
- Actual: One person with fresh track ID 6, confidence 0.9626, first seen at
    `2026-08-02T01:44:56.754Z`, and age 22.05 seconds. The pre-exit track ID was 1.
- Result: **PASS**. Re-entry produced a new active person track rather than reviving the expired track.

### Zones: outside `left_half` baseline — PASS

- Physical action: Operator moved to the right half of the displayed camera image and explicitly
    confirmed `on right`.
- Tool: `get_zone_occupancy(stream_id="webcam")` through the MCP stdio transport.
- Actual: `left_half` occupancy 0, empty `by_class`, no occupancy limit, and `over_limit: false`.
- Result: **PASS**. The person standing on the displayed right was outside the configured left zone.

### Zones and line: right-to-left movement — PARTIAL; initial failure retracted

- Physical action: Operator crossed from the displayed right into `left_half`, remained there, and
    explicitly confirmed `left`.
- Live tools: `get_zone_occupancy` reported one person in `left_half`; `get_active_tracks` reported
    the continuing person track ID 6 in `left_half`. Live occupancy and track continuity passed.
- Persisted tools: `get_dwell_times`, `get_line_crossing_events`, and
    `get_recent_detection_events`, all with a 5-minute window and 1-second interval.
- Initial concern: The intended entry appeared at `2026-08-02T01:48:03.074Z`, followed by line
    `out` at `01:48:11.077Z`, zone exit with 8.7 seconds dwell at `01:48:11.769Z`, then line `in`
    and zone re-entry at `01:48:12.433Z`. This was initially classified as boundary jitter.
- Operator clarification: The operator immediately stated, `no i moved twice`; the additional
    transitions were physical movements and must not be classified as false detections.
- Actual persisted results: Zone entries, a zone exit with 8.7 seconds dwell, and directional center
    crossings were stored for the continuing track ID 6. The 5-minute dwell summary contained seven
    completed samples with mean 16.05 seconds and maximum 75.91 seconds.
- Result: **PARTIAL**. Live zone occupancy, zone transitions, dwell, and line event persistence all
    worked. The initial failure is retracted; because multiple intentional movements and earlier events
    shared the query window, one controlled crossing is still required to validate the exact event delta.

### Zones and line: controlled left-to-right crossing — PASS

- Baseline physical action: Operator moved to `left_half`, stayed still, and explicitly confirmed
    `left steady`.
- Baseline MCP state: `left_half` contained one person; active person track ID 6 was in `left_half`;
    the newest event was at `2026-08-02T01:48:12.433Z`, with no newer events during the steady period.
- Crossing action: Operator crossed the center once into the displayed right half, stayed there, and
    explicitly confirmed `right steady`.
- Live result: `left_half` occupancy became 0 while the continuing person track ID 6 remained active
    outside all zones.
- Exact event delta after the baseline:
    - One center line `out` crossing for track 6 at `2026-08-02T01:51:55.541Z`.
    - One `left_half` zone exit for track 6 at `2026-08-02T01:51:56.201Z`, with 223.77 seconds dwell.
    - No other new spatial events appeared.
- Result: **PASS**. One physical crossing produced exactly one directional line event followed by
    exactly one zone exit, with continuous tracking and a positive persisted dwell duration.

## Metrics fix and live retest

### Fifteen-minute throughput denominator — PASS after fix

- Prior failure: `get_processing_throughput` divided a fresh run's processed frames by the entire
    requested wall-clock window, producing 0.0689 FPS against a 3.0 FPS viewer.
- Regression test: Added `test_throughput_uses_observed_bucket_seconds`; it failed because the query
    did not select `bucket_seconds`.
- Fix: Overall and response-bucket throughput now divide frame totals by their persisted observed
    aggregation seconds. The targeted regression and full focused suite passed.
- Live retest: `get_processing_throughput(stream_id="webcam", time_window="15m", interval="1m")`
    returned 2,019 processed frames, 2.852 FPS overall, and target 3.0 FPS. The overall value correctly
    includes the deliberate restart/degraded interval; the newest completed buckets were 2.9832 and
    3.0000 FPS while live stream status was 2.99 FPS.
- Result: **PASS**. Current stored throughput agrees with the live viewer/status without averaging
    over unobserved pre-start time.

### macOS rapid camera restart — PASS after fix

- Reproduced failure: The first post-metrics-fix restart opened `camera:0` at a reported 1.0 FPS and
    repeatedly reconnected instead of reaching the configured 3.0 FPS.
- Hardware probe: Explicit AVFoundation open reported 30.0 FPS and delivered 8/8 frames while paced
    at approximately 3 FPS.
- Regression test: Added `test_macos_webcam_open_selects_avfoundation_and_recovers_fps`.
- Fix: macOS integer webcams now open with `CAP_AVFOUNDATION` and request a stable 30 FPS camera mode
    before the first read. The full focused suite passed with 10 tests.
- Live retest: Camera connected at 30.0 FPS, model loaded on MPS, and the engine remained connected
    with 2.99 processed FPS, fresh frames, zero reconnect attempts, no last error, and health `healthy`.
- Result: **PASS**.

### Live metric contracts — PASS

- `get_inference_latency`, 15m/1m: 2,019 samples; mean 91.81 ms; p50 89.05 ms; p95
    111.16 ms; p99 131.78 ms; max 667.32 ms. Percentile ordering passed.
- `get_frame_drop_rate`, 15m/1m: 2,030 captured, 2,019 processed, 11 dropped, drop rate
    0.0054.
- `get_queue_metrics`: depth 0/2, high-water 2, inference queue depth 0.
- `get_stream_status`: connected and healthy; 2.99/3.0 FPS; queue 0/2; reconnect attempts 0;
    fresh `last_frame_at` `2026-08-02T01:59:47.361Z`.
- Result: **PASS**.

## Evidence artifact

### Annotated artifact creation — PASS; MCP URI readback — FAIL

- Tool: `get_latest_annotated_frame(stream_id="webcam")`.
- Created artifact ID `114e87901dd8498b8d6335fc9ec2bda9`, URI
    `vision://artifacts/114e87901dd8498b8d6335fc9ec2bda9`, JPEG, 14,050 bytes, at
    `2026-08-02T02:01:05.123Z`.
- Readback: MCP `resources/read` for the returned URI produced a text error envelope with code
    `INVALID_ARGUMENT` and message `Unknown vision resource URI`.
- Result: **FAIL**. Artifact creation works, but the resource URI advertised by the tool is not
    dereferenceable through MCP. Resource routing must be fixed and the same artifact flow retested.

### Annotated artifact MCP readback — PASS after fix

- Regression test: Added `test_artifact_resource_uri_is_routed_and_validated`; it initially failed
    because the artifact route did not exist.
- Fix: Registered `vision://artifacts/{artifact_id}`, validated artifact IDs before use, fetched bytes
    from the engine's contained artifact endpoint, and returned MCP `BlobResourceContents`.
- Test result: The focused suite passed with 11 tests.
- Same-URI retest: Reading `vision://artifacts/114e87901dd8498b8d6335fc9ec2bda9`
    returned `BlobResourceContents`, MIME type `image/jpeg`, 18,736 base64 characters, and exactly
    14,050 decoded bytes, matching the advertised artifact size.
- Disk path: `data/artifacts/frame/20260802/114e87901dd8498b8d6335fc9ec2bda9.jpg`.
- Visual inspection: The 384x216 image showed one annotated person, connected/healthy status,
    3.0/3.0 FPS, queue 0/2, the configured center line, and the shaded left zone.
- Result: **PASS**.

### Event order — PASS

- The controlled crossing produced `line_crossing` at `2026-08-02T01:51:55.541Z`, followed by
    `zone_exit` at `2026-08-02T01:51:56.201Z`; the returned event list was newest-first.
- Result: **PASS**. Event timestamps and query ordering were internally consistent.

## Failure and recovery behavior

### Physically covered camera — PASS

- Physical action: Operator completely covered the webcam lens and explicitly confirmed
    `camera covered`.
- MCP results: Stream remained connected and healthy at 3.0/3.0 FPS with a fresh frame timestamp
    `2026-08-02T02:05:56.636Z`, reconnect attempts 0, and no last error.
- `get_current_counts` returned zero objects and an empty class map; active tracks were 0.
- `get_queue_metrics` returned capture depth 0/2 and inference depth 0.
- Result: **PASS**. A dark/occluded scene clears detections without crashing, stalling, or corrupting
    the stream contract.

### Camera uncovered — PASS

- Physical action: Operator uncovered the webcam, remained visible, waited, and explicitly confirmed
    `camera restored`.
- MCP results: Stream remained connected and healthy at 3.0 FPS with fresh frame timestamp
    `2026-08-02T02:06:47.835Z`; `get_current_counts` returned exactly one person.
- Result: **PASS**. Detection recovered without restarting or reconnecting the stream.

### Engine shutdown and immediate recovery — PASS

- Deliberate action: Stopped the engine cleanly while leaving the MCP facade/viewer workflow intact.
- While down: `get_system_status()` returned MCP `is_error: true` with a schema-versioned
    `ENGINE_UNAVAILABLE` response naming `http://127.0.0.1:8765`; the MCP session itself did not crash.
- Immediate restart: Engine started at `2026-08-02T02:07:39.298Z`; AVFoundation reopened the camera
    at 30.0 FPS and the model loaded on MPS.
- Recovery MCP state: System healthy with database OK, 1/1 streams running, and 1 model loaded.
    Webcam connected/healthy at 2.99/3.0 FPS with fresh frame timestamp
    `2026-08-02T02:08:14.888Z`, queue 0/2, one current person, zero reconnect attempts, and no last error.
- `get_recent_errors(time_window="5m", interval="1s", limit=20)` returned a valid empty error list.
- Result: **PASS**. Transport failure is stable and structured, and immediate restart restores the
    full live pipeline without stale frames or camera-reopen failure.

## Ordered validation gate

- Initial assessment: The gate was briefly marked open after presence, discovery/status, active-track
    continuity, live zone/line events, metrics, evidence, occlusion, and engine restart passed.
- Objective audit correction: The written objective also requires a measured unique-object increment,
    isolated one-entry/one-exit and approximately 20-second dwell results, the opposite line direction,
    class-count buckets, all-tool unavailable behavior, and a persisted disconnection error. Those
    checks had not yet been directly evidenced.
- Result: **NOT YET OPEN**. This correction is retained in the transcript; M7/M8/M9 must wait until
    the remaining checks pass.

### Unique-object leave sample — INCONCLUSIVE; operator correction

- Confirmed baseline: One active person track ID 3 and one unique person in the 15-minute window.
- First leave sample: MCP returned a fresh active person track ID 23 and 20 unique person tracking
    instances, so the result initially appeared to show severe track churn.
- Follow-up: Raw counts still showed one person while active tracks had cleared; persisted rows showed
    short tracks during movement.
- Operator correction: The operator stated, `no maybe i stepped in by mistake, run test again`.
- Result: **INCONCLUSIVE**, not a product failure. The physical action was not held constant, so this
    sample cannot validate or invalidate unique-object semantics. A fully confirmed redo is required.

### Unique-object controlled re-entry — FAIL

- Confirmed empty baseline: Raw detections 0, active tracks 0, historical unique-person count 33.
- Physical action: Operator entered once, stood still, and explicitly confirmed
    `unique back steady`.
- Actual: One active person track ID 35, but `get_unique_object_count` rose from 33 to 35 instead of
    34\.
- Persistence diagnosis: Two rows were created after the baseline: genuine track 35 with 114 frames
    and a stable multi-second lifetime, plus duplicate track 36 with exactly one frame and zero-second
    lifetime.
- Result: **FAIL**. A one-frame association is counted as a unique tracked object. The historical
    metric needs to exclude unconfirmed single-frame track noise, then the same action must be retested.
