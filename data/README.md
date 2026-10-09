# Drop your footage + flight metadata here

This folder is the conventional home for your **real** mission inputs. Everything in `data/` is
**git-ignored** (except this README), so your footage and telemetry never get committed.

You are not *required* to use this folder — a dataset descriptor can point anywhere — but keeping one
folder per mission here is the tidy default.

## Layout (one folder per mission)

```
data/
└── my_mission/
    ├── flight.MP4          # your single-pass drone clip (1080p / 4K)
    ├── flight.SRT          # telemetry sidecar  (DJI)          — OR —
    ├── track.csv           #   a CSV GPS track                 — OR —
    ├── flight.tlog / .bin  #   an ArduPilot / PX4 log
    ├── intrinsics.yaml      # optional: known camera intrinsics
    └── check_points.geojson # optional: surveyed points for accuracy validation
```

## What DRISHTI needs

| Input | Required? | Notes |
|-------|-----------|-------|
| **Video** | ✅ yes | The recorded single-pass flight. |
| **Telemetry** (GPS track) | ✅ yes | One of: DJI `.SRT`, a `.csv` track, a MAVLink `.tlog`/`.bin`, or EXIF-geotagged media. Georeferencing needs real GPS — missing it fails loudly. |
| Camera intrinsics | optional | If absent, DRISHTI self-calibrates from the imagery. |
| IMU / baro | optional | Used to strengthen pose estimation when present. |
| RTK / PPK track | optional | A corrected track improves absolute accuracy. |
| Check points | optional | Surveyed control points. Without them the accuracy report is honestly marked **"unvalidated"** — never fabricated. |

## Then point a descriptor at it

Copy the template, set the paths, and run — full walkthrough in the root
[`README.md`](../README.md#prepare-your-data-where-your-footage--metadata-go):

```bash
cp configs/datasets/example.yaml configs/datasets/my_mission.yaml
# edit video: + telemetry.{format,path} to match the files above
drishti run --dataset configs/datasets/my_mission.yaml --profile balanced
```

Outputs land in a resumable, content-hashed bundle at `runs/<run_id>/` (also git-ignored).
