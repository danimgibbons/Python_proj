# MEG Skater Stimulus Pipeline

This pipeline prepares synchronized multi-camera skating videos for MEG stimulus
presentation. It is organized around a fixed input-folder structure, so each
stage can be run independently and the intermediate videos can be inspected
before continuing.

## Input Folder

Create an input folder with this structure:

```text
Vids/input/
  timing.csv
  color_map.csv                  optional; defaults are built in
  cameras/
    camera1/
      raw.mp4
      first_frame.txt
      background_mask.png        optional; used by clean_bg
      empty_frame.png            optional; used by masks and/or clean_bg
      overlay_mask.png           optional; used by masks
    camera2/
      raw.mp4
      first_frame.txt
      background_mask.png
      empty_frame.png
      overlay_mask.png
    camera3/
      raw.mp4
      first_frame.txt
      background_mask.png
      empty_frame.png
      overlay_mask.png
```

`input_template/` contains starter metadata files. Copy or rename it to
`Vids/input/`, then add the video and image assets.

`timing.csv` must contain:

```csv
chunk,start,end
1,00:09:15,00:17:49
```

`first_frame.txt` contains the first synchronized frame number for that camera.
The first camera is treated as the timing reference.

## Keeping Data Outside the Repo

The repository is set up so large input videos and generated outputs can live
outside the git checkout, for example on a mounted secure server folder.

The easiest setup is to create a local config file:

```bash
python stimulus_pipeline.py --write-config pipeline_config.json
```

Then edit `pipeline_config.json`:

```json
{
  "data_root": "/Volumes/secure-server/skater-data",
  "input_dir": "input",
  "output_dir": "output"
}
```

With that file in place, short commands such as this automatically use:
`/Volumes/secure-server/skater-data/input` and
`/Volumes/secure-server/skater-data/output`.

```bash
python stimulus_pipeline.py --show-paths
python stimulus_pipeline.py --steps preprocess
```

`pipeline_config.json` is ignored by git so each machine can point at its own
local mount path. `pipeline_config.example.json` is committed as a template.

Path settings are applied in this order, from lowest to highest priority:

1. Built-in defaults: `Vids/input` and `Vids/output`.
2. Config file values from `pipeline_config.json`, or from `--config PATH`.
3. Environment variables: `SKATER_DATA_ROOT`, `SKATER_INPUT_DIR`,
   `SKATER_OUTPUT_DIR`, and `SKATER_PIPELINE_CONFIG`.
4. Command-line options: `--data-root`, `--input-dir`, and `--output-dir`.

If `--data-root` or `data_root` is set, relative input/output paths are resolved
inside that root. Absolute input/output paths are used exactly as written.

## Environment

Create the Conda environment before running the pipeline:

```bash
conda env create -f environment.yml
conda activate skater-stimulus-pipeline
```

After activating the environment, use `python` rather than `python3` so the
pipeline runs with the Conda interpreter that has packages such as OpenCV
available. For a one-off command without activating first, use:

```bash
conda run -n skater-stimulus-pipeline python stimulus_pipeline.py --steps masks
```

## Workflow

1. `cut_big`: cut the long raw camera recordings into synchronized big chunks.
2. `cut_small`: split big chunks into shorter clips for inspection and editing.
3. `clean_bg`: optionally replace masked background pixels with a clean frame.
4. `masks`: generate one mask image per frame for available small chunks.
5. `overlays`: render colored landmark overlays from generated masks.
6. `camera_cut`: create a final stimulus clip that switches viewpoints every
   500 frames, equal to 10 seconds at 50 fps, and write a frame-level camera log.

## Commands

Run one stage at a time:

```bash
python stimulus_pipeline.py --steps cut_big
python stimulus_pipeline.py --steps cut_small
python stimulus_pipeline.py --steps clean_bg
python stimulus_pipeline.py --steps masks
python stimulus_pipeline.py --steps overlays
python stimulus_pipeline.py --steps camera_cut
```

Run both chunking stages:

```bash
python stimulus_pipeline.py --steps preprocess
```

Run the full workflow:

```bash
python stimulus_pipeline.py --steps all
```

Preview FFmpeg commands for cutting and switching without writing videos:

```bash
python stimulus_pipeline.py --steps cut_big --dry-run
```

## Useful Options

- `--fps 50`: frame rate used for timecode/frame conversion.
- `--config pipeline_config.json`: use a specific local path config file.
- `--data-root /Volumes/secure-server/skater-data`: base folder for relative
  input/output paths.
- `--input-dir input --output-dir output`: override the configured folders.
- `--show-paths`: print the effective input/output paths and exit.
- `--lead-in-seconds 4`: seconds added to each start time before cutting.
- `--small-duration 150`: fixed small-chunk duration in seconds.
- `--small-duration-min 120 --small-duration-max 180`: search range used when
  choosing a small-chunk duration automatically.
- `--chunk-pattern "big1_small*.mp4"`: restrict cleaning/masking to selected
  small chunks while testing parameters.
- `--big-index 1 --small-index 2`: choose which synchronized small chunk to use
  for the final switched-viewpoint stimulus.
- `--cut-frames 500`: number of frames shown before switching camera.

## Outputs

By default, outputs are written under `Vids/output/`:

```text
Vids/output/
  camera1/
    big/
      big1.mp4
    small/
      big1_small1.mp4
    cleaned/
    overlaid/
  camera2/
  camera3/
  masks/
    camera3/
      big1_small1_cleaned/
        frame_000000.png
  final/
    big1_small2_switch_500f.mp4
    big1_small2_switch_500f_camlog.csv
```

For final viewpoint switching, the pipeline chooses the best available version
for each camera in this order: overlaid, cleaned, then the original small chunk.
