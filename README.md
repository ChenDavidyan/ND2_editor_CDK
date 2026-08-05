# Chen Viewer

Chen Viewer is a desktop microscopy viewer and exporter for Nikon ND2 files. It uses Napari to inspect FOVs and Z ranges, create maximum-intensity projections, crop an XY region of interest, preview channel montages, and export OME-TIFF or JPEG images.

## Recommended launcher

The most complete version is:

```powershell
python run_app_faster_threaded_savequeue.py path\to\image.nd2
```

It provides:

- Cached Z planes and faster projection updates
- Fast, non-destructive ROI display cropping
- Background FOV-thumbnail loading
- Background montage preview generation
- A serial background save queue with progress and error reporting
- FOV-loading progress
- OME-TIFF export with channel and pixel-size metadata
- Per-channel and montage JPEG export
- Adaptive burned-in scale bars for JPEG output

## Installation

Python 3.13 is used by the current development environment.

```powershell
git clone https://github.com/ChenDavidyan/ND2_editor_CDK.git
cd ND2_editor_CDK
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then launch the viewer with an ND2 file:

```powershell
python run_app_faster_threaded_savequeue.py C:\path\to\image.nd2
```

## Main workflow

1. Select an FOV from the navigator.
2. Choose the desired Z range and update the view.
3. Set and position the yellow ROI, then crop if needed.
4. Adjust channel visibility, contrast, colormaps, and scale-bar settings.
5. Preview the montage or enqueue TIFF/JPEG exports.
6. Check the Save Queue before closing the application.

Changing the FOV clears the active full-resolution plane cache. Save jobs retain the image snapshot and display settings captured when the Save button was pressed.

## Other launchers

- `run_app.py`: original viewer
- `run_app_faster.py`: projection and ROI caching
- `run_app_faster_threaded.py`: caching plus background FOV thumbnails
- `run_orthogonal_view.py`: orthogonal-view workflow
- `run_segmentation.py`: segmentation workflow

## Output behavior

- OME-TIFF exports preserve numerical channel data and pixel-size metadata. Display contrast and colormaps do not alter TIFF intensities.
- JPEG exports use the current display contrast, colormaps, opacity, channel visibility, and scale-bar settings.
- The default saved scale bar is 50 µm and uses adaptive visual sizing.
- Output filenames receive numbered suffixes when necessary to prevent accidental overwrites during the same session.

## Data and privacy

ND2 files and generated microscopy images are intentionally excluded from Git. Do not commit datasets containing private, clinical, or identifying information.
