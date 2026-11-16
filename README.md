# ND2 Editor CDK

ND2 Editor CDK is a desktop microscopy viewer and exporter for Nikon ND2 files. It uses Napari to inspect FOVs and Z ranges, create maximum-intensity projections, crop an XY region of interest, preview channel montages, and export OME-TIFF or JPEG images.

## Recommended launcher

The most complete version is:

```powershell
python run_app_faster_threaded_savequeue.py path\to\image.nd2
```

It provides:

- Cached Z planes and faster projection updates
- Background prefetching for up to three neighboring FOV projections
- Fast, non-destructive ROI display cropping
- Background FOV-thumbnail loading
- Visible-first thumbnail loading followed by low-priority loading of every FOV
- Background montage preview generation
- A serial background save queue with progress and error reporting
- Terminal or in-app output-folder selection
- FOV-loading progress
- OME-TIFF export with channel and pixel-size metadata
- Per-channel and montage JPEG export
- Adaptive burned-in scale bars for JPEG output
- Optional montage marker labels taken from the editable Napari channel-layer names

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

Choose a specific output folder with `--out-dir` (or `-o`):

```powershell
python run_app_faster_threaded_savequeue.py C:\path\to\image.nd2 --out-dir D:\analysis
```

The selected destination is a parent folder. The app creates `D:\analysis\image_maxproj` in this example. OME-TIFF files are saved inside `image_maxproj`, with channel and montage JPEGs in its `jpeg_channels` and `jpeg_montage` subfolders. Without this option, `<ND2 filename>_maxproj` is created next to the input file.

The parent destination can also be changed while the app is running with **Choose output destination**. The app creates `<ND2 filename>_maxproj` inside the selected location. New save jobs use the destination selected at the moment they are queued.

On Windows, very deep destinations can exceed the legacy path-length limit when combined with descriptive image filenames. ND2 Editor CDK automatically removes the repeated ND2 stem and, if necessary, safely compacts the filename while retaining FOV/Z information and a collision-resistant hash.

## Main workflow

1. Select an FOV from the navigator.
2. Choose the desired Z range and update the view.
3. Set and position the yellow ROI, then crop if needed.
4. Adjust channel visibility, contrast, colormaps, and scale-bar settings.
5. Rename channel layers if you want custom marker labels, enable **Marker names**, and choose the marker font size. Labels use Tahoma Bold in the montage margin above each image, without a separate filled label box. Fluorescence text follows its channel color; brightfield and the combined **Merge** tile use white.
6. Preview the montage or enqueue TIFF/JPEG exports.
7. Check the Save Queue before closing the application.

## Keyboard shortcuts

With the Napari canvas focused:

- `M`: preview montage
- `Shift+M`: save montage
- `N`: move to the next FOV
- `S`: apply the ROI width and height currently shown in the ROI controls
- `C`: crop to the selected ROI

Changing the FOV clears the active full-resolution plane cache. Save jobs retain the image snapshot and display settings captured when the Save button was pressed.

## Faster orthogonal viewer

Launch the cached orthogonal-view workflow with:

```powershell
python run_orthogonal_view_faster.py C:\path\to\image.nd2
```

An output folder can be supplied from the terminal or selected later in Napari:

```powershell
python run_orthogonal_view_faster.py C:\path\to\image.nd2 --out-dir D:\analysis\orthogonal
```

This version keeps the original orthogonal viewer unchanged while adding:

- Background loading and progress for full-resolution FOV volumes
- A memory-bounded current-FOV volume cache
- Cached rotated volumes for faster Z-range and slab updates
- Cropping from cached data and in-place ROI resizing
- Background loading of the complete FOV thumbnail strip
- A serial background save queue for Fiji TIFF and merged JPEG output
- Rename-safe channel tracking and an in-app output-folder chooser
- `N`, `S`, and `C` shortcuts for next FOV, ROI sizing, and cropping

Orthogonal volumes can be large, so the app caches one complete FOV at a time instead of preloading several neighboring 3D volumes. Closing Napari shuts down the loading, thumbnail, and saving workers cleanly.

For interactive projection boundaries and batch slicing, use:

```powershell
python run_orthogonal_view_faster_interactive.py C:\path\to\image.nd2
```

This version adds draggable XY Z-range boundaries, editable crosshair coordinates, cached dependency-aware XY/XZ/YZ updates, individual-view export, and background XY/XZ/YZ batch exports with preview bands and CSV indexes.

The main `run_app_faster_threaded_savequeue.py` viewer also includes a single scrollable tools sidebar, an editable Napari `ANNOTATIONS` Shapes layer with optional solid or dashed JPEG rendering, and an in-app **Load new ND2 file** action. Loading another file preserves channel names, contrast limits, colormaps, visibility, opacity, blending, and display/export preferences while clearing the previous file's image caches and workers.

## Other launchers

- `run_app.py`: original viewer
- `run_app_faster.py`: projection and ROI caching
- `run_app_faster_threaded.py`: caching plus background FOV thumbnails
- `run_orthogonal_view.py`: original orthogonal-view workflow
- `run_orthogonal_view_faster.py`: cached, threaded orthogonal-view workflow
- `run_orthogonal_view_faster_interactive.py`: interactive boundaries and batch orthogonal exports
- `run_segmentation.py`: segmentation workflow

## Output behavior

- OME-TIFF exports preserve numerical channel data and pixel-size metadata. Display contrast and colormaps do not alter TIFF intensities.
- JPEG exports use the current display contrast, colormaps, opacity, channel visibility, and scale-bar settings.
- The default saved scale bar is 50 µm and uses adaptive visual sizing.
- Output filenames receive numbered suffixes when necessary to prevent accidental overwrites during the same session.

## Data and privacy

ND2 files and generated microscopy images are intentionally excluded from Git. Do not commit datasets containing private, clinical, or identifying information.
