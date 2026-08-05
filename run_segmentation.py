from nd2_editor.segmentation_app import launch

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_segmentation.py path/to/file.nd2")
    launch(sys.argv[1])
