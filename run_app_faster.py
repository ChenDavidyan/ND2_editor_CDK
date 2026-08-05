from nd2_editor.napari_app_faster import launch


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_app_faster.py path/to/file.nd2")
    launch(sys.argv[1])
