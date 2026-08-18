if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Open an ND2 file in ND2 Editor CDK."
    )
    parser.add_argument("nd2_path", help="Path to the input ND2 file")
    parser.add_argument(
        "-o",
        "--out-dir",
        help=(
            "Parent output destination. The app creates an <ND2 "
            "filename>_maxproj folder inside it. Defaults to the input "
            "file's parent folder."
        ),
    )
    args = parser.parse_args()

    from nd2_editor.napari_app_faster_threaded_savequeue import launch

    launch(args.nd2_path, out_dir=args.out_dir)
