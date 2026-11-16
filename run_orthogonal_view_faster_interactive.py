from nd2_editor.orthogonal_app_faster_interactive import launch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast cached ND2 orthogonal viewer with interactive Z-range lines"
    )
    parser.add_argument("nd2_path", help="Path to the ND2 file")
    parser.add_argument("-o", "--out-dir", default=None, help="Output directory")
    args = parser.parse_args()
    launch(args.nd2_path, out_dir=args.out_dir)
