import os
import argparse

EXCLUDED_DIRS = ["__pycache__", "__marimo__"]


def traverse_and_write(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Exclude directories named '__pycache__' and '__marimo__'
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(filepath, root_dir)
                out.write(f"# {rel_path}\n")
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    content = f"Error reading file: {e}\n"
                out.write(content)
                out.write("\n")  # extra newline between files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traverse a directory recursively and write file contents with relative paths as headers."
    )
    parser.add_argument("directory", help="Path of the directory to traverse.")
    parser.add_argument(
        "-o",
        "--output",
        default="output.txt",
        help="Output text file (default: output.txt)",
    )
    args = parser.parse_args()
    traverse_and_write(args.directory, args.output)