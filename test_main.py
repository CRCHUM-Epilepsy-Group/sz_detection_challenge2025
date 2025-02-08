import subprocess


def main() -> None:
    subprocess.run(["uv", "run", "02-feature_extraction.py"])
    subprocess.run(["uv", "run", "06-inference.py"])


if __name__ == "__main__":
    main()
