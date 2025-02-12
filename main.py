import subprocess


def main() -> None:
    subprocess.run(["uv", "run", "04-features_for_inference.py"])
    subprocess.run(["uv", "run", "06-inference.py"])


if __name__ == "__main__":
    main()
