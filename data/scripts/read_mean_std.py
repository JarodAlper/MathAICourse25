def read_mean_std(path: str):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    mean = float(lines[0])
    std = float(lines[1])
    return mean, std


def main() -> None:
    path = "/Users/jarod/gitwork/math-ai-course/data/fib_stats.txt"
    mean, std = read_mean_std(path)
    print(mean, std)


if __name__ == "__main__":
    main()
