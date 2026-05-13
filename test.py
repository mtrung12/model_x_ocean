def add(a, b):
    result = a + b
    return result


def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def process_list(items):
    results = []
    for item in items:
        squared = item ** 2
        results.append(squared)
    return results


def main():
    # Arithmetic
    x = 10
    y = 5
    total = add(x, y)
    print(f"add({x}, {y}) = {total}")

    # Factorial
    n = 6
    fact = factorial(n)
    print(f"factorial({n}) = {fact}")

    # List processing
    numbers = [1, 2, 3, 4, 5]
    squared = process_list(numbers)
    print(f"squares of {numbers} = {squared}")

    # Dict iteration
    scores = {"Alice": 95, "Bob": 82, "Carol": 88}
    for name, score in scores.items():
        grade = "A" if score >= 90 else "B"
        print(f"{name}: {score} -> {grade}")


if __name__ == "__main__":
    main()
