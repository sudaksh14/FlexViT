def generate_head_sizes(n, min_val, max_val, num_head=12):
    """
    Generate a list of `n` values between `min_val` and `max_val`
    (inclusive) where each value is divisible by `divisor`.

    Args:
        n (int): Number of values to generate.
        min_val (int): Minimum value.
        max_val (int): Maximum value.
        divisor (int): The number each value must be divisible by.

    Returns:
        list: List of integers meeting the criteria.
    """
    # Ensure bounds are multiples of divisor
    min_val = ((min_val + num_head - 1) // num_head) * num_head
    max_val = (max_val // num_head) * num_head

    # Generate all valid multiples in range
    possible_values = list(range(min_val, max_val + 1, num_head))

    if n > len(possible_values):
        raise ValueError(f"Cannot generate {n} unique values in range.")

    # Evenly spaced selection
    step = max(1, len(possible_values) // n)
    return possible_values[::step][:n]

if __name__ == "__main__":
    try:
        parts = generate_head_sizes(n=9, min_val=192, max_val=768)
        parts = [part / 12 for part in parts]  # Adjusting for num_head
        print(f"Split into parts: {parts}")
    except ValueError as e:
        print(f"Error: {e}")