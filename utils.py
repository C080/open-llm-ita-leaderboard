def print_colored(text, color):
    if color == 'red':
        print(f"\033[91m{text}\033[00m")
    elif color == 'green':
        print(f"\033[92m{text}\033[00m")
    elif color == 'yellow':
        print(f"\033[93m{text}\033[00m")
    elif color == 'blue':
        print(f"\033[94m{text}\033[00m")
    elif color == 'magenta':
        print(f"\033[95m{text}\033[00m")
    elif color == 'cyan':
        print(f"\033[96m{text}\033[00m")
    elif color == 'white':
        print(f"\033[97m{text}\033[00m")
    elif color == 'black':
        print(f"\033[98m{text}\033[00m")
    else:
        print(f"Color {color} not found")
        print(text)

