import os

def is_in_use(path):
    try:
        fd = os.open(path, os.O_RDWR | os.O_EXCL)
        os.close(fd)
        return False
    except OSError:
        return True

directories = [
    r"C:\Research\GameAgents\15puzzle-datasets\raw\local_20260304",
    r"C:\Research\GameAgents\15puzzle-datasets\raw\local_20260312",
    r"C:\Research\GameAgents\15puzzle-datasets\raw\local_20260324",
]

for d in directories:
    print(f"\nChecking: {d}")
    active = []
    for f in os.listdir(d):
        full = os.path.join(d, f)
        if os.path.isfile(full) and is_in_use(full):
            active.append(full)

    if active:
        for a in active:
            print("  ACTIVE:", a)
    else:
        print("  No active files")
