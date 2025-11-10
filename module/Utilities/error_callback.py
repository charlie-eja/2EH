import traceback
from pathlib import Path

def find_project_root(start_path: Path = None) -> Path:
    """Automatically detect the project root directory"""
    if start_path is None:
        start_path = Path(__file__).resolve()
    for parent in [start_path, *start_path.parents]:
        for marker in [".git", "pyproject.toml", "setup.py", "requirements.txt"]:
            if (parent / marker).exists():
                return parent
    return start_path.parent

def print_project_trace(e):
    '''print error massage'''
    RED_BOLD = "\033[1;31m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    project_root = find_project_root()
    frames = traceback.extract_tb(e.__traceback__)
    lines = []

    for fr in frames:
        path = Path(fr.filename).resolve()
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            continue
        lines.append(f'{GRAY}  File "{rel}", line {fr.lineno}, in {fr.name}{RESET}\n    {fr.line}')
    '''If lines is empty , but the overall frames is not empty '''
    if not lines and frames:
        fr = frames[-1]
        lines.append(f'{GRAY}  File "{fr.filename}", line {fr.lineno}, in {fr.name}{RESET}\n    {fr.line}')

    print("")
    print("\n".join(lines))
    print(f"{RED_BOLD}{type(e).__name__}: {e}{RESET}")

if __name__ == "__main__":
    try:
        int("s")
    except Exception as e:
        print_project_trace(e)
