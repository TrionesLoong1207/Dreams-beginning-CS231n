"""Adapt only infrastructure cells for Windows. Does not fill exercise TODOs."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
MARKER = '# CS231n local environment setup (exercise code unchanged)'


def main():
    for number in (1, 2):
        for path in sorted((ROOT / f'assignment{number}').glob('*.ipynb')):
            if path.name == 'collect_submission.ipynb':
                continue
            notebook = json.loads(path.read_text(encoding='utf-8'))
            changed = False
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                if 'drive.mount(' in source and 'from google.colab import drive' in source and not source.lstrip().startswith('# from'):
                    cell['source'] = f'''{MARKER}
from pathlib import Path
import os
import sys

_candidates = [Path.cwd(), *Path.cwd().parents]
_workspace = next((p for p in _candidates if (p / "requirements-local.txt").is_file()
                   and (p / "assignment{number}" / "cs231n").is_dir()), None)
if _workspace is None:
    raise RuntimeError("Launch Jupyter from the CS231n workspace root.")
ASSIGNMENT_DIR = _workspace / "assignment{number}"
os.chdir(ASSIGNMENT_DIR)
if str(ASSIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_DIR))
print("Python:", sys.executable)
print("Working directory:", Path.cwd())
'''.splitlines(keepends=True)
                    changed = True
                elif 'setup.py build_ext --inplace' in source:
                    cell['source'] = '''# Build the provided convolution extension using the active kernel's Python.
import subprocess
subprocess.check_call(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    cwd=ASSIGNMENT_DIR / "cs231n",
)
# Restart this notebook's kernel after the first successful build.
'''.splitlines(keepends=True)
                    changed = True
            if changed:
                path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
                print('Adapted setup:', path.relative_to(ROOT))


if __name__ == '__main__':
    main()
