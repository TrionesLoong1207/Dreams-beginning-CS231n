"""Check infrastructure only; never execute or fill exercise implementations."""
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local'
(LOCAL / 'tmp').mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(LOCAL / 'tmp')
os.environ['JUPYTER_RUNTIME_DIR'] = str(LOCAL / 'jupyter' / 'runtime')


def main():
    import numpy as np
    import scipy
    import matplotlib
    import h5py
    import torch
    import torchvision
    print('Python:', sys.version.split()[0], flush=True)
    print('NumPy:', np.__version__, 'SciPy:', scipy.__version__, flush=True)
    print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__, flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available in this environment.')
    x = torch.randn(32, 32, device='cuda')
    y = x @ x.T
    torch.cuda.synchronize()
    assert torch.isfinite(y).all()
    print('CUDA computation OK:', torch.cuda.get_device_name(0), flush=True)

    sys.path.insert(0, str(ROOT / 'assignment2'))
    from cs231n.im2col_cython import im2col_cython
    assert im2col_cython(np.zeros((1, 3, 8, 8)), 3, 3, 1, 1).shape == (27, 64)
    print('Provided Cython extension: OK', flush=True)

    import nbformat
    from nbclient import NotebookClient
    original = nbformat.read(ROOT / 'assignment1' / 'knn.ipynb', as_version=4)
    # Only setup, imports and the provided data-loading cell. No exercise cells.
    cells = [c for c in original.cells if c.cell_type == 'code'][:3]
    smoke = nbformat.v4.new_notebook(cells=cells)
    client = NotebookClient(smoke, timeout=180, kernel_name='cs231n',
                            resources={'metadata': {'path': str(ROOT / 'assignment1')}})
    client.execute()
    nbformat.write(smoke, LOCAL / 'knn-environment-check.ipynb')
    for cell in smoke.cells:
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stream':
                print(output.text.strip(), flush=True)
    print('kNN setup, imports and CIFAR-10 loading: OK. No exercise code executed.', flush=True)


if __name__ == '__main__':
    main()
