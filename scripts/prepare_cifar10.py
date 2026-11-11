"""Download the official CIFAR-10 archive, verify it, and share it with all assignments."""
from pathlib import Path
import hashlib
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / '.cache' / 'cifar-10-python.tar.gz'
CHECKSUM = 'c58f30108f718f92721af3b95e74349a'
URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
MIRROR = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-python.tar.gz'


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists() or hashlib.md5(ARCHIVE.read_bytes()).hexdigest() != CHECKSUM:
        print('Downloading official CIFAR-10 archive (~163 MiB)...', flush=True)
        # Same archive and checksum; this mirror is faster on the current network.
        request = urllib.request.Request(MIRROR, headers={'User-Agent': 'CS231n-local-setup'})
        with urllib.request.urlopen(request, timeout=60) as response, ARCHIVE.open('wb') as output:
            shutil.copyfileobj(response, output)
    if hashlib.md5(ARCHIVE.read_bytes()).hexdigest() != CHECKSUM:
        raise RuntimeError('CIFAR-10 checksum mismatch. Archive was not extracted.')
    roots = [ROOT / 'assignment1', ROOT / 'assignment2', ROOT / 'assignment3' / 'assignment3']
    with tarfile.open(ARCHIVE) as archive:
        members = archive.getmembers()
        for member in members:
            if (member.name != 'cifar-10-batches-py' and not member.name.startswith('cifar-10-batches-py/')) or member.issym() or member.islnk():
                raise RuntimeError(f'Unexpected archive entry: {member.name}')
        archive.extractall(roots[0] / 'cs231n' / 'datasets', filter='data')
    source = roots[0] / 'cs231n' / 'datasets' / 'cifar-10-batches-py'
    for root in roots[1:]:
        shutil.copytree(source, root / 'cs231n' / 'datasets' / source.name, dirs_exist_ok=True)
    print('CIFAR-10 checksum verified; available in all three assignment directories.', flush=True)


if __name__ == '__main__':
    main()
