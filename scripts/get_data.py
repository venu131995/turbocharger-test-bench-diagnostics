"""Download the Marine Engine Fault dataset (Zenodo 10.5281/zenodo.19857425, CC BY 4.0) into data/ and check it."""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://zenodo.org/api/records/19857425/files/Marine_Engine_Fault_Data_v1.zip/content'
MD5 = 'f4d246c1bb46e05b26e56221acc2606c'
DATA = Path(__file__).resolve().parents[1] / 'data'

if __name__ == '__main__':
    raw = urllib.request.urlopen(URL, timeout=120).read()
    got = hashlib.md5(raw).hexdigest()
    if got != MD5:
        raise SystemExit(f'checksum mismatch: {got} != {MD5}')
    DATA.mkdir(exist_ok=True)
    zipfile.ZipFile(io.BytesIO(raw)).extractall(DATA)
    print('extracted to', DATA / 'Marine_Engine_Fault_Data')
