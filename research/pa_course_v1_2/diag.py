from pathlib import Path
import hashlib,base64
here=Path(__file__).parent
exp=[
('532f5553f577e74f98152715f52040e929fd98ae4c24482da8b028ca9d4060f0',5500),
('e976b82662b5e43234a9ce9fdbc832fe028d04a19111679b84da61c3ac4574fd',5500),
('40dd481c9a9635254cb7c844a53f27e0422cee9e135c57f94aac64a0e7d5bd49',5500),
('ad3c28648211909d453c5ec7d30ff11887c77af9c530219340e59388ac8d5b2a',4876),
]
def getpart(i):
    if i != 1:
        return (here/f'archive_{i}.b64').read_bytes().strip()
    return b''.join((here/'archive_1_parts'/f'part_{j}.b64').read_bytes().strip() for j in range(6))
parts=[]
for i,(eh,el) in enumerate(exp):
    b=getpart(i); h=hashlib.sha256(b).hexdigest(); ok=(len(b)==el and h==eh); print(i,len(b),h,'OK' if ok else 'BAD');
    if not ok: raise SystemExit(f'BUNDLE PART {i} FAILED')
    parts.append(b)
s=b''.join(parts); print('total',len(s),'mod4',len(s)%4); print('decoded',len(base64.b64decode(s,validate=True))); print('TRANSFER_BUNDLE_CHECK=PASS')
