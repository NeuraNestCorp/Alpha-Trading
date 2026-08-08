from pathlib import Path
import hashlib,base64
here=Path(__file__).parent
exp=[
('532f5553f577e74f98152715f52040e929fd98ae4c24482da8b028ca9d4060f0',5500),
('e976b82662b5e43234a9ce9fdbc832fe028d04a19111679b84da61c3ac4574fd',5500),
('40dd481c9a9635254cb7c844a53f27e0422cee9e135c57f94aac64a0e7d5bd49',5500),
('ad3c28648211909d453c5ec7d30ff11887c77af9c530219340e59388ac8d5b2a',4876),
]
subexp=[
('2a909172a13f711d86e0eafa60a2517bb27d576ab005dc5afa531a261061f7b4',1000),
('e050e911579e10a0a7f2e01b2b8b4f368f9be4a00ce5ec0e60654826d1a1a2a9',1000),
('8ac0899dac01eac1891be19b10ca634d51e125fcf868e451b7f4d65ca16cfa0a',1000),
('80c151e548fd9f7dc5cbe7f2104859f1426603f22eaf6518bd62d9aa56222ad4',1000),
('4d8b664cd93bbefb5724a37f63c18e8a119570313f9556f5e8caf8e5f7fb970f',1000),
('c8827e73b34dbb8f09010804cd3292526175e1e157eedaec544af56865ec2771',500),
]
def getpart(i):
    if i != 1: return (here/f'archive_{i}.b64').read_bytes().strip()
    xs=[]
    for j,(eh,el) in enumerate(subexp):
        b=(here/'archive_1_parts'/f'part_{j}.b64').read_bytes().strip(); h=hashlib.sha256(b).hexdigest(); ok=(len(b)==el and h==eh); print('sub',j,len(b),h,'OK' if ok else 'BAD');
        if not ok: raise SystemExit(f'SUBPART {j} FAILED')
        xs.append(b)
    return b''.join(xs)
parts=[]
for i,(eh,el) in enumerate(exp):
    b=getpart(i); h=hashlib.sha256(b).hexdigest(); ok=(len(b)==el and h==eh); print(i,len(b),h,'OK' if ok else 'BAD');
    if not ok: raise SystemExit(f'BUNDLE PART {i} FAILED')
    parts.append(b)
s=b''.join(parts); print('total',len(s),'mod4',len(s)%4); print('decoded',len(base64.b64decode(s,validate=True))); print('TRANSFER_BUNDLE_CHECK=PASS')
