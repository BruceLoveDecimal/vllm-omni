import hashlib,json,tarfile
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
pairs=[]
for seed in [42,123,2026]:
    traces=[json.loads((root/f'{seed}-{arm}/trace.json').read_text()) for arm in ['fixed','dynamic']]
    pairs.append({'seed':seed,'identical_initial_prompt':traces[0][0]['event_prompt']==traces[1][0]['event_prompt'],'identical_first_chunk':traces[0][0]['pixels_sha256']==traces[1][0]['pixels_sha256'],'first_different_chunk':next((i for i,(a,b) in enumerate(zip(*traces)) if a['pixels_sha256']!=b['pixels_sha256']),None)})
    for arm,rows in zip(['fixed','dynamic'],traces):
        out=root/f'{seed}-{arm}'
        canvas=Image.new('RGB',(1280,2020),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows):
            im=Image.open(out/r['observation_files'][-1]).convert('RGB')
            x,y=i%4*320,i//4*505
            canvas.paste(im,(x,y+25));state=(r['blind_judge']['parsed'] or {}).get('state','invalid')
            draw.text((x+5,y+5),f'{seed}-{arm} chunk {i}: {state}',fill='black')
        canvas.save(out/'all-chunks.jpg',quality=93)
        tail=Image.new('RGB',(960,1010),'white');draw=ImageDraw.Draw(tail)
        for k,(step,filename) in enumerate((s,f) for s in [14,15] for f in rows[s]['observation_files']):
            x,y=k%3*320,k//3*505
            tail.paste(Image.open(out/filename),(x,y+25))
            draw.text((x+5,y+5),f'{seed}-{arm} {filename}',fill='black')
        tail.save(out/'tail.jpg',quality=93)
    readers=[imageio.get_reader(str(root/f'{seed}-{arm}/rollout.mp4')) for arm in ['fixed','dynamic']]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    with imageio.get_writer(str(root/f'comparison-{seed}.mp4'),fps=16,codec='libx264',quality=6) as writer:
        for i,frames in enumerate(zip(*readers)):
            step=0 if i<9 else 1+(i-9)//12
            canvas=Image.new('RGB',(640,528),'#202028');draw=ImageDraw.Draw(canvas)
            for k,(arm,frame) in enumerate(zip(['fixed','dynamic'],frames)):
                canvas.paste(Image.fromarray(frame),(k*320,48))
                state=(traces[k][step]['blind_judge']['parsed'] or {}).get('state','invalid')
                draw.text((k*320+5,5),f'{arm} | seed {seed} | chunk {step}',font=font,fill='white')
                draw.text((k*320+5,25),'VLM judge: '+state,font=font,fill='white')
            writer.append_data(np.asarray(canvas))
    for r in readers:r.close()
(root/'paired-checks.json').write_text(json.dumps(pairs,indent=2))
paths=[]
for p in root.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and 'cache' not in p.parts:
        if p.suffix in {'.json','.py','.md','.yaml','.log','.jpg'} or p.name in {'input.png','final.png'} or p.name.startswith('comparison-'):
            if p.name not in {'manifest.json','package.json'}: paths.append(p)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));paths.append(root/'manifest.json')
archive=root.parent/'t7-semantic-door-evidence-20261101.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in paths:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'pairs':pairs}))
