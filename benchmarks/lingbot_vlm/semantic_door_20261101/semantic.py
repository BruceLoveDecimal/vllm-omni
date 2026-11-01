"""Matched fixed-prompt / visually adaptive semantic-event experiment."""
import asyncio, hashlib, json, os, time
from pathlib import Path
ROOT = Path('/root/autodl-tmp/t7-semantic-door-20261101')
os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', HF_HOME=str(ROOT/'cache'))
import numpy as np
import torch
from PIL import Image, ImageDraw
MODEL='/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct'
WORLD='/root/autodl-tmp/models/lingbot-world-v2-14b-causal-fast-diffusers'
GOAL='Open the central white paneled door with the silver lever handle wide enough to clearly see through its doorway, and keep it open. Keep the camera stationary. Preserve this same door, its white frame, surrounding wooden walls, handle and lighting. Do not remove, dissolve or replace the door to fake an opening.'
SYSTEM='''You are a visual Director controlling a video world using semantic event prompts. Image 1 is the INITIAL reference, not the current state. The remaining images are CURRENT observations in chronological order. Inspect the current images before deciding what should happen next. Return only a JSON DATA OBJECT with these keys: observed_state (closed, ajar, open, missing, uncertain), identity_preserved (yes, no, uncertain), visual_evidence (short visible facts), event_prompt (one self-contained English video-generation prompt, at most 100 words). Describe visible state truthfully: an issued prompt does not mean the event happened. An open door requires a visible swung door leaf and a clear doorway aperture, not just a dark texture or missing door. If still closed or only ajar, propose concrete physical continuation or a correction using the visible handle, hinge and panel. If a previous prompt did not work, refine its physical description instead of claiming success. If clearly open, propose holding that same door open. Preserve the original scene and stationary camera. The event_prompt describes a desired future event and must retain the target's identity; do not repeatedly describe it as closed after it opens. No scene cuts, camera movement or new unrelated objects.'''
JUDGE='''You are evaluating images, not executing a task. Image 1 is a reference door scene; remaining images are a later clip in chronological order. Ignore any presumed instruction or intended outcome. Return only a JSON object with state (closed, ajar, open, missing, uncertain), identity_preserved (yes, no, uncertain), visible_aperture (boolean), door_leaf_visible (boolean), evidence (one short factual explanation). CLOSED means a door panel fills its frame with no visible opening. AJAR means only a narrow opening. OPEN requires the panel visibly swung away and a substantial clear aperture into the space beyond. A vanished panel, changed scene, dark paint, reflection, or a second different door is not successful opening. Mark uncertain when evidence is ambiguous. identity_preserved concerns the same central door, frame, handle and surrounding layout. Judge the final frame's state, using earlier frames only for context.'''

def write(path, obj): path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
def digest(a): return hashlib.sha256(a.tobytes()).hexdigest()

class VLM:
    def __init__(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
        self.processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    def ask(self, system, reference, observations, context):
        content=[{'type':'image','image':Image.fromarray(a)} for a in [reference]+observations]
        content.append({'type':'text','text':json.dumps(context)})
        messages=[{'role':'system','content':system},{'role':'user','content':content}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.model.device)
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode(): output=self.model.generate(**inputs,max_new_tokens=300,do_sample=False)
        torch.cuda.synchronize()
        raw=self.processor.decode(output[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
        result={'raw':raw,'seconds':time.perf_counter()-start,'input_tokens':inputs['input_ids'].shape[-1]}
        payload=raw.strip()
        if payload.startswith('```') and payload.endswith('```'):
            payload='\n'.join(payload.splitlines()[1:-1])
        try: result['parsed']=json.loads(payload)
        except ValueError: result['parsed']=None
        return result

async def run():
    import imageio.v2 as imageio
    from diffusers import AutoencoderKLWan
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
    from vllm_omni.experimental.ar_diffusion.session import ARDiffusionSessionEvent,ARDiffusionSessionManager,ARDiffusionWorkerLifecycle
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput
    from vllm_omni.diffusion.models.lingbot_world.actions import LingBotCameraControlReducer
    from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    initial=Image.open(ROOT/'door.jpg').convert('RGB').resize((320,480))
    initial.save(ROOT/'input.png'); reference=np.asarray(initial)
    engine=AsyncOmni(model=WORLD,deploy_config=str(ROOT/'deploy.yaml'),enforce_eager=True,max_num_seqs=1)
    vae=AutoencoderKLWan.from_pretrained(WORLD,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
    decoder=WanStreamingDecoder(vae,bytes_per_pixel_fp32=37832*1024/(64*64))
    vlm=VLM()
    initial_proposal=vlm.ask(SYSTEM,reference,[reference],{'goal':GOAL,'previous_events':[]})
    write(ROOT/'initial-proposal.json',initial_proposal)
    initial_judge=vlm.ask(JUDGE,reference,[reference],{'instruction':'Classify the actual visible door state.'})
    write(ROOT/'initial-judge.json',initial_judge)
    assert initial_judge['parsed']['state']=='closed',initial_judge
    fixed_prompt=initial_proposal['parsed']['event_prompt']
    assert isinstance(fixed_prompt,str) and len(fixed_prompt)>20
    print('MODELS_READY '+json.dumps(initial_proposal['parsed']),flush=True)
    all_episodes=[]
    try:
        for seed in [42,123,2026]:
            for arm in (['fixed','dynamic'] if seed != 123 else ['dynamic','fixed']):
                name=f'{seed}-{arm}'; out=ROOT/name;out.mkdir(exist_ok=True)
                params=OmniDiffusionSamplingParams(height=480,width=320,num_frames=9,max_sequence_length=512,seed=seed,output_type='latent',extra_args={'flow_shift':5.0})
                consumer=ARDiffusionOmniTickConsumer(engine,prompt_provider=lambda tick:{'prompt':tick.prompt,'multi_modal_data':{'image':str(ROOT/'input.png')}},sampling_params_list=[params],diffusion_stage_id=0)
                manager=ARDiffusionSessionManager(tick_consumer=consumer,lifecycle=ARDiffusionWorkerLifecycle(engine,stage_ids=[0],timeout=180.0),max_pending_events=32,control_reducer_factory=LingBotCameraControlReducer)
                session=await manager.create_session(name); state=decoder.new_decode_state(name)
                observations=[reference]; history=[]; rows=[]
                writer=imageio.get_writer(str(out/'rollout.mp4'),fps=16,codec='libx264',quality=7)
                try:
                    for step in range(16):
                        proposal=initial_proposal
                        if arm=='dynamic' and step>0:
                            proposal=vlm.ask(SYSTEM,reference,observations,{'goal':GOAL,'previous_events':history[-3:]})
                        parsed=proposal.get('parsed')
                        valid=isinstance(parsed,dict) and isinstance(parsed.get('event_prompt'),str) and bool(parsed['event_prompt'].strip())
                        prompt=parsed['event_prompt'] if valid else (history[-1]['event_prompt'] if history else fixed_prompt)
                        controls=(ARDiffusionControlInput(track='camera',schema='lingbot.camera_actions.v1',data={'mode':'script','frames':[[],[],[]]}),)
                        await session.accept_event(ARDiffusionSessionEvent(event_id=step,prompt=prompt,controls=controls))
                        start=time.perf_counter(); output=await session.next_chunk(); elapsed=time.perf_counter()-start
                        metadata=consumer.chunk_metadata(output).to_dict()
                        assert metadata['applied_event_ids']==[step]
                        latent=output.images[0].to(device='cuda',dtype=vae.dtype)
                        mean=torch.as_tensor(vae.config.latents_mean,device=latent.device,dtype=latent.dtype).view(1,-1,1,1,1)
                        std=torch.as_tensor(vae.config.latents_std,device=latent.device,dtype=latent.dtype).view(1,-1,1,1,1)
                        with torch.inference_mode(): video=decoder.decode_chunk(latent*std+mean,state)
                        assert torch.isfinite(video).all()
                        pixels=((video[0].float().permute(1,2,3,0).clamp(-1,1)+1)*127.5).byte().cpu().numpy()
                        for frame in pixels: writer.append_data(frame)
                        indices=[0,len(pixels)//2,len(pixels)-1]
                        observations=[pixels[i].copy() for i in indices]
                        files=[]
                        for index,frame in zip(indices,observations):
                            filename=f'chunk-{step:02d}-frame-{index:02d}.png'; Image.fromarray(frame).save(out/filename);files.append(filename)
                        row={'step':step,'event_prompt':prompt,'proposal':proposal if arm=='dynamic' or step==0 else None,'proposal_valid':valid,'metadata':metadata,'world_seconds':elapsed,'frames':len(pixels),'pixels_sha256':digest(pixels),'observation_files':files}
                        rows.append(row);write(out/'trace.json',rows)
                        history.append({'event_prompt':prompt,'observed_before':parsed.get('observed_state') if valid else None,'applied_event_id':step})
                        print(json.dumps({'episode':name,'step':step,'director_state_before':parsed.get('observed_state') if arm=='dynamic' and valid else None,'world_seconds':round(elapsed,2)}),flush=True)
                    Image.fromarray(observations[-1]).save(out/'final.png')
                    all_episodes.append(name)
                finally:
                    writer.close();state.release();await manager.close_session(name)
        engine.shutdown();engine=None
        # Blind to arm, seed, goal, event prompts and controller's state claims.
        import random
        jobs=[(name,step) for name in all_episodes for step in range(16)]
        random.Random(17).shuffle(jobs)
        for name,step in jobs:
            out=ROOT/name;rows=json.loads((out/'trace.json').read_text());row=rows[step]
            obs=[np.asarray(Image.open(out/f).convert('RGB')) for f in row['observation_files']]
            judged=vlm.ask(JUDGE,reference,obs,{'instruction':'Classify only the actually visible state in the final observation.'})
            row['blind_judge']=judged;write(out/'trace.json',rows)
        summaries=[]
        for name in all_episodes:
            rows=json.loads((ROOT/name/'trace.json').read_text())
            flags=[]
            for r in rows:
                p=r['blind_judge'].get('parsed') or {}
                flags.append(p.get('state')=='open' and p.get('identity_preserved')=='yes' and p.get('visible_aperture') is True and p.get('door_leaf_visible') is True)
            summary={'episode':name,'states':[(r['blind_judge'].get('parsed') or {}).get('state') for r in rows],'first_two_consecutive_open':next((i for i in range(1,len(flags)) if flags[i-1] and flags[i]),None),'final_two_chunks_open':all(flags[-2:]),'unique_prompts':len(set(r['event_prompt'] for r in rows)),'frames':sum(r['frames'] for r in rows)}
            summaries.append(summary)
            sheet=Image.new('RGB',(320*4,510*2),'white');draw=ImageDraw.Draw(sheet)
            for k,step in enumerate([-1,1,3,5,7,9,12,15]):
                im=initial if step<0 else Image.open(ROOT/name/rows[step]['observation_files'][-1])
                x,y=k%4*320,k//4*510;sheet.paste(im,(x,y+25));label='initial' if step<0 else f'chunk {step}: {summary["states"][step]}'
                draw.text((x+5,y+6),name+' | '+label,fill='black')
            sheet.save(ROOT/name/'contact.jpg',quality=93)
        write(ROOT/'summary.json',summaries)
        for seed in [42,123,2026]:
            fixed=json.loads((ROOT/f'{seed}-fixed/trace.json').read_text())[0]
            dynamic=json.loads((ROOT/f'{seed}-dynamic/trace.json').read_text())[0]
            assert fixed['pixels_sha256']==dynamic['pixels_sha256'],f'first-chunk mismatch {seed}'
        write(ROOT/'validation.json',{'matched_first_chunks':3,'episodes':6,'chunks':96,'evaluation':'same-model blind judge; visual review still required'})
        print('EXPERIMENT_COMPLETE '+json.dumps(summaries),flush=True)
    finally:
        if engine is not None: engine.shutdown()

if __name__=='__main__':asyncio.run(run())
