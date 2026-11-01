import hashlib,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
results=[];comparisons=[]
for sub,policy in [('', '扩展记忆'),('sliding-only','仅滑窗与 sink')]:
    base=root/sub
    summaries=json.loads((base/'summary.json').read_text())
    for arm in ['fixed','dynamic']:
        selected=[x for x in summaries if x['episode'].endswith('-'+arm)]
        observed=[];proposals=[];disagreement=0;invalid=0;online=[];world=[]
        for episode in selected:
            rows=json.loads((base/episode['episode']/'trace.json').read_text())
            states=[]
            for r in rows:
                assert r['metadata']['applied_event_ids']==[r['step']]
                p=r['blind_judge'].get('parsed')
                invalid+=not isinstance(p,dict)
                states.append(p.get('state') if isinstance(p,dict) else None)
                world.append(r['world_seconds'])
            observed.extend(states)
            if arm=='dynamic':
                for i,r in enumerate(rows[1:],1):
                    proposal=r['proposal'].get('parsed') or {}
                    disagreement+=proposal.get('observed_state')=='open' and states[i-1]!='open'
                    online.append(r['proposal']['seconds'])
        result={'memory':policy,'arm':arm,'final_success':sum(x['final_two_chunks_open'] for x in selected),'n':len(selected),'any_two_open':sum(x['first_two_consecutive_open'] is not None for x in selected),'open_endpoint_labels':observed.count('open'),'total_endpoint_labels':len(observed),'invalid_judges':invalid,'director_open_disagreements':disagreement,'dynamic_vlm_median_seconds':statistics.median(online) if online else None,'world_median_seconds':statistics.median(world)}
        results.append(result)
    for seed in [42,123,2026]:
        traces=[json.loads((base/f'{seed}-{a}/trace.json').read_text()) for a in ['fixed','dynamic']]
        assert traces[0][0]['pixels_sha256']==traces[1][0]['pixels_sha256']
        comparisons.append({'memory':policy,'seed':seed,'matched_first_chunk':True,'first_divergent_chunk':next((i for i,(a,b) in enumerate(zip(*traces)) if a['pixels_sha256']!=b['pixels_sha256']),None)})
lines=['| 记忆配置 | 策略 | 末两段末帧达标 | 中途连续两段末帧达标 |','|---|---|---:|---:|']
for r in results:lines.append(f"| {r['memory']} | {'固定提示词' if r['arm']=='fixed' else 'VLM 动态事件'} | {r['final_success']}/{r['n']} | {r['any_two_open']}/{r['n']} |")
report='''# 开门语义事件对照

本轮实际执行了：关闭门照片 → VLM 识别并提出开门事件 → 更新 LingBot session 的文本条件 → 解码新画面 → VLM 再观察并纠正。两组共用第一条 VLM 事件提示词；固定组之后不变，动态组依据最近生成画面改写。相机保持中性控制，没有通过左右转伪装开门，也没有程序直接指定门已打开。

## 自动视觉评估结果

'''+ '\n'.join(lines)+'''

上述“达标”来自与 Director 同一 Qwen3-VL-8B 模型的独立、无历史评估提示词，不是独立模型或人工真值。判定同时要求 open、同一门与场景、可见门扇和明显通道。“两段达标”只检查这两段的末帧，不能证明整个区间一直打开；另提供最后两段的首/中/末六帧核查。

每种记忆配置使用同一照片、3 个种子（42/123/2026）、2 种策略、每条16段，共12条视频、192段。每条189帧，16fps约11.8秒，分辨率320×480。第二轮是在观察到反复开合后做的限定消融，关闭检索、world pruning和PC压缩，保留滑窗18与sink3；它不能单独归因到检索，也不等同完整官方原生实现。

## 已验证与限制

- 六对固定/动态试验的第一段像素哈希完全相同；动态更新后有实际输出差异，配对检查见 metrics.json。
- 世界模型确实能生成门扇转开和门后空间。固定组也能出现开门，因此单张开门截图不构成动态策略获益的证据。
- 第一轮末段截图已确认反复开合，部分轨迹有门框/门扇几何变化。控制器关于“open”的判断与无历史复核存在分歧；这种分歧不能自动算作控制器全部判断错误。
- 单场景、3个种子的探索，不支持显著性或通用任务成功率结论。对照改变的是观察反馈和提示词重写这一组合，没有独立隔离每个因素。
- 本轮不是官方完整 Director–Pilot 部署复现，没有 SAM 跟踪、对象状态约束或独立物理状态引擎。实际使用内部 ARDiffusionSession API。

## 产物

- [扩展记忆：seed42 对照视频](comparison-42.mp4)
- [仅滑窗：seed42 对照视频](sliding-only/comparison-42.mp4)
- 每个 seed/arm 目录含 contact.jpg、all-chunks.jpg、tail.jpg、trace.json；完整观测PNG和原始视频保留在远端。
- [运行方法与判定规则](METHOD.md)
- [实际控制及评估代码](semantic.py)
- [统计与配对验证](metrics.json)

初始照片：[Closed white door.jpg](https://commons.wikimedia.org/wiki/File:Closed_white_door.jpg)，作者 [Shixart1985](https://commons.wikimedia.org/wiki/User:Shixart1985)，[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)。使用缩略图并缩放；生成视频为模型演绎，不是真实拍摄的开门录像。

远端：`/root/autodl-tmp/t7-semantic-door-20261101`。
'''
(root/'metrics.json').write_text(json.dumps({'results':results,'pairs':comparisons},indent=2,ensure_ascii=False))
(root/'REPORT.md').write_text(report)
print(json.dumps(results,ensure_ascii=False,indent=2))
