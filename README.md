# mt4g AMD extension and rewrite

Build using
`make GPU_TARGET_ARCH=<COMPUTE CAPABILITY / GFX ID>`

Make sure to have HIP_PATH set to hipccs "root" directory

#### Benchmark Overview - NVIDIA

| Cache     | L1 | L2         | RO | TXT | C1 | C1.5 | SM  | M   |
|-----------|----|------------|----|-----|----|------|-----|-----|
| Size      | Yes | API, Seg. | Yes | Yes | Yes | Yes | API | API |
| Line Size | Yes | Yes       | Yes | Yes | Yes | Yes |  /  |  /  |
| Fetch Granularity | Yes | Yes     | Yes | Yes | Yes | Yes |  /  |  /  |
| Latency   | Yes | Yes       | Yes | Yes | Yes | Yes | Yes | Yes |
| Count     | Yes | Yes, Seg. | Yes | Yes | Yes | No  |  /  |  /  |
| Miss Penalty | Yes | Yes     | Yes | Yes | Yes | No  |  /  |  /  |
| Bandwidth | No  | R/W       | No  | No  | No  | No  | No  | R/W |
| Shared With | RO, C1, TXT |     | L1, TXT |    |    |    |     |     |

#### Benchmark Overview - AMD

| Cache     | vL1d | L2         | L3 | sL1d | SM  | M   |
|-----------|------|------------|----|------|-----|-----|
| Size      | API, FB  | API, Seg.  | API | Yes  | API | API |
| Line Size | API, FB   | API, FB    | API | Yes  |  /  |  /  |
| Fetch Granularity     | Yes | Yes     | No | Yes  |  /  |  /  |
| Latency   | Yes  | Yes        | No | Yes  | Yes | Yes |
| Count     | Yes  | Yes, Seg.  | No | No   |  /  |  /  |
| Miss Penalty | Yes | Yes       | No | Yes   |  /  |  /  |
| Bandwidth | No   | R/W        | No | No   | No  | R/W |
| Shared With |     |            |    | CU   |     |     |
