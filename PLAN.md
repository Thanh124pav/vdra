# GEAR: Information-Gated Policy Optimization
## Online Segment-Level Pruning and Value Sharing via Answer Log-Probability

### 1. Problem Statement
SPO [Segment Policy Optimization] train LLM bằng RL: với mỗi problem, build tree theo depth, mỗi node = segment. SPO expand đầy đủ width W cho mỗi depth D, gây noise do:
1. **Redundant segments**: `s` có `P(·|trajectory tới s) ≈ P(·|trajectory tới s')` với `s'` bất kỳ đã có.
2. **Information-irrelevant segments**: `I(A*; Y_s | Y_pa(s)) ≈ 0`, segment `s` không giúp model tiến gần `A*`.

GEAR: Cắt tỉa và share value **ngay khi segment sinh ra**, trước khi expand con. Giảm variance gradient PPO.

### 2. Definitions

**Notation**
- `s`: segment/node. `traj(s)`: trajectory từ root tới s.
- `pa(s)`: parent segment.
- `π_θ(·|traj(s))`: policy LLM condition trên toàn trajectory, không chỉ text node s.
- `A*`: ground-truth answer.
- `Y = {y_1,...,y_m}`: global answer set. Sinh song với depth=1.
- `K << m`: fast subset. `R_max`: max reward.

**Def 2.1 – Answer Set Y**
Khi depth=1, với problem prompt `p` và GT answer `a*`, cho model sinh:
```
Y ← LLM(p, "Given answer="+a*+", list m diverse step-by-step solutions")
```
`m=100`. Y dùng cho toàn bộ tree của problem đó. Sinh parallel với W nodes depth=1.

**Def 2.2 – Segment Log-Probability**
Với segment `s` vừa được sinh, tính ngay:
```
LP = log π_θ(y_i | traj(s)), i=1..m
```
Dùng log-sum-exp để tránh underflow. `LP` lưu vào global matrix.[i][s]

**Def 2.3 – TV Distance**
```
AvgLP_K(s) := (1/K)*Σ_{i=1..K} LP
TV_m(a,b) := 0.5*Σ_{i=1..m} |exp(LP) - exp(LP)| + 0.5*(exp(δ_a)+exp(δ_b))
```

**Lemma 2.4 – Threshold từ Toán [To Prove]**
Cho sai số `ε`. Đặt `η = ε / R_max - exp(δ_avg)`.
Nếu `TV_m(s,s') ≤ η` thì `|V*(s)-V*(s')| ≤ ε` → share được.
Nếu `AvgLP_m(s) < AvgLP_m(pa(s)) - η` thì `I(A*;Y_s|Y_pa) ≈ 0` → prune được.

### 3. Algorithm: GEAR

GEAR kế thừa SPO: build tree theo depth, width W song song. Khác biệt: mỗi node sinh ra sẽ trigger check Share/Prune **ngay lập tức** trước khi cho phép expand con.

```
Algorithm: GEAR Training for one problem
Input: problem p, GT a*, policy π_θ, value V_φ, D, W, K, m, η
Output: updated π_θ, V_φ

1 Global LP[*] ← {}
2 Global BST ← BinarySortTree() // key=AvgLP_K(s), val=s
3 Y ← {} // sẽ fill ở depth 1
4
5 Procedure BuildDepth(d, parents):
6 If d > D: return
7 children ← []
8 For pa in parents in parallel do
9 segs ← LLM_Generate(π_θ, traj(pa), W) // W segments song song, SPO style
10 For s in segs do
11 // Trigger 1: Compute LP ngay khi s sinh ra
12 If d == 1 and Y == {} then // depth 1: sinh Y song song
13 Y ← LLM(p, "Answer="+a*+", list m solutions") // m calls parallel
14 End if
15 For i = 1 to K do // fast check
16 LP ← log π_θ(y_i | traj(s)) // 1 forward, batch K
17 End for
18 AvgLP_K ← (1/K)*Σ_i LP
19
20 // Trigger 2: ValueShare - so với node BẤT KỲ
21 s' ← FindNearest(BST, AvgLP_K)
22 If s'≠null and |AvgLP_K - AvgLP_K(s')| < τ_share(K,η) then
23 Compute LP for i=K+1..m // full check
24 If TV_m(s,s') ≤ η then
25 MarkShare(s, s') // V(s):=V(s'), không cho s expand
26 Continue // bỏ s, không thêm vào children
27 End if
28 End if
29
30 // Trigger 3: Prune - so với CHA
31 If AvgLP_K < AvgLP_K(pa) - τ_prune(K,η) then
32 Compute LP for i=K+1..m
33 If AvgLP_m(s) < AvgLP_m(pa) - η then
34 MarkPrune(s) // IG≈0, không cho s expand
35 Continue
36 End if
37 End if
38
39 Insert(BST, AvgLP_K, s) // chỉ insert nếu không bị share/prune
40 children.append(s)
41 End for
42 End parallel for
43
44 BuildDepth(d+1, children) // đệ quy xuống depth tiếp
45 // PPO update cho depth d trước khi xuống depth d+1
46 ComputeAdvantage(children, pa) // A_children = V(children) - V_(pa), dùng V share nếu có
46 UpdatePolicyPPO()
45 End Procedure
51
52 BuildDepth(1, )
```

**Key Points**
1. **Online**: Line 20, 26, 34 quyết định ngay khi `s` sinh. Node bị Share/Prune sẽ không có con, giảm cây ngay lập tức.
2. **Parallel**: Line 8, 10, 15 đều song song W nodes và K answers.
3. **LP = P[y_i | traj(s)]**: Line 16 dùng toàn bộ trajectory, không phải mỗi text node.
4. **Y sinh ở depth 1**: Line 12-14, chạy song song với việc sinh W segments đầu tiên.
5. **Threshold**: `τ(K,η) = η + sqrt(log(2/α)/(2K))`, α=0.05. `η` từ Lemma 2.4, không tune.

**Theorem 3.1 – Sample Complexity [To Prove]**
GEAR giảm số segment cần update từ `O(W^D)` của SPO xuống `O((ρW)^D)`, với `ρ<1` là tỉ lệ giữ lại sau prune/share.

### 4. Experiments to Run

Kế thừa 100% setup SPO paper.

**Models**: DeepSeek-Distill-Qwen-1.5B, Rho-math-1.1b-SFT.
**Data**: GSM8K, MATH train.
**Tree**: 4-4-4, 6-6-6, 8-8-8, như SPO.

**Baselines**
1. **SPO**: Segment Policy Optimization gốc.
2. **PPO**: không chia segment.
3. **RFT**: Rejection sampling Fine-Tuning.
4. **GEAR**: `K=10`, `m=100`, `η` từ Lemma 2.4.

**Exp 1: Sample Efficiency**
Metrics: Pass@1 trên test vs #problems seen trong train.
Goal: GEAR đạt SPO với 2-3x ít problems.

**Exp 2: Online Prune/Share Rate**
Metrics: `%segments bị Share tại depth d`, `%segments bị Prune tại depth d`, `Var[A_t]` trước/sau.
Goal: 40-60% segments bị loại ngay khi sinh, variance giảm 30%.

**Exp 3: Overhead**
Metrics: Time cho `LP[i][s]` với K=10, time BST, total wall-clock vs SPO.
Goal: Overhead <10% nhờ K<<m và song song.

### 5. Ablation Studies

**Abl 1: `K` vs `m`**
Vary `K∈{5,10,20}`, `m∈{50,100,200}`. Metrics: False positive của fast filter, time, final acc.
Hypothesis: `K=10,m=100` tối ưu.

**Abl 2: `η` theory vs grid**
So `η` từ Lemma 2.4 vs `{0.005,0.01,0.02,0.05}`. Metrics: final acc.
Hypothesis: theory `η` tốt nhất.

**Abl 3: Share vs Parent vs Nearest**
V1: ValueShare chỉ với `pa(s)`. V2: với `root`. V3: với `nearest` GEAR.
Metrics: Share rate, acc. Hypothesis: V3 >> V1,V2.

**Abl 4: Share-only vs Prune-only vs Both**
Tắt từng trigger. Metrics: sample efficiency. Hypothesis: Both tốt nhất.

**Abl 5: Y sinh depth=1 vs pre-compute**
V1: Y cố định cho cả dataset. V2: Y per-problem như GEAR.
Metrics: acc. Hypothesis: per-problem tốt hơn vì `y_i` sát GT.

**Abl 6: LogP vs Prob**
Dùng `exp(LP)` từ đầu. Metrics: NaN count. Hypothesis: LogP bắt buộc.

**Abl 7: Oracle**
Train SPO không prune, sau đó check các segment mà GEAR đã prune/share: nếu update policy trên chúng thì Δacc?
Metric: Δacc. Hypothesis: <0.5%, tức prune/share không mất thông tin.

### 6. Implementation Notes
1. **SPO Integration**: Hook vào vòng `for d in range(D)` của SPO. Thay `children = expand(pa)` bằng `BuildDepth`.
2. **LP Batch**: Với W segments, K answers → 1 forward pass với batch size W*K. Dùng `input_ids = [traj(s)+y_i]`.
3. **BST**: `sortedcontainers.SortedList` với key=lambda s: AvgLP_K(s). `FindNearest` = bisect.
4. **PPO**: Khi `MarkShare(s,s')`, gán `value_target[s] = V_φ(s')`. Khi `MarkPrune(s)`, gán `advantage[s]=0`, không backward.
5. **Y Cache**: Với mỗi problem, cache Y để nếu resample thì reuse.

### 7. Contributions
1. **Algorithm**: GEAR, online segment-level pruning/sharing cho SPO, trigger ngay khi segment sinh.
2. **Theory**: Thresholds `η,τ` từ bound, không tune. Giảm sample complexity.
3. **Practice**: 2-3x sample efficiency trên setup SPO gốc, overhead <10%.

### 8. Title
**Paper**: *Information-Gated Policy Optimization: Online Segment Pruning for Sample-Efficient LLM Reasoning*

**Acronym**: **GEAR**
