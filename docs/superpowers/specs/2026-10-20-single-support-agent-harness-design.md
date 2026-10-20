# 单 SupportAgent 工单 Harness 设计

## 状态

提案，尚未实施。本说明把当前“主 Agent + 辅助 Agent 并行 + Response Composer”的同步编排，收敛为一个能跨回合延续工具执行上下文的 SupportAgent。现有 HTTP 接口、知识库、会话记忆、Skills 和工具实现尽量复用。

## 目的

SupportNexus 的主要产品形态是 B2B SaaS 客服工单，而不是多个可独立交付的后台任务。用户应能在下一轮自然地说“继续查刚才的 request_id”或“按刚才的排查继续”，Agent 需要知道此前调用过哪些工具、拿到了哪些结果、哪些事项尚待核验。

正式对话仍然只展示用户消息与最终答复；工具调用不是用户可见聊天正文，但必须作为 Agent harness 的短期、可恢复执行历史保存。系统不保存模型推理文本。

## 设计原则

1. **一个执行 Agent。** 每个工单的所有 LLM 工具回合由同一个 SupportAgent 完成；不再创建主/辅专家实例，也不需要 Composer。
2. **专业领域是能力，不是执行者。** API、上手、企业、技术和账单继续存在，但以 Skill、SOP 与工具策略的形式按需激活。
3. **工具默认拒绝。** Agent 只能看到本轮激活 Skill 映射出的工具，加上少量通用分诊工具；不因改为单 Agent 而获得所有领域工具。
4. **工具协议历史与用户记忆分层。** 活跃 harness 保存协议级 `tool_use` / `tool_result`；长期记忆只保存经确认、仍有价值的工单事实。
5. **状态可能过期。** 账户、权限、账单和订单等工具结果有时间性；可恢复历史不能被当作永远有效的事实。

## 非目标

- 不实现主/辅 Agent、Response Composer、Agent 间通信或通用工作流引擎。
- 不把工具原始结果、工具参数、模型推理或内部异常直接展示给用户。
- 不在此阶段接入真实支付、权限、工单或订单写操作。
- 不让一个长工单永久携带全部原始工具输出；超出预算后必须压缩。

## 目标架构

```text
用户新消息 (user_id, conv_id)
  │
  ├─ 读取 CaseState + 活跃 AgentTranscript + 用户可见对话摘要
  ├─ 意图识别 / 能力激活：选择 0–2 个 Skills 和对应工具白名单
  ├─ SupportAgent：同一条消息链内执行 LLM → tool_use → tool_result 循环
  ├─ 每个完成的协议块立即写入 AgentTranscript
  ├─ 输出一条面向用户的最终答复
  └─ 更新 CaseState，并写入用户可见的 user / assistant 对话投影
```

### 1. 能力激活

现有 `IntentRecognizer` 保留领域分数、实体与意图识别能力，但输出从 `RoutingDecision(primary_agent, supporting_agents)` 改为 `CapabilityDecision`：

- `active_skills`：本轮需激活的 0–2 个 Skill ID。
- `active_domains`：供观测与前端展示的业务领域。
- `allowed_tools`：由 `active_skills` 映射出的工具集合，再加通用只读工具。
- `reason` / `confidence`：保留现有可解释性字段。

单域问题通常激活一个 Skill。复合问题可以激活两个 Skill，但仍由一个 Agent 按用户问题顺序处理；这不是并行的两个答案。低置信度、超过两个领域，或无法确定应激活的 Skill 时，启用 `general-triage` 并只开放分诊、检索和必要字段检查工具。

现有 Skill front matter 的 `agents` 字段迁移为 `domains`，或由一个集中 `skill_id → domains/tools` 映射暂时兼容。每个 Skill 的流程、边界和升级规则继续保留在 Skill 正文。工具策略必须在 LLM 调用前确定，不能仅依赖提示词要求模型“不要调用”。

### 2. 单个 SupportAgent

SupportAgent 使用一份稳定的公共系统提示词，包含：客服身份、通用安全边界、能力激活摘要、允许工具、继续工单的规则，以及“不要将未经当前或有时效标记的工具结果表述为已确认事实”的要求。

它按当前 Skill 目录决定是否读取详细 SOP。一次请求最多激活和读取两份 Skill，沿用当前预算限制。原先六个 Agent profile 中的领域使命、输入字段、输出约束与禁止事项迁入对应 Skill 或能力策略；它们不再作为六个独立模型调用者存在。

同一回合中，`assistant tool_use` 与对应 `tool_result` 必须原样按提供方协议加入消息链，直到模型输出最终文本。工具错误也必须以 tool result 形式返回给模型，由模型解释失败、重试、改走低风险步骤或请求最少补充信息。

### 3. 三种状态视图

| 视图 | 内容 | 生命周期 | 用途 |
| --- | --- | --- | --- |
| 用户可见对话 | 用户消息、SupportAgent 最终答复 | 现有会话生命周期 | 聊天 UI、最近对话、用户阅读 |
| AgentTranscript | user、assistant、tool_use、tool_result、完成/失败状态及必要的提供方关联 ID | 活跃工单，短期保留并可压缩 | 下一回合恢复 harness |
| CaseState | 已确认事实、待核验项、当前能力领域、上次核验时间、关联 request_id | 当前会话摘要及经审阅的长期记忆 | 快速恢复、压缩后的持续跟进 |

用户可见对话是 AgentTranscript 的受限投影，不应再额外存入专家草稿。CaseState 不保存完整工具响应，而保存诸如“用户提供的 request_id”“401 仍待确认鉴权头是否过期”“账单金额待人工核验”这类带来源和时间的短事实。

### 4. AgentTranscript 的持久化与恢复

每个工单以 `(user_id, conv_id)` 定位活跃 transcript，并为每轮分配 `turn_id`。每个 `tool_use` 和 `tool_result` 带 `turn_id`、工具名、工具调用 ID、完成状态、时间戳和脱敏后的内容。写入发生在每个工具回合完成后，而不是只在最终回答之后；这样网络中断或模型调用失败后仍可判断哪些只读查询已完成。

恢复时，系统先加载 CaseState 与最近未压缩的 transcript，再作能力激活：

- 若本轮是同一工单的继续，加载当前 transcript；新消息按提供方所需的 user message 结构追加。
- 若当前领域改变，保留用户可见对话与 CaseState，但不把不相关领域的原始工具结果直接注入模型。
- 已完成工具调用不会仅因模型重试或服务恢复而自动重复；若结果已过期或操作具有副作用，Agent 必须明确重新核验或请求用户确认。当前工具均为只读/计划型，未来写操作必须提供幂等键和人工确认机制。

活跃 transcript 应使用 Redis 等短期存储，并遵循当前会话保留期。长期用户记忆与已结算会话的语义历史只接收压缩后的 CaseState / 对话摘要，不能接收原始工具结果。原始结果的可保存字段、脱敏规则和 TTL 必须按工具声明；默认采用工具输出长度上限，并拒绝密码、完整 API Key、Token、银行卡号、验证码和原始支付凭证。

### 5. 压缩

压缩在调用模型之前，根据完整 transcript、当前系统提示词、激活 Skills 与工具定义计算输入预算。不能只按用户可见问答压缩。

压缩以**已完成的交互片段**为最小单位：一个 assistant tool call 与其所有 tool results 不能被拆开，也不能留下孤立 tool result。早期完成片段被整理为 CaseState 和 transcript summary；最近的原始片段保留。若出现未完成工具调用，先标记为中断/待恢复，再继续，不伪造 tool result。

### 6. 失败与输出规则

- 工具失败：写入失败 tool result；SupportAgent 获得该结果并选择解释、低风险替代步骤、请求信息或升级。
- 模型失败：保留已写入的 transcript；下一次相同工单可从最后一个完整协议边界恢复，不能重复完成的工具调用。
- 用户取消：保留已完成的协议块，标记本 turn 未完成；不写入虚构的 assistant 最终答复。
- 正常完成：写入一个用户可见最终答复，并更新 CaseState。

`ChatResponse` 在迁移期保留 `primary_agent`、`supporting_agents` 字段以兼容前端，但它们映射为 `active_domains`，且 `supporting_agents` 始终为空。新增 `active_skills`、`case_status` 与安全的工具摘要字段；前端后续改为展示“已激活能力”，不再展示多个 Agent。

## 迁移路径

1. 新增 transcript 和 CaseState 的存储模型、序列化/脱敏策略、压缩和恢复测试；先不切换现有编排。
2. 引入 SupportAgent，复用当前工具执行循环和 SkillManager；实现 `CapabilityDecision` 与 `skill → tool` 白名单映射。
3. 将现有六个 Agent profile 中不可缺少的领域契约迁入 Skills / 能力策略，保留现有 Skill 文本并迁移授权字段。
4. 以 feature flag 将新会话切换到 SupportAgent；保留旧路径以便回归比较。
5. 评测通过后移除 AgentPool、主辅并行、ResponseComposer 及其相关前端展示。

## 验收标准

- 用户在 API 排障后追问“继续查刚才的 request_id”时，同一 SupportAgent 能看到此前成对保存的工具调用与结果，且不会重复已完成查询。
- 同一工单从技术排障转为账单问题时，账单工具不会获得技术工具的原始输出；Agent 仅获得公共对话、CaseState 和激活的账单能力。
- 工具、模型或用户取消发生在中间回合时，transcript 不含孤立 tool result 或伪造的最终答复。
- 压缩后仍保留已确认事实、待核验项、最近工具片段及完整的调用/结果配对。
- 每个 LLM 请求仅携带能力策略允许的工具；未知/低置信度请求不暴露账户、账单或 API 专属工具。
- 用户界面只显示最终答复与安全的过程摘要，不显示工具参数、原始结果或模型推理。

## 待确认的产品选择

1. 活跃 transcript 的默认保留期，以及用户主动新开工单时是否允许从旧 transcript 恢复。
2. 哪些工具结果可作为 CaseState 事实保存、哪些必须每次重新查询。
3. 是否需要客服后台查看脱敏后的完整执行 transcript；这与用户聊天界面的展示范围无关。
