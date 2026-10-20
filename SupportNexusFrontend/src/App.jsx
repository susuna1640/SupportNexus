import { useEffect, useMemo, useRef, useState } from 'react'
import MarkdownIt from 'markdown-it'
import {
  addKnowledge,
  backendMeta,
  createInitialSettings,
  reloadSkills,
  requestChatStream,
  requestHealth,
  requestKnowledgeStats,
  requestMonitor,
  requestSearch,
  requestSkills,
  runEvaluation as requestEvaluation,
  saveSettings,
  uploadKnowledge
} from './lib/backends'

const AGENT_LABELS = {
  general: '通用客服 Agent',
  api: 'API 接入 Agent',
  onboarding: '新用户引导 Agent',
  enterprise: '企业账户 Agent',
  technical: '技术支持 Agent',
  billing: '订阅账单 Agent'
}

const INTENT_LABELS = {
  query: '产品咨询', request: '服务请求', complaint: '投诉反馈', greeting: '问候', feedback: '体验反馈',
  api_integration: 'API 接入', feature_inquiry: '功能咨询', onboarding: '新用户引导',
  enterprise_mgmt: '企业账户管理', subscription: '订阅管理', billing: '账单咨询', refund: '退款申请',
  invoice: '发票处理', payment_issue: '支付异常', account: '账户问题', account_security: '账户与密钥安全',
  technical: '技术支持', technical_login: '登录故障', technical_crash: '系统故障', other: '其他'
}

const INTENT_GROUP_LABELS = {
  query: '咨询', request: '请求', complaint: '投诉', greeting: '问候', feedback: '反馈',
  api_integration: 'API 接入', feature_inquiry: '功能咨询', onboarding: '新用户引导',
  enterprise_mgmt: '企业账户', billing: '订阅与账单', account: '账户与安全', technical: '技术支持', other: '其他'
}

const ENTITY_LABELS = {
  error_code: '错误码', request_id: '请求 ID', api_key: 'API Key', workspace: 'workspace',
  amount: '金额', order_id: '订单号', invoice: '发票', date: '日期', plan: '套餐', email: '邮箱'
}

const markdown = new MarkdownIt({ html: false, breaks: true, linkify: true, typographer: true })

export default function App() {
  const [settings, setSettings] = useState(createInitialSettings)
  const [activeView, setActiveView] = useState('chat')
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [healthOk, setHealthOk] = useState(false)
  const [healthLabel, setHealthLabel] = useState('未检查')
  const [knowledgeCount, setKnowledgeCount] = useState('-')
  const [searchQuery, setSearchQuery] = useState('调用 /v1/chat 返回 401，如何排查？')
  const [searchResults, setSearchResults] = useState([])
  const [docTitle, setDocTitle] = useState('API 鉴权与 workspace 排查规范')
  const [docContent, setDocContent] = useState('出现 401 时，先核对 Authorization Bearer 格式、API Key 所属 workspace、环境配置和 request_id；不要在工单或聊天中提交完整密钥。')
  const [monitorData, setMonitorData] = useState({ agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] })
  const [skillsData, setSkillsData] = useState({ count: 0, skills: [], errors: [] })
  const [lastResponse, setLastResponse] = useState(null)
  const [lastTrace, setLastTrace] = useState(null)
  const [evalData, setEvalData] = useState(null)
  const [toast, setToast] = useState('')
  const messageListRef = useRef(null)
  const sidebarRef = useRef(null)
  const toastTimerRef = useRef(null)
  const messageSequenceRef = useRef(0)

  const currentBackend = useMemo(() => backendMeta(settings.backend, settings), [settings])
  const docsUrl = `${currentBackend.baseUrl}/docs`
  const userInitial = (settings.userId || 'U').slice(0, 1).toUpperCase()
  const activeAlerts = monitorData.active_alerts || []
  const agentCount = Object.keys(monitorData.agent_stats || {}).length
  const totalRequests = Object.values(monitorData.agent_stats || {}).reduce((sum, item) => sum + Number(item.total || 0), 0)
  const responseEntities = useMemo(() => formatEntities(lastResponse?.entities), [lastResponse])
  const responseDomainScores = useMemo(() => Object.entries(lastResponse?.domainScores || {})
    .map(([key, value]) => ({ key, value: Number(value) }))
    .filter((item) => Number.isFinite(item.value) && item.value > 0)
    .sort((left, right) => right.value - left.value), [lastResponse])

  useEffect(() => {
    refreshConsole()
    const updateSidebarHeight = () => {
      const sidebar = sidebarRef.current
      if (!sidebar) return
      const height = Math.max(320, Math.floor(sidebar.getBoundingClientRect().height))
      sidebar.style.setProperty('--sidebar-height', `${height}px`)
    }
    updateSidebarHeight()
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(updateSidebarHeight)
    if (sidebarRef.current && observer) observer.observe(sidebarRef.current)
    window.addEventListener('resize', updateSidebarHeight)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', updateSidebarHeight)
      clearTimeout(toastTimerRef.current)
    }
  }, [])

  function updateSettings(key, value) {
    setSettings((previous) => {
      const next = { ...previous, [key]: value }
      saveSettings(next)
      return next
    })
  }

  function showToast(message) {
    setToast(message)
    clearTimeout(toastTimerRef.current)
    toastTimerRef.current = setTimeout(() => setToast(''), 2600)
  }

  function createMessageId() {
    messageSequenceRef.current += 1
    return `message-${Date.now()}-${messageSequenceRef.current}`
  }

  async function refreshConsole() {
    await Promise.allSettled([checkHealth(), loadStats(), loadMonitor(), loadSkills()])
  }

  async function checkHealth() {
    try {
      const data = await requestHealth(settings.backend, settings)
      setHealthOk(data.status === 'ok')
      setHealthLabel(data.status || 'ok')
    } catch {
      setHealthOk(false)
      setHealthLabel('不可用')
    }
  }

  async function loadStats() {
    try {
      const data = await requestKnowledgeStats(settings.backend, settings)
      setKnowledgeCount(data.total_chunks ?? data.totalChunks ?? '-')
    } catch {
      setKnowledgeCount('-')
    }
  }

  async function loadMonitor() {
    try {
      setMonitorData(await requestMonitor(settings.backend, settings))
    } catch {
      setMonitorData({ agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] })
    }
  }

  async function loadSkills() {
    try {
      setSkillsData(await requestSkills(settings.backend, settings))
    } catch {
      setSkillsData({ count: 0, skills: [], errors: [] })
    }
  }

  async function reloadSkillSet() {
    setBusy(true)
    try {
      setSkillsData(await reloadSkills(settings.backend, settings))
      showToast('Skills 已重新加载')
    } catch {
      showToast('Skills 加载失败')
    } finally {
      setBusy(false)
    }
  }

  async function sendMessage(event) {
    event?.preventDefault()
    const content = draft.trim()
    if (!content || busy) return
    const assistantId = createMessageId()
    setMessages((previous) => [
      ...previous,
      { id: createMessageId(), role: 'user', content },
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        meta: '正在连接流式服务',
        progress: [],
        streaming: true
      }
    ])
    setDraft('')
    setBusy(true)
    try {
      const { response, trace } = await requestChatStream(settings.backend, settings, content, {
        onEvent: (type, data) => {
          if (type === 'done') return
          const progress = formatStreamProgress(type, data)
          setMessages((previous) => previous.map((item) => item.id === assistantId
            ? {
                ...item,
                content: type === 'delta' ? `${item.content}${data?.text || ''}` : item.content,
                meta: type === 'delta' ? '正在生成回复' : progress.message,
                progress: type === 'delta' ? item.progress : [...(item.progress || []), progress]
              }
            : item))
          requestAnimationFrame(scrollMessagesToBottom)
        }
      })
      if (response.conversationId && !settings.conversationId) updateSettings('conversationId', response.conversationId)
      setLastResponse(response)
      setLastTrace(trace)
      const meta = [
        formatIntent(response.intent),
        formatAgent(response.primaryAgent || response.agentType),
        response.contextWindowUsagePercent !== null ? `窗口 ${formatContextUsage(response)}` : '',
        response.knowledgeUsed ? 'RAG' : ''
      ].filter(Boolean).join(' · ')
      setMessages((previous) => previous.map((item) => item.id === assistantId
        ? {
            ...item,
            content: response.response || item.content,
            meta,
            trace: trace?.trace || null,
            streaming: false
          }
        : item))
      await loadMonitor()
    } catch (error) {
      setMessages((previous) => previous.map((item) => item.id === assistantId
        ? {
            ...item,
            content: item.content || error.message,
            meta: '请求失败',
            streaming: false,
            progress: [...(item.progress || []), formatStreamProgress('error', { message: error.message })]
          }
        : item))
    } finally {
      setBusy(false)
      requestAnimationFrame(scrollMessagesToBottom)
    }
  }

  function scrollMessagesToBottom() {
    const list = messageListRef.current
    list?.scrollTo({ top: list.scrollHeight, behavior: 'smooth' })
  }

  function clearConversation() {
    setMessages([])
    setLastResponse(null)
    setLastTrace(null)
    updateSettings('conversationId', '')
  }

  async function searchKnowledge() {
    if (!searchQuery.trim()) return
    setBusy(true)
    try {
      const data = await requestSearch(settings.backend, settings, searchQuery, 5)
      const results = data.results || []
      setSearchResults(results)
      showToast(`检索完成，返回 ${results.length} 条结果`)
    } catch {
      showToast('检索失败，请检查连接')
    } finally {
      setBusy(false)
    }
  }

  async function submitKnowledge() {
    if (!docTitle.trim() || !docContent.trim()) return
    setBusy(true)
    try {
      await addKnowledge(settings.backend, settings, [{ title: docTitle.trim(), content: docContent.trim() }])
      await loadStats()
      showToast('文档已添加')
    } catch {
      showToast('文档导入失败')
    } finally {
      setBusy(false)
    }
  }

  async function handleUpload(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setBusy(true)
    try {
      await uploadKnowledge(settings.backend, settings, file)
      await loadStats()
      showToast(`${file.name} 导入成功`)
    } catch {
      showToast('文件导入失败')
    } finally {
      setBusy(false)
    }
  }

  async function runEvaluation() {
    setBusy(true)
    try {
      setEvalData(await requestEvaluation(settings.backend, settings))
      showToast('评测完成')
    } catch {
      showToast('评测运行失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className={`app-shell app-shell-${activeView}`}>
      <header className="topbar">
        <a className="brand" href="#chat" aria-label="SupportNexus：面向 B2B SaaS 客服与运营支持的多 Agent 编排平台首页" onClick={() => setActiveView('chat')}>
          <span className="brand-mark">M</span><span className="brand-name">SupportNexus</span>
        </a>
        <nav className="view-nav" aria-label="工作区">
          <button className={activeView === 'chat' ? 'active' : ''} onClick={() => setActiveView('chat')}>对话</button>
          <button className={activeView === 'knowledge' ? 'active' : ''} onClick={() => setActiveView('knowledge')}>知识库</button>
          <button className={activeView === 'evaluation' ? 'active' : ''} onClick={() => setActiveView('evaluation')}>评测</button>
        </nav>
        <div className="topbar-tools">
          <span className="environment-pill"><i className={healthOk ? 'online' : 'offline'} />{currentBackend.label}</span>
          <a className="docs-link" href={docsUrl} target="_blank" rel="noreferrer">API 文档</a>
          <button className="avatar-button" title="当前用户">{userInitial}</button>
        </div>
      </header>

      {toast && <div className="toast" role="status">{toast}</div>}

      {activeView === 'chat' && (
        <section className="page page-chat">
          <div className="page-heading">
            <div className="heading-copy"><span className="kicker">SupportNexus</span><h1>SupportNexus：面向 B2B SaaS 客服与运营支持的多 Agent 编排平台</h1><p>面向开发者和企业客户，验证请求分析、知识检索与专业 Agent 主辅协同。</p></div>
            <div className="heading-actions"><span className="session-label">{settings.conversationId || '新会话'}</span><button className="quiet-button" onClick={clearConversation}>清空</button></div>
          </div>
          <div className="chat-layout">
            <section className="chat-stage">
              <div className="stage-bar"><div className="stage-context"><span className="context-dot" /><span>{currentBackend.baseUrl}</span></div><span>{messages.length} 条消息</span></div>
              <div className="messages" ref={messageListRef}>
                {messages.map((item) => <Message key={item.id} item={item} />)}
                {messages.length === 0 && <EmptyChat onUsePrompt={setDraft} />}
              </div>
              <form className="composer" onSubmit={sendMessage}>
                <textarea value={draft} rows="3" placeholder="描述 API、workspace、权限、订阅或系统故障问题..." onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => {
                  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') sendMessage(event)
                }} />
                <div className="composer-bottom"><span>⌘ / Ctrl + Enter 发送</span><button type="submit" disabled={busy || !draft.trim()}>{busy ? '处理中' : '发送'}</button></div>
              </form>
            </section>

            <aside className="chat-sidebar" ref={sidebarRef}>
              <div className="chat-sidebar-scroll">
                <section className="side-card session-card">
                  <div className="card-heading"><div><span className="kicker">Session</span><h2>会话信息</h2></div><span className="status-copy muted">{settings.conversationId ? '已启用' : '新会话'}</span></div>
                  <div className="session-grid"><div><span>会话 ID</span><strong>{settings.conversationId || '自动生成'}</strong></div><div><span>用户 ID</span><strong>{settings.userId || 'anonymous'}</strong></div></div>
                </section>
                <section className="side-card connection-card">
                  <div className="card-heading"><div><span className="kicker">Connection</span><h2>连接配置</h2></div><span className={`status-copy ${healthOk ? 'success' : 'muted'}`}>{healthLabel}</span></div>
                  <label><span>用户 ID</span><input value={settings.userId} placeholder="u1001" onChange={(event) => updateSettings('userId', event.target.value)} /></label>
                  <label><span>会话 ID</span><input value={settings.conversationId} placeholder="自动生成" onChange={(event) => updateSettings('conversationId', event.target.value)} /></label>
                  <div className="side-actions"><button onClick={checkHealth}>检查连接</button><button className="quiet-button" onClick={refreshConsole}>刷新</button></div>
                </section>
                <TracePanel lastResponse={lastResponse} lastTrace={lastTrace} responseEntities={responseEntities} responseDomainScores={responseDomainScores} />
                <section className="side-card monitor-card">
                  <div className="card-heading"><div><span className="kicker">Runtime</span><h2>运行状态</h2></div><button className="link-button" onClick={loadMonitor}>刷新</button></div>
                  <div className="mini-stats"><div><strong>{totalRequests}</strong><span>请求</span></div><div><strong>{agentCount}</strong><span>Agent</span></div><div><strong>{activeAlerts.length}</strong><span>告警</span></div></div>
                  {activeAlerts.length ? <div className="alert-note">{activeAlerts[0].detail || activeAlerts[0].title}</div> : <p className="healthy-note">当前没有活跃告警。</p>}
                </section>
              </div>
            </aside>
          </div>
        </section>
      )}

      {activeView === 'knowledge' && (
        <section className="page page-knowledge">
          <div className="page-heading"><div className="heading-copy"><span className="kicker">Knowledge operations</span><h1>知识库</h1><p>搜索、补充和维护 B2B SaaS 客服 Agent 使用的产品、接入与运营知识。</p></div><div className="count-display"><strong>{knowledgeCount}</strong><span>chunks</span></div></div>
          <div className="knowledge-layout">
            <section className="workspace-card search-workspace">
              <div className="card-heading"><div><span className="kicker">Retrieval</span><h2>检索知识</h2></div><code>POST /search</code></div>
              <div className="search-line"><input value={searchQuery} placeholder="例如：API 调用返回 401，如何排查？" onChange={(event) => setSearchQuery(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && searchKnowledge()} /><button onClick={searchKnowledge} disabled={busy || !searchQuery.trim()}>搜索</button></div>
              {searchResults.length ? <div className="result-list">{searchResults.map((item, index) => <article key={item.id || item.title || index} className="result-item"><span className="result-number">{String(index + 1).padStart(2, '0')}</span><div><div className="result-title"><strong>{item.title || '未命名文档'}</strong><small>score {item.score ?? '-'}</small></div><p>{item.content}</p></div></article>)}</div> : <div className="workspace-empty">输入 API、workspace、权限或订阅问题开始搜索。</div>}
            </section>
            <section className="workspace-card import-workspace">
              <div className="card-heading"><div><span className="kicker">Ingestion</span><h2>添加知识</h2></div><code>ChromaDB</code></div>
              <label><span>标题</span><input value={docTitle} placeholder="API 鉴权与 workspace 排查规范" onChange={(event) => setDocTitle(event.target.value)} /></label>
              <label><span>内容</span><textarea value={docContent} rows="7" placeholder="输入 API 接入、企业账户、订阅账单或排障流程" onChange={(event) => setDocContent(event.target.value)} /></label>
              <div className="side-actions"><button onClick={submitKnowledge} disabled={busy || !docTitle.trim() || !docContent.trim()}>添加文档</button><label className="upload-button">上传文件<input type="file" accept=".txt,.md,.json" onChange={handleUpload} /></label></div>
            </section>
          </div>
          <section className="workspace-card skills-workspace">
            <div className="card-heading"><div><span className="kicker">Loaded skills</span><h2>已加载能力</h2></div><button className="link-button" onClick={reloadSkillSet}>重新加载</button></div>
            <div className="skill-table">{skillsData.skills.map((skill) => <div key={skill.name} className="skill-item"><span className="skill-dot" /><strong>{skill.name}</strong><span>{skill.description || '业务规范能力'}</span><small>{skill.content_chars || 0} chars</small></div>)}{!skillsData.skills.length && <div className="workspace-empty">暂无已加载 Skill。</div>}</div>
          </section>
        </section>
      )}

      {activeView === 'evaluation' && (
        <section className="page page-evaluation">
          <div className="page-heading"><div className="heading-copy"><span className="kicker">Evaluation lab</span><h1>评测 Agent</h1><p>运行 FastAPI 内置评测，查看意图识别、对话质量和回归结果。</p></div><button onClick={runEvaluation} disabled={busy}>{busy ? '运行中...' : '运行评测'}</button></div>
          {evalData ? <Evaluation data={evalData} /> : <div className="evaluation-empty"><div className="empty-symbol">◎</div><h2>还没有评测结果</h2><p>点击右上角运行一次评测。</p></div>}
        </section>
      )}
    </main>
  )
}

function Message({ item }) {
  return <article className={`message ${item.role}`}>
    <div className="message-meta"><span>{item.role === 'user' ? '你' : 'SupportNexus 编排结果'}</span>{item.meta && <small>{item.meta}</small>}</div>
    {item.role === 'user' ? <p>{item.content}</p> : <>
      {item.progress?.length > 0 && <ProcessTimeline events={item.progress} />}
      {item.content ? <div className="message-markdown" dangerouslySetInnerHTML={{ __html: markdown.render(String(item.content)) }} /> : item.streaming && <p className="stream-placeholder">正在等待生成回复…</p>}
    </>}
    {item.trace && <div className="message-trace"><div className="trace-head"><span>工具调用</span>{item.trace.requestId && <small>#{item.trace.requestId}</small>}</div>{item.trace.toolCalls?.length ? <div className="trace-calls">{item.trace.toolCalls.map((call, index) => <details key={`${item.id}-${index}`} open><summary><strong>{call.tool_name || 'unknown_tool'}</strong><span>{call.success ? '成功' : '失败'}</span></summary><div className="trace-call-caption">{formatToolCallMeta(call)}</div>{call.input && <pre>{formatJson(call.input)}</pre>}</details>)}</div> : <div className="trace-empty-block"><p>本次请求没有调用工具。</p>{item.trace.toolsUsed?.length && <p className="trace-note">已调用：{item.trace.toolsUsed.join(' · ')}</p>}</div>}</div>}
  </article>
}

function ProcessTimeline({ events }) {
  return <div className="process-timeline" aria-label="实时处理过程">
    <div className="process-title">实时处理过程</div>
    {events.map((event) => <div key={event.id} className={`process-event ${event.type} ${event.success === false ? 'failed' : ''}`}>
      <span className="process-dot" />
      <span>{event.message}</span>
      {event.latencyMs !== null && <small>{event.latencyMs} ms</small>}
    </div>)}
  </div>
}

function EmptyChat({ onUsePrompt }) {
  const prompts = [
    ['调用 /v1/chat 返回 401，如何排查？', 'API 鉴权'], ['如何创建 workspace 并邀请团队成员？', '首次接入'],
    ['企业版如何配置 SSO 和 SCIM？', '企业账户'], ['Webhook 签名校验失败，应该检查哪些字段？', 'Webhook 排障'],
    ['登录一直报 401，而且本月账单似乎重复扣款了', '复合问题']
  ]
  return <div className="empty-state"><div className="empty-symbol">✦</div><h2>从一个 B2B SaaS 支持请求开始</h2><p>可从 API、workspace、企业账户、订阅账单或复合故障场景开始测试。</p><div className="starter-prompts">{prompts.map(([prompt, label]) => <button key={label} onClick={() => onUsePrompt(prompt)}>{label}</button>)}</div></div>
}

function TracePanel({ lastResponse, lastTrace, responseEntities, responseDomainScores }) {
  return <section className="side-card trace-card">
    <div className="card-heading"><div><span className="kicker">Last trace</span><h2>最近一次请求</h2></div><span className={`trace-status ${lastResponse ? 'has-data' : ''}`} /></div>
    {lastResponse ? <div className="trace-body">
      <div className="latency"><span>响应耗时</span><strong>{lastResponse.latencyMs || '-'}<small> ms</small></strong></div>
      <dl className="detail-list">
        <div><dt>主处理 Agent</dt><dd>{formatAgent(lastResponse.primaryAgent || lastResponse.agentType)}</dd></div><div><dt>业务意图</dt><dd>{formatIntent(lastResponse.intent)}</dd></div><div><dt>意图组</dt><dd>{formatIntentGroup(lastResponse.intentGroup)}</dd></div><div><dt>路由置信度</dt><dd>{formatPercent(lastResponse.routingConfidence)}</dd></div><div><dt>意图置信度</dt><dd>{formatPercent(lastResponse.intentConfidence)}</dd></div>
        {lastResponse.contextWindowUsagePercent !== null && <div title={contextUsageTitle(lastResponse)}><dt>上下文窗口</dt><dd>{formatContextUsage(lastResponse)}</dd></div>}
        <div><dt>知识库</dt><dd className={lastResponse.knowledgeUsed ? 'success' : 'muted'}>{lastResponse.knowledgeUsed ? '已使用' : '未使用'}</dd></div>
      </dl>
      {lastResponse.agentTypes?.length > 0 && <TraceTagSection label="参与 Agent" items={lastResponse.agentTypes.map((agent) => formatAgent(agent))} />}
      {lastResponse.supportingAgents?.length > 0 && <TraceTagSection label="协同处理" items={lastResponse.supportingAgents.map((agent) => formatAgent(agent))} supporting />}
      {responseDomainScores.length > 0 && <TraceTagSection label="融合领域分数" items={responseDomainScores.map((item) => `${formatAgent(item.key)} ${formatPercent(item.value)}`)} />}
      {responseEntities.length > 0 && <div className="response-section"><span className="response-section-label">识别实体</span><div className="entity-list">{responseEntities.map((entity) => <span key={entity.key} className="entity-item"><b>{entity.label}</b>{entity.values.join(' · ')}</span>)}</div></div>}
      {lastResponse.routingReason && <p className="routing-reason">{lastResponse.routingReason}</p>}
      {lastTrace?.trace && <div className="trace-call-list"><div className="trace-call-title">工具调用</div>{lastTrace.trace.toolCalls?.map((call, index) => <div key={call.tool_use_id || index} className="trace-call-item"><div className="trace-call-meta"><strong>{call.tool_name || 'unknown_tool'}</strong><span>{call.latency_ms || 0} ms</span></div><div className="trace-call-caption">{formatToolCallMeta(call)}</div>{call.input && <pre>{formatJson(call.input)}</pre>}</div>)}{!lastTrace.trace.toolCalls?.length && <div className="trace-empty-block"><p>这次请求没有调用工具。</p>{lastTrace.trace.toolsUsed?.length && <p className="trace-note">已调用：{lastTrace.trace.toolsUsed.join(' · ')}</p>}</div>}</div>}
    </div> : <p className="side-empty">发送消息后，这里会显示意图、实体、主/辅 Agent 路由和耗时。</p>}
  </section>
}

function TraceTagSection({ label, items, supporting = false }) {
  return <div className="response-section"><span className="response-section-label">{label}</span><div className="tag-list">{items.map((item) => <span key={item} className={`data-tag ${supporting ? 'supporting' : ''}`}>{item}</span>)}</div></div>
}

function Evaluation({ data }) {
  return <div className="evaluation-content">
    <div className="evaluation-summary"><div className="score-hero"><span>Pass rate</span><strong>{formatPercent(data.pass_rate)}</strong><small>{data.passed} / {data.total} cases passed</small></div><div><span>通过</span><strong>{data.passed}</strong></div><div><span>总数</span><strong>{data.total}</strong></div><div><span>回归</span><strong className={data.regressions?.length ? 'danger' : 'success'}>{data.regressions?.length || 0}</strong></div></div>
    <div className="evaluation-layout"><section className="workspace-card"><div className="card-heading"><div><span className="kicker">Scores</span><h2>平均评分</h2></div></div><div className="score-list">{Object.entries(data.avg_scores || {}).map(([key, value]) => <div key={key}><span>{key}</span><i><b style={{ width: `${Math.min(Number(value) * 10, 100)}%` }} /></i><strong>{Number(value).toFixed(2)}</strong></div>)}</div></section><section className="workspace-card"><div className="card-heading"><div><span className="kicker">Recommendations</span><h2>优化建议</h2></div></div>{data.recommendations?.length ? <div className="recommendations">{data.recommendations.map((item, index) => <p key={index}>{item}</p>)}</div> : <div className="workspace-empty">本次评测没有返回额外建议。</div>}</section></div>
  </div>
}

function formatPercent(value) {
  const number = Number(value || 0)
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`
}

function formatStreamProgress(type, data = {}) {
  const primary = data.primary_agent ? formatAgent(data.primary_agent) : ''
  const supporting = Array.isArray(data.supporting_agents) && data.supporting_agents.length
    ? `，协同 ${data.supporting_agents.map(formatAgent).join('、')}`
    : ''
  const agent = data.agent_type ? formatAgent(data.agent_type) : ''
  const tool = data.tool_name || ''
  const fallback = {
    status: data.message || '正在处理请求',
    routing: primary ? `已完成路由：${primary}${supporting}` : '已完成 Agent 路由',
    agent_start: agent ? `${agent} 开始处理` : 'Agent 开始处理',
    agent_end: agent ? `${agent} ${data.success === false ? '未能完成' : '已完成'}` : 'Agent 已完成',
    tool_start: tool ? `正在调用工具：${tool}` : '正在调用工具',
    tool_end: tool ? `工具 ${tool} ${data.success === false ? '未能完成' : '已完成'}` : '工具调用完成',
    error: data.message || '请求处理失败'
  }
  return {
    id: `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    type,
    message: type === 'routing' ? fallback.routing : (data.message || fallback[type] || '收到服务端事件'),
    success: data.success,
    latencyMs: data.latency_ms === undefined || data.latency_ms === null ? null : data.latency_ms
  }
}

function formatToolCallMeta(call) {
  const parts = []
  if (call.agent_type) parts.push(formatAgent(call.agent_type))
  if (call.latency_ms !== undefined && call.latency_ms !== null) parts.push(`${call.latency_ms} ms`)
  if (call.cached) parts.push('缓存命中')
  if (call.reranked) parts.push('已重排')
  return parts.join(' · ') || '未返回额外公开元数据'
}

function formatAgent(agent) {
  const value = String(agent || '').trim()
  return AGENT_LABELS[value] || value || '-'
}

function formatIntent(intent) {
  const value = String(intent || '').trim()
  return INTENT_LABELS[value] || value || '-'
}

function formatIntentGroup(group) {
  const value = String(group || '').trim()
  return INTENT_GROUP_LABELS[value] || value || '-'
}

function formatEntities(entities) {
  if (!entities || typeof entities !== 'object') return []
  return Object.entries(entities).map(([key, rawValues]) => {
    const values = (Array.isArray(rawValues) ? rawValues : [rawValues]).map((value) => String(value).trim()).filter(Boolean)
    return { key, label: ENTITY_LABELS[key] || key.replaceAll('_', ' '), values }
  }).filter((entity) => entity.values.length)
}

function formatContextUsage(response) {
  const percent = Number(response.contextWindowUsagePercent)
  return Number.isFinite(percent) ? `${response.contextWindowUsageEstimated ? '≈' : ''}${percent.toFixed(1)}%` : '-'
}

function contextUsageTitle(response) {
  const base = '业务 Agent 首次生成前的输入窗口占用；并行协作时显示最高值。'
  return response.contextWindowUsageEstimated ? `${base} 当前兼容服务未提供精确 token 计数，数值为保守估算。` : base
}

function formatJson(value) {
  try { return JSON.stringify(value ?? {}, null, 2) } catch { return String(value ?? '') }
}
