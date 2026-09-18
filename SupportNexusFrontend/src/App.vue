<template>
  <main :class="['app-shell', `app-shell-${activeView}`]">
    <header class="topbar">
      <a class="brand" href="#" aria-label="SupportNexus：面向 B2B SaaS 客服与运营支持的多 Agent 编排平台首页" @click.prevent="activeView = 'chat'">
        <span class="brand-mark">M</span>
        <span class="brand-name">SupportNexus</span>
      </a>

      <nav class="view-nav" aria-label="工作区">
        <button :class="{ active: activeView === 'chat' }" @click="activeView = 'chat'">对话</button>
        <button :class="{ active: activeView === 'knowledge' }" @click="activeView = 'knowledge'">知识库</button>
        <button :class="{ active: activeView === 'evaluation' }" @click="activeView = 'evaluation'">评测</button>
      </nav>

      <div class="topbar-tools">
        <span class="environment-pill">
          <i :class="healthOk ? 'online' : 'offline'"></i>
          {{ currentBackend.label }}
        </span>
        <a class="docs-link" :href="docsUrl" target="_blank" rel="noreferrer">API 文档</a>
        <button class="avatar-button" title="当前用户">{{ userInitial }}</button>
      </div>
    </header>

    <div v-if="toast" class="toast" role="status">{{ toast }}</div>

    <section v-if="activeView === 'chat'" class="page page-chat">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">SupportNexus</span>
          <h1>SupportNexus：面向 B2B SaaS 客服与运营支持的多 Agent 编排平台</h1>
          <p>面向开发者和企业客户，验证请求分析、知识检索与专业 Agent 主辅协同。</p>
        </div>
        <div class="heading-actions">
          <span class="session-label">{{ settings.conversationId || '新会话' }}</span>
          <button class="quiet-button" @click="clearConversation">清空</button>
        </div>
      </div>

      <div class="chat-layout">
        <section class="chat-stage">
          <div class="stage-bar">
            <div class="stage-context">
              <span class="context-dot"></span>
              <span>{{ currentBackend.baseUrl }}</span>
            </div>
            <span>{{ messages.length }} 条消息</span>
          </div>

          <div class="messages" ref="messageList">
            <article v-for="item in messages" :key="item.id" :class="['message', item.role]">
              <div class="message-meta">
                <span>{{ item.role === 'user' ? '你' : 'SupportNexus 编排结果' }}</span>
                <small v-if="item.meta">{{ item.meta }}</small>
              </div>
              <p v-if="item.role === 'user'">{{ item.content }}</p>
              <div
                v-else
                class="message-markdown"
                v-html="renderMarkdown(item.content)"
              ></div>
              <div v-if="item.trace" class="message-trace">
                <div class="trace-head">
                  <span>工具调用</span>
                  <small v-if="item.trace.requestId">#{{ item.trace.requestId }}</small>
                </div>
                <div v-if="item.trace.toolCalls?.length" class="trace-calls">
                  <details v-for="(call, index) in item.trace.toolCalls" :key="`${item.id}-${index}`" open>
                    <summary>
                      <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
                      <span>{{ call.success ? '成功' : '失败' }}</span>
                    </summary>
                    <pre>{{ formatJson(call.input || {}) }}</pre>
                  </details>
                </div>
                <div v-else class="trace-empty-block">
                  <p>本次请求已生成 trace，但没有可展示的工具输入。</p>
                  <p v-if="item.trace.toolsUsed?.length" class="trace-note">已调用：{{ item.trace.toolsUsed.join(' · ') }}</p>
                </div>
              </div>
            </article>

            <div v-if="messages.length === 0" class="empty-state">
              <div class="empty-symbol">✦</div>
              <h2>从一个 B2B SaaS 支持请求开始</h2>
              <p>可从 API、workspace、企业账户、订阅账单或复合故障场景开始测试。</p>
              <div class="starter-prompts">
                <button @click="usePrompt('调用 /v1/chat 返回 401，如何排查？')">API 鉴权</button>
                <button @click="usePrompt('如何创建 workspace 并邀请团队成员？')">首次接入</button>
                <button @click="usePrompt('企业版如何配置 SSO 和 SCIM？')">企业账户</button>
                <button @click="usePrompt('Webhook 签名校验失败，应该检查哪些字段？')">Webhook 排障</button>
                <button @click="usePrompt('登录一直报 401，而且本月账单似乎重复扣款了')">复合问题</button>
              </div>
            </div>
          </div>

          <form class="composer" @submit.prevent="sendMessage">
            <textarea
              v-model="draft"
              rows="3"
              placeholder="描述 API、workspace、权限、订阅或系统故障问题..."
              @keydown.meta.enter.prevent="sendMessage"
              @keydown.ctrl.enter.prevent="sendMessage"
            ></textarea>
            <div class="composer-bottom">
              <span>⌘ / Ctrl + Enter 发送</span>
              <button type="submit" :disabled="busy || !draft.trim()">{{ busy ? '处理中' : '发送' }}</button>
            </div>
          </form>
        </section>

        <aside class="chat-sidebar" ref="sidebarRef">
          <div class="chat-sidebar-scroll">
            <section class="side-card session-card">
              <div class="card-heading">
                <div>
                  <span class="kicker">Session</span>
                  <h2>会话信息</h2>
                </div>
                <span class="status-copy muted">{{ settings.conversationId ? '已启用' : '新会话' }}</span>
              </div>
              <div class="session-grid">
                <div>
                  <span>会话 ID</span>
                  <strong>{{ settings.conversationId || '自动生成' }}</strong>
                </div>
                <div>
                  <span>用户 ID</span>
                  <strong>{{ settings.userId || 'anonymous' }}</strong>
                </div>
              </div>
            </section>

            <section class="side-card connection-card">
              <div class="card-heading">
                <div>
                  <span class="kicker">Connection</span>
                  <h2>连接配置</h2>
                </div>
                <span class="status-copy" :class="healthOk ? 'success' : 'muted'">{{ healthLabel }}</span>
              </div>

              <label>
                <span>用户 ID</span>
                <input v-model="settings.userId" @change="persist" placeholder="u1001" />
              </label>
              <label>
                <span>会话 ID</span>
                <input v-model="settings.conversationId" @change="persist" placeholder="自动生成" />
              </label>
              <div class="side-actions">
                <button @click="checkHealth">检查连接</button>
                <button class="quiet-button" @click="refreshConsole">刷新</button>
              </div>
            </section>

            <section class="side-card trace-card">
              <div class="card-heading">
                <div>
                  <span class="kicker">Last trace</span>
                  <h2>最近一次请求</h2>
                </div>
                <span class="trace-status" :class="lastResponse ? 'has-data' : ''"></span>
              </div>

              <div v-if="lastResponse" class="trace-body">
                <div class="latency">
                  <span>响应耗时</span>
                  <strong>{{ lastResponse.latencyMs || '-' }}<small> ms</small></strong>
                </div>
                <dl class="detail-list">
                  <div><dt>主处理 Agent</dt><dd>{{ formatAgent(lastResponse.primaryAgent || lastResponse.agentType) }}</dd></div>
                  <div><dt>业务意图</dt><dd>{{ formatIntent(lastResponse.intent) }}</dd></div>
                  <div><dt>意图组</dt><dd>{{ formatIntentGroup(lastResponse.intentGroup) }}</dd></div>
                  <div><dt>路由置信度</dt><dd>{{ formatPercent(lastResponse.routingConfidence) }}</dd></div>
                  <div><dt>意图置信度</dt><dd>{{ formatPercent(lastResponse.intentConfidence) }}</dd></div>
                  <div v-if="lastResponse.contextWindowUsagePercent !== null" :title="contextUsageTitle(lastResponse)"><dt>上下文窗口</dt><dd>{{ formatContextUsage(lastResponse) }}</dd></div>
                  <div><dt>知识库</dt><dd :class="lastResponse.knowledgeUsed ? 'success' : 'muted'">{{ lastResponse.knowledgeUsed ? '已使用' : '未使用' }}</dd></div>
                </dl>
                <div v-if="lastResponse.agentTypes?.length" class="response-section">
                  <span class="response-section-label">参与 Agent</span>
                  <div class="tag-list">
                    <span v-for="agent in lastResponse.agentTypes" :key="agent" class="data-tag">{{ formatAgent(agent) }}</span>
                  </div>
                </div>
                <div v-if="lastResponse.supportingAgents?.length" class="response-section">
                  <span class="response-section-label">协同处理</span>
                  <div class="tag-list">
                    <span v-for="agent in lastResponse.supportingAgents" :key="agent" class="data-tag supporting">{{ formatAgent(agent) }}</span>
                  </div>
                </div>
                <div v-if="responseDomainScores.length" class="response-section">
                  <span class="response-section-label">融合领域分数</span>
                  <div class="tag-list">
                    <span v-for="item in responseDomainScores" :key="item.key" class="data-tag">{{ formatAgent(item.key) }} {{ formatPercent(item.value) }}</span>
                  </div>
                </div>
                <div v-if="responseEntities.length" class="response-section">
                  <span class="response-section-label">识别实体</span>
                  <div class="entity-list">
                    <span v-for="entity in responseEntities" :key="entity.key" class="entity-item">
                      <b>{{ entity.label }}</b>{{ entity.values.join(' · ') }}
                    </span>
                  </div>
                </div>
                <p v-if="lastResponse.routingReason" class="routing-reason">{{ lastResponse.routingReason }}</p>
                <div v-if="lastTrace?.trace" class="trace-call-list">
                  <div class="trace-call-title">工具调用</div>
                  <div v-for="(call, index) in lastTrace.trace.toolCalls" :key="`${call.tool_use_id || index}`" class="trace-call-item">
                    <div class="trace-call-meta">
                      <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
                      <span>{{ call.latency_ms || 0 }} ms</span>
                    </div>
                    <pre>{{ formatJson(call.input || {}) }}</pre>
                  </div>
                  <div v-if="!lastTrace.trace.toolCalls?.length" class="trace-empty-block">
                    <p>这次 trace 没有记录到工具输入。</p>
                    <p v-if="lastTrace.trace.toolsUsed?.length" class="trace-note">已调用：{{ lastTrace.trace.toolsUsed.join(' · ') }}</p>
                  </div>
                </div>
              </div>
              <p v-else class="side-empty">发送消息后，这里会显示意图、实体、主/辅 Agent 路由和耗时。</p>
            </section>

            <section class="side-card monitor-card">
              <div class="card-heading">
                <div>
                  <span class="kicker">Runtime</span>
                  <h2>运行状态</h2>
                </div>
                <button class="link-button" @click="loadMonitor">刷新</button>
              </div>
              <div class="mini-stats">
                <div><strong>{{ totalRequests }}</strong><span>请求</span></div>
                <div><strong>{{ agentCount }}</strong><span>Agent</span></div>
                <div><strong>{{ activeAlerts.length }}</strong><span>告警</span></div>
              </div>
              <div v-if="activeAlerts.length" class="alert-note">{{ activeAlerts[0].detail || activeAlerts[0].title }}</div>
              <p v-else class="healthy-note">当前没有活跃告警。</p>
            </section>
          </div>
        </aside>
      </div>
    </section>

    <section v-else-if="activeView === 'knowledge'" class="page page-knowledge">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">Knowledge operations</span>
          <h1>知识库</h1>
          <p>搜索、补充和维护 B2B SaaS 客服 Agent 使用的产品、接入与运营知识。</p>
        </div>
        <div class="count-display"><strong>{{ knowledgeCount }}</strong><span>chunks</span></div>
      </div>

      <div class="knowledge-layout">
        <section class="workspace-card search-workspace">
          <div class="card-heading">
            <div><span class="kicker">Retrieval</span><h2>检索知识</h2></div>
            <code>POST /search</code>
          </div>
          <div class="search-line">
            <input v-model="searchQuery" placeholder="例如：API 调用返回 401，如何排查？" @keydown.enter="searchKnowledge" />
            <button @click="searchKnowledge" :disabled="busy || !searchQuery.trim()">搜索</button>
          </div>
          <div v-if="searchResults.length" class="result-list">
            <article v-for="(item, index) in searchResults" :key="item.id || item.title || index" class="result-item">
              <span class="result-number">{{ String(index + 1).padStart(2, '0') }}</span>
              <div>
                <div class="result-title"><strong>{{ item.title || '未命名文档' }}</strong><small>score {{ item.score ?? '-' }}</small></div>
                <p>{{ item.content }}</p>
              </div>
            </article>
          </div>
          <div v-else class="workspace-empty">输入 API、workspace、权限或订阅问题开始搜索。</div>
        </section>

        <section class="workspace-card import-workspace">
          <div class="card-heading">
            <div><span class="kicker">Ingestion</span><h2>添加知识</h2></div>
            <code>ChromaDB</code>
          </div>
          <label><span>标题</span><input v-model="docTitle" placeholder="API 鉴权与 workspace 排查规范" /></label>
          <label><span>内容</span><textarea v-model="docContent" rows="7" placeholder="输入 API 接入、企业账户、订阅账单或排障流程"></textarea></label>
          <div class="side-actions">
            <button @click="submitKnowledge" :disabled="busy || !docTitle.trim() || !docContent.trim()">添加文档</button>
            <label class="upload-button">上传文件<input type="file" accept=".txt,.md,.json" @change="handleUpload" /></label>
          </div>
        </section>
      </div>

      <section class="workspace-card skills-workspace">
        <div class="card-heading">
          <div><span class="kicker">Loaded skills</span><h2>已加载能力</h2></div>
          <button class="link-button" @click="reloadSkillSet">重新加载</button>
        </div>
        <div class="skill-table">
          <div v-for="skill in skillsData.skills" :key="skill.name" class="skill-item">
            <span class="skill-dot"></span><strong>{{ skill.name }}</strong><span>{{ skill.description || '业务规范能力' }}</span><small>{{ skill.content_chars || 0 }} chars</small>
          </div>
          <div v-if="!skillsData.skills.length" class="workspace-empty">暂无已加载 Skill。</div>
        </div>
      </section>
    </section>

    <section v-else class="page page-evaluation">
      <div class="page-heading">
        <div class="heading-copy">
          <span class="kicker">Evaluation lab</span>
          <h1>评测 Agent</h1>
          <p>运行 FastAPI 内置评测，查看意图识别、对话质量和回归结果。</p>
        </div>
        <button @click="runEvaluation" :disabled="busy">{{ busy ? '运行中...' : '运行评测' }}</button>
      </div>

      <div v-if="evalData" class="evaluation-content">
        <div class="evaluation-summary">
          <div class="score-hero"><span>Pass rate</span><strong>{{ formatPercent(evalData.pass_rate) }}</strong><small>{{ evalData.passed }} / {{ evalData.total }} cases passed</small></div>
          <div><span>通过</span><strong>{{ evalData.passed }}</strong></div>
          <div><span>总数</span><strong>{{ evalData.total }}</strong></div>
          <div><span>回归</span><strong :class="evalData.regressions?.length ? 'danger' : 'success'">{{ evalData.regressions?.length || 0 }}</strong></div>
        </div>
        <div class="evaluation-layout">
          <section class="workspace-card">
            <div class="card-heading"><div><span class="kicker">Scores</span><h2>平均评分</h2></div></div>
            <div class="score-list">
              <div v-for="(value, key) in evalData.avg_scores" :key="key"><span>{{ key }}</span><i><b :style="{ width: `${Math.min(Number(value) * 10, 100)}%` }"></b></i><strong>{{ Number(value).toFixed(2) }}</strong></div>
            </div>
          </section>
          <section class="workspace-card">
            <div class="card-heading"><div><span class="kicker">Recommendations</span><h2>优化建议</h2></div></div>
            <div v-if="evalData.recommendations?.length" class="recommendations"><p v-for="(item, index) in evalData.recommendations" :key="index">{{ item }}</p></div>
            <div v-else class="workspace-empty">本次评测没有返回额外建议。</div>
          </section>
        </div>
      </div>
      <div v-else class="evaluation-empty"><div class="empty-symbol">◎</div><h2>还没有评测结果</h2><p>点击右上角运行一次评测。</p></div>
    </section>
  </main>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import MarkdownIt from 'markdown-it'
import {
  addKnowledge,
  backendMeta,
  createInitialSettings,
  reloadSkills,
  requestChat,
  requestHealth,
  requestKnowledgeStats,
  requestMonitor,
  requestSearch,
  requestToolTrace,
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
  query: '产品咨询',
  request: '服务请求',
  complaint: '投诉反馈',
  greeting: '问候',
  feedback: '体验反馈',
  api_integration: 'API 接入',
  feature_inquiry: '功能咨询',
  onboarding: '新用户引导',
  enterprise_mgmt: '企业账户管理',
  subscription: '订阅管理',
  billing: '账单咨询',
  refund: '退款申请',
  invoice: '发票处理',
  payment_issue: '支付异常',
  account: '账户问题',
  account_security: '账户与密钥安全',
  technical: '技术支持',
  technical_login: '登录故障',
  technical_crash: '系统故障',
  other: '其他'
}

const INTENT_GROUP_LABELS = {
  query: '咨询',
  request: '请求',
  complaint: '投诉',
  greeting: '问候',
  feedback: '反馈',
  api_integration: 'API 接入',
  feature_inquiry: '功能咨询',
  onboarding: '新用户引导',
  enterprise_mgmt: '企业账户',
  billing: '订阅与账单',
  account: '账户与安全',
  technical: '技术支持',
  other: '其他'
}

const ENTITY_LABELS = {
  error_code: '错误码',
  request_id: '请求 ID',
  api_key: 'API Key',
  workspace: 'workspace',
  amount: '金额',
  order_id: '订单号',
  invoice: '发票',
  date: '日期',
  plan: '套餐',
  email: '邮箱'
}

const settings = reactive(createInitialSettings())
const markdown = new MarkdownIt({
  html: false,
  breaks: true,
  linkify: true,
  typographer: true
})
const activeView = ref('chat')
const messages = ref([])
const draft = ref('')
const busy = ref(false)
const healthOk = ref(false)
const healthLabel = ref('未检查')
const statusText = ref('')
const knowledgeCount = ref('-')
const searchQuery = ref('调用 /v1/chat 返回 401，如何排查？')
const searchResults = ref([])
const docTitle = ref('API 鉴权与 workspace 排查规范')
const docContent = ref('出现 401 时，先核对 Authorization Bearer 格式、API Key 所属 workspace、环境配置和 request_id；不要在工单或聊天中提交完整密钥。')
const messageList = ref(null)
const sidebarRef = ref(null)
const monitorData = ref({ agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] })
const skillsData = ref({ count: 0, skills: [], errors: [] })
const lastResponse = ref(null)
const lastTrace = ref(null)
const evalData = ref(null)
const toast = ref('')
let toastTimer
let messageSequence = 0
let sidebarObserver

const currentBackend = computed(() => backendMeta(settings.backend, settings))
const docsUrl = computed(() => `${currentBackend.value.baseUrl}/docs`)
const userInitial = computed(() => (settings.userId || 'U').slice(0, 1).toUpperCase())
const activeAlerts = computed(() => monitorData.value.active_alerts || [])
const agentCount = computed(() => Object.keys(monitorData.value.agent_stats || {}).length)
const totalRequests = computed(() => Object.values(monitorData.value.agent_stats || {}).reduce((sum, item) => sum + Number(item.total || 0), 0))
const responseEntities = computed(() => formatEntities(lastResponse.value?.entities))
const responseDomainScores = computed(() => Object.entries(lastResponse.value?.domainScores || {})
  .map(([key, value]) => ({ key, value: Number(value) }))
  .filter((item) => Number.isFinite(item.value) && item.value > 0)
  .sort((left, right) => right.value - left.value))

watch(() => settings.conversationId, persist)
onMounted(() => {
  refreshConsole()
  updateSidebarHeight()
  if (typeof ResizeObserver !== 'undefined') {
    sidebarObserver = new ResizeObserver(updateSidebarHeight)
    if (sidebarRef.value) sidebarObserver.observe(sidebarRef.value)
  }
  window.addEventListener('resize', updateSidebarHeight)
})

onBeforeUnmount(() => {
  sidebarObserver?.disconnect?.()
  window.removeEventListener('resize', updateSidebarHeight)
})

function persist() { saveSettings(settings) }

function updateSidebarHeight() {
  const sidebar = sidebarRef.value
  if (!sidebar) return
  const rect = sidebar.getBoundingClientRect()
  const height = Math.max(320, Math.floor(rect.height))
  sidebar.style.setProperty('--sidebar-height', `${height}px`)
}

async function refreshConsole() {
  await Promise.allSettled([checkHealth(), loadStats(), loadMonitor(), loadSkills()])
}

async function checkHealth() {
  try {
    const data = await requestHealth(settings.backend, settings)
    healthOk.value = data.status === 'ok'
    healthLabel.value = data.status || 'ok'
    statusText.value = JSON.stringify(data, null, 2)
  } catch (error) {
    healthOk.value = false
    healthLabel.value = '不可用'
    statusText.value = error.message
  }
}

async function loadStats() {
  try {
    const data = await requestKnowledgeStats(settings.backend, settings)
    knowledgeCount.value = data.total_chunks ?? data.totalChunks ?? '-'
  } catch {
    knowledgeCount.value = '-'
  }
}

async function loadMonitor() {
  try {
    monitorData.value = await requestMonitor(settings.backend, settings)
  } catch {
    monitorData.value = { agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] }
  }
}

async function loadSkills() {
  try {
    skillsData.value = await requestSkills(settings.backend, settings)
  } catch {
    skillsData.value = { count: 0, skills: [], errors: [] }
  }
}

async function reloadSkillSet() {
  busy.value = true
  try {
    skillsData.value = await reloadSkills(settings.backend, settings)
    showToast('Skills 已重新加载')
  } catch (error) {
    statusText.value = error.message
    showToast('Skills 加载失败')
  } finally { busy.value = false }
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content || busy.value) return
  messages.value.push({ id: createMessageId(), role: 'user', content })
  draft.value = ''
  busy.value = true
  try {
    const response = await requestChat(settings.backend, settings, content)
    if (response.conversationId && !settings.conversationId) {
      settings.conversationId = response.conversationId
      persist()
    }
    lastResponse.value = response
    lastTrace.value = await loadToolTrace(response.requestId)
    const meta = [
      formatIntent(response.intent),
      formatAgent(response.primaryAgent || response.agentType),
      response.contextWindowUsagePercent !== null ? `窗口 ${formatContextUsage(response)}` : '',
      response.knowledgeUsed ? 'RAG' : ''
    ].filter(Boolean).join(' · ')
    messages.value.push({ id: createMessageId(), role: 'assistant', content: response.response, meta, trace: lastTrace.value?.trace || null })
    await loadMonitor()
  } catch (error) {
    messages.value.push({ id: createMessageId(), role: 'assistant', content: error.message, meta: '请求失败' })
  } finally {
    busy.value = false
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
  }
}

function usePrompt(prompt) { draft.value = prompt }

function clearConversation() {
  messages.value = []
  lastResponse.value = null
  lastTrace.value = null
  settings.conversationId = ''
  persist()
}

async function searchKnowledge() {
  busy.value = true
  try {
    const data = await requestSearch(settings.backend, settings, searchQuery.value, 5)
    searchResults.value = data.results || []
    showToast(`检索完成，返回 ${searchResults.value.length} 条结果`)
  } catch (error) {
    statusText.value = error.message
    showToast('检索失败，请检查连接')
  } finally { busy.value = false }
}

async function submitKnowledge() {
  busy.value = true
  try {
    const data = await addKnowledge(settings.backend, settings, [{ title: docTitle.value.trim(), content: docContent.value.trim() }])
    statusText.value = JSON.stringify(data, null, 2)
    await loadStats()
    showToast('文档已添加')
  } catch (error) {
    statusText.value = error.message
    showToast('文档导入失败')
  } finally { busy.value = false }
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  busy.value = true
  try {
    const data = await uploadKnowledge(settings.backend, settings, file)
    statusText.value = JSON.stringify(data, null, 2)
    await loadStats()
    showToast(`${file.name} 导入成功`)
  } catch (error) {
    statusText.value = error.message
    showToast('文件导入失败')
  } finally { busy.value = false }
}

async function runEvaluation() {
  busy.value = true
  try {
    evalData.value = await requestEvaluation(settings.backend, settings)
    showToast('评测完成')
  } catch (error) {
    statusText.value = error.message
    showToast('评测运行失败')
  } finally { busy.value = false }
}

async function loadToolTrace(requestId) {
  try {
    return await requestToolTrace(settings.backend, settings, requestId)
  } catch {
    return null
  }
}

function formatPercent(value) {
  const number = Number(value || 0)
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`
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
  return Object.entries(entities)
    .map(([key, rawValues]) => {
      const values = Array.isArray(rawValues) ? rawValues : [rawValues]
      return {
        key,
        label: ENTITY_LABELS[key] || key.replaceAll('_', ' '),
        values: values.map(value => String(value).trim()).filter(Boolean)
      }
    })
    .filter(entity => entity.values.length)
}

function formatContextUsage(response) {
  const prefix = response.contextWindowUsageEstimated ? '≈' : ''
  const percent = Number(response.contextWindowUsagePercent)
  return Number.isFinite(percent) ? `${prefix}${percent.toFixed(1)}%` : '-'
}

function contextUsageTitle(response) {
  const base = '业务 Agent 首次生成前的输入窗口占用；并行协作时显示最高值。'
  return response.contextWindowUsageEstimated ? `${base} 当前兼容服务未提供精确 token 计数，数值为保守估算。` : base
}

function formatJson(value) {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch {
    return String(value ?? '')
  }
}

function createMessageId() {
  messageSequence += 1
  return `message-${Date.now()}-${messageSequence}`
}

function renderMarkdown(content) {
  return markdown.render(String(content || ''))
}

function showToast(message) {
  toast.value = message
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 2600)
}
</script>
