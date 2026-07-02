from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from config import AppPaths, ensure_dirs
from core_rag import RAGIndex
from chatbot_agent.tool_decorator import agent_tool
from chatbot_agent.chatbot_agent import ChatBot

ensure_dirs()
paths = AppPaths()
rag = RAGIndex(paths)
app = FastAPI(title="聊天机器人服务", version="1.0.0")


class ChatRequest(BaseModel):
    session_id: str = Field(default="default_chat", description="会话ID")
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    source: str
    contexts: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)


class BuildResponse(BaseModel):
    document_count: int
    message: str


class KnowledgeBaseSearchParams(BaseModel):
    question: str = Field(..., description="用户问题")


@agent_tool(
    name="knowledge_base_search",
    description="根据用户问题检索本地知识库，并返回最相关的文档片段",
    param_model=KnowledgeBaseSearchParams,
)
def knowledge_base_search(params: KnowledgeBaseSearchParams) -> str:
    question = getattr(params, "question", "").strip()
    if not question:
        return "[知识库检索失败: question 不能为空]"

    contexts = rag.hybrid_search(question)
    if not contexts:
        return "知识库中未找到足够相关的内容。"

    blocks: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"[{i}] 标题路径：{ctx['title_path']}\n来源：{ctx['source_file']}\n内容：{ctx['content']}"
        )
    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def get_chatbot() -> ChatBot:
    return ChatBot(tools=None)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>聊天机器人</title>
  <style>
    body{margin:0;font-family:Inter,Arial,sans-serif;background:linear-gradient(180deg,#eef4ff,#f8fafc);color:#0f172a}
    .wrap{max-width:980px;margin:0 auto;padding:24px}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:20px;box-shadow:0 20px 40px rgba(15,23,42,.08);overflow:hidden}
    .header{padding:20px 24px;border-bottom:1px solid #e5e7eb}
    .header h1{margin:0;font-size:24px}
    .header p{margin:8px 0 0;color:#64748b}
    .chat{height:68vh;overflow:auto;padding:24px;display:flex;flex-direction:column;gap:14px}
    .msg{max-width:82%;padding:14px 16px;border-radius:16px;line-height:1.65;white-space:pre-wrap}
    .user{align-self:flex-end;background:#2563eb;color:#fff;border-bottom-right-radius:6px}
    .bot{align-self:flex-start;background:#f1f5f9;color:#111827;border-bottom-left-radius:6px}
    .input{display:flex;gap:12px;padding:18px 24px;border-top:1px solid #e5e7eb;background:#fff}
    textarea{flex:1;resize:none;border:1px solid #cbd5e1;border-radius:14px;padding:14px 16px;font-size:15px;min-height:56px;outline:none}
    button{border:none;background:linear-gradient(135deg,#2563eb,#7c3aed);color:#fff;padding:0 20px;border-radius:14px;font-size:15px;min-width:110px;cursor:pointer}
    button:disabled{opacity:.6;cursor:not-allowed}
    .meta{font-size:12px;color:#64748b;margin-top:8px}
    .toolbar{display:flex;gap:10px;padding:0 24px 18px}
    .ghost{background:#e2e8f0;color:#0f172a}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <h1>聊天机器人</h1>
        <p>LLM 会先判断是否需要调用知识库工具，命中后直接返回最终答案。</p>
      </div>
      <div id="chat" class="chat"></div>
      <div class="toolbar">
        <button class="ghost" id="build">重建知识库索引</button>
      </div>
      <div class="input">
        <textarea id="msg" placeholder="请输入问题，例如：项目里的知识库工具怎么用？"></textarea>
        <button id="send">发送</button>
      </div>
    </div>
  </div>
<script>
const chat=document.getElementById('chat');
const msg=document.getElementById('msg');
const send=document.getElementById('send');
const build=document.getElementById('build');

function escapeHtml(text) {
  if (!text) return '';
  const escapeMap = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  };
  return text.replace(/[&<>"']/g, function(m) { return escapeMap[m]; });
}

function addMessage(text, cls, meta='') {
  const el=document.createElement('div');
  el.className=`msg ${cls}`;
  el.innerHTML=`<div>${escapeHtml(text).replace(/\\n/g,'<br>')}</div>${meta?`<div class="meta">${escapeHtml(meta)}</div>`:''}`;
  chat.appendChild(el);
  chat.scrollTop=chat.scrollHeight;
}

async function postJson(url, body){
  const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json().catch(()=>({}));
  if(!res.ok){
    throw new Error(data.detail || data.message || '请求失败');
  }
  return data;
}

async function handleSend(){
  const text=msg.value.trim();
  if(!text) return;
  addMessage(text,'user');
  msg.value='';
  send.disabled=true;
  const think=document.createElement('div');
  think.className='msg bot';
  think.textContent='思考中...';
  chat.appendChild(think);
  chat.scrollTop=chat.scrollHeight;
  try{
    const data=await postJson('/api/chat',{session_id:'web_chat',message:text});
    think.remove();
    const meta=data.source==='knowledge_base'
      ? `来源：知识库；工具返回 ${data.tool_results.length} 条`
      : '来源：LLM 直接回答';
    addMessage(data.answer,'bot',meta);
  }catch(e){
    think.remove();
    addMessage(e.message || '请求失败，请稍后重试。','bot');
  }finally{
    send.disabled=false;
  }
}

send.addEventListener('click',handleSend);
msg.addEventListener('keydown',(e)=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();handleSend();}});
build.addEventListener('click',async()=>{
  build.disabled=true;
  const think=document.createElement('div');
  think.className='msg bot';
  think.textContent='正在重建索引...';
  chat.appendChild(think);
  chat.scrollTop=chat.scrollHeight;
  try{
    const data=await postJson('/api/rebuild',{});
    think.remove();
    addMessage(`索引重建完成，共处理 ${data.document_count} 个切片。`,'bot');
  }catch(e){
    think.remove();
    addMessage(e.message || '重建索引失败，请检查文档和依赖。','bot');
  }finally{
    build.disabled=false;
  }
});

addMessage('你好，我是聊天机器人。你可以直接提问，LLM 会自动决定是否检索知识库。','bot');
</script>
</body>
</html>
"""


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/rebuild", response_model=BuildResponse)
def rebuild() -> BuildResponse:
    try:
        count = rag.build(paths.doc_dir)
        return BuildResponse(document_count=count, message="索引重建成功")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat_api(payload: ChatRequest) -> ChatResponse:
    try:
        bot = get_chatbot()
        reply = await bot.async_reply(payload.message, context=None)
        tool_results = bot.last_tool_results
        source = bot.last_source

        contexts: list[dict] = []
        if source == "knowledge_base" and tool_results:
            contexts = rag.hybrid_search(payload.message)

        return ChatResponse(
            answer=reply.content or "",
            source=source,
            contexts=contexts,
            tool_results=tool_results,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"聊天请求失败: {exc}") from exc
