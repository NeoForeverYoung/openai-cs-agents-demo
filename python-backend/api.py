"""
模块说明（中文）：
本模块基于 FastAPI 提供一个聊天接口（/chat），对接多智能体编排（见 main.py）。
职责：
- 管理会话状态（内存存储演示版）
- 调用 Runner.run 执行当前智能体，处理消息、工具调用、转办事件
- 收集与返回消息列表、事件流、上下文变化、可用智能体信息、守护（guardrail）检查结果

说明：
- 演示环境使用内存存储 InMemoryConversationStore。生产请替换为外部持久化与分布式会话管理。
- 仅添加中文注释，不改变功能与数据契约（response_model/字段等保持不变）。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging

from main import (
    triage_agent,
    faq_agent,
    seat_booking_agent,
    flight_status_agent,
    cancellation_agent,
    create_initial_context,
)

from agents import (
    Runner,
    ItemHelpers,
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    InputGuardrailTripwireTriggered,
    Handoff,
)

# 基础日志配置（INFO 级别），生产可接入更完善的日志体系
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI()

# CORS 配置：前端默认在 http://localhost:3000 开发
# 部署时可改为具体域名或使用更严格的白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic 模型（请求/响应/事件）
# =========================

class ChatRequest(BaseModel):
    """
    聊天请求：
    - conversation_id：可选，会话 ID；若为空或不存在，将创建新会话
    - message：用户输入消息
    """
    conversation_id: Optional[str] = None
    message: str

class MessageResponse(BaseModel):
    """
    单条回复消息：
    - content：文本内容
    - agent：产生该消息的智能体名称
    """
    content: str
    agent: str

class AgentEvent(BaseModel):
    """
    事件记录（用于前端时间轴展示）：
    - type：事件类型（message/handoff/tool_call/tool_output/context_update 等）
    - agent：产生该事件的智能体
    - content：事件内容/描述
    - metadata：可选的附加数据（如工具参数、结果等）
    - timestamp：时间戳（毫秒），未设置时可由服务端填充
    """
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class GuardrailCheck(BaseModel):
    """
    守护（guardrail）检查结果：
    - name：守护名称
    - input：被判断的输入内容
    - reasoning：守护给出的理由
    - passed：是否通过（True 表示安全/相关；False 表示触发/不通过）
    - timestamp：时间戳（毫秒）
    """
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float

class ChatResponse(BaseModel):
    """
    聊天响应总览：
    - conversation_id：会话 ID
    - current_agent：当前会话接管的智能体
    - messages：新产生的消息列表
    - events：事件流（含转办、工具调用、上下文变更）
    - context：序列化后的上下文对象（dict）
    - agents：当前系统中可用的智能体列表与其元数据
    - guardrails：守护检查结果（本轮）
    """
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []

# =========================
# In-memory 会话存储（演示）
# =========================

class ConversationStore:
    """
    会话存储抽象：
    - get：通过会话 ID 获取状态
    - save：保存会话状态
    生产环境可实现为 Redis/数据库/自建服务等
    """
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        pass

    def save(self, conversation_id: str, state: Dict[str, Any]):
        pass

class InMemoryConversationStore(ConversationStore):
    """
    简单的内存版会话存储：
    - 仅适用于单进程演示；多实例/重启后会丢失
    """
    _conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state

# TODO: 生产部署时务必替换为可扩展、可靠的会话存储实现
conversation_store = InMemoryConversationStore()

# =========================
# 辅助函数
# =========================

def _get_agent_by_name(name: str):
    """
    根据名称获取智能体对象。
    默认回退到 triage_agent（保证始终有可用智能体）。
    """
    agents = {
        triage_agent.name: triage_agent,
        faq_agent.name: faq_agent,
        seat_booking_agent.name: seat_booking_agent,
        flight_status_agent.name: flight_status_agent,
        cancellation_agent.name: cancellation_agent,
    }
    return agents.get(name, triage_agent)

def _get_guardrail_name(g) -> str:
    """
    获取友好的守护名称：
    - 优先读取对象的 name
    - 其次从 guardrail_function 或函数名推断
    - 最终回退到 str(g)
    """
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    return str(g)

def _build_agents_list() -> List[Dict[str, Any]]:
    """
    构建当前可用智能体及其元数据的列表，以便前端展示。
    包含：名称、描述、可转办对象、可用工具、输入守护。
    """
    def make_agent_dict(agent):
        return {
            "name": agent.name,
            "description": getattr(agent, "handoff_description", ""),
            "handoffs": [getattr(h, "agent_name", getattr(h, "name", "")) for h in getattr(agent, "handoffs", [])],
            "tools": [getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(agent, "tools", [])],
            "input_guardrails": [_get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])],
        }
    return [
        make_agent_dict(triage_agent),
        make_agent_dict(faq_agent),
        make_agent_dict(seat_booking_agent),
        make_agent_dict(flight_status_agent),
        make_agent_dict(cancellation_agent),
    ]

# =========================
# 主聊天接口
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    处理一次聊天轮次：
    1) 初始化/获取会话状态（input_items、上下文、当前智能体）
    2) 追加用户输入为新的 input item
    3) 调用 Runner.run 执行当前智能体
       - 捕获 InputGuardrailTripwireTriggered：若守护触发则直接返回拒答与守护结果
    4) 解析 result.new_items：
       - MessageOutputItem：收集为 messages 与 events
       - HandoffOutputItem：记录转办事件并切换 current_agent（必要时记录 on_handoff 回调名）
       - ToolCallItem/ToolCallOutputItem：记录工具调用/输出为事件；特殊处理 display_seat_map -> 向消息中注入 DISPLAY_SEAT_MAP
    5) 比对上下文变化，记录 context_update 事件
    6) 保存会话最新状态，构造 guardrail 检查结果并返回
    """
    # 1) 初始化或恢复会话状态
    is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
    if is_new:
        # 新会话：生成会话 ID、初始化上下文与默认智能体（分诊）
        conversation_id: str = uuid4().hex
        ctx = create_initial_context()
        current_agent_name = triage_agent.name
        state: Dict[str, Any] = {
            "input_items": [],        # 历史对话输入项（Runner 需要）
            "context": ctx,           # 业务上下文
            "current_agent": current_agent_name,  # 当前接管的智能体
        }
        # 若首次消息是空字符串，直接返回基础信息（前端可用来获取 agents/context 等）
        if req.message.strip() == "":
            conversation_store.save(conversation_id, state)
            return ChatResponse(
                conversation_id=conversation_id,
                current_agent=current_agent_name,
                messages=[],
                events=[],
                context=ctx.model_dump(),
                agents=_build_agents_list(),
                guardrails=[],
            )
    else:
        # 继续已有会话
        conversation_id = req.conversation_id  # type: ignore
        state = conversation_store.get(conversation_id)

    # 2) 将用户消息追加到 input_items（Runner 输入协议）
    current_agent = _get_agent_by_name(state["current_agent"])
    state["input_items"].append({"content": req.message, "role": "user"})
    # 记录旧的上下文快照，用于后续对比变更
    old_context = state["context"].model_dump().copy()
    guardrail_checks: List[GuardrailCheck] = []

    # 3) 执行当前智能体；若守护触发，返回拒答与守护结果
    try:
        result = await Runner.run(current_agent, state["input_items"], context=state["context"])
    except InputGuardrailTripwireTriggered as e:
        # 有守护触发：收集守护名称、是否通过、理由等
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = req.message
        gr_timestamp = time.time() * 1000
        for g in current_agent.input_guardrails:
            guardrail_checks.append(GuardrailCheck(
                id=uuid4().hex,
                name=_get_guardrail_name(g),
                input=gr_input,
                reasoning=(gr_reasoning if g == failed else ""),
                passed=(g != failed),
                timestamp=gr_timestamp,
            ))
        # 返回一个统一的拒答文案（演示文案，可按需定制国际化）
        refusal = "Sorry, I can only answer questions related to airline travel."
        state["input_items"].append({"role": "assistant", "content": refusal})
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )

    # 4) 解析新产生的输出项（消息/转办/工具调用/工具输出）
    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []

    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            # 标准文本消息：提取文本与来源智能体
            text = ItemHelpers.text_message_output(item)
            messages.append(MessageResponse(content=text, agent=item.agent.name))
            events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))
        # 处理转办输出并切换智能体
        elif isinstance(item, HandoffOutputItem):
            # 记录转办事件：来源智能体 -> 目标智能体
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="handoff",
                    agent=item.source_agent.name,
                    content=f"{item.source_agent.name} -> {item.target_agent.name}",
                    metadata={"source_agent": item.source_agent.name, "target_agent": item.target_agent.name},
                )
            )
            # 若手动定义了 on_handoff 回调（通过 handoff(...) 包装），尝试记录回调函数名以供前端展示
            from_agent = item.source_agent
            to_agent = item.target_agent
            # 在来源智能体的 handoffs 列表中查找对应目标的 Handoff 对象
            ho = next(
                (h for h in getattr(from_agent, "handoffs", [])
                 if isinstance(h, Handoff) and getattr(h, "agent_name", None) == to_agent.name),
                None,
            )
            if ho:
                # 通过闭包变量取出 on_handoff 回调并记录其名称
                fn = ho.on_invoke_handoff
                fv = fn.__code__.co_freevars
                cl = fn.__closure__ or []
                if "on_handoff" in fv:
                    idx = fv.index("on_handoff")
                    if idx < len(cl) and cl[idx].cell_contents:
                        cb = cl[idx].cell_contents
                        cb_name = getattr(cb, "__name__", repr(cb))
                        events.append(
                            AgentEvent(
                                id=uuid4().hex,
                                type="tool_call",
                                agent=to_agent.name,
                                content=cb_name,
                            )
                        )
            # 切换当前智能体为转办目标
            current_agent = item.target_agent
        elif isinstance(item, ToolCallItem):
            # 工具调用项：提取工具名与参数，记录事件
            tool_name = getattr(item.raw_item, "name", None)
            raw_args = getattr(item.raw_item, "arguments", None)
            tool_args: Any = raw_args
            if isinstance(raw_args, str):
                # 尝试解析 JSON 字符串参数（若失败则保留原字符串）
                try:
                    import json
                    tool_args = json.loads(raw_args)
                except Exception:
                    pass
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=item.agent.name,
                    content=tool_name or "",
                    metadata={"tool_args": tool_args},
                )
            )
            # 特殊处理：若为 display_seat_map 工具，则给前端发送 DISPLAY_SEAT_MAP 信号
            if tool_name == "display_seat_map":
                messages.append(
                    MessageResponse(
                        content="DISPLAY_SEAT_MAP",
                        agent=item.agent.name,
                    )
                )
        elif isinstance(item, ToolCallOutputItem):
            # 工具输出项：记录工具返回值到事件
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_output",
                    agent=item.agent.name,
                    content=str(item.output),
                    metadata={"tool_result": item.output},
                )
            )

    # 5) 比对上下文变化，若有变更则记录 context_update 事件（便于前端展示关键字段更新）
    new_context = state["context"].dict()
    changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
    if changes:
        events.append(
            AgentEvent(
                id=uuid4().hex,
                type="context_update",
                agent=current_agent.name,
                content="",
                metadata={"changes": changes},
            )
        )

    # 6) 更新会话状态：将结果转换为下轮 input_items，并保存当前智能体
    state["input_items"] = result.to_input_list()
    state["current_agent"] = current_agent.name
    conversation_store.save(conversation_id, state)

    # 7) 构造本轮守护检查结果：
    #    - 若之前因触发而记录过失败项，则保留其详细信息
    #    - 其余守护标记为通过（passed=True），reasoning 置空
    final_guardrails: List[GuardrailCheck] = []
    for g in getattr(current_agent, "input_guardrails", []):
        name = _get_guardrail_name(g)
        failed = next((gc for gc in guardrail_checks if gc.name == name), None)
        if failed:
            final_guardrails.append(failed)
        else:
            final_guardrails.append(GuardrailCheck(
                id=uuid4().hex,
                name=name,
                input=req.message,
                reasoning="",
                passed=True,
                timestamp=time.time() * 1000,
            ))

    # 返回完整响应
    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].dict(),
        agents=_build_agents_list(),
        guardrails=final_guardrails,
    )