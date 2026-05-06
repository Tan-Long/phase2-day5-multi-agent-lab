# Lab 20: Multi-Agent Research System Starter

Starter repo cho bài lab **Multi-Agent Systems**: xây dựng hệ thống nghiên cứu theo pipeline **Supervisor → Researcher → Analyst → Writer**, có **guardrails**, **tracing** và **benchmark** để so sánh với single-agent baseline.

Repo này là một **scaffold “production-grade”**: kiến trúc, schema, workflow, benchmark đã có khung; phần “trí tuệ” của từng agent được cố ý để dạng `TODO(student)`/`StudentTodoError` để học viên tự triển khai.

## Chạy nhanh (5 phút)

### 1) Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install (includes dev + optional llm extras)
pip install -e ".[dev,llm]"
cp .env.example .env
```

### 2) Cấu hình `.env`

Mở `.env` và điền key cần thiết:

```bash
OPENAI_API_KEY=...

# optional
LANGSMITH_API_KEY=...
TAVILY_API_KEY=...
```

### 3) Smoke test + xem CLI

```bash
make test
python3 -m multi_agent_research_lab.cli --help
```

### 4) Chạy baseline (single-agent)

```bash
make run-baseline
```

Hoặc chạy trực tiếp:

```bash
python3 -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

### 5) Chạy multi-agent workflow

```bash
make run-multi
```

Hoặc chạy trực tiếp:

```bash
python3 -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Lưu ý: trong `Makefile` hiện tại các target `run-baseline`/`run-multi` dùng `python` (không phải `python3`). Nếu máy bạn không có alias `python`, hãy chạy theo lệnh `python3 -m ...` như trên, hoặc tự chỉnh `Makefile` để dùng `python3`.

Nếu bạn chưa implement các phần `TODO(student)`, CLI có thể dừng với `StudentTodoError` — đó là hành vi “đúng thiết kế” của starter.

## Kiến trúc multi-agent (tổng quan)

### Data flow

```text
CLI (src/multi_agent_research_lab/cli.py)
  └─> ResearchState (core/state.py)  ← shared state (Pydantic)
        └─> MultiAgentWorkflow.run() (graph/workflow.py)
              ├─> SupervisorAgent   (agents/supervisor.py)  decides next: researcher | analyst | writer | done
              ├─> ResearcherAgent   (agents/researcher.py)  fills: sources + research_notes
              ├─> AnalystAgent      (agents/analyst.py)     fills: analysis_notes
              ├─> WriterAgent       (agents/writer.py)      fills: final_answer (+ citations)
              └─> (optional) CriticAgent                   quality gate
```

### Shared state (vì sao quan trọng?)

Mọi node trong graph **không truyền “output rời rạc” cho nhau**, mà cùng đọc/ghi vào một object duy nhất: `ResearchState`.

Thiết kế này giúp:
- Debug dễ: xem state ở mỗi bước để biết agent nào “đổ dữ liệu” gì.
- Handoff rõ: Supervisor route dựa trên state (đã có sources chưa? đã có analysis chưa?).
- Benchmark/tracing thuận tiện: ghi lại latency/tokens/cost theo từng agent.

### Các agent làm gì?

- **Supervisor (router)**: quyết định bước kế tiếp dựa trên `ResearchState` và guardrails (iteration/timeout/errors).
- **Researcher**: gọi search/web client, tổng hợp nguồn, đẩy vào `state.sources` + `state.research_notes`.
- **Analyst**: biến research notes thành lập luận/so sánh/insights, đẩy vào `state.analysis_notes`.
- **Writer**: viết câu trả lời cuối cùng, ưu tiên cấu trúc rõ và có citations, đẩy vào `state.final_answer`.

## Guardrails (an toàn vận hành)

Scaffold đã đặt các guardrails tối thiểu, mục tiêu là tránh “agent chạy vô hạn” và có cơ chế fail-safe:
- **Max iterations**: Supervisor phải dừng khi `state.iteration` vượt ngưỡng.
- **Timeout**: workflow có deadline `timeout_seconds`; quá hạn sẽ raise lỗi và CLI báo rõ.
- **Retry / fallback**: khi một agent lỗi nhiều lần, workflow có thể skip agent đó và dùng giá trị fallback (tùy bài lab).
- **Schema-first I/O**: input/output quan trọng dùng Pydantic types trong `core/schemas.py` để giảm “stringly-typed bugs”.

## Tracing & observability

Tracing hooks nằm ở `src/multi_agent_research_lab/observability/tracing.py`.

Mục tiêu tracing trong lab:
- Thấy rõ route: `supervisor → researcher → analyst → writer → done`
- Đo **latency/tokens/cost** theo từng agent
- Lưu “artifact” để chụp screenshot hoặc gửi link trace khi nộp bài

## Benchmark (single vs multi-agent)

Benchmark harness ở `src/multi_agent_research_lab/evaluation/benchmark.py`.

Repo hướng tới việc so sánh theo nhiều trục:
- **Latency**: thời gian wall-clock
- **Cost**: token usage / provider usage
- **Quality**: rubric peer-review (0–10)
- **Failure rate**: số query fail / tổng query

Phần report (ví dụ `reports/benchmark_report.md`) được tạo bởi code trong `evaluation/` (tuỳ scaffold hiện tại).

## Cấu trúc repo (điểm vào quan trọng)

Các file bạn sẽ chạm nhiều nhất:
- `src/multi_agent_research_lab/cli.py`: CLI entrypoint (baseline + multi-agent)
- `src/multi_agent_research_lab/core/state.py`: `ResearchState` (shared state)
- `src/multi_agent_research_lab/agents/*.py`: Supervisor/Researcher/Analyst/Writer skeleton
- `src/multi_agent_research_lab/graph/workflow.py`: LangGraph workflow build/run
- `src/multi_agent_research_lab/services/llm_client.py`: LLM wrapper (thường là thứ implement đầu tiên)
- `src/multi_agent_research_lab/services/search_client.py`: Search/web wrapper
- `src/multi_agent_research_lab/observability/tracing.py`: tracing hooks
- `src/multi_agent_research_lab/evaluation/benchmark.py`: benchmark runner

## Lộ trình 2 giờ lab (gợi ý)

- **0–15'**: setup, chạy baseline, implement LLM client tối thiểu
- **15–45'**: implement Supervisor routing + stop conditions
- **45–75'**: implement Researcher/Analyst/Writer (schema-first)
- **75–95'**: bật tracing + chạy benchmark
- **95–115'**: peer review theo `docs/peer_review_rubric.md`
- **115–120'**: trả lời exit ticket trong `docs/lab_guide.md`

## Tìm các phần cần làm (TODO)

```bash
grep -R "TODO(student)" -n src tests docs
```

## Deliverables (nộp bài)

- Repo GitHub cá nhân (fork + commit + push)
- Screenshot trace hoặc link trace
- Benchmark report so sánh baseline vs multi-agent
- Mô tả vài failure mode và cách fix (liên hệ guardrails)

## Tài liệu tham khảo

- Anthropic: Building effective agents — `https://www.anthropic.com/engineering/building-effective-agents`
- OpenAI: Orchestration & handoffs — `https://developers.openai.com/api/docs/guides/agents/orchestration`
- LangGraph concepts — `https://langchain-ai.github.io/langgraph/concepts/`
- LangSmith tracing — `https://docs.smith.langchain.com/`
- Langfuse tracing — `https://langfuse.com/docs`
