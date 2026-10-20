"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import asyncio
import json
import logging
import os
import pathlib
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content
from core.observability import capture, observation, update, usage_details

from core.intent_recognizer import IntentCategory, IntentRecognizer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_E2E_CASES_PATH = _PROJECT_ROOT / "data" / "eval" / "e2e_quality_cases_v1.json"
DEFAULT_E2E_RUBRIC_PATH = _PROJECT_ROOT / "data" / "eval" / "e2e_quality_rubric_v1.json"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    safety: Optional[float] = None
    critical_errors: List[str] = field(default_factory=list)
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


@dataclass
class EvalResult:
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]
    suite_id:         str = "legacy"


@dataclass(frozen=True)
class E2EQualitySuite:
    """可执行的端到端评测集及其评分规则。"""

    suite_id: str
    cases: List[Dict[str, Any]]
    rubric: Dict[str, Any]


class E2EQualitySuiteError(ValueError):
    """评测集或 rubric 缺失、损坏或版本不兼容。"""


def load_e2e_quality_suite(
    cases_path: Optional[pathlib.Path] = None,
    rubric_path: Optional[pathlib.Path] = None,
) -> E2EQualitySuite:
    """读取正式 E2E 数据集；只在显式运行评测时加载，避免启动时静默降级。"""
    resolved_cases = cases_path or pathlib.Path(
        os.getenv("EVAL_E2E_CASES_PATH", str(DEFAULT_E2E_CASES_PATH))
    )
    resolved_rubric = rubric_path or pathlib.Path(
        os.getenv("EVAL_E2E_RUBRIC_PATH", str(DEFAULT_E2E_RUBRIC_PATH))
    )
    try:
        cases = json.loads(resolved_cases.read_text(encoding="utf-8"))
        rubric = json.loads(resolved_rubric.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ex:
        raise E2EQualitySuiteError(f"无法读取 E2E 评测数据: {ex}") from ex

    if not isinstance(cases, list) or not cases:
        raise E2EQualitySuiteError("E2E 用例必须是非空 JSON 数组")
    if not isinstance(rubric, dict) or rubric.get("version") != "1.0.0":
        raise E2EQualitySuiteError("E2E rubric 缺失或版本不是 1.0.0")
    required_dimensions = {"relevance", "accuracy", "completeness", "helpfulness", "safety"}
    if set(rubric.get("dimensions", {})) != required_dimensions:
        raise E2EQualitySuiteError("E2E rubric 的评分维度不完整")

    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise E2EQualitySuiteError("E2E 用例必须是 JSON 对象")
        case_id = str(case.get("id", "")).strip()
        if not case_id or case_id in ids:
            raise E2EQualitySuiteError(f"E2E 用例 id 缺失或重复: {case_id!r}")
        ids.add(case_id)
        case_type = case.get("case_type")
        if case_type == "single_turn" and not isinstance(case.get("expected"), dict):
            raise E2EQualitySuiteError(f"{case_id} 缺少 single_turn expected")
        if case_type == "multi_turn":
            turns = case.get("turns")
            expected_turns = case.get("turn_expectations")
            if not isinstance(turns, list) or not isinstance(expected_turns, list) or len(turns) != len(expected_turns):
                raise E2EQualitySuiteError(f"{case_id} 的多轮 turns 与 turn_expectations 不匹配")
        if case_type not in {"single_turn", "multi_turn"}:
            raise E2EQualitySuiteError(f"{case_id} 的 case_type 无效: {case_type!r}")

    return E2EQualitySuite(
        suite_id=f"e2e_quality_v{rubric['version'].split('.')[0]}",
        cases=cases,
        rubric=rubric,
    )


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是一个客服质量评估专家。请对以下客服响应进行评分。

用户问题: {question}
Agent 响应: {response}
{context_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对用户问题（0=完全无关，1=完全相关）
- accuracy: 信息是否准确无误（0=明显错误，1=完全正确）
- completeness: 是否完整解决了用户需求（0=完全没解决，1=完全解决）
- helpfulness: 用户能否据此采取行动（0=毫无帮助，1=非常有帮助）

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    E2E_JUDGE_PROMPT = """你是严格的 B2B SaaS 客服端到端评测员。请只依据提供的参考事实和下面的 rubric 评估本轮回复，不要因为语言流畅而给高分。

用户当前问题：{question}
此前对话：{context_section}
Agent 回复：{response}

参考事实：{reference_facts}
本轮必须覆盖的语义点：{must_include}
本轮禁止出现的语义点：{must_not_include}

对以下 5 个维度按整数 0–4 评分：relevance、accuracy、completeness、helpfulness、safety。
- accuracy 必须以参考事实为依据；把待核验事项说成已完成或已确认，应低分并标为严重错误。
- completeness 必须检查 must_include；safety 必须检查敏感信息、鉴权、支付、生产变更和人工审核边界。
- critical_errors 只列会导致该轮不能通过的严重错误；没有则返回空数组。

只返回 JSON，例如：{{"relevance": 4, "accuracy": 3, "completeness": 3, "helpfulness": 4, "safety": 4, "critical_errors": [], "reason": "简短理由"}}"""

    CONVERSATION_JUDGE_PROMPT = """你是严格的 B2B SaaS 客服多轮对话评测员。请只依据下列对话和预期评估整段对话，不要假设没有出现的事实。

对话记录：
{conversation}

对话级预期：{expectations}

按整数 0–4 评分：context_retention、state_progression、consistency、correction_handling、resolution。
critical_errors 只列导致这段多轮对话不能通过的严重错误；没有则返回空数组。
只返回 JSON，例如：{{"context_retention": 4, "state_progression": 3, "consistency": 4, "correction_handling": 4, "resolution": 3, "critical_errors": [], "reason": "简短理由"}}"""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> QualityScores:
        ctx_section = f"背景信息: {context}" if context else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
        )
        prompt = self._clean_text(prompt)
        try:
            with observation(
                "llm-judge-generation",
                as_type="evaluator",
                model=self._model,
                input={"messages": [{"role": "user", "content": capture(prompt)}]},
            ) as judge_observation:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=int(os.getenv("SUPPORT_NEXUS_JUDGE_MAX_TOKENS", "4096")),
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
            raw = extract_text_content(resp.content)
            data = self._parse_json_object(raw)
            scores = QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5)),
            )
            update(
                judge_observation,
                output={"raw": capture(raw), "scores": asdict(scores)},
                usage_details=usage_details(resp),
            )
            return scores
        except Exception as ex:
            raw = locals().get("raw", "")
            resp = locals().get("resp", None)
            block_types = self._content_block_types(getattr(resp, "content", None))
            detail = (
                f"{type(ex).__name__}: {ex}; "
                f"block_types={block_types}; raw_preview={self._preview(raw)}"
            )
            logger.warning(f"LLM Judge 失败: {detail}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error=detail,
            )

    async def judge_e2e(
        self,
        question: str,
        response: str,
        expected: Dict[str, Any],
        context: Optional[str] = None,
    ) -> QualityScores:
        """用正式 rubric 评分一轮，并将 0–4 分数归一化为 API 既有的 0–1 标度。"""
        prompt = self.E2E_JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=context or "（无）",
            reference_facts=self._json_text(expected.get("reference_facts", [])),
            must_include=self._json_text(expected.get("must_include", [])),
            must_not_include=self._json_text(expected.get("must_not_include", [])),
        )
        try:
            data, raw = await self._request_json("e2e-llm-judge-generation", prompt)
            scores = QualityScores(
                relevance=self._normalize_rubric_score(data.get("relevance")),
                accuracy=self._normalize_rubric_score(data.get("accuracy")),
                completeness=self._normalize_rubric_score(data.get("completeness")),
                helpfulness=self._normalize_rubric_score(data.get("helpfulness")),
                safety=self._normalize_rubric_score(data.get("safety")),
                critical_errors=self._string_list(data.get("critical_errors")),
            )
            logger.debug("E2E Judge 完成: %s", self._preview(raw))
            return scores
        except Exception as ex:
            detail = f"{type(ex).__name__}: {ex}"
            logger.warning("E2E LLM Judge 失败: %s", detail)
            return QualityScores(
                0.0, 0.0, 0.0, 0.0,
                safety=0.0,
                judge_failed=True,
                error=detail,
            )

    async def judge_conversation(
        self,
        conversation: Sequence[Dict[str, str]],
        expectations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """评分多轮承接；返回归一化指标和失败信息，方便直接写入报告。"""
        conversation_text = "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in conversation
        )
        prompt = self.CONVERSATION_JUDGE_PROMPT.format(
            conversation=conversation_text,
            expectations=self._json_text(expectations),
        )
        names = (
            "context_retention",
            "state_progression",
            "consistency",
            "correction_handling",
            "resolution",
        )
        try:
            data, raw = await self._request_json("e2e-conversation-judge-generation", prompt)
            scores = {name: self._normalize_rubric_score(data.get(name)) for name in names}
            logger.debug("E2E 多轮 Judge 完成: %s", self._preview(raw))
            return {
                "scores": scores,
                "critical_errors": self._string_list(data.get("critical_errors")),
                "judge_failed": False,
                "error": None,
            }
        except Exception as ex:
            detail = f"{type(ex).__name__}: {ex}"
            logger.warning("E2E 多轮 Judge 失败: %s", detail)
            return {
                "scores": {name: 0.0 for name in names},
                "critical_errors": [],
                "judge_failed": True,
                "error": detail,
            }

    async def _request_json(self, observation_name: str, prompt: str) -> tuple[Dict[str, Any], str]:
        clean_prompt = self._clean_text(prompt)
        with observation(
            observation_name,
            as_type="evaluator",
            model=self._model,
            input={"messages": [{"role": "user", "content": capture(clean_prompt)}]},
        ) as judge_observation:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=int(os.getenv("SUPPORT_NEXUS_JUDGE_MAX_TOKENS", "4096")),
                temperature=0.0,
                messages=[{"role": "user", "content": clean_prompt}],
            )
        raw = extract_text_content(resp.content)
        data = self._parse_json_object(raw)
        update(
            judge_observation,
            output={"raw": capture(raw), "scores": capture(data)},
            usage_details=usage_details(resp),
        )
        return data, raw

    @staticmethod
    def _normalize_rubric_score(value: Any) -> float:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            raw = 0.0
        return round(max(0.0, min(4.0, raw)) / 4.0, 4)

    @staticmethod
    def _string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _json_text(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @staticmethod
    def _parse_json_object(raw: str) -> Dict[str, Any]:
        """从 Judge 输出中解析第一个 JSON 对象，并给出可诊断错误。"""
        if not raw or not raw.strip():
            raise ValueError("empty_judge_text")

        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no_json_object_found")

        return json.loads(raw[start:end + 1])

    @staticmethod
    def _content_block_types(content: Any) -> List[str]:
        if not content:
            return []
        types = []
        for block in content:
            if isinstance(block, dict):
                types.append(str(block.get("type", type(block).__name__)))
            else:
                types.append(str(getattr(block, "type", type(block).__name__)))
        return types

    @classmethod
    def _preview(cls, value: Any, limit: int = 300) -> str:
        text = cls._clean_text(value)
        if len(text) > limit:
            return text[:limit] + "...[truncated]"
        return text


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            result = await self._recognizer.recognize(case.message)
            predicted = result.intent.value
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 每类 F1
        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._orchestrator     = orchestrator
        self._judge            = LLMJudge(client, model)
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": []
        }

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={"accuracy": intent_metrics["accuracy"], "macro_f1": intent_metrics["macro_f1"]},
                detail=f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用 LLM Judge 评分）
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])

        # 3. 汇总
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 4. 回归检测
        regressions = self._detect_regressions(avg_scores, suite_id="legacy")

        # 5. 优化建议
        recommendations = self._recommendations(avg_scores, intent_metrics)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
            suite_id="legacy",
        )
        self._history.append(report)
        self._save_baseline(report)
        return report

    async def run_e2e_quality_suite(self, suite: E2EQualitySuite) -> EvalReport:
        """运行正式评测集，执行确定性断言、逐轮质量评估和多轮对话评估。"""
        results: List[EvalResult] = []
        score_values: Dict[str, List[float]] = {}

        def collect(result: EvalResult) -> None:
            for name, value in result.scores.items():
                if isinstance(value, (int, float)):
                    score_values.setdefault(name, []).append(float(value))

        for case in suite.cases:
            case_results = await self._evaluate_e2e_case(case, suite.rubric)
            results.extend(case_results)
            for result in case_results:
                collect(result)

        avg_scores = {
            name: round(statistics.mean(values), 4)
            for name, values in score_values.items()
            if values
        }
        passed_count = sum(1 for result in results if result.passed)
        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(passed_count / len(results), 4) if results else 0.0,
            avg_scores=avg_scores,
            regressions=self._detect_regressions(avg_scores, suite_id=suite.suite_id),
            recommendations=self._recommendations(avg_scores, {}),
            results=results,
            suite_id=suite.suite_id,
        )
        self._history.append(report)
        self._save_baseline(report)
        return report

    async def _evaluate_e2e_case(
        self,
        case: Dict[str, Any],
        rubric: Dict[str, Any],
    ) -> List[EvalResult]:
        """执行一条正式用例；多轮额外生成一条对话级结果。"""
        from agents.agent_orchestrator import Request as OrcReq

        case_id = str(case["id"])
        turns = self._dialog_turns(case)
        expectations = (
            [dict(case["expected"])]
            if case["case_type"] == "single_turn"
            else [dict(item) for item in case["turn_expectations"]]
        )
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []
        transcript: List[Dict[str, str]] = []

        for turn_idx, (question, expected) in enumerate(zip(turns, expectations)):
            context = self._history_context(history)
            orch_result = await self._orchestrator.run(OrcReq(
                message=question,
                user_id=f"eval_{case_id}",
                conv_id=f"eval_{case_id}",
                context=context,
                history=history[-6:] if history else None,
            ))
            response = str(orch_result.response)
            judgement = await self._judge.judge_e2e(question, response, expected, context=context or None)
            checks = self._deterministic_checks(orch_result, expected)
            result = self._make_e2e_turn_result(
                case=case,
                turn_idx=turn_idx,
                question=question,
                response=response,
                expected=expected,
                judgement=judgement,
                checks=checks,
                rubric=rubric,
            )
            results.append(result)
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ])
            transcript.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ])

        if case["case_type"] == "multi_turn":
            results.append(await self._make_conversation_result(case, transcript, rubric))
        return results

    def _make_e2e_turn_result(
        self,
        *,
        case: Dict[str, Any],
        turn_idx: int,
        question: str,
        response: str,
        expected: Dict[str, Any],
        judgement: QualityScores,
        checks: Dict[str, Dict[str, Any]],
        rubric: Dict[str, Any],
    ) -> EvalResult:
        policy = rubric["pass_policy"]
        scores = {
            "relevance": judgement.relevance,
            "accuracy": judgement.accuracy,
            "completeness": judgement.completeness,
            "helpfulness": judgement.helpfulness,
            "safety": judgement.safety if judgement.safety is not None else 0.0,
            "overall": judgement.overall,
            "intent_accuracy": float(checks["intent"]["passed"]),
            "primary_agent_accuracy": float(checks["primary_agent"]["passed"]),
            "supporting_agent_compliance": float(checks["supporting_agents"]["passed"]),
            "escalation_compliance": float(checks["escalation"]["passed"]),
            "tool_compliance": float(checks["tools"]["passed"]),
        }
        hard_assertions_passed = all(check["passed"] for check in checks.values())
        scores["hard_assertion_pass_rate"] = float(hard_assertions_passed)
        quality_passed = (
            judgement.overall >= float(policy["normalized_overall_min"])
            and judgement.accuracy >= float(policy["normalized_accuracy_min"])
            and scores["safety"] >= float(policy["normalized_safety_min"])
            and (not policy.get("no_critical_error", True) or not judgement.critical_errors)
            and not judgement.judge_failed
        )
        passed = hard_assertions_passed and quality_passed
        case_id = str(case["id"])
        test_id = case_id if case["case_type"] == "single_turn" else f"{case_id}_turn_{turn_idx}"
        failed_checks = [name for name, check in checks.items() if not check["passed"]]
        return EvalResult(
            test_id=test_id,
            passed=passed,
            scores=scores,
            detail=(
                f"E2E {case_id} 第 {turn_idx + 1} 轮："
                f"质量 {judgement.overall:.3f}，"
                f"确定性断言 {'通过' if hard_assertions_passed else '失败: ' + ', '.join(failed_checks)}"
            ),
            metadata={
                "case_id": case_id,
                "case_type": case["case_type"],
                "turn": turn_idx,
                "difficulty": case.get("difficulty"),
                "risk": case.get("risk"),
                "tags": case.get("tags", []),
                "question": question,
                "response": response,
                "expected": expected,
                "assertions": checks,
                "judge_failed": judgement.judge_failed,
                "judge_error": judgement.error,
                "critical_errors": judgement.critical_errors,
            },
        )

    async def _make_conversation_result(
        self,
        case: Dict[str, Any],
        transcript: List[Dict[str, str]],
        rubric: Dict[str, Any],
    ) -> EvalResult:
        judgement = await self._judge.judge_conversation(
            transcript,
            dict(case.get("conversation_expectations", {})),
        )
        scores = dict(judgement["scores"])
        scores["conversation_overall"] = statistics.mean(scores.values()) if scores else 0.0
        passed = (
            scores["conversation_overall"] >= float(rubric["pass_policy"]["normalized_overall_min"])
            and not judgement["critical_errors"]
            and not judgement["judge_failed"]
        )
        return EvalResult(
            test_id=f"{case['id']}_conversation",
            passed=passed,
            scores=scores,
            detail=f"E2E {case['id']} 多轮承接：综合评分 {scores['conversation_overall']:.3f}",
            metadata={
                "case_id": case["id"],
                "case_type": "multi_turn_conversation",
                "conversation_expectations": case.get("conversation_expectations", {}),
                "transcript": transcript,
                "judge_failed": judgement["judge_failed"],
                "judge_error": judgement["error"],
                "critical_errors": judgement["critical_errors"],
            },
        )

    @staticmethod
    def _deterministic_checks(orch_result: Any, expected: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """对 intent、路由、交接和工具调用执行可复查的硬断言。"""
        actual_intent = EndToEndEvaluator._enum_value(getattr(orch_result, "intent", None))
        expected_intents = [str(expected.get("intent", ""))]
        expected_intents.extend(str(value) for value in expected.get("allowed_intents", []))
        actual_primary = EndToEndEvaluator._enum_value(getattr(orch_result, "primary_agent", None))
        actual_supporting = {
            EndToEndEvaluator._enum_value(value)
            for value in getattr(orch_result, "supporting_agents", [])
        }
        actual_supporting.discard("")
        routing = dict(expected.get("routing", {}))
        required_supporting = {str(value) for value in routing.get("required_supporting", [])}
        allowed_supporting = {str(value) for value in routing.get("allowed_supporting", [])}
        missing_supporting = sorted(required_supporting - actual_supporting)
        extra_supporting = sorted(actual_supporting - allowed_supporting)
        tools_used = {str(value) for value in getattr(orch_result, "tools_used", [])}
        tool_rules = dict(expected.get("tools", {}))
        required_all = {str(value) for value in tool_rules.get("required_all", [])}
        required_any = {str(value) for value in tool_rules.get("required_any", [])}
        forbidden = {str(value) for value in tool_rules.get("forbidden", [])}
        response = str(getattr(orch_result, "response", ""))
        escalation_seen = EndToEndEvaluator._contains_handoff_signal(response, tools_used)
        escalation_expected = str(expected.get("escalation", "not_required"))
        if escalation_expected == "required":
            escalation_passed = escalation_seen
        elif escalation_expected == "not_required":
            escalation_passed = not escalation_seen
        else:
            # allowed 的边界解释由 rubric Judge 检查；不把升级本身当作违规。
            escalation_passed = True

        return {
            "intent": {
                "passed": actual_intent in expected_intents,
                "expected": expected_intents,
                "actual": actual_intent,
            },
            "primary_agent": {
                "passed": actual_primary == str(routing.get("primary", "")),
                "expected": routing.get("primary"),
                "actual": actual_primary,
            },
            "supporting_agents": {
                "passed": not missing_supporting and not extra_supporting,
                "required": sorted(required_supporting),
                "allowed": sorted(allowed_supporting),
                "actual": sorted(actual_supporting),
                "missing": missing_supporting,
                "unexpected": extra_supporting,
            },
            "escalation": {
                "passed": escalation_passed,
                "expected": escalation_expected,
                "actual": escalation_seen,
            },
            "tools": {
                "passed": (
                    required_all.issubset(tools_used)
                    and (not required_any or bool(required_any & tools_used))
                    and not (forbidden & tools_used)
                ),
                "required_all": sorted(required_all),
                "required_any": sorted(required_any),
                "forbidden": sorted(forbidden),
                "actual": sorted(tools_used),
                "missing_all": sorted(required_all - tools_used),
                "missing_any": sorted(required_any) if required_any and not (required_any & tools_used) else [],
                "forbidden_used": sorted(forbidden & tools_used),
            },
        }

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")

    @staticmethod
    def _contains_handoff_signal(response: str, tools_used: set[str]) -> bool:
        if "create_handoff_summary" in tools_used:
            return True
        return any(marker in response for marker in ("转人工", "人工交接", "升级处理", "升级至", "交由人工"))

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            orch_req = OrcReq(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None,
            )
            orch_result = await self._orchestrator.run(orch_req)
            actual_answer = orch_result.response

            scores = await self._judge.judge(question, actual_answer, context=context or None)
            passed = scores.overall >= self.PASS_THRESHOLD

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                },
                detail=f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}",
                metadata={
                    "question": question,
                    "response": actual_answer,
                    "agent_type": orch_result.agent_type.value,
                    "intent": orch_result.intent.value if orch_result.intent else None,
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "judge_failed": scores.judge_failed,
                    "judge_error": scores.error,
                },
            ))

        return results

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float], suite_id: str) -> List[str]:
        """只在同一评测集内与上一次结果对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None or prev_report.suite_id != suite_id:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
    ) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append("意图识别准确率 < 90%：增加 Few-shot 示例，或对低 F1 的意图类别补充训练数据")
        if scores.get("relevance", 1.0) < 0.75:
            recs.append("相关性偏低：检查 Agent system_prompt，确保 Agent 聚焦于用户问题")
        if scores.get("completeness", 1.0) < 0.75:
            recs.append("完整性偏低：Agent 可能过早结束回答，考虑在 prompt 中要求提供完整解决方案")
        if scores.get("helpfulness", 1.0) < 0.75:
            recs.append("有用性偏低：回答可能过于抽象，考虑要求 Agent 提供具体操作步骤")
        if "safety" in scores and scores["safety"] < 0.90:
            recs.append("安全评分未达 90%：检查敏感信息、鉴权、支付和生产变更边界，避免承诺未执行的后台动作")
        if "hard_assertion_pass_rate" in scores and scores["hard_assertion_pass_rate"] < 1.0:
            recs.append("确定性断言未全部通过：优先修复意图、路由、人工交接或工具调用规则，再优化回复措辞")
        if "conversation_overall" in scores and scores["conversation_overall"] < 0.80:
            recs.append("多轮承接偏弱：减少重复追问，复用已提供的标识和上下文，并随用户新信息推进处理")
        if not recs:
            recs.append("所有指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
            suite_id=str(data.get("suite_id", "legacy")),
        )


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    IntentTestCase("调用 /v1/chat 接口一直返回 401", "api_integration"),
    IntentTestCase("Webhook 签名校验失败，request_id 是 req_9ab32c", "api_integration"),
    IntentTestCase("企业版支持审计日志和 SSO 吗？", "enterprise_mgmt"),
    IntentTestCase("团队成员权限怎么分配？", "enterprise_mgmt"),
    IntentTestCase("我是新用户，怎么完成首次接入？", "onboarding"),
    IntentTestCase("如何创建 workspace 并邀请成员？", "onboarding"),
    IntentTestCase("专业版和企业版套餐差异是什么？", "subscription"),
    IntentTestCase("怎么取消自动续费并导出发票？", "subscription"),
    IntentTestCase("是否支持自定义 Agent 数量？", "feature_inquiry"),
    IntentTestCase("控制台登录一直报 401", "technical_login"),
    IntentTestCase("我要投诉，这个问题一直没有解决。", "complaint"),
    IntentTestCase("你好", "greeting"),
]

DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = [
    {"question": "调用 /v1/chat 接口一直返回 401，应该怎么排查？"},
    {"question": "我们团队想配置 SSO 和审计日志，需要什么权限？"},
    {"question": "为什么这个月订阅账单比上个月多了 50 块钱？"},
    {"question": "我是新用户，想从创建 workspace 到上线 API 接入走一遍"},
    {"turns": ["你好，我想取消自动续费", "workspace_id 是 ws_prod_1234", "还能导出这个月的发票吗？"]},
]
