"""
title: Ask Jev
author: Aryan Ebrahimpour
description: Ask TypeSafe's Jev "System One" model typed questions - yes/no, choice, or score - about text, the current conversation, or attached files. Jev returns calibrated, typed JSON far faster and cheaper than an LLM can write the same answer, so it is the fast path for structured decisions and bulk labelling.
version: 0.1.0
licence: MIT
required_open_webui_version: 0.4.0
"""

import asyncio
import json
import random
import re
from typing import Any, Optional

import aiohttp
from pydantic import BaseModel, Field

NOUL_THRESHOLD = 0.5
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
VALID_TYPES = ("noul", "choice", "score")

# Why route a decision to Jev at all, when the chat model could answer it directly?
# Speed and cost. Jev is a "System One" classifier: it emits a typed JSON answer in one
# shot instead of generating text token by token, and TypeSafe reports up to 200x faster
# inference and 400x lower cost than an LLM asked for the same structured answer. So these
# tools are the fast path whenever a decision needs to be *structured* rather than
# *explained* - routing, triage, scoring, gating - and especially bulk labelling, where the
# per-item latency gap compounds over every row. The trade-off is that Jev cannot generate
# text: it only ever answers the typed questions defined below.


class JevError(Exception):
    """Raised when the Jev API is unreachable or rejects a request."""


class Tools:
    class Valves(BaseModel):
        JEV_API_KEY: str = Field(
            default="",
            description="TypeSafe API key. Sent as the Bearer token on every Jev request.",
        )
        JEV_BASE_URL: str = Field(
            default="https://api.typesafe.ai",
            description="Base URL of the Jev API. Use https://openrouter.ai to go through OpenRouter, or change it for a proxy or self-hosted gateway. Set JEV_ENDPOINT_PATH to match.",
        )
        JEV_ENDPOINT_PATH: str = Field(
            default="/v1/systemone",
            description="Path appended to JEV_BASE_URL. TypeSafe native: /v1/systemone. OpenRouter Decisions API: /api/alpha/decisions.",
        )
        JEV_MODEL: str = Field(
            default="jev-latest",
            description="Jev model identifier. TypeSafe native: 'jev-latest' or a pinned version like 'jev-1.13.0'. Via OpenRouter: 'typesafe/jev-1.13'.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=30,
            description="Per-request timeout in seconds.",
        )
        MAX_RETRIES: int = Field(
            default=3,
            description="Retries after a 429 (rate limit), 529 (overload), 5xx, timeout, or network error.",
        )
        BACKOFF_FACTOR: float = Field(
            default=0.75,
            description="Base seconds for exponential backoff: factor * 2^attempt, plus jitter. A Retry-After header wins when present.",
        )
        MAX_STATE_CHARS: int = Field(
            default=24000,
            description="Hard cap on the characters sent as 'state'. Longer content is truncated.",
        )
        CONTEXT_MESSAGES: int = Field(
            default=12,
            description="How many trailing chat messages to use when the model does not supply a state.",
        )
        BATCH_CONCURRENCY: int = Field(
            default=8,
            description="Parallel Jev requests during batch classification.",
        )
        BATCH_MAX_ITEMS: int = Field(
            default=200,
            description="Maximum items accepted by a single batch call.",
        )
        RAW_JSON_OUTPUT: bool = Field(
            default=False,
            description="Return the raw Jev JSON response instead of a formatted summary.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # Tools exposed to the model
    # ------------------------------------------------------------------

    async def jev_yes_no(
        self,
        question: str,
        state: str = "",
        __event_emitter__=None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Ask Jev a calibrated yes/no question about some content, and get back the
        probability that the statement is true (0 = definitely no, 1 = definitely yes).

        Jev returns this typed JSON verdict far faster and cheaper than an LLM can write
        the same answer, so use it instead of judging yourself whenever the user wants a
        defensible, calibrated probability rather than an opinion, or whenever a structured
        yes/no is needed quickly.

        :param question: The yes/no question to evaluate, e.g. "Does this message convey urgency?".
        :param state: The content to evaluate. Leave empty to use attached files, or failing that the recent conversation.
        :return: The probability and verdict.
        """
        try:
            resolved = self._resolve_state(state, __messages__, __files__)
            payload = self._payload(
                resolved, {"answer": {"type": "noul", "instructions": question}}
            )
            await self._emit(__event_emitter__, "Asking Jev...", False)
            data = await self._request(payload, __event_emitter__)
        except JevError as exc:
            await self._emit(__event_emitter__, "Jev request failed", True)
            return f"Jev error: {exc}"

        await self._emit(__event_emitter__, "Jev answered", True)
        return self._render(data, {"answer": question})

    async def jev_classify(
        self,
        question: str,
        options: dict,
        state: str = "",
        __event_emitter__=None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Ask Jev to pick exactly one option from a closed set, with a probability for
        every option and an overall confidence. Good for routing, triage, intent and
        category labelling.

        Jev answers in typed JSON in a fraction of the time an LLM needs to produce the
        same classification, so prefer it for any latency-sensitive routing or triage
        decision rather than deciding yourself.

        :param question: What is being decided, e.g. "Which team should handle this ticket?".
        :param options: Mapping of option name to a short description of what that option means, e.g. {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations"}. 2 to 255 options.
        :param state: The content to evaluate. Leave empty to use attached files, or failing that the recent conversation.
        :return: The chosen option, its confidence, and the full probability distribution.
        """
        try:
            resolved = self._resolve_state(state, __messages__, __files__)
            criteria = self._coerce_options(options)
            payload = self._payload(
                resolved,
                {
                    "answer": {
                        "type": "choice",
                        "instructions": question,
                        "criteria": criteria,
                    }
                },
            )
            await self._emit(__event_emitter__, "Asking Jev...", False)
            data = await self._request(payload, __event_emitter__)
        except JevError as exc:
            await self._emit(__event_emitter__, "Jev request failed", True)
            return f"Jev error: {exc}"

        await self._emit(__event_emitter__, "Jev answered", True)
        return self._render(data, {"answer": question})

    async def jev_score(
        self,
        question: str,
        levels: list,
        state: str = "",
        __event_emitter__=None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Ask Jev to place content on an ordered scale you define, returning a
        probability-weighted score between 0 and len(levels)-1. Good for severity,
        sentiment intensity, risk and rubric grading.

        Scoring this way is both quicker and more consistent than reasoning it out: Jev
        returns the typed score directly, and applies the same calibrated scale every
        time it is asked.

        :param question: What is being rated, e.g. "How frustrated is the customer?".
        :param levels: Ordered list of level labels from lowest to highest, e.g. ["Calm", "Frustrated", "Very angry"]. Between 2 and 10 levels.
        :param state: The content to evaluate. Leave empty to use attached files, or failing that the recent conversation.
        :return: The weighted score, the closest level, confidence, and the distribution.
        """
        try:
            resolved = self._resolve_state(state, __messages__, __files__)
            criteria = self._coerce_levels(levels)
            payload = self._payload(
                resolved,
                {
                    "answer": {
                        "type": "score",
                        "instructions": question,
                        "criteria": criteria,
                    }
                },
            )
            await self._emit(__event_emitter__, "Asking Jev...", False)
            data = await self._request(payload, __event_emitter__)
        except JevError as exc:
            await self._emit(__event_emitter__, "Jev request failed", True)
            return f"Jev error: {exc}"

        await self._emit(__event_emitter__, "Jev answered", True)
        return self._render(data, {"answer": question})

    async def ask_jev(
        self,
        questions: dict,
        state: str = "",
        __event_emitter__=None,
        __messages__: Optional[list] = None,
        __files__: Optional[list] = None,
    ) -> str:
        """
        Ask Jev several typed questions about the same content in a single request.
        Jev evaluates them in parallel with barely any added latency, so this is far
        faster and cheaper than calling the other tools one at a time - and faster still
        than working the answers out yourself. Use it whenever more than one question is
        being asked about the same content.

        Each entry in `questions` is keyed by a name you choose, and its value is an
        object with a "type" of "noul", "choice" or "score", an "instructions" string,
        and for choice a "criteria" object of option to description, or for score a
        "criteria" array of 2 to 10 ordered level labels.

        Example: {"urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {"type": "choice", "instructions": "Who should handle it?",
        "criteria": {"billing": "Payments and refunds", "tech": "Bugs and outages"}}}

        :param questions: Map of question name to a typed Jev question object.
        :param state: The content to evaluate. Leave empty to use attached files, or failing that the recent conversation.
        :return: One formatted answer per question.
        """
        try:
            resolved = self._resolve_state(state, __messages__, __files__)
            spec = self._coerce_questions(questions)
            payload = self._payload(resolved, spec)
            await self._emit(
                __event_emitter__, f"Asking Jev {len(spec)} question(s)...", False
            )
            data = await self._request(payload, __event_emitter__)
        except JevError as exc:
            await self._emit(__event_emitter__, "Jev request failed", True)
            return f"Jev error: {exc}"

        await self._emit(__event_emitter__, "Jev answered", True)
        labels = {k: str(v.get("instructions", k)) for k, v in spec.items()}
        return self._render(data, labels)

    async def jev_classify_batch(
        self,
        question: str,
        options: dict,
        items: list,
        __event_emitter__=None,
    ) -> str:
        """
        Classify many items with the same Jev question, running the calls in parallel.
        Use this for lists, pasted rows, or any "label each of these" request.

        This is the biggest speedup the tool offers: Jev returns each typed label almost
        immediately and the calls are issued concurrently, where writing out the same
        labels token by token would take orders of magnitude longer and drift partway
        down the list. Always prefer it over labelling the items yourself.

        :param question: What is being decided for every item, e.g. "Which team should handle this?".
        :param options: Mapping of option name to a short description. 2 to 255 options.
        :param items: The list of text items to classify, one per row.
        :return: A table of per-item results plus a tally of each label.
        """
        try:
            criteria = self._coerce_options(options)
            texts = self._coerce_items(items)
        except JevError as exc:
            return f"Jev error: {exc}"

        question_spec = {
            "answer": {"type": "choice", "instructions": question, "criteria": criteria}
        }
        semaphore = asyncio.Semaphore(max(1, self.valves.BATCH_CONCURRENCY))
        done = 0
        total = len(texts)

        await self._emit(__event_emitter__, f"Classifying {total} items...", False)

        async def classify(session, index: int, text: str):
            nonlocal done
            async with semaphore:
                try:
                    payload = self._payload(self._truncate(text), question_spec)
                    data = await self._request(payload, None, session=session)
                    result: dict[str, Any] = {"index": index, "text": text, "data": data}
                except JevError as exc:
                    result = {"index": index, "text": text, "error": str(exc)}
            done += 1
            if done % 10 == 0 or done == total:
                await self._emit(
                    __event_emitter__, f"Classified {done}/{total}...", False
                )
            return result

        timeout = aiohttp.ClientTimeout(total=max(1, self.valves.REQUEST_TIMEOUT))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results = await asyncio.gather(
                *(classify(session, i, t) for i, t in enumerate(texts, start=1))
            )

        await self._emit(__event_emitter__, f"Classified {total} items", True)
        return self._render_batch(question, results, criteria)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _payload(self, state: str, questions: dict) -> dict:
        return {
            "state": state,
            "model": self.valves.JEV_MODEL or "jev-latest",
            "questions": questions,
        }

    async def _request(
        self,
        payload: dict,
        emitter,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> dict:
        if not self.valves.JEV_API_KEY.strip():
            raise JevError(
                "no API key configured. An administrator must set JEV_API_KEY in this tool's valves."
            )
        if session is not None:
            return await self._send(session, payload, emitter)
        timeout = aiohttp.ClientTimeout(total=max(1, self.valves.REQUEST_TIMEOUT))
        async with aiohttp.ClientSession(timeout=timeout) as owned:
            return await self._send(owned, payload, emitter)

    async def _send(
        self, session: aiohttp.ClientSession, payload: dict, emitter
    ) -> dict:
        path = (self.valves.JEV_ENDPOINT_PATH or "/v1/systemone").lstrip("/")
        url = f"{self.valves.JEV_BASE_URL.rstrip('/')}/{path}"
        headers = {
            "Authorization": f"Bearer {self.valves.JEV_API_KEY.strip()}",
            "Content-Type": "application/json",
        }
        attempts = max(0, self.valves.MAX_RETRIES) + 1
        last = "unknown error"

        for attempt in range(attempts):
            retry_after: Optional[float] = None
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        try:
                            return json.loads(body)
                        except json.JSONDecodeError as exc:
                            raise JevError(
                                f"Jev returned a non-JSON body: {self._short(body)}"
                            ) from exc
                    if resp.status in (401, 403):
                        raise JevError(
                            f"HTTP {resp.status} - the API key was rejected. Check JEV_API_KEY in the tool valves."
                        )
                    if resp.status == 422:
                        raise JevError(
                            f"HTTP 422 - Jev rejected the request body: {self._short(body)}"
                        )
                    if resp.status in (429, 529) or resp.status >= 500:
                        last = f"HTTP {resp.status}: {self._short(body)}"
                        retry_after = self._retry_after(resp.headers.get("Retry-After"))
                    else:
                        raise JevError(f"HTTP {resp.status}: {self._short(body)}")
            except asyncio.TimeoutError:
                last = f"timed out after {self.valves.REQUEST_TIMEOUT}s"
            except aiohttp.ClientError as exc:
                last = f"network error: {exc}"

            if attempt == attempts - 1:
                break
            delay = retry_after
            if delay is None:
                delay = self.valves.BACKOFF_FACTOR * (2**attempt)
            delay = max(0.1, delay * (1 + random.random() * 0.2))
            await self._emit(emitter, f"Jev unavailable ({last}), retrying...", False)
            await asyncio.sleep(delay)

        raise JevError(f"request failed after {attempts} attempt(s) - {last}")

    @staticmethod
    def _retry_after(header: Optional[str]) -> Optional[float]:
        if not header:
            return None
        try:
            return min(60.0, max(0.0, float(header.strip())))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _short(text: str, limit: int = 300) -> str:
        collapsed = re.sub(r"\s+", " ", (text or "")).strip()
        if not collapsed:
            return "(empty)"
        return collapsed[:limit] + ("..." if len(collapsed) > limit else "")

    # ------------------------------------------------------------------
    # Input coercion
    # ------------------------------------------------------------------

    def _resolve_state(
        self, state: str, messages: Optional[list], files: Optional[list]
    ) -> str:
        explicit = (state or "").strip()
        if explicit:
            return self._truncate(explicit)

        chunks = []
        for entry in files or []:
            name, content = self._file_text(entry)
            if content:
                chunks.append(f"# {name}\n{content}")
        if chunks:
            return self._truncate("\n\n".join(chunks))

        window = max(1, self.valves.CONTEXT_MESSAGES)
        lines = []
        for message in (messages or [])[-window:]:
            if not isinstance(message, dict):
                continue
            text = self._message_text(message.get("content"))
            if text:
                lines.append(f"{message.get('role', 'user')}: {text}")
        if lines:
            return self._truncate("\n".join(lines), keep_tail=True)

        raise JevError(
            "nothing to evaluate. Pass the text as 'state', attach a file, or ask about a conversation that has messages."
        )

    @staticmethod
    def _file_text(entry: Any) -> tuple:
        if not isinstance(entry, dict):
            return ("file", "")
        inner = entry.get("file") if isinstance(entry.get("file"), dict) else entry
        name = str(
            inner.get("filename") or inner.get("name") or entry.get("name") or "file"
        )
        data = inner.get("data") if isinstance(inner.get("data"), dict) else {}
        content = data.get("content") or inner.get("content") or ""
        return (name, content if isinstance(content, str) else "")

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(p for p in parts if p).strip()
        return ""

    def _truncate(self, text: str, keep_tail: bool = False) -> str:
        limit = max(200, self.valves.MAX_STATE_CHARS)
        if len(text) <= limit:
            return text
        if keep_tail:
            return "[...truncated...]\n" + text[-limit:]
        return text[:limit] + "\n[...truncated...]"

    @staticmethod
    def _as_obj(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def _coerce_options(self, options: Any) -> dict:
        parsed = self._as_obj(options)
        if isinstance(parsed, list):
            parsed = {str(item): str(item) for item in parsed}
        if not isinstance(parsed, dict) or not parsed:
            raise JevError(
                "'options' must be an object mapping each option name to a short description."
            )
        criteria = {str(k): str(v) for k, v in parsed.items()}
        if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise JevError(
                f"a choice question needs between 2 and {MAX_CHOICE_OPTIONS} options, got {len(criteria)}."
            )
        return criteria

    def _coerce_levels(self, levels: Any) -> list:
        parsed = self._as_obj(levels)
        if isinstance(parsed, dict):
            parsed = list(parsed.values())
        if not isinstance(parsed, list):
            raise JevError(
                "'levels' must be an ordered list of level labels, lowest first."
            )
        labels = [str(item) for item in parsed if str(item).strip()]
        if not MIN_SCORE_LEVELS <= len(labels) <= MAX_SCORE_LEVELS:
            raise JevError(
                f"a score question needs between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels, got {len(labels)}."
            )
        return labels

    def _coerce_questions(self, questions: Any) -> dict:
        parsed = self._as_obj(questions)
        if not isinstance(parsed, dict) or not parsed:
            raise JevError(
                "'questions' must be an object mapping a question name to a typed Jev question."
            )
        spec = {}
        for key, raw in parsed.items():
            question = self._as_obj(raw)
            if not isinstance(question, dict):
                raise JevError(f"question '{key}' must be an object.")
            qtype = str(question.get("type", "")).lower()
            if qtype not in VALID_TYPES:
                raise JevError(
                    f"question '{key}' has type '{qtype or 'missing'}', expected one of {', '.join(VALID_TYPES)}."
                )
            instructions = question.get("instructions")
            if not instructions:
                raise JevError(f"question '{key}' is missing 'instructions'.")
            built: dict[str, Any] = {"type": qtype, "instructions": instructions}
            criteria = question.get("criteria")
            if qtype == "choice":
                built["criteria"] = self._coerce_options(criteria)
            elif qtype == "score":
                built["criteria"] = self._coerce_levels(criteria)
            elif criteria is not None:
                built["criteria"] = self._as_obj(criteria)
            spec[str(key)] = built
        return spec

    def _coerce_items(self, items: Any) -> list:
        parsed = self._as_obj(items)
        if isinstance(parsed, str):
            parsed = [line for line in parsed.splitlines() if line.strip()]
        if not isinstance(parsed, list) or not parsed:
            raise JevError("'items' must be a non-empty list of strings to classify.")
        texts = [str(item).strip() for item in parsed if str(item).strip()]
        if not texts:
            raise JevError("'items' contained no non-empty entries.")
        cap = max(1, self.valves.BATCH_MAX_ITEMS)
        if len(texts) > cap:
            raise JevError(
                f"{len(texts)} items exceeds the batch limit of {cap}. Split the list or raise BATCH_MAX_ITEMS."
            )
        return texts

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    async def _emit(self, emitter, description: str, done: bool) -> None:
        if not emitter:
            return
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done, "hidden": done},
                }
            )
        except Exception:
            pass

    def _render(self, data: dict, labels: dict) -> str:
        if self.valves.RAW_JSON_OUTPUT:
            return json.dumps(data, indent=2, ensure_ascii=False)

        answers = data.get("answers") or {}
        if not answers:
            return (
                "Jev returned no answers. Raw response: "
                f"{self._short(json.dumps(data))}"
            )

        lines = []
        for key, answer in answers.items():
            lines.append(self._render_answer(key, labels.get(key, key), answer))
        lines.append("")
        lines.append(self._footer(data))
        return "\n".join(lines)

    def _render_answer(self, key: str, label: str, answer: Any) -> str:
        if not isinstance(answer, dict):
            return f"**{label}** -> {answer}"
        atype = answer.get("type")

        if atype == "noul":
            p = self._num(answer.get("noul"))
            verdict = "yes" if p is not None and p >= NOUL_THRESHOLD else "no"
            shown = f"{p:.2f}" if p is not None else "n/a"
            return f"**{label}** -> **{verdict}** (probability {shown})"

        if atype == "choice":
            choice = answer.get("choice", "n/a")
            out = f"**{label}** -> **{choice}**{self._confidence(answer)}"
            dist = self._distribution(answer.get("probabilities"))
            return out + (f"\n  {dist}" if dist else "")

        if atype == "score":
            score = self._num(answer.get("score"))
            legend = answer.get("legend") if isinstance(answer.get("legend"), dict) else {}
            level = legend.get(str(round(score))) if score is not None else None
            maximum = len(legend) - 1 if legend else None
            shown = f"{score:.2f}" if score is not None else "n/a"
            scale = f"/{maximum}" if maximum is not None else ""
            level_text = f' - closest level: "{level}"' if level else ""
            out = (
                f"**{label}** -> **{shown}{scale}**{level_text}"
                f"{self._confidence(answer)}"
            )
            dist = self._distribution(answer.get("probabilities"), legend)
            return out + (f"\n  {dist}" if dist else "")

        return f"**{label}** ({key}) -> {json.dumps(answer, ensure_ascii=False)}"

    def _render_batch(self, question: str, results: list, criteria: dict) -> str:
        if self.valves.RAW_JSON_OUTPUT:
            return json.dumps(results, indent=2, ensure_ascii=False)

        rows = ["| # | Item | Answer | Confidence |", "|---|---|---|---|"]
        tally = {name: 0 for name in criteria}
        failures = []
        input_tokens = 0
        output_tokens = 0
        total_cost = 0.0

        for result in sorted(results, key=lambda r: r["index"]):
            preview = self._cell(result["text"])
            if "error" in result:
                failures.append(f"{result['index']}: {result['error']}")
                rows.append(f"| {result['index']} | {preview} | error | - |")
                continue
            data = result["data"]
            usage = data.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            total_cost += self._num(usage.get("cost")) or 0.0
            answer = (data.get("answers") or {}).get("answer") or {}
            choice = str(answer.get("choice", "n/a"))
            tally[choice] = tally.get(choice, 0) + 1
            conf = self._num(answer.get("confidence"))
            conf_text = f"{conf:.2f}" if conf is not None else "-"
            rows.append(f"| {result['index']} | {preview} | {choice} | {conf_text} |")

        summary = ", ".join(f"{name}: {count}" for name, count in tally.items() if count)
        out = [f"**{question}** - {len(results)} item(s) classified by Jev", ""]
        out.extend(rows)
        out.append("")
        out.append(f"Tally: {summary or 'none'}")
        if failures:
            out.append(f"Failed: {len(failures)} - " + "; ".join(failures[:5]))
        footer = (
            f"Model: {self.valves.JEV_MODEL} | "
            f"tokens in/out: {input_tokens}/{output_tokens}"
        )
        if total_cost:
            footer += f" | cost: ${total_cost:.6f}"
        out.append(footer)
        return "\n".join(out)

    @staticmethod
    def _cell(text: str, limit: int = 70) -> str:
        flat = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
        return flat[:limit] + ("..." if len(flat) > limit else "")

    @staticmethod
    def _num(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _confidence(self, answer: dict) -> str:
        conf = self._num(answer.get("confidence"))
        return f" (confidence {conf:.2f})" if conf is not None else ""

    def _distribution(self, probabilities: Any, legend: Any = None) -> str:
        if not isinstance(probabilities, dict) or not probabilities:
            return ""
        parts = []
        for key, value in sorted(
            probabilities.items(), key=lambda kv: -(self._num(kv[1]) or 0)
        ):
            p = self._num(value)
            if p is None or p < 0.005:
                continue
            name = f"{key} {legend[key]}" if isinstance(legend, dict) and key in legend else key
            parts.append(f"{name} {p * 100:.0f}%")
        return " | ".join(parts)

    def _footer(self, data: dict) -> str:
        usage = data.get("usage") or {}
        parts = [f"Jev {data.get('model', self.valves.JEV_MODEL)}"]
        if usage:
            parts.append(
                "tokens in/out: "
                f"{usage.get('input_tokens', '?')}/{usage.get('output_tokens', '?')}"
            )
            cost = self._num(usage.get("cost"))
            if cost is not None:
                parts.append(f"cost: ${cost:.6f}")
        return "_" + " | ".join(parts) + "_"
