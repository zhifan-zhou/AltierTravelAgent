"""Deterministic FakeLLM used by tests and the acceptance script."""

from __future__ import annotations

import json
import re


class FakeRequirementLLM:
    """Returns valid TravelRequirementContractUpdate JSON for acceptance scenarios."""

    def __init__(self):
        self.last_meta = {
            "model": "fake-deepseek",
            "latency_ms": 0.0,
            "token_usage": {},
        }

    async def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        message = payload["user_message"]
        current = payload.get("current_contract") or {}
        update = _with_defaults(self._update_for(message, current))
        self.last_meta = {
            "model": "fake-deepseek",
            "latency_ms": 0.0,
            "token_usage": {"prompt_tokens": len(user_prompt), "completion_tokens": len(json.dumps(update))},
        }
        return json.dumps(update, ensure_ascii=False)

    def _trace(self, step: str, evidence: str, decision: str, fields: list[str]) -> list[dict]:
        return [
            {
                "step": step,
                "evidence": evidence,
                "decision": decision,
                "affected_fields": fields,
            }
        ]

    def _update_for(self, message: str, current: dict) -> dict:
        selected = _extract_option_index(message)
        if selected is not None:
            return {
                "update_type": "explain_option",
                "field_updates": {},
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": False,
                "should_rerun_search": False,
                "should_rerank_only": False,
                "selected_option_index": selected,
                "user_facing_ack_zh": f"我来展开第 {selected} 个方案。",
                "reasoning_summary": "用户要求解释已展示方案。",
                "decision_trace": self._trace(
                    "detect_explain_option",
                    f"用户说：{message}",
                    f"选择第 {selected} 个已展示方案做详情解释。",
                    ["selected_option_index"],
                ),
                "confidence": 0.92,
            }
        if message.lower() in {"导出", "export"}:
            return {
                "update_type": "export",
                "field_updates": {},
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": False,
                "should_rerun_search": False,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "准备导出本轮对话。",
                "reasoning_summary": "用户要求导出。",
                "decision_trace": self._trace(
                    "detect_export",
                    f"用户说：{message}",
                    "导出当前对话、schema 更新和推荐结果。",
                    ["update_type"],
                ),
                "confidence": 0.96,
            }
        if message.strip() in {"你好", "您好", "hi", "hello"}:
            return _non_action(
                "smalltalk",
                "你好，我可以帮你用自然语言规划路线。比如：温州到匹兹堡，可以从上海走，越便宜越好。",
            )
        if "怎么用" in message or message.lower() in {"help", "帮助"}:
            return _non_action(
                "help",
                "可以直接描述路线、限制或排序偏好。例如：温州到匹兹堡，可以从上海走，越便宜越好；也可以追加：不要纽约转、主流航司优先、解释第1个。",
            )
        if "天气" in message:
            return _tool_update(
                tool_name="weather",
                arguments={},
                reason="用户询问天气，需要实时工具。",
                ack="你问的是天气信息。",
                requires_current_contract=True,
            )
        if "几点" in message or "当地时间" in message or "现在时间" in message:
            location = _extract_known_location(message)
            return _tool_update(
                tool_name="time",
                arguments={"location": location} if location else {},
                reason="用户询问地点当前时间，需要时区解析。",
                ack="我来查当地时间。",
                requires_current_contract=not bool(location),
            )
        if _looks_like_currency_query(message):
            amount, source, target = _currency_arguments(message)
            return _tool_update(
                tool_name="currency",
                arguments={"amount": amount, "from_currency": source, "to_currency": target},
                reason="用户询问货币换算，需要 Frankfurter 参考汇率。",
                ack="我来做货币换算。",
                requires_current_contract=False,
            )
        if "机场" in message and ("哪个" in message or "哪些" in message):
            location = "奥斯丁" if "奥斯丁" in message or "奥斯汀" in message else "成都" if "成都" in message else ""
            return _tool_update(
                tool_name="airport_lookup",
                arguments={"location": location},
                reason="用户询问机场信息，可以使用本地机场数据。",
                ack="我来查机场信息。",
                requires_current_contract=not bool(location),
            )
        if any(token in message for token in ["有什么好玩", "介绍一下", "旅行亮点", "值得注意"]):
            location = _extract_known_location(message)
            return _tool_update(
                tool_name="destination_brief",
                arguments={"location": location} if location else {},
                reason="用户询问目的地简介。",
                ack="我来查一份简短的目的地介绍。",
                requires_current_contract=not bool(location),
            )
        if ("不带" in message or "取消" in message or "算了" in message) and ("狗" in message or "宠物" in message):
            return _cancel_pet_update(message, self._trace)
        if "狗" in message or "宠物" in message:
            update = _special_update(
                message=message,
                category="pet_travel",
                description="用户想带狗同行" if "狗" in message else "用户有宠物同行需求",
                structured_values={
                    "animal": "dog" if "狗" in message else "unknown",
                    "size": "unknown",
                    "in_cabin_or_checked": "unknown",
                },
                impact_areas=["airline_policy", "baggage", "risk", "self_transfer"],
                hard_constraint=False,
                requires_clarification=True,
                clarification_question="狗狗大概是小型犬、中型犬还是大型犬？这会影响能否进客舱或需要托运。",
                ack="已记录：宠物同行。后续会更偏向少转机、低风险、主流航司，并提醒你确认宠物政策。",
                advisory="可以把宠物同行作为需求记录下来，但是否能进客舱或托运取决于航司、狗狗体型重量、航线和名额。你可以告诉我狗狗大概是小型犬、中型犬还是大型犬？我后续会更偏向少转机、主流航司、低风险的方案，并在推荐里提醒你确认宠物政策。",
                trace_step="capture_special_requirement",
            )
            update["field_updates"] = {
                "companions": {
                    "pets": [
                        {
                            "kind": "dog" if "狗" in message else "pet",
                            "count": 1,
                            "active": True,
                            "source": "user",
                        }
                    ]
                }
            }
            update["constraints_to_add"] = [
                {
                    "type": "pet_companion",
                    "category": "companions",
                    "value": "dog" if "狗" in message else "pet",
                    "priority": "high",
                    "reason": "用户明确提出宠物同行",
                    "source_user_message": message,
                    "active": True,
                }
            ]
            return update
        if "预算" in message:
            return _budget_update(message, self._trace)
        if "红眼" in message:
            return _flight_preference_update(
                message,
                field_updates={"preferences": {"avoid_red_eye": True}},
                preference={"type": "avoid_red_eye", "value": True},
                ack="已记录：避开红眼航班。",
                trace=self._trace,
            )
        if "不要转机" in message or "不想转机" in message or "直飞" in message:
            return _flight_preference_update(
                message,
                field_updates={"preferences": {"nonstop_preferred": True, "max_stops": 0}},
                preference={"type": "prefer_nonstop", "value": True},
                ack="已记录：优先直飞、尽量不转机。",
                trace=self._trace,
            )
        if "很多行李" in message or "行李" in message:
            return _special_update(
                message=message,
                category="heavy_baggage",
                description="用户有较多行李，需要降低重新托运和自助转机风险",
                structured_values={"baggage_amount": "many"},
                impact_areas=["baggage", "risk", "self_transfer", "connection_time"],
                hard_constraint=False,
                requires_clarification=False,
                clarification_question=None,
                ack="较多行李的需求已记录，我会更谨慎对待拆票和自助转机。",
                advisory="带很多行李时，拆票、自助转机或跨航司衔接会更麻烦，可能需要重新托运行李。后续我会更偏向少转机、衔接更稳的方案。",
                trace_step="capture_special_requirement",
            )
        if "签证" in message or "过境" in message:
            return _special_update(
                message=message,
                category="visa_constraint",
                description="用户担心签证或过境要求",
                structured_values={"visa_sensitive": True, "avoid_uncertain_transit": True},
                impact_areas=["routing", "documentation", "risk"],
                hard_constraint=True,
                requires_clarification=True,
                clarification_question="你是想避免某些国家/地区转机，还是只想要入境政策更简单的路线？",
                ack="签证/过境顾虑已记录。本 demo 不判断签证政策，我会把它作为高风险提醒处理。",
                advisory="签证和过境政策必须按护照、签证、转机国家和航司规则单独确认。本 demo 不判断签证可行性，我会把它作为高风险约束提醒。",
                trace_step="capture_special_requirement",
            )
        if ("成都" in message and ("奥斯丁" in message or "奥斯汀" in message)):
            has_time = _has_searchable_time_in_text(message)
            return {
                "update_type": "create_new",
                "field_updates": {
                    "trip": {
                        # Intentionally swapped so RouteSemanticValidator tests repair.
                        "origin_text": "奥斯丁",
                        "origin_airport": "AUS",
                        "destination_text": "成都",
                        "destination_airport": "TFU",
                        "trip_type": "one_way",
                    },
                    "time": _time_update(message),
                },
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": has_time,
                "should_rerun_search": has_time,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "我来处理成都和奥斯丁之间的路线。",
                "clarification_question_zh": None
                if has_time
                else "你大概什么时候出发？可以给一个日期或时间范围，比如 6月初、下周、8月20日前后。",
                "reasoning_summary": "用户提到成都和奥斯丁。",
                "decision_trace": self._trace(
                    "extract_route",
                    f"用户说：{message}",
                    "抽取成都/奥斯丁路线。",
                    ["trip.origin_airport", "trip.destination_airport"],
                ),
                "confidence": 0.8,
            }
        if "宁波" in message and "迈阿密" in message:
            has_time = _has_searchable_time_in_text(message)
            return {
                "update_type": "create_new",
                "field_updates": {
                    "trip": {
                        "origin_text": "宁波",
                        "origin_airport": "NGB",
                        "origin_city": "Ningbo",
                        "destination_text": "迈阿密",
                        "destination_airport": "MIA",
                        "destination_city": "Miami",
                        "trip_type": "one_way",
                    },
                    "time": _time_update(message),
                    "ranking": {"profile": "balanced"},
                },
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": _has_searchable_time_in_text(message),
                "should_rerun_search": _has_searchable_time_in_text(message),
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": (
                    "好的，已切换为宁波到迈阿密的新搜索。"
                    if has_time
                    else "已切换到新的搜索：宁波 NGB → 迈阿密 MIA。你大概什么时候出发？"
                ),
                "clarification_question_zh": None
                if has_time
                else "你大概什么时候出发？可以给一个日期或时间范围，比如 6月初、下周、8月20日前后。",
                "reasoning_summary": "用户明确说想看看另一组起终点。",
                "decision_trace": self._trace(
                    "detect_new_search",
                    "用户提出新的路线：宁波到迈阿密",
                    "创建新搜索，并清空上一轮温州到匹兹堡结果。",
                    ["update_type", "trip.origin_airport", "trip.destination_airport"],
                ),
                "confidence": 0.97,
            }
        if "温州" in message and "匹兹堡" in message:
            has_time = _has_searchable_time_in_text(message)
            return {
                "update_type": "create_new",
                "field_updates": {
                    "trip": {
                        "origin_text": "温州",
                        "origin_airport": "WNZ",
                        "origin_city": "Wenzhou",
                        "destination_text": "匹兹堡",
                        "destination_airport": "PIT",
                        "destination_city": "Pittsburgh",
                        "trip_type": "one_way",
                    },
                    "time": _time_update(message),
                    "geography": {"acceptable_origin_hubs": ["PVG", "SHA"]} if "上海" in message else {},
                    "hub_policy": {"nearby_hub_policy": "prefer", "allow_ground_access": True},
                    "ranking": {"profile": "cheapest", "price_priority": "high"},
                },
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [
                    {
                        "type": "prefer_low_price",
                        "value": "越便宜越好",
                        "normalized_values": [],
                        "weight_hint": "high",
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "preferences_to_remove": [],
                "should_search": has_time,
                "should_rerun_search": has_time,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": (
                    "收到，我会按温州到匹兹堡、可从上海走、价格优先来找方案。"
                    if has_time
                    else "可以。你大概什么时候出发？给我一个日期或时间范围就行，比如 6月初、下周、8月20日前后。"
                ),
                "clarification_question_zh": None
                if has_time
                else "你大概什么时候出发？可以给一个日期或时间范围，比如 6月初、下周、8月20日前后。",
                "reasoning_summary": "用户给出起终点、可接受上海出发枢纽，并强调低价。",
                "decision_trace": self._trace(
                    "extract_new_search",
                    f"用户说：{message}",
                    "建立新搜索并设置价格优先、附近枢纽优先。",
                    [
                        "trip.origin_airport",
                        "trip.destination_airport",
                        "geography.acceptable_origin_hubs",
                        "ranking.profile",
                    ],
                ),
                "confidence": 0.98,
            }
        if _has_route(current) and _has_date_hint(message):
            return _date_followup_update(message)
        if "杭州" in message:
            return {
                "update_type": "modify_existing",
                "field_updates": {"geography": {"acceptable_origin_hubs": ["HGH"]}},
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": True,
                "should_rerun_search": True,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "好的，我把杭州加入可接受出发枢纽。",
                "reasoning_summary": "用户是在现有路线下增加出发枢纽。",
                "decision_trace": self._trace(
                    "detect_modify_existing",
                    "用户说：如果从杭州走呢",
                    "保留当前起终点，只追加 HGH。",
                    ["geography.acceptable_origin_hubs"],
                ),
                "confidence": 0.94,
            }
        if "不去浦东" in message and "虹桥" in message:
            return {
                "update_type": "modify_existing",
                "field_updates": {"geography": {"acceptable_origin_hubs": ["SHA"], "avoid_airports": ["PVG"]}},
                "constraints_to_add": [
                    {
                        "type": "avoid_airport",
                        "value": "PVG",
                        "normalized_values": ["PVG"],
                        "reason": "用户明确不去浦东",
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": True,
                "should_rerun_search": True,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "好的，避开浦东，允许虹桥。",
                "reasoning_summary": "用户重新允许 SHA，排除 PVG。",
                "decision_trace": self._trace(
                    "reallow_specific_airport",
                    "用户说：不去浦东，可以去虹桥",
                    "将 SHA 从排除中移除并保留 PVG 排除。",
                    ["geography.acceptable_origin_hubs", "geography.avoid_airports"],
                ),
                "confidence": 0.94,
            }
        if "上海浦东" in message or "浦东" in message:
            return {
                "update_type": "modify_existing",
                "field_updates": {"geography": {"acceptable_origin_hubs": ["PVG"], "avoid_airports": ["SHA"]}},
                "constraints_to_add": [
                    {
                        "type": "avoid_airport",
                        "value": "SHA",
                        "normalized_values": ["SHA"],
                        "reason": "用户明确不去虹桥",
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": True,
                "should_rerun_search": True,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "好的，允许浦东，避开虹桥。",
                "reasoning_summary": "用户做了机场级别区分。",
                "decision_trace": self._trace(
                    "disambiguate_shanghai_airports",
                    f"用户说：{message}",
                    "允许浦东机场，排除虹桥机场。",
                    ["geography.acceptable_origin_hubs", "geography.avoid_airports"],
                ),
                "confidence": 0.96,
            }
        if "不想去上海" in message:
            return {
                "update_type": "add_constraint",
                "field_updates": {"geography": {"avoid_cities": ["上海"], "avoid_airports": ["PVG", "SHA"]}},
                "constraints_to_add": [
                    {
                        "type": "avoid_city",
                        "value": "上海",
                        "normalized_values": ["PVG", "SHA"],
                        "reason": "用户不想去上海",
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": True,
                "should_rerun_search": True,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "好的，我会避开上海的浦东和虹桥。",
                "reasoning_summary": "城市级排除扩展为 PVG/SHA。",
                "decision_trace": self._trace(
                    "city_level_exclusion",
                    "用户说：我不想去上海",
                    "上海的两个机场都加入排除，并从候选中移除。",
                    ["geography.avoid_cities", "geography.avoid_airports"],
                ),
                "confidence": 0.96,
            }
        if "纽约" in message:
            return {
                "update_type": "add_constraint",
                "field_updates": {"geography": {"avoid_cities": ["纽约"], "avoid_airports": ["JFK", "EWR", "LGA"]}},
                "constraints_to_add": [
                    {
                        "type": "avoid_city",
                        "value": "纽约",
                        "normalized_values": ["JFK", "EWR", "LGA"],
                        "reason": "用户不要纽约转",
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "should_search": True,
                "should_rerun_search": True,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "好的，我会避开纽约三机场。",
                "reasoning_summary": "纽约转机排除 JFK/EWR/LGA。",
                "decision_trace": self._trace(
                    "exclude_new_york_transfer",
                    "用户说：不要纽约转",
                    "纽约三机场都加入排除。",
                    ["geography.avoid_cities", "geography.avoid_airports"],
                ),
                "confidence": 0.96,
            }
        if "主流航司" in message:
            return {
                "update_type": "add_preference",
                "field_updates": {
                    "ranking": {"profile": "airline_priority", "airline_quality_priority": "high"},
                    "airline_preferences": {"prefer_major_airlines": True, "avoid_low_cost_carriers": True},
                },
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [
                    {
                        "type": "prefer_airline",
                        "value": "major_airlines",
                        "normalized_values": [],
                        "weight_hint": "high",
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "preferences_to_remove": [],
                "should_search": False,
                "should_rerun_search": False,
                "should_rerank_only": True,
                "selected_option_index": None,
                "user_facing_ack_zh": "好的，我按主流航司优先重新排序。",
                "reasoning_summary": "这是排序偏好，不需要重新请求 provider。",
                "decision_trace": self._trace(
                    "detect_rerank_only",
                    "用户说：主流航司优先",
                    "只调整排序权重，不重新搜索航班。",
                    ["ranking.profile", "airline_preferences.prefer_major_airlines"],
                ),
                "confidence": 0.95,
            }
        if "爸妈" in message or "别太折腾" in message:
            return {
                "update_type": "add_special_requirement",
                "field_updates": {
                    "passengers": {"family_or_parents": True},
                    "ranking": {"profile": "low_risk", "risk_priority": "high"},
                    "ticketing": {"split_ticket_policy": "avoid", "allow_self_transfer": False},
                },
                "constraints_to_add": [
                    {
                        "type": "no_split_ticket",
                        "value": "avoid",
                        "normalized_values": [],
                        "reason": "家人同行，降低折腾和自转机风险",
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "constraints_to_remove": [],
                "preferences_to_add": [],
                "preferences_to_remove": [],
                "special_requirements_to_add": [
                    {
                        "category": "family_or_elderly",
                        "description_zh": "用户与父母同行，需要更稳妥、少折腾的路线",
                        "structured_values": {"family_travel": True, "elderly_or_parent_travel": True},
                        "impact_areas": ["risk", "connection_time", "self_transfer", "routing"],
                        "hard_constraint": False,
                        "preference_weight": "high",
                        "requires_clarification": False,
                        "clarification_question_zh": None,
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "special_requirements_to_remove": [],
                "clarification_questions": [],
                "should_search": True,
                "should_rerun_search": True,
                "should_rerank_only": False,
                "selected_option_index": None,
                "user_facing_ack_zh": "好的，我会优先稳妥、少折腾。",
                "reasoning_summary": "家庭同行触发低风险和避免拆票。",
                "decision_trace": self._trace(
                    "family_low_risk",
                    f"用户说：{message}",
                    "设置 family_or_parents=true，并避免拆票。",
                    ["passengers.family_or_parents", "ticketing.split_ticket_policy", "ranking.profile"],
                ),
                "confidence": 0.94,
            }
        if "便宜" in message:
            return {
                "update_type": "add_preference",
                "field_updates": {
                    "ranking": {"profile": "cheapest", "price_priority": "high"},
                    "hub_policy": {"nearby_hub_policy": "prefer"},
                },
                "constraints_to_add": [],
                "constraints_to_remove": [],
                "preferences_to_add": [
                    {
                        "type": "prefer_low_price",
                        "value": "越便宜越好",
                        "normalized_values": [],
                        "weight_hint": "high",
                        "source_user_message": message,
                        "active": True,
                    }
                ],
                "preferences_to_remove": [],
                "should_search": False,
                "should_rerun_search": False,
                "should_rerank_only": True,
                "selected_option_index": None,
                "user_facing_ack_zh": "好的，按价格优先。",
                "reasoning_summary": "用户表达低价偏好。",
                "decision_trace": self._trace(
                    "detect_price_priority",
                    f"用户说：{message}",
                    "设置 cheapest/price_priority=high，不追问泛化偏好。",
                    ["ranking.profile", "ranking.price_priority"],
                ),
                "confidence": 0.92,
            }
        if message.lower() in {"quit", "q", "退出"}:
            return _non_action("quit", "好的，退出。")
        return _non_action("unknown", "我没有把这句话转换成可执行需求更新。")


def _non_action(update_type: str, ack: str) -> dict:
    return {
        "update_type": update_type,
        "field_updates": {},
        "constraints_to_add": [],
        "constraints_to_remove": [],
        "preferences_to_add": [],
        "preferences_to_remove": [],
        "special_requirements_to_add": [],
        "special_requirements_to_remove": [],
        "clarification_questions": [],
        "next_action": {
            "quit": "quit",
            "help": "help",
            "smalltalk": "smalltalk",
        }.get(update_type, "no_op"),
        "user_intent_summary_zh": "",
        "advisory_response_zh": None,
        "clarification_question_zh": None,
        "should_search": False,
        "should_rerun_search": False,
        "should_rerank_only": False,
        "selected_option_index": None,
        "user_facing_ack_zh": ack,
        "reasoning_summary": "",
        "decision_trace": [],
        "confidence": 0.5,
    }


def _with_defaults(update: dict) -> dict:
    update.setdefault("special_requirements_to_add", [])
    update.setdefault("special_requirements_to_remove", [])
    update.setdefault("clarification_questions", [])
    update.setdefault("tool_requests", [])
    update.setdefault("user_intent_summary_zh", update.get("reasoning_summary", ""))
    update.setdefault("advisory_response_zh", None)
    update.setdefault("clarification_question_zh", None)
    update.setdefault("next_action", _infer_next_action(update))
    return update


def _tool_update(
    *,
    tool_name: str,
    arguments: dict,
    reason: str,
    ack: str,
    requires_current_contract: bool,
) -> dict:
    return {
        "update_type": "advisory_question",
        "field_updates": {},
        "constraints_to_add": [],
        "constraints_to_remove": [],
        "preferences_to_add": [],
        "preferences_to_remove": [],
        "special_requirements_to_add": [],
        "special_requirements_to_remove": [],
        "clarification_questions": [],
        "tool_requests": [
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "reason_zh": reason,
                "requires_current_contract": requires_current_contract,
            }
        ],
        "next_action": "tool_query",
        "user_intent_summary_zh": reason,
        "advisory_response_zh": None,
        "clarification_question_zh": None,
        "should_search": False,
        "should_rerun_search": False,
        "should_rerank_only": False,
        "selected_option_index": None,
        "user_facing_ack_zh": ack,
        "reasoning_summary": reason,
        "decision_trace": [
            {
                "step": "detect_tool_query",
                "evidence": "用户询问非航班搜索信息",
                "decision": f"调用 {tool_name} 工具。",
                "affected_fields": ["next_action", "tool_requests"],
            }
        ],
        "confidence": 0.9,
    }


def _infer_next_action(update: dict) -> str:
    update_type = update.get("update_type")
    if update_type == "explain_option":
        return "explain_result"
    if update_type == "export":
        return "export"
    if update_type == "help":
        return "help"
    if update_type == "smalltalk":
        return "smalltalk"
    if update_type == "quit":
        return "quit"
    if update.get("clarification_question_zh"):
        return "ask_clarification"
    if update.get("should_rerank_only"):
        return "rerank"
    if update.get("should_search") or update.get("should_rerun_search"):
        return "run_search"
    return "no_op"


def _has_route(current: dict) -> bool:
    trip = current.get("trip") or {}
    return bool(trip.get("origin_airport") and trip.get("destination_airport"))


def _has_date_hint(message: str) -> bool:
    return any(token in message for token in ["六月", "6月", "七月", "7月", "下个月", "下周", "日期", "时间", "随便看看", "灵活"])


def _has_flexible_hint(message: str) -> bool:
    return any(token in message for token in ["日期还没定", "先随便看看", "大概看看", "用默认日期", "灵活日期", "先按灵活"])


def _has_searchable_time_in_text(message: str) -> bool:
    return _has_flexible_hint(message) or any(
        token in message
        for token in ["六月", "6月", "七月", "7月", "下个月", "下周", "明天", "后天", "日前后"]
    )


def _time_update(message: str) -> dict:
    if _has_flexible_hint(message):
        return {"departure_window_text": "灵活日期", "flexible": True, "flexible_date_confirmed": True}
    for token in ["六月初", "6月初", "七月初", "7月初", "下个月", "下周"]:
        if token in message:
            return {"departure_window_text": token, "departure_text": token, "flexible": True}
    return {}


def _date_followup_update(message: str) -> dict:
    time = _time_update(message) or {"departure_window_text": message.strip(), "departure_text": message.strip()}
    field_updates = {"time": time}
    preferences = []
    if "上海" in message:
        field_updates["geography"] = {"acceptable_origin_hubs": ["PVG", "SHA"]}
    if "便宜" in message:
        field_updates["ranking"] = {"profile": "cheapest", "price_priority": "high"}
        field_updates["hub_policy"] = {"nearby_hub_policy": "prefer"}
        preferences.append(
            {
                "type": "prefer_low_price",
                "value": "越便宜越好",
                "normalized_values": [],
                "weight_hint": "high",
                "source_user_message": message,
                "active": True,
            }
        )
    return {
        "update_type": "clarification_answer",
        "field_updates": field_updates,
        "constraints_to_add": [],
        "constraints_to_remove": [],
        "preferences_to_add": preferences,
        "preferences_to_remove": [],
        "special_requirements_to_add": [],
        "special_requirements_to_remove": [],
        "clarification_questions": [],
        "next_action": "run_search",
        "user_intent_summary_zh": "用户补充了出发时间窗口，当前搜索条件完整。",
        "advisory_response_zh": None,
        "clarification_question_zh": None,
        "should_search": True,
        "should_rerun_search": True,
        "should_rerank_only": False,
        "selected_option_index": None,
        "user_facing_ack_zh": "好的，已补充出发时间，我来搜索方案。",
        "reasoning_summary": "用户补充了缺失的时间窗口。",
        "decision_trace": [
            {
                "step": "answer_missing_departure_window",
                "evidence": f"用户说：{message}",
                "decision": "将用户回复作为出发时间窗口，并开始搜索。",
                "affected_fields": ["time.departure_window_text", "next_action"],
            }
        ],
        "confidence": 0.92,
    }


def _special_update(
    *,
    message: str,
    category: str,
    description: str,
    structured_values: dict,
    impact_areas: list[str],
    hard_constraint: bool,
    requires_clarification: bool,
    clarification_question: str | None,
    ack: str,
    advisory: str,
    trace_step: str,
) -> dict:
    return {
        "update_type": "advisory_question",
        "field_updates": {},
        "constraints_to_add": [],
        "constraints_to_remove": [],
        "preferences_to_add": [],
        "preferences_to_remove": [],
        "special_requirements_to_add": [
            {
                "category": category,
                "description_zh": description,
                "structured_values": structured_values,
                "impact_areas": impact_areas,
                "hard_constraint": hard_constraint,
                "preference_weight": "high",
                "requires_clarification": requires_clarification,
                "clarification_question_zh": clarification_question,
                "source_user_message": message,
                "active": True,
            }
        ],
        "special_requirements_to_remove": [],
        "clarification_questions": [clarification_question] if clarification_question else [],
        "next_action": "answer_advisory",
        "user_intent_summary_zh": "用户提出旅行政策/可行性问题，并补充了特殊需求。",
        "advisory_response_zh": advisory,
        "clarification_question_zh": clarification_question,
        "should_search": False,
        "should_rerun_search": False,
        "should_rerank_only": False,
        "selected_option_index": None,
        "user_facing_ack_zh": ack,
        "reasoning_summary": "用户补充了生活/特殊出行需求，映射为通用 special_requirements。",
        "decision_trace": [
            {
                "step": trace_step,
                "evidence": f"用户说：{message}",
                "decision": f"记录为 {category}，由 deterministic pipeline 根据 impact_areas 调整风险和提醒。",
                "affected_fields": ["special_requirements", "ranking.risk_priority", "ticketing.allow_self_transfer"],
            }
        ],
        "confidence": 0.9,
    }


def _cancel_pet_update(message: str, trace) -> dict:
    return {
        "update_type": "remove_special_requirement",
        "field_updates": {
            "companions": {
                "pets": [
                    {"kind": "dog" if "狗" in message else "pet", "count": 1, "active": False, "source": "user"}
                ]
            }
        },
        "constraints_to_add": [],
        "constraints_to_remove": ["pet_companion", "dog", "pet"],
        "preferences_to_add": [],
        "preferences_to_remove": [],
        "special_requirements_to_add": [],
        "special_requirements_to_remove": ["pet_travel", "dog", "pet"],
        "clarification_questions": [],
        "next_action": "answer_advisory",
        "user_intent_summary_zh": "用户取消宠物同行约束。",
        "advisory_response_zh": None,
        "clarification_question_zh": None,
        "should_search": False,
        "should_rerun_search": False,
        "should_rerank_only": False,
        "selected_option_index": None,
        "user_facing_ack_zh": "好的，已取消宠物同行需求。",
        "reasoning_summary": "将已有宠物相关 constraint 与 special requirement 标记为 inactive。",
        "decision_trace": trace(
            "cancel_companion_constraint",
            f"用户说：{message}",
            "停用匹配的宠物同行约束，保留历史记录。",
            ["companions.pets", "constraints.hard_constraints", "special_requirements"],
        ),
        "confidence": 0.96,
    }


def _budget_update(message: str, trace) -> dict:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(美元|美金|人民币|元|欧元|英镑|日元|USD|CNY|EUR|GBP|JPY)?", message, re.I)
    amount = float(match.group(1)) if match else None
    currency_aliases = {"美元": "USD", "美金": "USD", "人民币": "CNY", "元": "CNY", "欧元": "EUR", "英镑": "GBP", "日元": "JPY"}
    currency = currency_aliases.get(match.group(2), (match.group(2) or "").upper()) if match else None
    budget_update = {"priority": "high", "preference": "lower"}
    if amount is not None:
        budget_update["amount"] = amount
    if currency:
        budget_update["currency"] = currency
    return {
        "update_type": "add_preference",
        "field_updates": {
            "budget": budget_update,
            "ranking": {"profile": "cheapest", "price_priority": "high"},
        },
        "constraints_to_add": [],
        "constraints_to_remove": [],
        "preferences_to_add": [
            {
                "type": "prefer_low_price",
                "value": "lower_budget",
                "weight_hint": "high",
                "source_user_message": message,
                "active": True,
            }
        ],
        "preferences_to_remove": [],
        "special_requirements_to_add": [],
        "special_requirements_to_remove": [],
        "clarification_questions": [],
        "next_action": "rerank",
        "user_intent_summary_zh": "用户补充预算或低价偏好。",
        "advisory_response_zh": None,
        "clarification_question_zh": None,
        "should_search": False,
        "should_rerun_search": False,
        "should_rerank_only": True,
        "selected_option_index": None,
        "user_facing_ack_zh": "已记录预算与价格优先偏好。",
        "reasoning_summary": "预算信息映射到通用 budget 与低价 preference。",
        "decision_trace": trace(
            "capture_budget",
            f"用户说：{message}",
            "更新预算字段并提高价格优先级。",
            ["budget", "ranking.price_priority"],
        ),
        "confidence": 0.92,
    }


def _flight_preference_update(message: str, *, field_updates: dict, preference: dict, ack: str, trace) -> dict:
    preference = {
        **preference,
        "weight_hint": "high",
        "source_user_message": message,
        "active": True,
    }
    return {
        "update_type": "add_preference",
        "field_updates": field_updates,
        "constraints_to_add": [],
        "constraints_to_remove": [],
        "preferences_to_add": [preference],
        "preferences_to_remove": [],
        "special_requirements_to_add": [],
        "special_requirements_to_remove": [],
        "clarification_questions": [],
        "next_action": "rerank",
        "user_intent_summary_zh": "用户补充航班偏好。",
        "advisory_response_zh": None,
        "clarification_question_zh": None,
        "should_search": False,
        "should_rerun_search": False,
        "should_rerank_only": True,
        "selected_option_index": None,
        "user_facing_ack_zh": ack,
        "reasoning_summary": "航班偏好映射到通用 preferences。",
        "decision_trace": trace(
            "capture_flight_preference",
            f"用户说：{message}",
            "更新航班偏好字段。",
            list(field_updates.keys()),
        ),
        "confidence": 0.94,
    }


def _extract_known_location(message: str) -> str:
    for name in ["奥斯丁", "奥斯汀", "Austin", "成都", "匹兹堡", "温州", "迈阿密", "宁波"]:
        if name.casefold() in message.casefold():
            return name
    return ""


def _looks_like_currency_query(message: str) -> bool:
    return any(token in message.upper() for token in ["USD", "CNY", "EUR", "GBP", "JPY"]) or (
        any(token in message for token in ["美元", "美金", "人民币", "欧元", "英镑", "日元", "汇率"])
        and any(token in message for token in ["多少", "换", "汇率", "TO", "to"])
    )


def _currency_arguments(message: str) -> tuple[float, str, str]:
    amount_match = re.search(r"(\d+(?:\.\d+)?)", message)
    amount = float(amount_match.group(1)) if amount_match else 1.0
    aliases = {
        "美元": "USD",
        "美金": "USD",
        "人民币": "CNY",
        "欧元": "EUR",
        "英镑": "GBP",
        "日元": "JPY",
        "新加坡元": "SGD",
        "新币": "SGD",
    }
    found: list[str] = []
    upper = message.upper()
    for alias, code in aliases.items():
        if alias in message and code not in found:
            found.append(code)
    for code in ["USD", "CNY", "EUR", "GBP", "JPY", "SGD"]:
        if code in upper and code not in found:
            found.append(code)
    return amount, found[0] if found else "", found[1] if len(found) > 1 else ""


def _extract_option_index(message: str) -> int | None:
    text = message.strip().lower()
    patterns = [
        r"解释第\s*(\d+)\s*个",
        r"第\s*(\d+)\s*个.*(为什么|便宜|展开|看看|看一下)",
        r"(看一下|展开)\s*第\s*(\d+)\s*个",
        r"why option\s*(\d+)",
        r"option\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        for group in match.groups():
            if group and group.isdigit():
                return int(group)
    chinese_digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    for word, value in chinese_digits.items():
        if f"第{word}个" in text or f"第 {word} 个" in text:
            return value
    return None
