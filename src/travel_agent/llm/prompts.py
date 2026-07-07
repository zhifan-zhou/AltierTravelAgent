"""Prompt construction for the DeepSeek requirement agent."""

from __future__ import annotations

import json
from typing import Any

from travel_agent.contract.models import TravelRequirementContract
from travel_agent.llm.schemas import TravelRequirementContractUpdate


SYSTEM_PROMPT = """你是 DeepSeekRequirementAgent，只负责把用户自然语言更新为严格 JSON。

你不是航班搜索器，不得编造航班、价格、航司、库存或可订状态。
你只能更新 TravelRequirementContractUpdate schema。
输出必须是一个 JSON object，不得在 JSON 前后写任何解释。
每个可执行更新必须包含 decision_trace。
你是通用出行需求 schema updater，不要依赖固定命令或关键词。
用户可能表达生活约束、同行人、证件、行李、偏好、闲聊或求助；你的工作是分类并映射为 schema update。
特殊/生活需求必须写入 special_requirements_to_add；不要编造航司政策事实。
必须设置 next_action，决定这轮是问澄清、回答建议、搜索、重排、行程规划、预算估算、约束检查、解释、导出、帮助、闲聊、退出还是 no_op。
如果用户询问实时/工具类信息（天气、时间、汇率、机场查询、目的地简介等），设置 next_action=tool_query 并填写 tool_requests。不要编造实时事实。
同行人、宠物、预算、时间偏好和航班偏好必须映射为通用 contract 字段与 constraint/preference，不要为某个案例写单点规则。
取消或否定已有约束时使用 remove_* update，并保留 inactive 历史语义。
“六月初 / 7月初 / 下个月 / 下周 / 8月20日前后”都算有效 departure_window_text。不要因为不是精确日期而追问具体哪一天。
如果用户只是问解释第几个方案，使用 update_type=explain_option，不要改需求。
"""


def build_requirement_prompt(
    *,
    contract: TravelRequirementContract | None,
    user_message: str,
    history_summary: str,
    airport_alias_map: dict[str, list[str]],
    displayed_recommendations_summary: str,
) -> str:
    schema = TravelRequirementContractUpdate.model_json_schema()
    payload: dict[str, Any] = {
        "task": "Return exactly one TravelRequirementContractUpdate JSON object.",
        "current_contract": contract.model_dump(mode="json") if contract else None,
        "user_message": user_message,
        "conversation_history_summary": history_summary,
        "airport_alias_map": airport_alias_map,
        "current_displayed_recommendations_summary": displayed_recommendations_summary,
        "schema": schema,
        "rules": [
            "DeepSeek is a general travel requirement schema updater, not a flight searcher.",
            "Never invent flights, prices, airlines, availability, or booking status.",
            "Classify the user message as one of: new_search, route_modification, hard_constraint, soft_preference, special_requirement, clarification_answer, explain_option, export, help, smalltalk, unknown.",
            "Use update_type=create_new for new_search; modify_existing/add_constraint/add_preference for normal travel requirement changes; add_special_requirement for life/special requirements; smalltalk for greetings; help for usage questions; unknown when not convertible.",
            "Always set next_action. Use run_search only when the user asks for itinerary recommendations or explicitly asks to rerun/search.",
            "Use ask_clarification when a route request lacks mandatory search fields such as departure date/window.",
            "Use answer_advisory for policy/feasibility/travel-advice questions such as pets, baggage, visa, split tickets, transfer risk, cabin feasibility.",
            "For advisory answers, fill advisory_response_zh and do not invent specific airline policy facts. Mention airline/official-source confirmation where relevant.",
            "Use rerank for preference-only updates when recommendations already exist.",
            "For day-by-day planning requests, set next_action=itinerary and store time.duration_days when stated; do not require an origin or flight date if destination is known.",
            "For rough trip budget requests, set next_action=cost_estimate and preserve duration, destination, budget amount, and currency when provided.",
            "For requests to review constraints or risks, set next_action=constraint_check.",
            "Planning actions do not invent tickets, opening hours, prices, visa conclusions, or airline policies.",
            "Use tool_query for weather/time/currency/airport/destination brief questions. Allowed tools: weather, airport_lookup, time, currency, destination_brief.",
            "For 目的地天气怎么样, if current contract has destination, use weather with destination city/airport in arguments and requires_current_contract=true.",
            "For 奥斯丁机场是哪个 / 成都有哪些机场, use airport_lookup with location text.",
            "For 奥斯丁现在几点, use time with location=奥斯丁.",
            "For 100美元是多少人民币, use currency with amount=100, from_currency=USD, to_currency=CNY.",
            "For 奥斯丁有什么好玩的/介绍目的地, use destination_brief.",
            "Pet companions go to companions.pets and a generic pet_companion constraint; cancellation marks/removes the matching pet constraint rather than adding a contradictory active one.",
            "Budget belongs in budget.amount/currency/priority or budget.preference and may also add prefer_low_price.",
            "Red-eye and nonstop statements belong in preferences.avoid_red_eye, preferences.nonstop_preferred, and preferences.max_stops.",
            "For actionable update types, decision_trace must be non-empty.",
            "For special/life requirements, put one or more SpecialRequirement objects in special_requirements_to_add.",
            "For SpecialRequirement.category choose a general category such as pet_travel, family_or_elderly, child_travel, accessibility, heavy_baggage, visa_constraint, alliance_preference, overnight_avoidance, stopover_request, meal_or_medical, or unknown.",
            "Fill structured_values from user evidence only; use unknown when information is missing.",
            "Use impact_areas such as airline_policy, baggage, risk, connection_time, self_transfer, cabin, routing, documentation.",
            "Mark requires_clarification=true only when a short answer would materially improve routing or warnings; do not block search if origin/destination are complete unless feasibility is impossible.",
            "For smalltalk like 你好, use update_type=smalltalk, should_search=false, and a helpful Chinese response.",
            "For help/usage questions, use update_type=help, should_search=false, and show examples.",
            "New search must use update_type=create_new and should_rerun_search=true.",
            "If route is known but date/window is missing, next_action should be ask_clarification unless the user explicitly says 日期还没定/先随便看看/大概看看/用默认日期/灵活日期.",
            "Approximate windows like 六月初, 7月初, 下个月, 下周, 8月20日前后 are sufficient for this demo; set time.departure_window_text and next_action=run_search.",
            "Do not ask for an exact calendar date when the user gave a usable date window.",
            "If user explicitly accepts flexible/default dates, set time.flexible_date_confirmed=true and next_action=run_search.",
            "Modification like 如果从杭州走呢 appends acceptable_origin_hubs and keeps the existing primary origin/destination.",
            "City-level exclusion expands airports: 上海 -> PVG/SHA, 纽约 -> JFK/EWR/LGA.",
            "Airport-specific disambiguation: 上海浦东/浦东/PVG -> PVG; 上海虹桥/虹桥/SHA -> SHA.",
            "Preference-only updates such as 主流航司优先 should use should_rerank_only=true unless they alter hard search constraints.",
            "越便宜越好 means profile=cheapest, price_priority=high, nearby_hub_policy=prefer, without generic preference questions.",
        ],
        "examples": [
            {
                "name": "new_search",
                "existing": "WNZ -> PIT",
                "user": "我其实想看看宁波到迈阿密",
                "expected": {
                    "update_type": "create_new",
                    "field_updates": {
                        "trip": {
                            "origin_text": "宁波",
                            "origin_airport": "NGB",
                            "destination_text": "迈阿密",
                            "destination_airport": "MIA",
                        }
                    },
                    "next_action": "run_search",
                    "should_rerun_search": True,
                },
            },
            {
                "name": "missing_departure_window",
                "user": "我要从温州去匹兹堡",
                "expected": {
                    "update_type": "create_new",
                    "field_updates": {
                        "trip": {
                            "origin_text": "温州",
                            "origin_airport": "WNZ",
                            "destination_text": "匹兹堡",
                            "destination_airport": "PIT",
                        }
                    },
                    "next_action": "ask_clarification",
                    "clarification_question_zh": "你大概什么时候出发？可以给一个日期或时间范围，比如 6月初、下周、8月20日前后。",
                    "should_search": False,
                    "should_rerun_search": False,
                },
            },
            {
                "name": "complete_search_with_date",
                "user": "温州到匹兹堡，六月初，越便宜越好",
                "expected": {
                    "update_type": "create_new",
                    "field_updates": {
                        "trip": {"origin_airport": "WNZ", "destination_airport": "PIT"},
                        "time": {"departure_window_text": "六月初"},
                        "ranking": {"profile": "cheapest", "price_priority": "high"},
                    },
                    "next_action": "run_search",
                    "should_rerun_search": True,
                },
            },
            {
                "name": "flexible_date_confirmation",
                "user": "温州到匹兹堡，日期还没定，先随便看看",
                "expected": {
                    "update_type": "create_new",
                    "field_updates": {
                        "trip": {"origin_airport": "WNZ", "destination_airport": "PIT"},
                        "time": {"flexible_date_confirmed": True, "departure_window_text": "灵活日期"},
                    },
                    "next_action": "run_search",
                    "should_rerun_search": True,
                },
            },
            {
                "name": "modify_route",
                "existing": "WNZ -> PIT",
                "user": "如果从杭州走呢",
                "expected": {
                    "update_type": "modify_existing",
                    "field_updates": {"geography": {"acceptable_origin_hubs": ["HGH"]}},
                    "next_action": "run_search",
                    "should_rerun_search": True,
                },
            },
            {
                "name": "airport_disambiguation",
                "user": "其实可以去上海浦东，但不去虹桥",
                "expected": {
                    "update_type": "modify_existing",
                    "field_updates": {
                        "geography": {
                            "acceptable_origin_hubs": ["PVG"],
                            "avoid_airports": ["SHA"],
                        }
                    },
                    "next_action": "run_search",
                    "should_rerun_search": True,
                },
            },
            {
                "name": "city_exclusion",
                "user": "我不想去上海",
                "expected": {
                    "update_type": "add_constraint",
                    "field_updates": {"geography": {"avoid_cities": ["上海"], "avoid_airports": ["PVG", "SHA"]}},
                    "next_action": "run_search",
                    "should_rerun_search": True,
                },
            },
            {
                "name": "preference_only",
                "user": "主流航司优先",
                "expected": {
                    "update_type": "add_preference",
                    "field_updates": {
                        "ranking": {"profile": "airline_priority", "airline_quality_priority": "high"},
                        "airline_preferences": {"prefer_major_airlines": True},
                    },
                    "next_action": "rerank",
                    "should_rerank_only": True,
                },
            },
            {
                "name": "family_low_risk",
                "user": "我爸妈也一起，别太折腾",
                "expected": {
                    "update_type": "add_special_requirement",
                    "field_updates": {
                        "passengers": {"family_or_parents": True},
                        "ranking": {"profile": "low_risk", "risk_priority": "high"},
                        "ticketing": {"split_ticket_policy": "avoid", "allow_self_transfer": False},
                    },
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
                            "source_user_message": "我爸妈也一起，别太折腾",
                            "active": True,
                        }
                    ],
                    "next_action": "run_search",
                    "should_rerun_search": True,
                },
            },
            {
                "name": "pet_travel_advisory_question",
                "user": "我可以带狗吗？",
                "expected": {
                    "update_type": "advisory_question",
                    "special_requirements_to_add": [
                        {
                            "category": "pet_travel",
                            "description_zh": "用户询问宠物同行可行性",
                            "structured_values": {
                                "animal": "dog",
                                "size": "unknown",
                                "in_cabin_or_checked": "unknown",
                            },
                            "impact_areas": ["airline_policy", "baggage", "risk", "self_transfer"],
                            "hard_constraint": False,
                            "preference_weight": "high",
                            "requires_clarification": True,
                            "clarification_question_zh": "狗狗大概是小型犬、中型犬还是大型犬？这会影响能否进客舱或需要托运。",
                            "source_user_message": "我可以带狗吗？",
                            "active": True,
                        }
                    ],
                    "next_action": "answer_advisory",
                    "advisory_response_zh": "可以把宠物同行作为需求记录下来，但是否能进客舱/托运取决于航司、狗狗体型重量、航线和名额。你可以告诉我狗狗大概是小型犬、中型犬还是大型犬？",
                    "should_rerun_search": False,
                },
            },
            {
                "name": "heavy_baggage",
                "user": "我有很多行李",
                "expected": {
                    "update_type": "add_special_requirement",
                    "special_requirements_to_add": [
                        {
                            "category": "heavy_baggage",
                            "description_zh": "用户有较多行李，需要降低自助转机和重新托运风险",
                            "structured_values": {"baggage_amount": "many"},
                            "impact_areas": ["baggage", "risk", "self_transfer", "connection_time"],
                            "hard_constraint": False,
                            "preference_weight": "high",
                            "requires_clarification": False,
                            "clarification_question_zh": None,
                            "source_user_message": "我有很多行李",
                            "active": True,
                        }
                    ],
                    "next_action": "answer_advisory",
                    "should_rerun_search": False,
                },
            },
            {
                "name": "visa_concern",
                "user": "我没有美国签证，不想过境麻烦",
                "expected": {
                    "update_type": "add_special_requirement",
                    "special_requirements_to_add": [
                        {
                            "category": "visa_constraint",
                            "description_zh": "用户担心签证或过境要求",
                            "structured_values": {"visa_sensitive": True, "avoid_uncertain_transit": True},
                            "impact_areas": ["routing", "documentation", "risk"],
                            "hard_constraint": True,
                            "preference_weight": "high",
                            "requires_clarification": True,
                            "clarification_question_zh": "你是想避免某些国家/地区转机，还是只想要入境政策更简单的路线？",
                            "source_user_message": "我没有美国签证，不想过境麻烦",
                            "active": True,
                        }
                    ],
                    "next_action": "answer_advisory",
                    "should_rerun_search": False,
                },
            },
            {
                "name": "smalltalk",
                "user": "你好",
                "expected": {
                    "update_type": "smalltalk",
                    "next_action": "smalltalk",
                    "should_search": False,
                    "user_facing_ack_zh": "你好，我可以帮你用自然语言规划路线。比如：温州到匹兹堡，可以从上海走，越便宜越好。",
                },
            },
            {
                "name": "destination_weather_tool",
                "existing": "TFU -> AUS",
                "user": "目的地天气怎么样？",
                "expected": {
                    "update_type": "advisory_question",
                    "next_action": "tool_query",
                    "tool_requests": [
                        {
                            "tool_name": "weather",
                            "arguments": {"location": "Austin"},
                            "reason_zh": "用户询问当前目的地天气，需要实时工具。",
                            "requires_current_contract": True,
                        }
                    ],
                    "should_search": False,
                },
            },
            {
                "name": "three_day_itinerary",
                "user": "帮我安排奥斯丁三天行程，预算低一点",
                "expected": {
                    "update_type": "create_new",
                    "field_updates": {
                        "trip": {"destination_text": "奥斯丁", "destination_airport": "AUS"},
                        "time": {"duration_days": 3},
                        "budget": {"preference": "lower", "priority": "high"},
                    },
                    "next_action": "itinerary",
                    "should_search": False,
                },
            },
            {
                "name": "cost_estimate_followup",
                "existing": "TFU -> AUS, duration=3",
                "user": "估算一下预算",
                "expected": {
                    "update_type": "modify_existing",
                    "next_action": "cost_estimate",
                    "should_search": False,
                },
            },
            {
                "name": "constraint_check_followup",
                "existing": "pet + avoid_red_eye + nonstop_preferred",
                "user": "检查一下当前约束和风险",
                "expected": {
                    "update_type": "modify_existing",
                    "next_action": "constraint_check",
                    "should_search": False,
                },
            },
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
