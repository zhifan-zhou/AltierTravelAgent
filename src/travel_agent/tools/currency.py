"""Currency conversion tool backed by Frankfurter."""

from __future__ import annotations

from travel_agent.services.currency_service import FrankfurterCurrencyClient, normalize_currency
from travel_agent.tools.base import BaseTool, ToolRequestContext, ToolResult, clarification, unavailable
from travel_agent.tools.http_client import HttpClientError


class CurrencyTool(BaseTool):
    name = "currency"
    description = "Currency conversion from Frankfurter public rates."
    input_schema = {
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        },
    }

    def __init__(self, client: FrankfurterCurrencyClient):
        self.client = client

    def execute(self, args: dict, context: ToolRequestContext) -> ToolResult:
        del context
        source = normalize_currency(args.get("from_currency") or args.get("from") or args.get("base"))
        target = normalize_currency(args.get("to_currency") or args.get("to") or args.get("quote"))
        if not source or not target:
            return clarification(
                self.name,
                "请告诉我要换算的两种货币，例如“100 美元是多少人民币”或“USD to CNY”。",
                error_code="invalid_currency",
            )
        try:
            amount = float(args.get("amount", 1))
        except (TypeError, ValueError):
            return clarification(self.name, "换算金额是多少？请给一个数字。", error_code="invalid_amount")
        if amount < 0:
            return clarification(self.name, "换算金额不能是负数。", error_code="invalid_amount")
        try:
            conversion = self.client.convert(amount, source, target)
            return ToolResult(
                tool_name=self.name,
                status="ok",
                data=conversion.model_dump(mode="json"),
                message=(
                    f"按 Frankfurter {conversion.date} 的参考汇率，"
                    f"{conversion.amount:g} {source} ≈ {conversion.converted_amount:.2f} {target} "
                    f"（1 {source} ≈ {conversion.rate:.6g} {target}）。仅作货币换算，不构成金融建议。"
                ),
                source=conversion.source,
                fetched_at=conversion.fetched_at,
                is_live=True,
            )
        except HttpClientError as exc:
            return unavailable(
                self.name,
                "Frankfurter 当前不可用或不支持该货币，无法给出可靠换算；我不会编造汇率。",
                source="frankfurter",
                error_code=exc.code,
                debug={"exception": exc.__class__.__name__},
            )
