"""NVIDIA NIM compatibility: sequential tool-calls for TradingAgents.

NVIDIA NIM rejects requests where the LLM generates multiple tool_calls at
once (500: "This model only supports single tool-calls at once!"). This
module provides replacement analyst functions that bind tools one at a time,
letting LangGraph's tool-calling loop handle sequential execution.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)


def _get_next_tool(tools, messages):
    """Find the next uncalled tool from the message history."""
    called = set()
    for msg in messages:
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                called.add(tc["name"])
    for tool in tools:
        if tool.name not in called:
            return tool
    return None


def create_market_analyst_nim(llm):
    """Market analyst with sequential tool binding for NVIDIA NIM."""
    from tradingagents.agents.utils.agent_utils import (
        get_indicators,
        get_instrument_context_from_state,
        get_language_instruction,
        get_stock_data,
        get_verified_market_snapshot,
    )

    tools = [get_stock_data, get_indicators, get_verified_market_snapshot]

    system_message = (
        """You are a senior technical analyst at a top hedge fund. Your task is to provide a comprehensive technical analysis of the stock, selecting the **most relevant indicators** for the current market condition. Choose up to **8 indicators** that provide complementary insights without redundancy.

**CRITICAL RULES:**
- Every claim MUST cite specific data from tool output (dates, prices, indicator values)
- Never fabricate support/resistance levels, percentage moves, or historical patterns
- If tool data conflicts with your expectation, flag the discrepancy
- State your time horizon explicitly (swing trade: 5-10 days, position: 2-4 weeks, etc.)

**Indicator Categories:**

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names.

Before writing the final report, call get_verified_market_snapshot for this ticker and the current date, and treat it as the source of truth for any exact OHLCV, price-level, or indicator-value claim. If another tool's output conflicts with the verified snapshot, flag the discrepancy rather than inventing a reconciled number. Do not claim historical validation, support/resistance bounces, or exact percentage moves unless they are directly supported by tool output with concrete dates and prices.

Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions. End with a clear technical verdict: bullish, bearish, or neutral with a confidence level (0-100%)."""
        + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        + get_language_instruction()
    )

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # NVIDIA NIM: bind one tool at a time
        next_tool = _get_next_tool(tools, state["messages"])
        if next_tool:
            chain = prompt | llm.bind_tools([next_tool])
        else:
            chain = prompt | llm.bind_tools([])

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node


def create_news_analyst_nim(llm):
    """News analyst with sequential tool binding for NVIDIA NIM."""
    from tradingagents.agents.utils.agent_utils import (
        get_global_news,
        get_instrument_context_from_state,
        get_language_instruction,
        get_macro_indicators,
        get_news,
        get_prediction_markets,
    )

    tools = [get_news, get_global_news, get_macro_indicators, get_prediction_markets]

    system_message = (
        "You are a senior news analyst at a top hedge fund. Your task is to provide a comprehensive news analysis that directly impacts the stock's near-term outlook.\n\n"
        "**CRITICAL RULES:**\n"
        "- Every claim MUST cite the specific headline, source, or data point\n"
        "- Differentiate between material news (earnings, guidance, M&A, regulatory) vs noise (analyst opinions, general market commentary)\n"
        "- Weight recency: news from today/yesterday matters more than news from 5 days ago\n"
        "- Identify sentiment direction and magnitude for each significant news item\n"
        "- Note any upcoming catalysts (earnings date, FDA decision, product launch, etc.)\n\n"
        "**Analysis Framework:**\n"
        "1. Company-Specific News: Earnings, guidance, product launches, management changes, insider transactions\n"
        "2. Sector/Industry News: Regulatory changes, competitor actions, supply chain disruptions\n"
        "3. Macro Context: Fed policy, interest rates, geopolitical events affecting the sector\n"
        "4. Market-Implied Probabilities: Use prediction markets to quantify forward-looking risks\n\n"
        "**Tools:**\n"
        "- get_news(ticker, start_date, end_date): Company-specific news by ticker\n"
        "- get_global_news(curr_date, look_back_days, limit): Broader macroeconomic news\n"
        "- get_macro_indicators(indicator, curr_date, look_back_days): FRED data (cpi, core_pce, unemployment, fed_funds_rate, 10y_treasury, yield_curve)\n"
        "- get_prediction_markets(topic, limit): Market-implied probabilities\n\n"
        "Provide specific, actionable insights with supporting evidence to help traders make informed decisions. End with a clear sentiment verdict: bullish, bearish, or neutral with a confidence level (0-100%)."
        + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        + get_language_instruction()
    )

    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # NVIDIA NIM: bind one tool at a time
        next_tool = _get_next_tool(tools, state["messages"])
        if next_tool:
            chain = prompt | llm.bind_tools([next_tool])
        else:
            chain = prompt | llm.bind_tools([])

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node


def create_fundamentals_analyst_nim(llm):
    """Fundamentals analyst with sequential tool binding for NVIDIA NIM."""
    from tradingagents.agents.utils.agent_utils import (
        get_balance_sheet,
        get_cashflow,
        get_fundamentals,
        get_income_statement,
        get_instrument_context_from_state,
        get_language_instruction,
    )

    tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]

    system_message = (
        "You are a senior fundamental analyst at a top hedge fund. Your task is to provide a comprehensive fundamental analysis that identifies the stock's intrinsic value and key financial risks.\n\n"
        "**CRITICAL RULES:**\n"
        "- Every claim MUST cite specific numbers from financial statements (revenue, margins, FCF, etc.)\n"
        "- Never fabricate financial data — only use what tools provide\n"
        "- Compare current metrics to historical trends and industry benchmarks\n"
        "- Identify material changes quarter-over-quarter or year-over-year\n\n"
        "**Analysis Framework:**\n"
        "1. Valuation: P/E, P/S, EV/EBITDA, PEG ratio vs historical and sector averages\n"
        "2. Profitability: Gross margin, operating margin, net margin trends\n"
        "3. Growth: Revenue growth, earnings growth, guidance vs consensus\n"
        "4. Balance Sheet: Debt levels, current ratio, cash position\n"
        "5. Cash Flow: Operating cash flow, free cash flow, capex trends\n"
        "6. Capital Allocation: Dividends, buybacks, M&A activity\n\n"
        "**Tools:**\n"
        "- get_fundamentals: Comprehensive company analysis\n"
        "- get_balance_sheet: Balance sheet details\n"
        "- get_cashflow: Cash flow statement\n"
        "- get_income_statement: Income statement details\n\n"
        "Provide specific, actionable insights with supporting evidence to help traders make informed decisions. End with a clear fundamental verdict: bullish, bearish, or neutral with a confidence level (0-100%)."
        + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
        + get_language_instruction(),
    )

    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # NVIDIA NIM: bind one tool at a time
        next_tool = _get_next_tool(tools, state["messages"])
        if next_tool:
            chain = prompt | llm.bind_tools([next_tool])
        else:
            chain = prompt | llm.bind_tools([])

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node


# ---------------------------------------------------------------------------
# Debate agent patches: add citation requirements and steel-manning
# ---------------------------------------------------------------------------

_CITATION_SUFFIX = (
    "\n\n**MANDATORY RULES:**\n"
    "- Cite specific numbers, dates, and data points from the analyst reports above\n"
    "- Do NOT make vague claims — every assertion must reference a specific data point\n"
    "- Steel-man the opposing argument before refuting it\n"
    "- Keep your response under 400 words — quality over quantity"
)


def create_bull_researcher_nim(llm):
    """Bull researcher with citation requirements."""
    from tradingagents.agents.utils.agent_utils import (
        get_instrument_context_from_state,
        get_language_instruction,
    )

    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")
        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        prompt = f"""You are a Bull Analyst advocating for investing in the {target_label}. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
{_CITATION_SUFFIX}
""" + get_language_instruction()

        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node


def create_bear_researcher_nim(llm):
    """Bear researcher with citation requirements."""
    from tradingagents.agents.utils.agent_utils import (
        get_instrument_context_from_state,
        get_language_instruction,
    )

    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")
        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        prompt = f"""You are a Bear Analyst making the case against investing in the {target_label}. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators.

Key points to focus on:
- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news.
- Bull Counterpoints: Critically analyze the bull argument with specific data, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
{_CITATION_SUFFIX}
""" + get_language_instruction()

        response = llm.invoke(prompt)
        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node


def create_aggressive_debator_nim(llm):
    """Aggressive risk debator with citation requirements."""
    from tradingagents.agents.utils.agent_utils import (
        get_instrument_context_from_state,
        get_language_instruction,
    )

    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Aggressive Risk Analyst, your role is to champion high-reward, high-risk opportunities. Focus on upside potential, growth potential, and innovative benefits.

When evaluating the trader's decision, incorporate insights from:
{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}

Trader's decision: {trader_decision}
Conversation history: {history}
Last conservative argument: {current_conservative_response}
Last neutral argument: {current_neutral_response}

**MANDATORY RULES:**
- Cite specific numbers and data points from the reports above
- Respond directly to each point made by conservative and neutral analysts
- Challenge assumptions with data-driven rebuttals
- Keep your response under 400 words
{get_language_instruction()}"""

        response = llm.invoke(prompt)
        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node


def create_conservative_debator_nim(llm):
    """Conservative risk debator with citation requirements."""
    from tradingagents.agents.utils.agent_utils import (
        get_instrument_context_from_state,
        get_language_instruction,
    )

    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")
        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. Focus on downside protection, risk mitigation, and capital preservation.

When evaluating the trader's decision, incorporate insights from:
{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}

Trader's decision: {trader_decision}
Conversation history: {history}
Last aggressive argument: {current_aggressive_response}
Last neutral argument: {current_neutral_response}

**MANDATORY RULES:**
- Cite specific numbers and data points from the reports above
- Respond directly to each point made by aggressive and neutral analysts
- Challenge assumptions with data-driven rebuttals
- Keep your response under 400 words
{get_language_instruction()}"""

        response = llm.invoke(prompt)
        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "conservative_history": conservative_history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_conservative_response": argument,
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node


def create_neutral_debator_nim(llm):
    """Neutral risk debator with citation requirements."""
    from tradingagents.agents.utils.agent_utils import (
        get_instrument_context_from_state,
        get_language_instruction,
    )

    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")
        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both potential benefits and risks. Consider the trader's decision and evaluate it against both aggressive and conservative viewpoints, using the provided market data and reports.

When evaluating the trader's decision, incorporate insights from:
{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}

Trader's decision: {trader_decision}
Conversation history: {history}
Last aggressive argument: {current_aggressive_response}
Last conservative argument: {current_conservative_response}

**MANDATORY RULES:**
- Cite specific numbers and data points from the reports above
- Respond directly to each point made by aggressive and conservative analysts
- Identify the strongest argument from each side and explain why
- Keep your response under 400 words
{get_language_instruction()}"""

        response = llm.invoke(prompt)
        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "neutral_history": neutral_history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "latest_speaker": "Neutral",
            "current_neutral_response": argument,
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
