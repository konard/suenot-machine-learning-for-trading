"""
Agent Communication Module

This module provides communication patterns for multi-agent trading systems:
- Message passing between agents
- Debate mechanisms (Bull vs Bear)
- Round table discussions
- Hierarchical communication
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging

import pandas as pd

from .agents import BaseAgent, Analysis, Signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages between agents."""
    ANALYSIS = "analysis"
    QUERY = "query"
    RESPONSE = "response"
    DEBATE_ARGUMENT = "debate_argument"
    DEBATE_REBUTTAL = "debate_rebuttal"
    DECISION = "decision"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """A message between agents."""
    sender: str
    receiver: str  # "broadcast" for all agents
    message_type: MessageType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.message_type.value,
            "content": str(self.content) if not isinstance(self.content, dict) else self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentMessage:
    """A structured message from an agent containing analysis."""
    agent_name: str
    agent_type: str
    analysis: Analysis
    arguments: List[str] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate a summary for other agents."""
        return (
            f"{self.agent_name} ({self.agent_type}): "
            f"{self.analysis.signal.value} with {self.analysis.confidence:.0%} confidence. "
            f"Reasoning: {self.analysis.reasoning}"
        )


class MessageBus:
    """
    Central message bus for agent communication.

    Supports:
    - Point-to-point messaging
    - Broadcast messaging
    - Message history and replay
    """

    def __init__(self):
        self._messages: List[Message] = []
        self._subscribers: Dict[str, List[Callable]] = {}

    def send(self, message: Message):
        """Send a message."""
        self._messages.append(message)

        # Notify subscribers
        if message.receiver == "broadcast":
            for subscriber_id, callbacks in self._subscribers.items():
                if subscriber_id != message.sender:
                    for callback in callbacks:
                        callback(message)
        else:
            if message.receiver in self._subscribers:
                for callback in self._subscribers[message.receiver]:
                    callback(message)

        logger.debug(f"Message sent: {message.sender} -> {message.receiver}")

    def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe to messages."""
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(callback)

    def get_history(self, agent_id: Optional[str] = None) -> List[Message]:
        """Get message history, optionally filtered by agent."""
        if agent_id:
            return [
                m for m in self._messages
                if m.sender == agent_id or m.receiver in [agent_id, "broadcast"]
            ]
        return self._messages.copy()

    def clear(self):
        """Clear message history."""
        self._messages.clear()


class Debate:
    """
    A structured debate between two opposing agents (Bull vs Bear).

    The debate consists of multiple rounds where each side presents
    arguments and rebuttals.
    """

    def __init__(
        self,
        bull_agent: BaseAgent,
        bear_agent: BaseAgent,
        num_rounds: int = 3
    ):
        if num_rounds <= 0:
            raise ValueError("num_rounds must be > 0")
        self.bull_agent = bull_agent
        self.bear_agent = bear_agent
        self.num_rounds = num_rounds
        self._rounds: List[Dict] = []
        self._message_bus = MessageBus()

    def conduct(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Dict:
        """
        Conduct a full debate.

        Args:
            symbol: Trading symbol to debate
            data: Market data
            context: Additional context

        Returns:
            Debate results including all arguments and final scores
        """
        context = context or {}
        self._rounds = []

        # Initial analyses
        bull_analysis = self.bull_agent.analyze(symbol, data, context)
        bear_analysis = self.bear_agent.analyze(symbol, data, context)

        # Store initial positions
        debate_context = {
            "bull_position": bull_analysis.reasoning,
            "bear_position": bear_analysis.reasoning,
            "previous_arguments": []
        }

        for round_num in range(self.num_rounds):
            round_data = {
                "round": round_num + 1,
                "bull_argument": None,
                "bear_argument": None
            }

            # Bull presents argument
            bull_context = {**context, "debate": debate_context, "opponent_last": debate_context.get("bear_last")}
            bull_round = self.bull_agent.analyze(symbol, data, bull_context)
            round_data["bull_argument"] = {
                "reasoning": bull_round.reasoning,
                "confidence": bull_round.confidence,
                "metrics": bull_round.metrics
            }
            debate_context["bull_last"] = bull_round.reasoning

            # Bear presents rebuttal
            bear_context = {**context, "debate": debate_context, "opponent_last": debate_context.get("bull_last")}
            bear_round = self.bear_agent.analyze(symbol, data, bear_context)
            round_data["bear_argument"] = {
                "reasoning": bear_round.reasoning,
                "confidence": bear_round.confidence,
                "metrics": bear_round.metrics
            }
            debate_context["bear_last"] = bear_round.reasoning

            # Track arguments
            debate_context["previous_arguments"].append({
                "round": round_num + 1,
                "bull": bull_round.reasoning,
                "bear": bear_round.reasoning
            })

            self._rounds.append(round_data)

            # Log messages
            self._message_bus.send(Message(
                sender=self.bull_agent.name,
                receiver=self.bear_agent.name,
                message_type=MessageType.DEBATE_ARGUMENT,
                content=bull_round.reasoning,
                metadata={"round": round_num + 1}
            ))

            self._message_bus.send(Message(
                sender=self.bear_agent.name,
                receiver=self.bull_agent.name,
                message_type=MessageType.DEBATE_REBUTTAL,
                content=bear_round.reasoning,
                metadata={"round": round_num + 1}
            ))

        # Calculate final scores
        avg_bull_confidence = sum(r["bull_argument"]["confidence"] for r in self._rounds) / len(self._rounds)
        avg_bear_confidence = sum(r["bear_argument"]["confidence"] for r in self._rounds) / len(self._rounds)

        return {
            "symbol": symbol,
            "rounds": self._rounds,
            "initial_bull_analysis": bull_analysis.to_dict(),
            "initial_bear_analysis": bear_analysis.to_dict(),
            "final_scores": {
                "bull_confidence": avg_bull_confidence,
                "bear_confidence": avg_bear_confidence,
                "winner": "bull" if avg_bull_confidence > avg_bear_confidence else "bear"
            },
            "message_history": [m.to_dict() for m in self._message_bus.get_history()]
        }


class DebateModerator:
    """
    Moderates debates and produces balanced conclusions.

    Weighs arguments from both sides and provides a final recommendation.
    """

    def __init__(self, name: str = "Moderator"):
        self.name = name

    def evaluate(self, debate_result: Dict) -> Dict:
        """
        Evaluate debate results and provide conclusion.

        Args:
            debate_result: Output from Debate.conduct()

        Returns:
            Moderated conclusion with final recommendation
        """
        bull_score = debate_result["final_scores"]["bull_confidence"]
        bear_score = debate_result["final_scores"]["bear_confidence"]

        # Analyze argument quality
        bull_arguments = [r["bull_argument"]["reasoning"] for r in debate_result["rounds"]]
        bear_arguments = [r["bear_argument"]["reasoning"] for r in debate_result["rounds"]]

        # Simple scoring: argument length and keyword diversity
        bull_quality = sum(len(arg.split()) for arg in bull_arguments) / len(bull_arguments)
        bear_quality = sum(len(arg.split()) for arg in bear_arguments) / len(bear_arguments)

        # Calculate weighted score
        bull_total = bull_score * 0.7 + (bull_quality / 100) * 0.3
        bear_total = bear_score * 0.7 + (bear_quality / 100) * 0.3

        # Determine signal
        score_diff = bull_total - bear_total

        if score_diff > 0.2:
            signal = Signal.STRONG_BUY
            conclusion = "Bull arguments significantly outweigh bear concerns"
        elif score_diff > 0.05:
            signal = Signal.BUY
            conclusion = "Bull arguments slightly favored"
        elif score_diff < -0.2:
            signal = Signal.STRONG_SELL
            conclusion = "Bear arguments significantly outweigh bull case"
        elif score_diff < -0.05:
            signal = Signal.SELL
            conclusion = "Bear arguments slightly favored"
        else:
            signal = Signal.NEUTRAL
            conclusion = "Arguments are evenly balanced"

        return {
            "signal": signal.value,
            "confidence": min(0.5 + abs(score_diff), 1.0),
            "conclusion": conclusion,
            "bull_total_score": bull_total,
            "bear_total_score": bear_total,
            "recommendation": self._generate_recommendation(signal, debate_result)
        }

    def _generate_recommendation(self, signal: Signal, debate_result: Dict) -> str:
        """Generate a detailed recommendation."""
        symbol = debate_result["symbol"]

        if signal in [Signal.STRONG_BUY, Signal.BUY]:
            return (
                f"Consider entering a long position in {symbol}. "
                f"Bull case: {debate_result['initial_bull_analysis']['reasoning'][:100]}... "
                f"Monitor: {debate_result['initial_bear_analysis']['reasoning'][:50]}..."
            )
        elif signal in [Signal.STRONG_SELL, Signal.SELL]:
            return (
                f"Consider avoiding or shorting {symbol}. "
                f"Bear case: {debate_result['initial_bear_analysis']['reasoning'][:100]}... "
                f"But watch: {debate_result['initial_bull_analysis']['reasoning'][:50]}..."
            )
        else:
            return (
                f"Stay neutral on {symbol} until clearer signals emerge. "
                f"Bull/Bear arguments are balanced."
            )


class RoundTable:
    """
    A round table discussion where multiple agents share views sequentially.

    Each agent can see previous agents' analyses before providing their own.
    """

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self._discussion: List[AgentMessage] = []
        self._message_bus = MessageBus()

    def conduct(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Dict:
        """
        Conduct a round table discussion.

        Args:
            symbol: Trading symbol
            data: Market data
            context: Additional context

        Returns:
            Discussion results with all agent inputs
        """
        context = context or {}
        self._discussion = []

        running_context = {**context, "previous_analyses": []}

        for agent in self.agents:
            # Agent analyzes with knowledge of previous analyses
            analysis = agent.analyze(symbol, data, running_context)

            # Create structured message
            message = AgentMessage(
                agent_name=agent.name,
                agent_type=agent.agent_type,
                analysis=analysis,
                arguments=[analysis.reasoning],
                evidence=analysis.metrics
            )

            self._discussion.append(message)

            # Broadcast to other agents
            self._message_bus.send(Message(
                sender=agent.name,
                receiver="broadcast",
                message_type=MessageType.ANALYSIS,
                content=message.to_summary()
            ))

            # Update context for next agent
            running_context["previous_analyses"].append({
                "agent": agent.name,
                "type": agent.agent_type,
                "signal": analysis.signal.value,
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning
            })

        return self._compile_results(symbol)

    def _compile_results(self, symbol: str) -> Dict:
        """Compile discussion results."""
        analyses = [m.analysis for m in self._discussion]

        # Count signals
        signal_counts = {}
        for analysis in analyses:
            sig = analysis.signal.value
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        # Calculate consensus
        buy_signals = signal_counts.get("STRONG_BUY", 0) + signal_counts.get("BUY", 0)
        sell_signals = signal_counts.get("STRONG_SELL", 0) + signal_counts.get("SELL", 0)
        total_agents = len(self._discussion)

        if buy_signals > total_agents * 0.6:
            consensus = "BULLISH"
        elif sell_signals > total_agents * 0.6:
            consensus = "BEARISH"
        else:
            consensus = "MIXED"

        return {
            "symbol": symbol,
            "participants": [m.agent_name for m in self._discussion],
            "analyses": [m.analysis.to_dict() for m in self._discussion],
            "signal_counts": signal_counts,
            "consensus": consensus,
            "buy_ratio": buy_signals / total_agents if total_agents > 0 else 0,
            "sell_ratio": sell_signals / total_agents if total_agents > 0 else 0,
            "message_history": [m.to_dict() for m in self._message_bus.get_history()]
        }


class HierarchicalCommunication:
    """
    Hierarchical communication pattern where a manager coordinates analysts.

    Manager agent distributes tasks, collects results, and makes decisions.
    """

    def __init__(
        self,
        manager: BaseAgent,
        analysts: List[BaseAgent]
    ):
        self.manager = manager
        self.analysts = analysts
        self._message_bus = MessageBus()

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Dict:
        """
        Coordinate analysis through hierarchical communication.

        Args:
            symbol: Trading symbol
            data: Market data
            context: Additional context

        Returns:
            Analysis results with manager's final decision
        """
        context = context or {}

        # Step 1: Manager broadcasts analysis request
        self._message_bus.send(Message(
            sender=self.manager.name,
            receiver="broadcast",
            message_type=MessageType.QUERY,
            content=f"Analyze {symbol} and report findings"
        ))

        # Step 2: Analysts perform analyses
        analyst_results = []
        for analyst in self.analysts:
            analysis = analyst.analyze(symbol, data, context)
            analyst_results.append(analysis)

            # Analyst reports to manager
            self._message_bus.send(Message(
                sender=analyst.name,
                receiver=self.manager.name,
                message_type=MessageType.RESPONSE,
                content=analysis.to_dict()
            ))

        # Step 3: Manager aggregates and decides
        manager_context = {**context, "analyses": analyst_results}
        manager_decision = self.manager.analyze(symbol, data, manager_context)

        # Step 4: Manager broadcasts decision
        self._message_bus.send(Message(
            sender=self.manager.name,
            receiver="broadcast",
            message_type=MessageType.DECISION,
            content=manager_decision.to_dict()
        ))

        return {
            "symbol": symbol,
            "manager": self.manager.name,
            "analysts": [a.name for a in self.analysts],
            "analyst_analyses": [a.to_dict() for a in analyst_results],
            "manager_decision": manager_decision.to_dict(),
            "message_history": [m.to_dict() for m in self._message_bus.get_history()]
        }


if __name__ == "__main__":
    print("Communication Demo\n" + "=" * 50)

    # Import agents
    from .agents import TechnicalAgent, FundamentalsAgent, BullAgent, BearAgent, TraderAgent
    import numpy as np

    # Create mock data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    close = 100 * (1 + np.random.randn(252) * 0.02).cumprod()

    data = pd.DataFrame({
        "open": close * (1 + np.random.randn(252) * 0.005),
        "high": close * (1 + abs(np.random.randn(252) * 0.01)),
        "low": close * (1 - abs(np.random.randn(252) * 0.01)),
        "close": close,
        "volume": np.random.randint(1e6, 1e8, 252)
    }, index=dates)

    # Test Debate
    print("\n1. Bull vs Bear Debate")
    print("-" * 30)
    bull = BullAgent("Optimist")
    bear = BearAgent("Skeptic")

    debate = Debate(bull, bear, num_rounds=2)
    result = debate.conduct("DEMO", data)

    print(f"Winner: {result['final_scores']['winner']}")
    print(f"Bull confidence: {result['final_scores']['bull_confidence']:.0%}")
    print(f"Bear confidence: {result['final_scores']['bear_confidence']:.0%}")

    # Test Moderator
    moderator = DebateModerator()
    conclusion = moderator.evaluate(result)
    print(f"\nModerated conclusion: {conclusion['conclusion']}")
    print(f"Signal: {conclusion['signal']}")

    # Test Round Table
    print("\n2. Round Table Discussion")
    print("-" * 30)
    agents = [
        TechnicalAgent("Tech-1"),
        FundamentalsAgent("Fund-1"),
        BullAgent("Bull-1"),
        BearAgent("Bear-1")
    ]

    round_table = RoundTable(agents)
    rt_result = round_table.conduct("DEMO", data)

    print(f"Participants: {rt_result['participants']}")
    print(f"Consensus: {rt_result['consensus']}")
    print(f"Buy ratio: {rt_result['buy_ratio']:.0%}")
    print(f"Sell ratio: {rt_result['sell_ratio']:.0%}")
