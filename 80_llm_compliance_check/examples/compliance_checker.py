"""
LLM Compliance Check - Python Implementation

This module provides LLM-based compliance checking for trading activities
using both stock market and cryptocurrency (Bybit) data.

Example usage:
    python compliance_checker.py

Note: For actual LLM integration, you need to provide a valid API key.
This demo shows the structure and pre-check functionality without API calls.
"""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib


def utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


class ComplianceStatus(Enum):
    """Compliance check status."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    PENDING = "PENDING"


class ViolationSeverity(Enum):
    """Violation severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActivityType(Enum):
    """Trading activity types."""
    ORDER_SUBMISSION = "order_submission"
    ORDER_MODIFICATION = "order_modification"
    ORDER_CANCELLATION = "order_cancellation"
    TRADE_EXECUTION = "trade_execution"
    POSITION_CHANGE = "position_change"
    STRATEGY_DEPLOYMENT = "strategy_deployment"


class Jurisdiction(Enum):
    """Jurisdiction for compliance checking."""
    UNITED_STATES = "US"
    EUROPEAN_UNION = "EU"
    UNITED_KINGDOM = "UK"
    CRYPTO = "CRYPTO"
    GLOBAL = "GLOBAL"


@dataclass
class Violation:
    """Represents a compliance violation."""
    rule: str
    description: str
    severity: ViolationSeverity
    evidence: List[str] = field(default_factory=list)


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    activity_id: str
    status: ComplianceStatus
    confidence: float
    violations: List[Violation]
    explanation: str
    recommendations: List[str]
    regulations_checked: List[str]
    checked_at: datetime
    audit_id: str


@dataclass
class TradingActivity:
    """Trading activity to be checked for compliance."""
    id: str
    activity_type: ActivityType
    symbol: str
    quantity: float
    side: str  # 'buy' or 'sell'
    price: Optional[float] = None
    order_type: Optional[str] = None
    timestamp: datetime = field(default_factory=utc_now)
    account_type: str = "retail"
    jurisdiction: str = "US"
    strategy_id: Optional[str] = None
    leverage: float = 1.0  # For crypto
    metadata: Dict[str, Any] = field(default_factory=dict)

    def notional_value(self) -> Optional[float]:
        """Calculate notional value of the trade."""
        if self.price:
            return self.quantity * self.price
        return None

    def is_crypto(self) -> bool:
        """Check if this is a cryptocurrency symbol."""
        crypto_suffixes = ["USDT", "USD", "BTC", "ETH", "USDC", "PERP"]
        return any(self.symbol.upper().endswith(suffix) for suffix in crypto_suffixes)


class RegulationDatabase:
    """
    Simple regulation database for compliance checking.
    In production, this would be a vector database with RAG.
    """

    def __init__(self):
        self.regulations = self._load_regulations()

    def _load_regulations(self) -> Dict[str, Dict]:
        """Load regulation definitions."""
        return {
            "SEC_15C3_5": {
                "name": "SEC Rule 15c3-5 (Market Access Rule)",
                "jurisdiction": "US",
                "description": "Requires broker-dealers with market access to implement risk controls",
                "checks": [
                    "pre_trade_risk_controls",
                    "position_limits",
                    "order_size_limits",
                    "credit_limits"
                ],
                "thresholds": {
                    "single_order_limit_pct": 0.05,
                    "position_limit_pct": 0.10,
                }
            },
            "REG_SHO": {
                "name": "Regulation SHO",
                "jurisdiction": "US",
                "description": "Short selling regulations",
                "checks": [
                    "locate_requirement",
                    "close_out_requirement",
                    "threshold_securities"
                ],
                "thresholds": {
                    "settlement_days": 3,
                }
            },
            "FINRA_BEST_EXEC": {
                "name": "FINRA Best Execution",
                "jurisdiction": "US",
                "description": "Best execution requirements for customer orders",
                "checks": [
                    "price_improvement",
                    "execution_quality",
                    "routing_analysis"
                ]
            },
            "MIFID_II_ALGO": {
                "name": "MiFID II Algorithmic Trading",
                "jurisdiction": "EU",
                "description": "Requirements for algorithmic trading systems",
                "checks": [
                    "kill_switch",
                    "throttling",
                    "system_resilience",
                    "pre_trade_controls"
                ],
                "thresholds": {
                    "order_to_trade_ratio_max": 100,
                }
            },
            "BYBIT_LEVERAGE": {
                "name": "Bybit Leverage Limits",
                "jurisdiction": "CRYPTO",
                "description": "Bybit exchange leverage and position limits",
                "checks": [
                    "leverage_limit",
                    "position_limit",
                    "margin_requirement"
                ],
                "thresholds": {
                    "max_leverage": 100,
                    "default_max_leverage": 25,
                }
            },
            "BYBIT_POSITION": {
                "name": "Bybit Position Limits",
                "jurisdiction": "CRYPTO",
                "description": "Position size limits for Bybit contracts",
                "checks": [
                    "position_size",
                    "notional_limit",
                    "tier_validation"
                ]
            },
            "AML_KYC": {
                "name": "AML/KYC Requirements",
                "jurisdiction": "GLOBAL",
                "description": "Anti-money laundering and know your customer requirements",
                "checks": [
                    "kyc_verification",
                    "transaction_monitoring",
                    "suspicious_activity"
                ],
                "thresholds": {
                    "large_transaction_usd": 10000,
                }
            },
            "WASH_TRADING": {
                "name": "Wash Trading Prevention",
                "jurisdiction": "GLOBAL",
                "description": "Prohibition of wash trading and self-dealing",
                "checks": [
                    "self_matching_check",
                    "circular_trading_check",
                    "artificial_volume_check"
                ]
            }
        }

    def get_applicable_regulations(
        self,
        activity: TradingActivity,
        jurisdiction: str
    ) -> List[Dict]:
        """Get regulations applicable to a trading activity."""
        applicable = []

        for reg_id, reg in self.regulations.items():
            if reg["jurisdiction"] in [jurisdiction, "GLOBAL"]:
                applicable.append({"id": reg_id, **reg})
            elif reg["jurisdiction"] == "CRYPTO" and activity.is_crypto():
                applicable.append({"id": reg_id, **reg})

        return applicable


class RuleBasedPreChecker:
    """
    Fast rule-based pre-checks before LLM analysis.
    Catches obvious violations without LLM latency.
    """

    def __init__(
        self,
        max_leverage: float = 100.0,
        notional_limit: float = 1_000_000.0,
        position_limits: Optional[Dict[str, float]] = None
    ):
        self.max_leverage = max_leverage
        self.notional_limit = notional_limit
        self.position_limits = position_limits or {}

    def check(self, activity: TradingActivity) -> Optional[ComplianceResult]:
        """
        Perform fast rule-based pre-checks.

        Returns ComplianceResult if violations found, None otherwise.
        """
        violations = []

        # Check leverage limits (for crypto)
        if activity.is_crypto() and activity.leverage > self.max_leverage:
            violations.append(Violation(
                rule="LEVERAGE_LIMIT",
                description=f"Leverage {activity.leverage}x exceeds maximum allowed {self.max_leverage}x",
                severity=ViolationSeverity.HIGH,
                evidence=[f"Requested leverage: {activity.leverage}x"]
            ))

        # Check for negative quantities
        if activity.quantity <= 0:
            violations.append(Violation(
                rule="INVALID_QUANTITY",
                description="Quantity must be positive",
                severity=ViolationSeverity.CRITICAL,
                evidence=[f"Quantity: {activity.quantity}"]
            ))

        # Check notional value
        notional = activity.notional_value()
        if notional and notional > self.notional_limit:
            violations.append(Violation(
                rule="NOTIONAL_LIMIT",
                description=f"Notional value ${notional:,.2f} exceeds limit ${self.notional_limit:,.2f}",
                severity=ViolationSeverity.MEDIUM,
                evidence=[f"Notional: ${notional:,.2f}, Limit: ${self.notional_limit:,.2f}"]
            ))

        # Check position limits
        if activity.symbol in self.position_limits:
            limit = self.position_limits[activity.symbol]
            if activity.quantity > limit:
                violations.append(Violation(
                    rule="POSITION_LIMIT",
                    description=f"Position {activity.quantity} exceeds limit {limit} for {activity.symbol}",
                    severity=ViolationSeverity.HIGH,
                    evidence=[f"Requested: {activity.quantity}, Limit: {limit}"]
                ))

        if violations:
            has_critical = any(v.severity == ViolationSeverity.CRITICAL for v in violations)
            status = ComplianceStatus.REJECTED if has_critical else ComplianceStatus.REVIEW_REQUIRED

            return ComplianceResult(
                activity_id=activity.id,
                status=status,
                confidence=1.0,
                violations=violations,
                explanation="Pre-check violations detected",
                recommendations=["Review and correct the flagged issues"],
                regulations_checked=["INTERNAL_RULES"],
                checked_at=utc_now(),
                audit_id=self._generate_audit_id(activity)
            )

        return None

    def _generate_audit_id(self, activity: TradingActivity) -> str:
        """Generate unique audit ID for the compliance check."""
        data = f"{activity.id}{utc_now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class ComplianceMetrics:
    """Track compliance checking metrics."""

    def __init__(self):
        self.total_checks = 0
        self.approved = 0
        self.rejected = 0
        self.review_required = 0
        self.violations_by_rule: Dict[str, int] = {}
        self.total_latency_ms = 0.0

    def record(self, result: ComplianceResult, latency_ms: float = 0.0):
        """Record a compliance result."""
        self.total_checks += 1
        self.total_latency_ms += latency_ms

        if result.status == ComplianceStatus.APPROVED:
            self.approved += 1
        elif result.status == ComplianceStatus.REJECTED:
            self.rejected += 1
        elif result.status == ComplianceStatus.REVIEW_REQUIRED:
            self.review_required += 1

        for violation in result.violations:
            self.violations_by_rule[violation.rule] = \
                self.violations_by_rule.get(violation.rule, 0) + 1

    def approval_rate(self) -> float:
        """Calculate approval rate."""
        if self.total_checks == 0:
            return 0.0
        return self.approved / self.total_checks

    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_checks == 0:
            return 0.0
        return self.total_latency_ms / self.total_checks

    def summary(self) -> str:
        """Generate summary report."""
        top_violations = sorted(
            self.violations_by_rule.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        violations_str = "\n".join(
            f"  - {rule}: {count}" for rule, count in top_violations
        ) or "  (none)"

        return f"""
Compliance Metrics Summary
==========================
Total Checks: {self.total_checks}
Approved: {self.approved} ({self.approval_rate() * 100:.1f}%)
Rejected: {self.rejected}
Review Required: {self.review_required}
Average Latency: {self.avg_latency_ms():.1f}ms

Top Violations:
{violations_str}
"""


def create_sample_activities() -> List[TradingActivity]:
    """Create sample trading activities for demonstration."""
    import uuid

    return [
        # Normal stock order
        TradingActivity(
            id=str(uuid.uuid4())[:8],
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="AAPL",
            quantity=100,
            side="buy",
            price=175.0,
            order_type="limit",
            jurisdiction="US"
        ),
        # Large stock order
        TradingActivity(
            id=str(uuid.uuid4())[:8],
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="GOOGL",
            quantity=5000,
            side="buy",
            price=140.0,
            order_type="limit",
            jurisdiction="US"
        ),
        # Crypto order with moderate leverage
        TradingActivity(
            id=str(uuid.uuid4())[:8],
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="BTCUSDT",
            quantity=1.0,
            side="buy",
            price=45000.0,
            order_type="limit",
            leverage=10.0,
            jurisdiction="CRYPTO"
        ),
        # Crypto order with high leverage
        TradingActivity(
            id=str(uuid.uuid4())[:8],
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="ETHUSDT",
            quantity=10.0,
            side="buy",
            price=2500.0,
            order_type="market",
            leverage=50.0,
            jurisdiction="CRYPTO"
        ),
        # Invalid order (negative quantity)
        TradingActivity(
            id=str(uuid.uuid4())[:8],
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="AAPL",
            quantity=-100,
            side="buy",
            price=175.0,
            order_type="limit",
            jurisdiction="US"
        ),
        # Order exceeding leverage limit
        TradingActivity(
            id=str(uuid.uuid4())[:8],
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="BTCUSDT",
            quantity=0.5,
            side="buy",
            price=45000.0,
            leverage=150.0,
            jurisdiction="CRYPTO"
        ),
    ]


def print_result(activity: TradingActivity, result: ComplianceResult):
    """Pretty print a compliance result."""
    status_icons = {
        ComplianceStatus.APPROVED: "✅",
        ComplianceStatus.REJECTED: "🚫",
        ComplianceStatus.REVIEW_REQUIRED: "⚠️",
        ComplianceStatus.PENDING: "⏳",
    }

    severity_icons = {
        ViolationSeverity.LOW: "🟡",
        ViolationSeverity.MEDIUM: "🟠",
        ViolationSeverity.HIGH: "🔴",
        ViolationSeverity.CRITICAL: "⛔",
    }

    print(f"\n{'='*60}")
    print(f"Activity: {activity.side.upper()} {activity.quantity} {activity.symbol}")
    if activity.price:
        print(f"Price: ${activity.price:,.2f}")
    if activity.leverage > 1:
        print(f"Leverage: {activity.leverage}x")
    if activity.notional_value():
        print(f"Notional: ${activity.notional_value():,.2f}")

    print(f"\nStatus: {status_icons.get(result.status, '?')} {result.status.value}")
    print(f"Confidence: {result.confidence * 100:.1f}%")
    print(f"Audit ID: {result.audit_id}")

    if result.violations:
        print("\nViolations:")
        for v in result.violations:
            icon = severity_icons.get(v.severity, "?")
            print(f"  {icon} [{v.rule}] {v.description}")
            for evidence in v.evidence:
                print(f"      Evidence: {evidence}")

    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")

    print(f"\nRegulations Checked: {', '.join(result.regulations_checked)}")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("LLM Compliance Check - Python Demo")
    print("=" * 60)

    # Create pre-checker with configuration
    pre_checker = RuleBasedPreChecker(
        max_leverage=100.0,
        notional_limit=1_000_000.0,
        position_limits={
            "GME": 1000.0,
            "TSLA": 5000.0,
        }
    )

    # Create regulation database
    reg_db = RegulationDatabase()

    # Create metrics tracker
    metrics = ComplianceMetrics()

    # Get sample activities
    activities = create_sample_activities()

    print(f"\nChecking {len(activities)} trading activities...\n")

    for activity in activities:
        # Run pre-check
        result = pre_checker.check(activity)

        if result is None:
            # Activity passed pre-checks, simulate LLM approval
            regulations = reg_db.get_applicable_regulations(
                activity,
                activity.jurisdiction
            )

            result = ComplianceResult(
                activity_id=activity.id,
                status=ComplianceStatus.APPROVED,
                confidence=0.95,
                violations=[],
                explanation=f"Order to {activity.side} {activity.quantity} {activity.symbol} "
                           f"complies with applicable regulations.",
                recommendations=[],
                regulations_checked=[r["id"] for r in regulations],
                checked_at=utc_now(),
                audit_id=hashlib.sha256(activity.id.encode()).hexdigest()[:16]
            )

            # Add recommendations for large orders
            if activity.notional_value() and activity.notional_value() > 100_000:
                result.recommendations.append(
                    "Consider using algorithmic execution for large orders."
                )

        # Record metrics
        metrics.record(result, latency_ms=25.0)

        # Print result
        print_result(activity, result)

    # Print metrics summary
    print("\n" + "=" * 60)
    print(metrics.summary())

    # Display applicable regulations
    print("=" * 60)
    print("Available Regulations")
    print("=" * 60)

    for reg_id, reg in reg_db.regulations.items():
        print(f"\n📋 {reg['name']} ({reg_id})")
        print(f"   Jurisdiction: {reg['jurisdiction']}")
        print(f"   Description: {reg['description']}")
        print(f"   Checks: {', '.join(reg['checks'])}")


if __name__ == "__main__":
    main()
