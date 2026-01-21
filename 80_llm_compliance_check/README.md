# Chapter 80: LLM Compliance Check

## Overview

LLM Compliance Check represents a critical advancement in algorithmic trading where Large Language Models (LLMs) automatically verify that trading strategies, orders, and market activities comply with regulatory requirements, internal risk policies, and ethical guidelines. Instead of relying solely on rule-based systems with hardcoded logic, LLMs can understand context, interpret complex regulations, and adapt to evolving compliance requirements.

This chapter explores how to build an LLM-powered compliance checking system that can analyze trading strategies across both traditional equity markets and cryptocurrency trading on platforms like Bybit.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Regulatory Landscape](#regulatory-landscape)
4. [LLM-Based Compliance Architecture](#llm-based-compliance-architecture)
5. [Compliance Check Categories](#compliance-check-categories)
6. [System Architecture](#system-architecture)
7. [Implementation Strategy](#implementation-strategy)
8. [Cryptocurrency Compliance](#cryptocurrency-compliance)
9. [Risk Assessment](#risk-assessment)
10. [Code Examples](#code-examples)
11. [References](#references)

---

## Introduction

### What is LLM Compliance Check?

An LLM Compliance Check system uses Large Language Models to automatically verify that trading activities meet regulatory requirements:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              LLM Compliance Check Overview                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional Compliance:              LLM-Based Compliance:            │
│   ┌──────────────────┐                ┌──────────────────┐              │
│   │ Rule-Based       │                │ Context-Aware    │              │
│   │ Checks           │                │ LLM Analysis     │              │
│   │      ↓           │                │      ↓           │              │
│   │ Binary Pass/Fail │                │ Nuanced          │              │
│   │      ↓           │                │ Assessment       │              │
│   │ Manual Review    │                │      ↓           │              │
│   │ Required         │                │ Explainable      │              │
│   │ (hours/days)     │                │ Recommendations  │              │
│   └──────────────────┘                │ (seconds)        │              │
│                                       └──────────────────┘              │
│                                                                          │
│   Key Capabilities:                                                      │
│   • Interpret complex regulatory text                                   │
│   • Understand strategy context and intent                              │
│   • Detect subtle compliance violations                                 │
│   • Adapt to new regulations without code changes                       │
│   • Generate explainable compliance reports                             │
│   • Multi-jurisdiction awareness                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Use LLMs for Compliance?

| Aspect | Traditional Compliance | LLM-Based Compliance |
|--------|----------------------|----------------------|
| Rule interpretation | Hardcoded logic | Context-aware understanding |
| New regulations | Code changes required | Prompt updates sufficient |
| Edge cases | Often missed | Contextual reasoning |
| Explanations | Generic error codes | Natural language reports |
| Multi-jurisdiction | Multiple codebases | Single model, multiple contexts |
| Adaptability | Slow iteration | Rapid adjustment |
| Audit trail | Structured logs | Human-readable narratives |

## Theoretical Foundation

### Compliance Framework

Trading compliance can be modeled as a function mapping trading activities to compliance outcomes:

$$C: \text{Trading Activity} \rightarrow \{Compliant, Non-Compliant, Review Required\}$$

The LLM extends this by adding:
- **Confidence scores**: $P(Compliant | Activity, Regulations)$
- **Explanations**: Natural language justification
- **Recommendations**: Suggested remediation steps

### Regulatory Text Understanding

LLMs process regulatory text through:

$$f_{LLM}: (R, A) \rightarrow (C, E, S)$$

Where:
- $R$ = Regulatory requirements (text)
- $A$ = Activity to check (trade details, strategy description)
- $C$ = Compliance classification
- $E$ = Explanation
- $S$ = Severity score

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM Compliance Processing Pipeline                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Trading Activity + Regulatory Context                           │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                │   │
│  │   "activity_type": "order_submission",                          │   │
│  │   "symbol": "AAPL",                                             │   │
│  │   "quantity": 50000,                                            │   │
│  │   "order_type": "market",                                       │   │
│  │   "side": "buy",                                                │   │
│  │   "timestamp": "2024-01-15T09:30:00Z",                          │   │
│  │   "account_type": "proprietary",                                │   │
│  │   "jurisdiction": "US"                                          │   │
│  │ }                                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  STEP 1: REGULATION RETRIEVAL (RAG)                                    │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • Retrieve relevant SEC regulations                             │   │
│  │ • Fetch FINRA rules                                             │   │
│  │ • Load internal compliance policies                             │   │
│  │ • Include recent regulatory updates                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  STEP 2: CONTEXT ASSEMBLY                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • Historical trading patterns for account                        │   │
│  │ • Current market conditions                                      │   │
│  │ • Position limits and thresholds                                 │   │
│  │ • Account restrictions and permissions                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  STEP 3: LLM ANALYSIS                                                  │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • Interpret regulations against activity                         │   │
│  │ • Identify potential violations                                  │   │
│  │ • Assess severity and intent                                     │   │
│  │ • Generate compliance recommendations                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  OUTPUT: Compliance Decision                                           │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                │   │
│  │   "status": "REVIEW_REQUIRED",                                  │   │
│  │   "confidence": 0.78,                                           │   │
│  │   "violations": [                                                │   │
│  │     {                                                            │   │
│  │       "rule": "SEC Rule 15c3-5",                                │   │
│  │       "description": "Large order may require pre-trade check", │   │
│  │       "severity": "MEDIUM"                                       │   │
│  │     }                                                            │   │
│  │   ],                                                             │   │
│  │   "explanation": "This 50,000 share market order exceeds...",   │   │
│  │   "recommendations": [                                           │   │
│  │     "Split order into smaller tranches",                        │   │
│  │     "Use TWAP algorithm to reduce market impact"                │   │
│  │   ]                                                              │   │
│  │ }                                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Regulatory Landscape

### Key Regulatory Frameworks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Global Regulatory Framework                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  UNITED STATES                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • SEC Rule 15c3-5 (Market Access Rule)                          │   │
│  │   - Pre-trade risk controls for broker-dealers                  │   │
│  │   - Position limits and order size checks                       │   │
│  │                                                                  │   │
│  │ • FINRA Rules                                                    │   │
│  │   - Best execution requirements                                  │   │
│  │   - Suitability obligations                                      │   │
│  │   - Supervision requirements                                     │   │
│  │                                                                  │   │
│  │ • Regulation SHO                                                 │   │
│  │   - Short selling rules                                          │   │
│  │   - Locate requirements                                          │   │
│  │                                                                  │   │
│  │ • Dodd-Frank Act                                                 │   │
│  │   - Volcker Rule (proprietary trading restrictions)             │   │
│  │   - Swap dealer regulations                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  EUROPEAN UNION                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • MiFID II / MiFIR                                               │   │
│  │   - Best execution policies                                      │   │
│  │   - Transaction reporting                                        │   │
│  │   - Algorithmic trading requirements                             │   │
│  │                                                                  │   │
│  │ • MAR (Market Abuse Regulation)                                  │   │
│  │   - Insider trading prohibitions                                 │   │
│  │   - Market manipulation detection                                │   │
│  │                                                                  │   │
│  │ • EU AI Act (2025-2026)                                          │   │
│  │   - AI system transparency requirements                          │   │
│  │   - High-risk AI system compliance                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  CRYPTOCURRENCY / DIGITAL ASSETS                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • AML/KYC Requirements                                           │   │
│  │   - Customer verification                                        │   │
│  │   - Transaction monitoring                                       │   │
│  │                                                                  │   │
│  │ • Travel Rule (FATF)                                             │   │
│  │   - Information sharing between exchanges                        │   │
│  │                                                                  │   │
│  │ • Exchange-Specific Rules (Bybit, Binance, etc.)                │   │
│  │   - Position limits                                              │   │
│  │   - Leverage restrictions                                        │   │
│  │   - Wash trading prohibitions                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Compliance Timeline 2025-2026

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Regulatory Compliance Timeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  2025                                                                   │
│  ────                                                                   │
│  • August 2025: EU AI Act - General purpose AI models compliance       │
│  • FINRA continues focus on GenAI and emerging technology risks        │
│  • SEC Regulation S-P amendments enforcement                            │
│                                                                          │
│  2026                                                                   │
│  ────                                                                   │
│  • August 2026: EU AI Act - High-risk AI systems (credit scoring)      │
│  • FINRA 2026 Report emphasizes cybersecurity and AI governance        │
│  • Continued SEC focus on fiduciary duty and custody rules             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## LLM-Based Compliance Architecture

### System Components

```rust
/// Core compliance check system components
#[derive(Debug, Clone)]
pub struct ComplianceChecker {
    /// LLM client for compliance analysis
    pub llm_client: LlmClient,

    /// Regulation database with RAG retrieval
    pub regulation_db: RegulationDatabase,

    /// Risk assessment engine
    pub risk_engine: RiskEngine,

    /// Configuration
    pub config: ComplianceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Jurisdiction(s) to check against
    pub jurisdictions: Vec<Jurisdiction>,

    /// Risk tolerance level
    pub risk_tolerance: RiskLevel,

    /// Confidence threshold for auto-approval
    pub auto_approve_threshold: f64,

    /// Whether to enable real-time checking
    pub real_time_enabled: bool,

    /// Audit logging level
    pub audit_level: AuditLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Jurisdiction {
    UnitedStates,
    EuropeanUnion,
    UnitedKingdom,
    Singapore,
    HongKong,
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Conservative,
    Moderate,
    Aggressive,
}
```

### Compliance Check Data Structures

```rust
/// Trading activity to be checked for compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingActivity {
    /// Unique identifier
    pub id: String,

    /// Type of activity
    pub activity_type: ActivityType,

    /// Trading symbol
    pub symbol: String,

    /// Quantity
    pub quantity: f64,

    /// Price (if applicable)
    pub price: Option<f64>,

    /// Order type
    pub order_type: Option<OrderType>,

    /// Trade side
    pub side: TradeSide,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Account information
    pub account: AccountInfo,

    /// Strategy identifier
    pub strategy_id: Option<String>,

    /// Additional context
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityType {
    OrderSubmission,
    OrderModification,
    OrderCancellation,
    TradeExecution,
    PositionChange,
    FundsTransfer,
    StrategyDeployment,
}

/// Compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Activity that was checked
    pub activity_id: String,

    /// Overall status
    pub status: ComplianceStatus,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// List of potential violations
    pub violations: Vec<Violation>,

    /// Natural language explanation
    pub explanation: String,

    /// Recommended actions
    pub recommendations: Vec<String>,

    /// Regulations checked
    pub regulations_checked: Vec<String>,

    /// Timestamp
    pub checked_at: DateTime<Utc>,

    /// Audit trail identifier
    pub audit_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Approved,
    Rejected,
    ReviewRequired,
    Pending,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// Rule or regulation violated
    pub rule: String,

    /// Description of violation
    pub description: String,

    /// Severity level
    pub severity: ViolationSeverity,

    /// Supporting evidence
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}
```

## Compliance Check Categories

### Pre-Trade Compliance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pre-Trade Compliance Checks                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. POSITION LIMITS                                                     │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Check against regulatory position limits                     │  │
│     │ • Verify internal risk limits are not breached                │  │
│     │ • Calculate concentration risk                                 │  │
│     │ • Assess portfolio-level exposure                              │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  2. ORDER SIZE VALIDATION                                               │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Verify order doesn't exceed single order limits             │  │
│     │ • Check market impact potential                                │  │
│     │ • Assess liquidity sufficiency                                 │  │
│     │ • Validate notional value limits                               │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  3. SHORT SELLING COMPLIANCE                                            │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Verify locate requirement satisfied (Reg SHO)               │  │
│     │ • Check for restricted securities                              │  │
│     │ • Validate uptick rule compliance                              │  │
│     │ • Confirm borrow availability                                  │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  4. ACCOUNT RESTRICTIONS                                                │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Verify account is authorized for instrument type            │  │
│     │ • Check for trading restrictions or freezes                   │  │
│     │ • Validate margin requirements                                 │  │
│     │ • Confirm account type permits strategy                        │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  5. STRATEGY COMPLIANCE                                                 │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Verify strategy is approved for account                     │  │
│     │ • Check algorithmic trading requirements (MiFID II)           │  │
│     │ • Validate kill switch functionality                           │  │
│     │ • Confirm risk parameter boundaries                            │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Post-Trade Compliance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Post-Trade Compliance Checks                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. BEST EXECUTION ANALYSIS                                             │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Compare execution price to benchmark (VWAP, arrival price)  │  │
│     │ • Analyze market conditions at execution time                 │  │
│     │ • Document execution quality metrics                          │  │
│     │ • Generate best execution reports                             │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  2. TRANSACTION REPORTING                                               │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Verify all required fields present                          │  │
│     │ • Check reporting deadlines                                    │  │
│     │ • Validate counterparty information                           │  │
│     │ • Ensure regulatory submission compliance                      │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  3. MARKET MANIPULATION DETECTION                                       │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Detect potential spoofing patterns                          │  │
│     │ • Identify layering activity                                   │  │
│     │ • Check for wash trading                                       │  │
│     │ • Analyze order-to-trade ratios                               │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  4. AUDIT TRAIL GENERATION                                              │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Create immutable compliance records                         │  │
│     │ • Store decision rationale with explanations                  │  │
│     │ • Link to source regulations                                   │  │
│     │ • Enable regulatory investigation support                      │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM Compliance Check Architecture                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌─────────────────┐                             │
│                         │  Trading System  │                             │
│                         └────────┬────────┘                             │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    COMPLIANCE GATEWAY                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │ Pre-Trade   │  │ Real-Time   │  │ Post-Trade  │               │  │
│  │  │ Checks      │  │ Monitoring  │  │ Analysis    │               │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │
│  └────────────────────────────┬──────────────────────────────────────┘  │
│                               │                                          │
│                               ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    LLM COMPLIANCE ENGINE                          │  │
│  │                                                                    │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐  │  │
│  │  │ Regulation RAG  │ →  │ LLM Analysis    │ →  │ Decision     │  │  │
│  │  │ (Vector DB)     │    │ (OpenAI/Claude) │    │ Engine       │  │  │
│  │  └─────────────────┘    └─────────────────┘    └──────────────┘  │  │
│  │          ↑                      ↑                     ↓           │  │
│  │  ┌───────────────┐      ┌───────────────┐    ┌──────────────┐   │  │
│  │  │ Regulation    │      │ Context       │    │ Audit Log    │   │  │
│  │  │ Updates       │      │ Assembler     │    │ Generator    │   │  │
│  │  └───────────────┘      └───────────────┘    └──────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    DATA STORES                                     │  │
│  │                                                                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │  │
│  │  │ Regulation  │  │ Position    │  │ Compliance  │               │  │
│  │  │ Database    │  │ Database    │  │ Audit Logs  │               │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Prompt Engineering for Compliance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Compliance Analysis Prompt Template                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SYSTEM PROMPT:                                                         │
│  """                                                                    │
│  You are a regulatory compliance expert specializing in trading and     │
│  financial markets. Your task is to analyze trading activities against  │
│  applicable regulations and determine compliance status.                │
│                                                                          │
│  When analyzing activities:                                             │
│  1. Consider all applicable regulations for the jurisdiction            │
│  2. Assess the intent and context of the activity                       │
│  3. Identify specific rule violations if any                            │
│  4. Provide clear, auditable explanations                               │
│  5. Suggest remediation steps when issues are found                     │
│                                                                          │
│  Key regulatory frameworks to consider:                                  │
│  - SEC Rules (US): 15c3-5, Regulation SHO, Regulation NMS               │
│  - FINRA Rules: Best execution, suitability, supervision                │
│  - MiFID II (EU): Best execution, algo trading requirements             │
│  - MAR (EU): Market abuse, insider trading                              │
│  - Exchange-specific rules                                              │
│                                                                          │
│  Output format:                                                         │
│  - Status: APPROVED / REJECTED / REVIEW_REQUIRED                        │
│  - Confidence: 0.0 to 1.0                                               │
│  - Violations: List of specific rule breaches                           │
│  - Explanation: Clear reasoning for the decision                        │
│  - Recommendations: Actionable steps to address issues                  │
│  """                                                                    │
│                                                                          │
│  USER PROMPT:                                                           │
│  """                                                                    │
│  Analyze the following trading activity for compliance:                 │
│                                                                          │
│  Activity Details:                                                      │
│  {activity_json}                                                        │
│                                                                          │
│  Account Information:                                                   │
│  {account_json}                                                         │
│                                                                          │
│  Current Positions:                                                     │
│  {positions_json}                                                       │
│                                                                          │
│  Applicable Regulations:                                                │
│  {regulations_text}                                                     │
│                                                                          │
│  Market Context:                                                        │
│  {market_context}                                                       │
│                                                                          │
│  Please provide your compliance assessment.                             │
│  """                                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### RAG-Based Regulation Retrieval

```rust
/// Regulation retrieval using RAG (Retrieval-Augmented Generation)
pub struct RegulationRetriever {
    /// Vector database for semantic search
    vector_db: VectorDatabase,

    /// Text embeddings model
    embedder: EmbeddingModel,

    /// Configuration
    config: RetrieverConfig,
}

impl RegulationRetriever {
    /// Retrieve relevant regulations for a trading activity
    pub async fn retrieve(
        &self,
        activity: &TradingActivity,
        jurisdiction: &Jurisdiction,
    ) -> Result<Vec<RegulationChunk>> {
        // Generate query from activity
        let query = self.build_query(activity);

        // Get embeddings
        let query_embedding = self.embedder.embed(&query).await?;

        // Search vector database
        let results = self.vector_db
            .search(query_embedding, self.config.top_k)
            .await?;

        // Filter by jurisdiction
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| r.metadata.jurisdiction == *jurisdiction)
            .collect();

        Ok(filtered)
    }

    fn build_query(&self, activity: &TradingActivity) -> String {
        format!(
            "{} trading {} {} {} shares",
            activity.activity_type,
            activity.side,
            activity.symbol,
            activity.quantity
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulationChunk {
    /// Regulation identifier
    pub regulation_id: String,

    /// Section or rule number
    pub section: String,

    /// Regulation text content
    pub content: String,

    /// Metadata
    pub metadata: RegulationMetadata,

    /// Relevance score
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulationMetadata {
    pub jurisdiction: Jurisdiction,
    pub effective_date: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub source: String,
    pub keywords: Vec<String>,
}
```

### Real-Time Compliance Checking

```rust
/// Real-time compliance checker
pub struct RealTimeComplianceChecker {
    /// LLM client
    llm: LlmClient,

    /// Regulation retriever
    retriever: RegulationRetriever,

    /// Position tracker
    positions: Arc<RwLock<PositionTracker>>,

    /// Configuration
    config: ComplianceConfig,
}

impl RealTimeComplianceChecker {
    /// Check a trading activity for compliance
    pub async fn check(&self, activity: TradingActivity) -> Result<ComplianceResult> {
        let start = std::time::Instant::now();

        // Step 1: Retrieve relevant regulations
        let regulations = self.retriever
            .retrieve(&activity, &self.config.jurisdictions[0])
            .await?;

        // Step 2: Get current position context
        let positions = self.positions.read().await;
        let position_context = positions.get_context(&activity.symbol);

        // Step 3: Build compliance prompt
        let prompt = self.build_compliance_prompt(
            &activity,
            &regulations,
            &position_context,
        );

        // Step 4: Get LLM analysis
        let analysis = self.llm.analyze(&prompt).await?;

        // Step 5: Parse response into structured result
        let result = self.parse_response(analysis, &activity)?;

        // Step 6: Log audit trail
        self.log_audit(&activity, &result, start.elapsed())?;

        Ok(result)
    }

    fn build_compliance_prompt(
        &self,
        activity: &TradingActivity,
        regulations: &[RegulationChunk],
        position_context: &PositionContext,
    ) -> String {
        let reg_text: String = regulations
            .iter()
            .map(|r| format!("[{}] {}: {}", r.regulation_id, r.section, r.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        format!(
            r#"
Analyze the following trading activity for compliance:

Activity Details:
- Type: {:?}
- Symbol: {}
- Quantity: {}
- Side: {:?}
- Order Type: {:?}
- Timestamp: {}

Account Information:
- Type: {}
- Jurisdiction: {:?}
- Restrictions: {:?}

Current Position:
- Existing Position: {} shares
- Max Position Limit: {} shares
- Concentration: {:.2}%

Applicable Regulations:
{}

Provide your compliance assessment in JSON format with:
- status: "APPROVED" | "REJECTED" | "REVIEW_REQUIRED"
- confidence: 0.0 to 1.0
- violations: array of {{ rule, description, severity }}
- explanation: string
- recommendations: array of strings
"#,
            activity.activity_type,
            activity.symbol,
            activity.quantity,
            activity.side,
            activity.order_type,
            activity.timestamp,
            activity.account.account_type,
            activity.account.jurisdiction,
            activity.account.restrictions,
            position_context.current_position,
            position_context.position_limit,
            position_context.concentration * 100.0,
            reg_text,
        )
    }
}
```

## Cryptocurrency Compliance

### Bybit-Specific Compliance Checks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Cryptocurrency Compliance Checks                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. LEVERAGE LIMITS                                                     │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Verify leverage within platform limits (e.g., 100x max)     │  │
│     │ • Check user-specific leverage restrictions                   │  │
│     │ • Validate margin requirements                                 │  │
│     │ • Monitor liquidation risk                                     │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  2. POSITION SIZE LIMITS                                                │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Check contract position limits                               │  │
│     │ • Verify notional value limits                                 │  │
│     │ • Monitor open interest contribution                           │  │
│     │ • Validate account tier permissions                            │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  3. WASH TRADING DETECTION                                              │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Detect self-matching orders                                  │  │
│     │ • Identify circular trading patterns                          │  │
│     │ • Monitor for artificial volume creation                      │  │
│     │ • Flag suspicious order patterns                               │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  4. AML/KYC COMPLIANCE                                                  │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Verify KYC status before large trades                       │  │
│     │ • Check withdrawal limits based on verification level         │  │
│     │ • Monitor for suspicious transaction patterns                 │  │
│     │ • Validate compliance with Travel Rule                        │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  5. MARKET MANIPULATION PREVENTION                                      │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ • Detect spoofing and layering                                │  │
│     │ • Identify pump-and-dump patterns                             │  │
│     │ • Monitor order book manipulation                              │  │
│     │ • Check for coordinated trading                                │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Crypto Compliance Data Structures

```rust
/// Cryptocurrency-specific compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoComplianceConfig {
    /// Exchange (e.g., Bybit)
    pub exchange: CryptoExchange,

    /// Maximum leverage allowed
    pub max_leverage: f64,

    /// Maximum position size (in contracts or base currency)
    pub max_position_size: f64,

    /// Wash trading detection sensitivity
    pub wash_trade_sensitivity: f64,

    /// KYC level required for trading
    pub min_kyc_level: KycLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CryptoExchange {
    Bybit,
    Binance,
    OKX,
    Coinbase,
    Kraken,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KycLevel {
    None,
    Basic,
    Intermediate,
    Advanced,
}

/// Crypto-specific activity data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoTradingActivity {
    /// Base trading activity
    #[serde(flatten)]
    pub base: TradingActivity,

    /// Leverage used
    pub leverage: f64,

    /// Contract type
    pub contract_type: ContractType,

    /// Margin mode
    pub margin_mode: MarginMode,

    /// Reduce only flag
    pub reduce_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContractType {
    Spot,
    LinearPerpetual,
    InversePerpetual,
    Futures,
    Options,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarginMode {
    Cross,
    Isolated,
}
```

## Risk Assessment

### LLM for Risk-Based Compliance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Risk-Based Compliance Assessment                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Risk Scoring Matrix:                                                   │
│                                                                          │
│  ┌────────────────┬─────────────┬──────────────┬──────────────┐        │
│  │   Factor       │   Low       │   Medium     │   High       │        │
│  ├────────────────┼─────────────┼──────────────┼──────────────┤        │
│  │ Order Size     │ < $100K     │ $100K-$1M    │ > $1M        │        │
│  │ Market Impact  │ < 0.1%      │ 0.1%-1%      │ > 1%         │        │
│  │ Position Conc. │ < 5%        │ 5%-15%       │ > 15%        │        │
│  │ Leverage       │ < 5x        │ 5x-20x       │ > 20x        │        │
│  │ Account Age    │ > 1 year    │ 3mo-1yr      │ < 3 months   │        │
│  │ Trading Freq.  │ < 10/day    │ 10-100/day   │ > 100/day    │        │
│  └────────────────┴─────────────┴──────────────┴──────────────┘        │
│                                                                          │
│  LLM Risk Assessment:                                                   │
│  ───────────────────                                                    │
│  • Combines quantitative risk factors                                   │
│  • Incorporates qualitative context (market conditions, news)           │
│  • Considers historical behavior patterns                               │
│  • Adapts thresholds based on regulatory changes                        │
│                                                                          │
│  Output: Risk Score (0-100) + Detailed Breakdown                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Compliance Metrics

```rust
/// Compliance metrics for monitoring and reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMetrics {
    /// Total activities checked
    pub total_checks: u64,

    /// Auto-approved count
    pub auto_approved: u64,

    /// Rejected count
    pub rejected: u64,

    /// Manual review required count
    pub review_required: u64,

    /// Average confidence score
    pub avg_confidence: f64,

    /// Average check latency (milliseconds)
    pub avg_latency_ms: f64,

    /// Violations by severity
    pub violations_by_severity: HashMap<String, u64>,

    /// Violations by rule
    pub violations_by_rule: HashMap<String, u64>,

    /// Period start
    pub period_start: DateTime<Utc>,

    /// Period end
    pub period_end: DateTime<Utc>,
}

impl ComplianceMetrics {
    /// Calculate approval rate
    pub fn approval_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.auto_approved as f64 / self.total_checks as f64
    }

    /// Calculate rejection rate
    pub fn rejection_rate(&self) -> f64 {
        if self.total_checks == 0 {
            return 0.0;
        }
        self.rejected as f64 / self.total_checks as f64
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            r#"
Compliance Summary ({} to {})
================================
Total Checks: {}
Auto-Approved: {} ({:.1}%)
Rejected: {} ({:.1}%)
Review Required: {} ({:.1}%)
Average Confidence: {:.2}
Average Latency: {:.1}ms

Top Violations:
{}
"#,
            self.period_start.format("%Y-%m-%d"),
            self.period_end.format("%Y-%m-%d"),
            self.total_checks,
            self.auto_approved,
            self.approval_rate() * 100.0,
            self.rejected,
            self.rejection_rate() * 100.0,
            self.review_required,
            (self.review_required as f64 / self.total_checks as f64) * 100.0,
            self.avg_confidence,
            self.avg_latency_ms,
            self.format_top_violations(),
        )
    }

    fn format_top_violations(&self) -> String {
        let mut violations: Vec<_> = self.violations_by_rule.iter().collect();
        violations.sort_by(|a, b| b.1.cmp(a.1));

        violations
            .iter()
            .take(5)
            .map(|(rule, count)| format!("  - {}: {}", rule, count))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
```

## Code Examples

### Python Implementation

```python
"""
LLM Compliance Check - Python Implementation

This module provides LLM-based compliance checking for trading activities
using both stock market and cryptocurrency (Bybit) data.
"""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib

import httpx
import numpy as np


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
    timestamp: datetime = field(default_factory=datetime.utcnow)
    account_type: str = "retail"
    jurisdiction: str = "US"
    strategy_id: Optional[str] = None
    leverage: float = 1.0  # For crypto
    metadata: Dict[str, Any] = field(default_factory=dict)


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
                    "single_order_limit_pct": 0.05,  # 5% of ADV
                    "position_limit_pct": 0.10,  # 10% of market cap
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
            # Check jurisdiction match
            if reg["jurisdiction"] in [jurisdiction, "GLOBAL"]:
                applicable.append({"id": reg_id, **reg})
            # Add crypto regulations for crypto symbols
            elif reg["jurisdiction"] == "CRYPTO" and self._is_crypto_symbol(activity.symbol):
                applicable.append({"id": reg_id, **reg})

        return applicable

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency."""
        crypto_suffixes = ["USDT", "USD", "BTC", "ETH", "USDC"]
        return any(symbol.upper().endswith(suffix) for suffix in crypto_suffixes)


class LLMComplianceChecker:
    """
    LLM-based compliance checker for trading activities.

    Uses Large Language Models to analyze trading activities against
    regulatory requirements and internal compliance policies.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: str = "https://api.openai.com/v1",
        auto_approve_threshold: float = 0.95
    ):
        """
        Initialize the compliance checker.

        Args:
            api_key: API key for LLM provider
            model: Model to use for compliance analysis
            base_url: Base URL for API
            auto_approve_threshold: Confidence threshold for auto-approval
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.auto_approve_threshold = auto_approve_threshold
        self.regulation_db = RegulationDatabase()
        self.client = httpx.AsyncClient(timeout=60.0)

    async def check(self, activity: TradingActivity) -> ComplianceResult:
        """
        Check a trading activity for compliance.

        Args:
            activity: Trading activity to check

        Returns:
            ComplianceResult with status, violations, and recommendations
        """
        # Get applicable regulations
        regulations = self.regulation_db.get_applicable_regulations(
            activity,
            activity.jurisdiction
        )

        # Build compliance prompt
        prompt = self._build_compliance_prompt(activity, regulations)

        # Get LLM analysis
        response = await self._call_llm(prompt)

        # Parse response
        result = self._parse_response(response, activity, regulations)

        return result

    def _build_compliance_prompt(
        self,
        activity: TradingActivity,
        regulations: List[Dict]
    ) -> str:
        """Build the compliance analysis prompt."""

        reg_text = "\n".join([
            f"- {r['id']}: {r['name']}\n  {r['description']}\n  Checks: {', '.join(r['checks'])}"
            for r in regulations
        ])

        return f"""You are a regulatory compliance expert. Analyze the following trading activity for compliance.

TRADING ACTIVITY:
- ID: {activity.id}
- Type: {activity.activity_type.value}
- Symbol: {activity.symbol}
- Side: {activity.side}
- Quantity: {activity.quantity}
- Price: {activity.price}
- Order Type: {activity.order_type}
- Timestamp: {activity.timestamp.isoformat()}
- Account Type: {activity.account_type}
- Jurisdiction: {activity.jurisdiction}
- Leverage: {activity.leverage}x
- Strategy: {activity.strategy_id or 'N/A'}

APPLICABLE REGULATIONS:
{reg_text}

Analyze this activity and provide your assessment in the following JSON format:
{{
    "status": "APPROVED" | "REJECTED" | "REVIEW_REQUIRED",
    "confidence": 0.0 to 1.0,
    "violations": [
        {{
            "rule": "regulation identifier",
            "description": "what was violated",
            "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        }}
    ],
    "explanation": "detailed explanation of your assessment",
    "recommendations": ["list of recommended actions"]
}}

Consider:
1. Pre-trade risk controls and position limits
2. Order size relative to typical market volume
3. Leverage and margin requirements (for crypto)
4. Potential market manipulation indicators
5. KYC/AML concerns for large transactions

Respond ONLY with the JSON object."""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a regulatory compliance expert for trading systems. Provide precise, auditable compliance assessments."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent compliance decisions
            "max_tokens": 1000
        }

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _parse_response(
        self,
        response: str,
        activity: TradingActivity,
        regulations: List[Dict]
    ) -> ComplianceResult:
        """Parse LLM response into ComplianceResult."""

        try:
            # Extract JSON from response
            data = json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a review-required result
            return ComplianceResult(
                activity_id=activity.id,
                status=ComplianceStatus.REVIEW_REQUIRED,
                confidence=0.0,
                violations=[],
                explanation="Failed to parse LLM response. Manual review required.",
                recommendations=["Review activity manually"],
                regulations_checked=[r["id"] for r in regulations],
                checked_at=datetime.utcnow(),
                audit_id=self._generate_audit_id(activity)
            )

        # Parse violations
        violations = []
        for v in data.get("violations", []):
            violations.append(Violation(
                rule=v.get("rule", "UNKNOWN"),
                description=v.get("description", ""),
                severity=ViolationSeverity[v.get("severity", "MEDIUM")],
                evidence=v.get("evidence", [])
            ))

        # Map status
        status_map = {
            "APPROVED": ComplianceStatus.APPROVED,
            "REJECTED": ComplianceStatus.REJECTED,
            "REVIEW_REQUIRED": ComplianceStatus.REVIEW_REQUIRED
        }
        status = status_map.get(
            data.get("status", "REVIEW_REQUIRED"),
            ComplianceStatus.REVIEW_REQUIRED
        )

        # Apply auto-approve threshold
        confidence = data.get("confidence", 0.5)
        if status == ComplianceStatus.APPROVED and confidence < self.auto_approve_threshold:
            status = ComplianceStatus.REVIEW_REQUIRED

        return ComplianceResult(
            activity_id=activity.id,
            status=status,
            confidence=confidence,
            violations=violations,
            explanation=data.get("explanation", ""),
            recommendations=data.get("recommendations", []),
            regulations_checked=[r["id"] for r in regulations],
            checked_at=datetime.utcnow(),
            audit_id=self._generate_audit_id(activity)
        )

    def _generate_audit_id(self, activity: TradingActivity) -> str:
        """Generate unique audit ID for the compliance check."""
        data = f"{activity.id}{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class RuleBasedPreCheck:
    """
    Fast rule-based pre-checks before LLM analysis.
    Catches obvious violations without LLM latency.
    """

    def __init__(self):
        self.position_limits = {
            "default": 1_000_000,  # $1M default
            "crypto": 100_000,  # $100K for crypto
        }
        self.leverage_limits = {
            "default": 1.0,
            "crypto": 100.0,
            "margin": 4.0,
        }

    def check(self, activity: TradingActivity) -> Optional[ComplianceResult]:
        """
        Perform fast rule-based pre-checks.

        Returns ComplianceResult if obvious violation found, None otherwise.
        """
        violations = []

        # Check leverage limits
        if activity.leverage > self.leverage_limits.get(
            "crypto" if self._is_crypto(activity.symbol) else "default",
            1.0
        ):
            violations.append(Violation(
                rule="LEVERAGE_LIMIT",
                description=f"Leverage {activity.leverage}x exceeds limit",
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
        if activity.price:
            notional = activity.quantity * activity.price
            limit = self.position_limits.get(
                "crypto" if self._is_crypto(activity.symbol) else "default"
            )
            if notional > limit:
                violations.append(Violation(
                    rule="NOTIONAL_LIMIT",
                    description=f"Notional value ${notional:,.2f} may require review",
                    severity=ViolationSeverity.MEDIUM,
                    evidence=[f"Notional: ${notional:,.2f}, Limit: ${limit:,.2f}"]
                ))

        if violations:
            # Return early rejection for critical violations
            has_critical = any(v.severity == ViolationSeverity.CRITICAL for v in violations)
            return ComplianceResult(
                activity_id=activity.id,
                status=ComplianceStatus.REJECTED if has_critical else ComplianceStatus.REVIEW_REQUIRED,
                confidence=1.0,
                violations=violations,
                explanation="Pre-check violations detected",
                recommendations=["Review and correct the flagged issues"],
                regulations_checked=["INTERNAL_RULES"],
                checked_at=datetime.utcnow(),
                audit_id=hashlib.sha256(activity.id.encode()).hexdigest()[:16]
            )

        return None

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is cryptocurrency."""
        crypto_suffixes = ["USDT", "USD", "BTC", "ETH", "USDC", "PERP"]
        return any(symbol.upper().endswith(suffix) for suffix in crypto_suffixes)


class ComplianceCheckPipeline:
    """
    Complete compliance checking pipeline with pre-checks and LLM analysis.
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.pre_checker = RuleBasedPreCheck()
        self.llm_checker = LLMComplianceChecker(api_key, model)

    async def check(self, activity: TradingActivity) -> ComplianceResult:
        """
        Run complete compliance check pipeline.

        1. Fast rule-based pre-checks
        2. LLM-based deep analysis (if pre-checks pass)
        """
        # Step 1: Pre-checks
        pre_result = self.pre_checker.check(activity)
        if pre_result and pre_result.status == ComplianceStatus.REJECTED:
            return pre_result

        # Step 2: LLM analysis
        llm_result = await self.llm_checker.check(activity)

        # Merge pre-check violations if any
        if pre_result:
            llm_result.violations.extend(pre_result.violations)
            if llm_result.status == ComplianceStatus.APPROVED:
                llm_result.status = ComplianceStatus.REVIEW_REQUIRED

        return llm_result

    async def close(self):
        """Close resources."""
        await self.llm_checker.close()


# Example usage
async def main():
    """Example usage of the compliance checker."""

    # Create sample trading activities
    activities = [
        # Normal stock order
        TradingActivity(
            id="order_001",
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="AAPL",
            quantity=100,
            side="buy",
            price=150.0,
            order_type="limit",
            jurisdiction="US"
        ),
        # Large crypto order with high leverage
        TradingActivity(
            id="order_002",
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="BTCUSDT",
            quantity=10.0,
            side="buy",
            price=45000.0,
            order_type="market",
            leverage=50.0,
            jurisdiction="CRYPTO"
        ),
        # Suspicious order pattern
        TradingActivity(
            id="order_003",
            activity_type=ActivityType.ORDER_SUBMISSION,
            symbol="GME",
            quantity=100000,
            side="buy",
            price=25.0,
            order_type="market",
            jurisdiction="US"
        ),
    ]

    # Note: In production, use actual API key
    print("LLM Compliance Checker Demo")
    print("=" * 50)
    print("\nNote: This demo shows the structure without actual LLM calls.")
    print("In production, provide a valid API key.\n")

    # Demo pre-checks only (no API key needed)
    pre_checker = RuleBasedPreCheck()

    for activity in activities:
        print(f"\nChecking activity: {activity.id}")
        print(f"  Symbol: {activity.symbol}")
        print(f"  Type: {activity.activity_type.value}")
        print(f"  Quantity: {activity.quantity}")
        print(f"  Leverage: {activity.leverage}x")

        result = pre_checker.check(activity)
        if result:
            print(f"  Pre-check Status: {result.status.value}")
            for v in result.violations:
                print(f"    - {v.rule}: {v.description} ({v.severity.value})")
        else:
            print("  Pre-check Status: PASSED (would proceed to LLM analysis)")


if __name__ == "__main__":
    asyncio.run(main())
```

### Rust Implementation

The complete Rust implementation is available in the `src/` directory with the following modules:

- `lib.rs` - Main library entry point
- `compliance/` - Core compliance checking logic
- `regulations/` - Regulation database and RAG
- `data/` - Data structures and fetching
- `reports/` - Report generation

## References

### Academic Papers

1. **LLMs for Regulatory Compliance in Finance**
   - URL: https://arxiv.org/abs/2402.01758
   - Year: 2024

2. **Responsible Innovation: A Strategic Framework for Financial LLM Integration**
   - URL: https://arxiv.org/pdf/2504.02165
   - Year: 2025

### Regulatory Resources

1. **FINRA 2026 Regulatory Oversight Report**
   - Source: FINRA
   - Focus: GenAI risks, cybersecurity, Reg BI compliance

2. **SEC 2026 Examination Priorities**
   - Source: SEC Division of Examinations
   - Focus: Fiduciary duty, custody rules, cybersecurity

3. **EU AI Act**
   - Effective: August 2025 (general AI), August 2026 (high-risk)
   - Requirements: Transparency, risk management, human oversight

### Industry Resources

1. **Best Open Source LLMs for Finance** - SiliconFlow
2. **How Financial Services Can Harness LLMs Safely** - FinTech Magazine
3. **AML and Financial Crime Compliance Trends** - Silent Eight

### Exchange Documentation

1. **Bybit API Documentation** - https://bybit-exchange.github.io/docs/
2. **Binance API Documentation** - https://binance-docs.github.io/apidocs/

---

## Summary

LLM Compliance Check systems represent a paradigm shift in regulatory compliance for trading:

1. **Context-Aware Analysis**: LLMs understand regulatory nuance and trading context
2. **Adaptable**: Update compliance rules through prompts, not code changes
3. **Explainable**: Generate human-readable compliance reports and audit trails
4. **Multi-Jurisdiction**: Single model handles multiple regulatory frameworks
5. **Real-Time Capable**: Fast enough for pre-trade compliance checks

Key implementation considerations:
- Combine fast rule-based pre-checks with deep LLM analysis
- Use RAG for relevant regulation retrieval
- Maintain comprehensive audit trails
- Implement confidence thresholds for auto-approval vs. manual review
- Regular validation against regulatory updates

The combination of traditional rule-based systems with LLM intelligence provides the best of both worlds: speed and interpretability of rules with the adaptability and nuanced understanding of LLMs.
