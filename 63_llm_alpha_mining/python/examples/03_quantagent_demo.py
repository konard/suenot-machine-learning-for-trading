#!/usr/bin/env python3
"""
Example 03: QuantAgent Demo

This example demonstrates the self-improving QuantAgent that learns
from experience to generate better alpha factors over time.

Run with: python 03_quantagent_demo.py
"""

import sys
sys.path.insert(0, '..')

from data_loader import generate_synthetic_data
from quantagent import QuantAgent, KnowledgeBase, Experience
from datetime import datetime


def main():
    print("=" * 60)
    print("LLM Alpha Mining - QuantAgent Demo")
    print("=" * 60)

    # Load data
    print("\n1. LOADING DATA")
    print("-" * 40)

    data = generate_synthetic_data(["BTCUSDT"], days=300, seed=42)
    btc_data = data["BTCUSDT"].ohlcv
    print(f"Loaded {len(btc_data)} records for BTCUSDT")

    # Initialize QuantAgent
    print("\n2. INITIALIZING QUANTAGENT")
    print("-" * 40)

    agent = QuantAgent(
        quality_threshold=30.0,  # Lower threshold for demo
        model="mock"  # Use mock LLM for demo
    )

    # Classify market condition
    market_condition = agent.classify_market(btc_data)
    print(f"Current market condition: {market_condition}")

    # Run mining
    print("\n3. RUNNING ALPHA MINING")
    print("-" * 40)

    successful_factors = agent.mine(
        btc_data,
        symbol="BTCUSDT",
        n_iterations=5,
        factors_per_iteration=3,
        verbose=True
    )

    # Display results
    print("\n4. MINING RESULTS")
    print("-" * 40)

    print(f"\nTotal successful factors: {len(successful_factors)}")

    if successful_factors:
        print("\nTop factors found:")
        # Sort by quality score
        sorted_factors = sorted(
            successful_factors,
            key=lambda x: x["metrics"].quality_score(),
            reverse=True
        )

        for i, result in enumerate(sorted_factors[:5], 1):
            factor = result["factor"]
            metrics = result["metrics"]
            print(f"\n  {i}. {factor.name}")
            print(f"     Expression: {factor.expression}")
            print(f"     IC: {metrics.ic:.4f}")
            print(f"     Sharpe: {metrics.sharpe_ratio:.2f}")
            print(f"     Quality Score: {metrics.quality_score():.1f}/100")
            print(f"     Iteration found: {result['iteration']}")

    # Knowledge base analysis
    print("\n5. KNOWLEDGE BASE ANALYSIS")
    print("-" * 40)

    summary = agent.kb.summary()
    print(f"\nTotal experiences: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Success rate: {summary['success_rate']:.1%}")

    if summary.get("best_patterns"):
        print("\nBest performing patterns:")
        for pattern, score in summary["best_patterns"][:5]:
            print(f"  - {pattern}: {score:.1%} success rate")

    if summary.get("avoid_patterns"):
        print("\nPatterns to avoid:")
        for pattern, score in summary["avoid_patterns"][:3]:
            print(f"  - {pattern}: {score:.1%} failure rate")

    # Get recommendations
    print("\n6. FACTOR RECOMMENDATIONS")
    print("-" * 40)

    recommendations = agent.get_recommendations(btc_data, n_recommendations=5)

    if recommendations:
        print("\nRecommended factors for current market:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. {rec.factor_name}")
            print(f"     Expression: {rec.factor_expression}")
            print(f"     Historical IC: {rec.metrics.get('ic', 0):.4f}")
            print(f"     Market: {rec.market_condition}")
    else:
        print("\n  No recommendations available (need more experience)")

    # Manual experience addition
    print("\n7. MANUAL KNOWLEDGE BASE USAGE")
    print("-" * 40)

    # Create and add a manual experience
    manual_exp = Experience(
        factor_expression="ts_mean(close, 10) / ts_mean(close, 50) - 1",
        factor_name="golden_cross_lite",
        metrics={"ic": 0.08, "sharpe": 1.5, "max_drawdown": -0.12},
        market_condition="bullish",
        success=True,
        notes="Manual addition: works well in trending markets"
    )

    added = agent.kb.add_experience(manual_exp)
    print(f"\nManually added experience: {added}")

    # Query knowledge base
    print("\nQuerying for 'mean' factors:")
    mean_factors = agent.kb.query(keyword="mean", limit=3)
    for exp in mean_factors:
        print(f"  - {exp.factor_name}: IC={exp.metrics.get('ic', 0):.4f}")

    # Factor refinement
    print("\n8. FACTOR REFINEMENT")
    print("-" * 40)

    if successful_factors:
        best_factor = sorted_factors[0]["factor"]
        print(f"\nRefining best factor: {best_factor.name}")
        print(f"Original expression: {best_factor.expression}")

        refined = agent.refine_factor(
            best_factor.expression,
            btc_data,
            n_variations=3
        )

        print("\nRefined variations:")
        for i, result in enumerate(refined[:3], 1):
            print(f"  {i}. {result['factor'].expression}")
            print(f"     Quality: {result['improvement']:.1f}")

    # Export knowledge base
    print("\n9. KNOWLEDGE BASE EXPORT")
    print("-" * 40)

    kb_json = agent.kb.to_json()
    print(f"\nKnowledge base exported ({len(kb_json)} bytes)")
    print("Can be saved to file and reloaded later:")
    print("  with open('kb.json', 'w') as f: f.write(kb_json)")
    print("  kb = KnowledgeBase.from_json(open('kb.json').read())")

    # Simulate learning over time
    print("\n10. LEARNING SIMULATION")
    print("-" * 40)

    print("\nSimulating multiple mining sessions...")

    # Create fresh agent for simulation
    sim_agent = QuantAgent(quality_threshold=30.0, model="mock")

    success_rates = []
    for session in range(3):
        results = sim_agent.mine(
            btc_data,
            symbol="BTCUSDT",
            n_iterations=3,
            factors_per_iteration=3,
            verbose=False
        )

        success_rate = len(results) / 9  # 3 iterations * 3 factors
        success_rates.append(success_rate)

        summary = sim_agent.kb.summary()
        print(f"  Session {session + 1}: Success rate={success_rate:.1%}, "
              f"KB size={summary['total']}")

    print(f"\nLearning trend: {' -> '.join([f'{r:.0%}' for r in success_rates])}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
