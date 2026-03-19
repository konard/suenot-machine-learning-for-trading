use relation_extraction_finance::*;

fn main() {
    println!("=== Relation Extraction in Finance ===\n");

    // ── 1. Pattern-based extraction ──────────────────────────────
    println!("1. Pattern-Based Relation Extraction");
    println!("────────────────────────────────────\n");

    let extractor = PatternExtractor::new();
    let corpus = generate_synthetic_corpus();

    for (sentence, entities, _expected) in &corpus {
        let relations = extractor.extract(sentence, entities);
        if let Some(rel) = relations.first() {
            println!("  Sentence: {}", sentence);
            println!(
                "  Extracted: {} {} {} (confidence: {:.2})",
                rel.entity1.name,
                rel.relation_type.label(),
                rel.entity2.name,
                rel.confidence
            );
            println!();
        }
    }

    // ── 2. Build knowledge graph ─────────────────────────────────
    println!("2. Knowledge Graph Construction");
    println!("───────────────────────────────\n");

    let mut kg = KnowledgeGraph::new();
    for (sentence, entities, _) in &corpus {
        let relations = extractor.extract(sentence, entities);
        for rel in &relations {
            kg.add_relation(rel);
        }
    }

    println!("  Entities: {}", kg.entity_count());
    println!("  Relations: {}", kg.edge_count());
    println!();

    // Query the graph
    let companies = kg.entities_of_type(EntityType::Company);
    println!("  Companies in graph: {:?}", companies);

    let exchanges = kg.entities_of_type(EntityType::Exchange);
    println!("  Exchanges in graph: {:?}", exchanges);
    println!();

    // Find neighbors
    if !companies.is_empty() {
        let company = companies[0];
        let neighbors = kg.neighbors(company);
        println!("  Neighbors of '{}': {:?}", company, neighbors);
        println!();
    }

    // ── 3. Train relation scorer ─────────────────────────────────
    println!("3. Relation Scorer Training");
    println!("───────────────────────────\n");

    let data = generate_training_data(1000);
    let (train, test) = data.split_at(800);

    let mut scorer = RelationScorer::new(8, 0.01);
    let acc_before = scorer.accuracy(test);
    println!("  Accuracy before training: {:.2}%", acc_before * 100.0);

    scorer.train(&train.to_vec(), 100);
    let acc_after = scorer.accuracy(test);
    println!("  Accuracy after training:  {:.2}%", acc_after * 100.0);
    println!("  Improvement: {:.2}%", (acc_after - acc_before) * 100.0);
    println!();

    // ── 4. Feature extraction demo ───────────────────────────────
    println!("4. Feature Extraction Demo");
    println!("──────────────────────────\n");

    let sentence = "Microsoft acquired Activision Blizzard in a landmark deal.";
    let e1 = Entity::new("Microsoft", EntityType::Company, 0, 9);
    let e2 = Entity::new("Activision Blizzard", EntityType::Company, 19, 38);
    let features = extract_features(sentence, &e1, &e2, RelationType::Acquired, &extractor);

    println!("  Sentence: {}", sentence);
    println!("  Entity 1: {} ({:?})", e1.name, e1.entity_type);
    println!("  Entity 2: {} ({:?})", e2.name, e2.entity_type);
    println!("  Features:");
    println!("    Normalized distance:   {:.3}", features[0]);
    println!("    Type compatibility:    {:.3}", features[1]);
    println!("    Trigger word present:  {:.3}", features[2]);
    println!("    Between-context words: {:.3}", features[3]);
    println!("    Sentence length:       {:.3}", features[4]);
    println!("    Entity 1 position:     {:.3}", features[5]);
    println!("    Entity 2 position:     {:.3}", features[6]);
    println!("    Name overlap:          {:.3}", features[7]);
    println!();

    // Score with trained model
    let (is_valid, confidence) = scorer.predict(&features);
    println!(
        "  Scorer prediction: {} (confidence: {:.2}%)",
        if is_valid { "VALID" } else { "INVALID" },
        confidence * 100.0
    );
    println!();

    // ── 5. Crypto listing detection ──────────────────────────────
    println!("5. Crypto Listing Detection (Bybit)");
    println!("────────────────────────────────────\n");

    let listing_sentences = vec![
        (
            "SMARTAI token has been listed on Bybit for spot trading.",
            vec![
                Entity::new("SMARTAI", EntityType::Token, 0, 7),
                Entity::new("Bybit", EntityType::Exchange, 31, 36),
            ],
        ),
        (
            "Bybit announced the listing of NEWCOIN starting Monday.",
            vec![
                Entity::new("Bybit", EntityType::Exchange, 0, 5),
                Entity::new("NEWCOIN", EntityType::Token, 31, 38),
            ],
        ),
        (
            "ARB token debut on Bybit futures market.",
            vec![
                Entity::new("ARB", EntityType::Token, 0, 3),
                Entity::new("Bybit", EntityType::Exchange, 19, 24),
            ],
        ),
    ];

    for (sentence, entities) in &listing_sentences {
        let relations = extractor.extract(sentence, entities);
        for rel in &relations {
            println!("  Sentence: {}", sentence);
            println!(
                "  Signal: {} {} {} (confidence: {:.2})",
                rel.entity1.name,
                rel.relation_type.label(),
                rel.entity2.name,
                rel.confidence
            );

            // Map to trading pair
            let token = if rel.entity1.entity_type == EntityType::Token {
                &rel.entity1.name
            } else {
                &rel.entity2.name
            };
            println!("  Trading pair: {}USDT", token);
            println!("  Action: Monitor for listing-event price impact");
            println!();
        }
    }

    println!("=== Done ===");
}
