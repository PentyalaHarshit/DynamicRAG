"""
Test Suite for OmniKnowledge Quiz Agent
=======================================
Validates:
- Query parser for multiple-choice quiz formats (with/without progress indicators).
- Top 10 document chunking & Question + Option-Aware Top 3 reranking.
- Dual-path Option Evidence Matching (SUPPORTED / CONTRADICTED / UNSUPPORTED stance matrix).
- Contradiction & Uniqueness checking.
- Question Quality Score (QQS) evaluation engine.
- Evidence-First Question Generation & Traceability.
- User Answer Evaluation, Weak Concept Detection & Knowledge Graph.
- DQN Option Selector feature extraction & Q-value ranking.
- Answer validation & self-correction retry mechanism.
- End-to-end solve_quiz_query and run_pipeline integration.
"""

import pytest
from agents.quiz_agent import (
    parse_quiz_query,
    is_quiz_query,
    chunk_retrieved_documents,
    rerank_chunks_option_aware,
    evaluate_option_evidence_matching,
    check_contradiction_and_uniqueness,
    calculate_question_quality_score,
    generate_evidence_grounded_quiz,
    evaluate_user_quiz_answer,
    UserKnowledgeGraph,
    DQNQuizOptionSelector,
    validate_quiz_answer,
    solve_quiz_query,
    OptionState,
    OptionEvidenceMatch,
    GeneratedQuizQuestion,
    EvidenceSnippet
)
from agents.search_tool import SearchResult
from graph import run_pipeline


def test_parse_quiz_query_standard():
    query = "What does AI stand for? 1 of 5 A Automated Information B Applied Interface C Advanced Internet D Artificial Intelligence"
    parsed = parse_quiz_query(query)
    assert parsed is not None
    assert parsed.question == "What does AI stand for?"
    assert parsed.progress == "1 of 5"
    assert parsed.current_index == 1
    assert parsed.total_count == 5
    assert len(parsed.options) == 4
    assert parsed.options["A"] == "Automated Information"
    assert parsed.options["B"] == "Applied Interface"
    assert parsed.options["C"] == "Advanced Internet"
    assert parsed.options["D"] == "Artificial Intelligence"
    assert is_quiz_query(query) is True


def test_parse_quiz_query_parentheses_and_colons():
    query = "Question 3 of 10: Which planet is known as the Red Planet? A) Venus B) Mars C) Jupiter D) Saturn"
    parsed = parse_quiz_query(query)
    assert parsed is not None
    assert "Red Planet" in parsed.question
    assert parsed.progress == "3 of 10"
    assert parsed.options["A"] == "Venus"
    assert parsed.options["B"] == "Mars"
    assert parsed.options["C"] == "Jupiter"
    assert parsed.options["D"] == "Saturn"


def test_parse_quiz_query_brackets():
    query = "What is the capital of France? [A] Berlin [B] Madrid [C] Paris [D] Rome"
    parsed = parse_quiz_query(query)
    assert parsed is not None
    assert parsed.question == "What is the capital of France?"
    assert parsed.options["C"] == "Paris"


def test_chunk_retrieved_documents():
    mock_results = [
        SearchResult(title="Database Isolation", link="http://db.org", snippet="Read Committed prevents dirty reads. It is the default in PostgreSQL. Read Uncommitted allows dirty reads."),
        SearchResult(title="SQL Standard", link="http://sql.org", snippet="Serializable provides the highest isolation. Repeatable Read prevents non-repeatable reads.")
    ]
    chunks = chunk_retrieved_documents(mock_results, max_chunks=10)
    assert len(chunks) >= 1
    combined_chunks = " ".join(chunks)
    assert "Read Committed prevents dirty reads" in combined_chunks


def test_rerank_chunks_option_aware():
    question = "Which database isolation level prevents dirty reads?"
    options = {
        "A": "Read Uncommitted",
        "B": "Read Committed",
        "C": "NoSQL",
        "D": "Eventual Consistency"
    }
    candidate_chunks = [
        "NoSQL databases often emphasize horizontal scaling.",
        "Read Committed prevents dirty reads by reading only committed data.",
        "Read Uncommitted allows dirty reads where transactions read uncommitted changes.",
        "Eventual consistency is common in distributed key-value stores.",
        "Database indexes improve query performance on large tables."
    ]
    top3 = rerank_chunks_option_aware(question, options, candidate_chunks, top_k=3)
    assert len(top3) == 3
    combined_top3 = " ".join(top3)
    assert "Read Committed" in combined_top3
    assert "Read Uncommitted" in combined_top3


def test_option_evidence_matching_matrix():
    question = "Which database isolation level prevents dirty reads?"
    options = {
        "A": "Read Uncommitted",
        "B": "Read Committed",
        "C": "NoSQL",
        "D": "Eventual Consistency"
    }
    top_chunks = [
        "Read Committed prevents dirty reads by ensuring only committed data is read.",
        "Read Uncommitted allows dirty reads and does not prevent them.",
        "Serializable isolation prevents phantom reads and serialization anomalies."
    ]
    matches = evaluate_option_evidence_matching(question, options, top_chunks)
    assert "B" in matches and "A" in matches and "C" in matches
    assert matches["B"].stance == "SUPPORTED"
    assert "Read Committed prevents dirty reads" in matches["B"].evidence_span
    assert matches["C"].stance == "UNSUPPORTED"


def test_contradiction_and_uniqueness_checker():
    matches_unique = {
        "A": OptionEvidenceMatch("A", "Read Uncommitted", "CONTRADICTED", 0.1, "Allows dirty reads", 0),
        "B": OptionEvidenceMatch("B", "Read Committed", "SUPPORTED", 0.95, "Prevents dirty reads", 0),
        "C": OptionEvidenceMatch("C", "NoSQL", "UNSUPPORTED", 0.0, "", 0),
        "D": OptionEvidenceMatch("D", "Eventual Consistency", "UNSUPPORTED", 0.0, "", 0),
    }
    is_valid, reason, opt = check_contradiction_and_uniqueness(matches_unique)
    assert is_valid is True
    assert opt == "B"

    # Ambiguous test case (multiple supported)
    matches_ambiguous = {
        "A": OptionEvidenceMatch("A", "Read Committed", "SUPPORTED", 0.95, "Prevents dirty reads", 0),
        "B": OptionEvidenceMatch("B", "Serializable", "SUPPORTED", 0.95, "Also prevents dirty reads", 0),
    }
    is_valid_amb, reason_amb, _ = check_contradiction_and_uniqueness(matches_ambiguous)
    assert is_valid_amb is False
    assert "Multiple options" in reason_amb


def test_question_quality_scoring_engine():
    options = {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"}
    matches = {
        "A": OptionEvidenceMatch("A", "Opt A", "CONTRADICTED", 0.1, "", 0),
        "B": OptionEvidenceMatch("B", "Opt B", "SUPPORTED", 0.96, "Exact evidence", 0),
        "C": OptionEvidenceMatch("C", "Opt C", "UNSUPPORTED", 0.0, "", 0),
        "D": OptionEvidenceMatch("D", "Opt D", "UNSUPPORTED", 0.0, "", 0),
    }
    qqs, breakdown, status = calculate_question_quality_score(
        "Which option is correct for database isolation?",
        options,
        "B",
        matches,
        difficulty="HARD"
    )
    assert qqs >= 85.0
    assert status == "ACCEPT"
    assert breakdown["evidence_support"] >= 25.0
    assert breakdown["answer_uniqueness"] == 25.0


def test_user_knowledge_graph_and_adaptive_learning():
    kg = UserKnowledgeGraph()
    # 1. User answers SQL JOIN correctly
    kg.record_attempt("SQL", "JOIN", is_correct=True)
    kg.record_attempt("SQL", "JOIN", is_correct=True)
    # 2. User fails Window Functions
    kg.record_attempt("SQL", "Window Functions", is_correct=False)
    
    assert kg.concepts["sql:join"].mastery_level == "STRONG"
    assert kg.concepts["sql:window functions"].mastery_level == "WEAK"
    
    weak = kg.get_weak_concepts("SQL")
    assert "Window Functions" in weak

    # Test duplicate question prevention
    q1 = "Which isolation level prevents dirty reads?"
    kg.register_question(q1)
    assert kg.is_duplicate_question(q1) is True


def test_evaluate_user_quiz_answer():
    kg = UserKnowledgeGraph()
    quiz = GeneratedQuizQuestion(
        question_id="test_1",
        question="Which database isolation level prevents dirty reads?",
        options={"A": "Read Uncommitted", "B": "Read Committed", "C": "NoSQL", "D": "Eventual Consistency"},
        correct_answer="B",
        topic="Databases",
        concept_tag="Isolation Levels",
        difficulty="MEDIUM",
        evidence=[EvidenceSnippet(source="PostgreSQL Docs", chunk="Read Committed prevents dirty reads.", span="Read Committed prevents dirty reads.")],
        quality_score=92.0,
        confidence=0.98,
        explanation="Read committed ensures only committed data is read.",
        validation_passed=True
    )
    # User picks B (1/1 correct -> STRONG)
    feedback = evaluate_user_quiz_answer(quiz, "B", knowledge_graph=kg)
    assert feedback["is_correct"] is True
    assert feedback["user_selected"] == "B"
    assert "PostgreSQL Docs" in feedback["grounded_evidence"]

    # User picks wrong option (1/2 correct -> MEDIUM)
    feedback_wrong = evaluate_user_quiz_answer(quiz, "A", knowledge_graph=kg)
    assert feedback_wrong["is_correct"] is False
    assert feedback_wrong["current_mastery"] in ["WEAK", "MEDIUM"]


def test_evidence_grounded_quiz_generation():
    kg = UserKnowledgeGraph()
    generated = generate_evidence_grounded_quiz(
        topic="Databases",
        difficulty="HARD",
        concept="Write-Ahead Logging",
        knowledge_graph=kg
    )
    assert generated is not None
    assert len(generated.options) >= 2
    assert generated.correct_answer in generated.options
    assert generated.quality_score >= 60.0
    assert len(generated.evidence) >= 1
    assert generated.evidence[0].span != ""


def test_dqn_selector_scoring():
    selector = DQNQuizOptionSelector()
    question = "What does AI stand for?"
    options = {
        "A": "Automated Information",
        "B": "Applied Interface",
        "C": "Advanced Internet",
        "D": "Artificial Intelligence"
    }
    context_chunks = [
        "Artificial intelligence (AI) is the intelligence of machines or software.",
        "In computer science, artificial intelligence is a field of study."
    ]
    states = selector.build_option_states(question, options, context_chunks)
    assert "D" in states
    assert states["D"].exact_phrase_match == 1.0
    assert states["D"].acronym_alignment >= 0.85
    assert states["D"].doc_support_count == 2
    assert states["D"].doc_occurrence_count >= 2

    counts = selector.count_option_support_in_docs(options, context_chunks)
    assert "A" in counts and "D" in counts
    assert counts["D"]["doc_support"] == 2
    assert counts["A"]["doc_support"] == 0

    q_vals = selector.evaluate_q_values(states)
    best_key, conf, probs = selector.select_best_option(q_vals)
    assert best_key == "D"
    assert conf > 0.50
    assert probs["D"] > probs["A"]


def test_validate_quiz_answer():
    state_valid = OptionState(
        option_key="D",
        option_text="Artificial Intelligence",
        exact_phrase_match=1.0,
        token_overlap_ratio=1.0,
        cross_encoder_score=0.9,
        acronym_alignment=1.0,
        semantic_similarity=0.95,
        entity_cooccurrence=1.0,
        doc_support_ratio=1.0,
        doc_occurrence_normalized=1.0,
        length_normalized_score=0.8,
        negation_penalty=0.0
    )
    probs = {"A": 0.05, "B": 0.05, "C": 0.05, "D": 0.85}
    is_valid, reason = validate_quiz_answer(
        "D",
        "Artificial Intelligence",
        0.85,
        probs,
        "Artificial intelligence (AI) is machine intelligence.",
        state_valid
    )
    assert is_valid is True


def test_solve_quiz_query_end_to_end():
    query = "What does AI stand for? 1 of 5 A Automated Information B Applied Interface C Advanced Internet D Artificial Intelligence"
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "D"
    assert res["selected_option"] == "Artificial Intelligence"
    assert res["validation_passed"] is True
    assert res["llm_required"] is False
    assert "Artificial Intelligence" in res["final_answer"]
    assert "D)" in res["final_answer"]
    assert len(res["top3_chunks"]) <= 3


def test_run_pipeline_quiz_route():
    query = "What does AI stand for? 1 of 5 A Automated Information B Applied Interface C Advanced Internet D Artificial Intelligence"
    pipeline_res = run_pipeline(query)
    assert pipeline_res["route"] == "quiz_agent"
    assert pipeline_res["domain"] == "QUIZ"
    assert pipeline_res["passed"] is True
    quiz_data = pipeline_res["funnel_meta"]["quiz_data"]
    assert quiz_data["selected_letter"] == "D"
    assert quiz_data["selected_option"] == "Artificial Intelligence"
    assert "Correct Answer: D) Artificial Intelligence" in pipeline_res["final_answer"]


def test_solve_rag_conceptual_quiz_query():
    query = (
        "A RAG system retrieves documents using dense embeddings. "
        "Which change most directly addresses the case where the relevant document contains the right concept "
        "but the query and document use very different wording? 2 of 10 "
        "A Reduce the retrieved context to one token "
        "B Use a stronger semantic embedding or hybrid retrieval strategy "
        "C Increase the LLM temperature "
        "D Disable document chunking entirely"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "B"
    assert "hybrid retrieval" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: B)" in res["final_answer"]


def test_solve_imbalance_classification_quiz():
    query = (
        "A classifier has 99% accuracy on a dataset where 99% of examples belong to the negative class. "
        "Which metric is generally more informative for evaluating performance on the rare positive class? 6 of 10 "
        "A F1 score "
        "B Number of model layers "
        "C Training-set size "
        "D Learning rate"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "A"
    assert res["selected_option"] == "F1 score"
    assert res["validation_passed"] is True
    assert "Correct Answer: A) F1 score" in res["final_answer"]


def test_solve_knowledge_graph_quiz_query():
    query = (
        "In a knowledge graph used for retrieval, what is the main advantage of explicitly representing "
        "entities and relationships instead of storing only independent text chunks? "
        "A It eliminates the need for any language model "
        "B It guarantees that every extracted fact is correct "
        "C It prevents all duplicate entities automatically "
        "D It can support relationship-aware traversal and multi-hop reasoning over connected entities"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "D"
    assert "multi-hop reasoning" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: D)" in res["final_answer"]


def test_solve_chunk_size_recall_quiz_query():
    query = (
        "Why can retrieval recall decrease when a document collection is chunked into extremely small fragments? "
        "A Important contextual signals can be separated across fragments, weakening retrieval representations "
        "B Vector databases cannot store short text "
        "C Embedding models stop working below a fixed universal word count "
        "D Smaller chunks always contain more factual errors Next Give feedback"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "A"
    assert "contextual signals" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: A)" in res["final_answer"]


def test_solve_btree_composite_index_quiz():
    query = (
        "A composite B+ tree index exists on (customer_id, order_date). "
        "Which query can most directly benefit from the index's leftmost-prefix property? "
        "A WHERE customer_id = 42 AND order_date >= '2026-01-01' "
        "B WHERE YEAR(order_date) = 2026 "
        "C WHERE customer_id + 1 = 42 "
        "D WHERE order_date >= '2026-01-01'"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "A"
    assert "customer_id = 42" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: A)" in res["final_answer"]


def test_solve_functional_dependency_transitivity_quiz():
    query = (
        "A relation has functional dependencies A → B and B → C, with A as a candidate key. "
        "Which statement best describes the dependency A → C? "
        "A It violates reflexivity "
        "B It follows by transitivity "
        "C It cannot be inferred from the given dependencies "
        "D It follows only if C is a candidate key"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "B"
    assert "transitivity" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: B)" in res["final_answer"]


def test_solve_sql_correlated_subquery_quiz():
    query = (
        "Which SQL query pattern is most likely to produce a correlated subquery that is evaluated conceptually with reference to each outer-row value? "
        "A SELECT department_id, AVG(salary) FROM employees GROUP BY department_id "
        "B SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees) "
        "C SELECT name FROM employees WHERE department_id = 10 "
        "D SELECT e.name FROM employees e WHERE e.salary > (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.department_id = e.department_id)"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "D"
    assert "WHERE e2.department_id" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: D)" in res["final_answer"]


def test_solve_wal_crash_recovery_quiz():
    query = (
        "A database transaction executes successfully but crashes immediately before its commit record is made durable. "
        "Under a correctly implemented write-ahead logging recovery protocol, how should the transaction normally be treated after restart? "
        "A As committed only when no other transaction was running "
        "B As committed only if the transaction modified an indexed table "
        "C As permanently committed because its SQL statements completed "
        "D As uncommitted and therefore subject to rollback during recovery"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "D"
    assert "uncommitted and therefore subject to rollback" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: D)" in res["final_answer"]


def test_solve_optimizer_separate_indexes_quiz():
    query = (
        "A query filters a table using WHERE customer_id = 100 AND order_date > '2026-01-01'. "
        "The table has separate indexes on customer_id and order_date. "
        "Why might the optimizer choose only the customer_id index rather than combining both indexes? 2 of 10 "
        "A The order_date predicate is always ignored when an equality predicate exists "
        "B Using one selective index followed by filtering may be cheaper than combining two index access paths "
        "C SQL databases cannot use more than one index in any query "
        "D Separate indexes are always slower than a composite index regardless of workload"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "B"
    assert "cheaper than combining" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: B)" in res["final_answer"]


def test_solve_sql_not_exists_antijoin_quiz():
    query = (
        "Which SQL technique is most appropriate for finding rows that exist in one table but have no matching row in another table? 3 of 10 "
        "A NOT EXISTS "
        "B CROSS JOIN "
        "C UNION ALL "
        "D GROUP BY without a condition"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "A"
    assert "NOT EXISTS" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: A)" in res["final_answer"]


def test_solve_candidate_key_2nf_violation_quiz():
    query = (
        "A relation has a candidate key (A,B) and a non-key attribute C depends only on A. "
        "Which normal form is violated? 5 of 10 "
        "A Second Normal Form "
        "B Fourth Normal Form "
        "C Only BCNF "
        "D First Normal Form"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "A"
    assert "Second Normal Form" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: A)" in res["final_answer"]


def test_solve_distributed_replication_factor_quiz():
    query = (
        "In a distributed database, why can increasing the replication factor improve fault tolerance but also increase write overhead? 9 of 10 "
        "A More replicas always reduce storage usage "
        "B More replicas guarantee zero network latency "
        "C Replication eliminates all consistency concerns "
        "D More replicas provide more copies but require additional coordination or propagation of writes"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "D"
    assert "require additional coordination" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: D)" in res["final_answer"]


def test_solve_btree_functional_index_least_likely_quiz():
    query = (
        "A database has a B-tree index on last_name. Which query is least likely to use that ordinary index efficiently without an additional functional index or specialized optimization? 6 of 10 "
        "A WHERE last_name LIKE 'Smi%' "
        "B WHERE last_name = 'Smith' "
        "C WHERE last_name >= 'Smith' AND last_name < 'Smitz' Range predicates can generally use a B-tree index. "
        "D WHERE UPPER(last_name) = 'SMITH' Applying UPPER to the indexed column changes the indexed expression, so an ordinary index on last_name may not efficiently support the predicate."
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "D"
    assert "UPPER(last_name)" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: D)" in res["final_answer"]
    assert "quiz_options" in res
    assert res["selected_option_id"] == res["correct_option_id"]


def test_immutable_option_id_and_deterministic_grader():
    from agents.quiz_agent import QuizOption, GeneratedQuizQuestion, UserQuizSubmission, grade_user_submission, QuestionLifecycleState

    options = [
        QuizOption(option_id="opt_91f2", label="A", text="1NF"),
        QuizOption(option_id="opt_38aa", label="B", text="2NF"),
        QuizOption(option_id="opt_72bc", label="C", text="3NF"),
        QuizOption(option_id="opt_44de", label="D", text="BCNF"),
    ]

    question = GeneratedQuizQuestion(
        question_id="db_001",
        question="Which normal form removes partial dependencies?",
        options=options,
        correct_option_id="opt_38aa",
        explanation="2NF requires 1NF and no partial dependencies on candidate keys.",
        topic="Database Normalization",
        concept_tag="2NF",
        difficulty="MEDIUM",
        state=QuestionLifecycleState.PUBLISHED
    )

    # Correct submission
    sub_correct = UserQuizSubmission(question_id="db_001", selected_option_id="opt_38aa")
    result_correct = grade_user_submission(question, sub_correct)
    assert result_correct["is_correct"] is True
    assert result_correct["selected_label"] == "B"
    assert result_correct["correct_label"] == "B"
    assert result_correct["selected_option_id"] == "opt_38aa"
    assert question.state == QuestionLifecycleState.GRADED

    # Incorrect submission
    sub_wrong = UserQuizSubmission(question_id="db_001", selected_option_id="opt_44de")
    result_wrong = grade_user_submission(question, sub_wrong)
    assert result_wrong["is_correct"] is False
    assert result_wrong["selected_label"] == "D"
    assert result_wrong["correct_label"] == "B"
    assert result_wrong["correct_option_id"] == "opt_38aa"


def test_solve_ssi_write_skew_dependency_cycle_quiz():
    query = (
        "A distributed database implements Serializable Snapshot Isolation (SSI) using MVCC. Two concurrent transactions execute: "
        "Initial state: A = 100, B = 100. "
        "T1: Read A, Read B, Write A = 0. "
        "T2: Read A, Read B, Write B = 0. "
        "Both transactions see the same initial snapshot. There are no direct write-write conflicts because T1 writes A and T2 writes B. "
        "However, the system detects the following dependency pattern: T1 -> rw -> T2 and T2 -> rw -> T1. "
        "What should SSI do to preserve serializability? "
        "A. Allow both transactions to commit because they modify different rows. "
        "B. Abort one of the transactions because the dangerous dependency cycle can produce a non-serializable execution. "
        "C. Force both transactions to restart because MVCC cannot support concurrent writes. "
        "D. Allow both transactions to commit because snapshot isolation guarantees serializability."
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "B"
    assert "Abort one of the transactions" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: B)" in res["final_answer"]


def test_solve_ancient_numeral_zero_quiz():
    query = (
        "Which civilization is credited with developing the world's earliest known decimal place-value numeral system with a symbol for zero? "
        "A. Ancient Greeks B. Ancient Romans C. Ancient Indians D. Ancient Egyptians"
    )
    res = solve_quiz_query(query)
    assert res["is_quiz"] is True
    assert res["selected_letter"] == "C"
    assert "Indians" in res["selected_option"] or "Indian" in res["selected_option"]
    assert res["validation_passed"] is True
    assert "Correct Answer: C)" in res["final_answer"]
