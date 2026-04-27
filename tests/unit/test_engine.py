import pytest
from app.engine import route_after_critique
from app.schema import ResearchState, CritiqueResult
from app.config import MAX_ITERATIONS

def test_route_after_critique_logic():
    # 1. Satisfied -> write_report
    state_satisfied: ResearchState = {
        "critique": CritiqueResult(satisfied=True, gaps=[], new_sub_questions=[]),
        "iteration": 1
    }
    assert route_after_critique(state_satisfied) == "write_report"
    
    # 2. Not satisfied but at MAX_ITERATIONS -> write_report
    state_max_iter: ResearchState = {
        "critique": CritiqueResult(satisfied=False, gaps=["some gaps"], new_sub_questions=["q1"]),
        "iteration": MAX_ITERATIONS
    }
    assert route_after_critique(state_max_iter) == "write_report"
    
    # 3. Not satisfied and below MAX_ITERATIONS -> search
    state_loop: ResearchState = {
        "critique": CritiqueResult(satisfied=False, gaps=["some gaps"], new_sub_questions=["q1"]),
        "iteration": 0
    }
    assert route_after_critique(state_loop) == "search"
