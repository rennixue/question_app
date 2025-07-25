import pytest

from question_app.models import QuestionType
from question_app.services import (
    OllamaService,
    QdrantService,
    QuestionGenerateService,
    QuestionImitateService,
    QuestionRewriteService,
    QuestionSearchService,
)


@pytest.mark.skip
async def test_qdrant_query_chunks(ollama: OllamaService, qdrant: QdrantService):
    kp = "hypothesis test"
    vec = await ollama.embed_one("definition of " + kp)
    pairs = await qdrant.query_chunks(kp, vec, 614639, 5)
    assert len(pairs) == 4
    for it in pairs:
        print(it)


@pytest.mark.skip
async def test_question_search_find_majors(question_search: QuestionSearchService):
    majors = await question_search.find_majors("math")
    assert len(majors) > 1
    assert majors[0] == "Mathematics"


@pytest.mark.skip
async def test_question_search_find_questions(question_search: QuestionSearchService):
    questions_same_course, questions_historical = await question_search.find_questions(
        exam_kp="t-distribution",
        context="compare with normal distribution",
        question_type=QuestionType.MultipleChoice,
        major_name="Business Analytics",
        course_name="Statistical Methods for Business",
        course_code="MSCI212",
        university="兰卡斯特大学(Lancaster University)",
        limit_same_course=2,
        limit_historical=2,
    )
    assert len(questions_same_course) == 2
    assert len(questions_historical) == 0


@pytest.mark.skip
async def test_question_verify(question_search: QuestionSearchService, question_imitate: QuestionImitateService):
    qs_same_course, qs_historical = await question_search.find_questions(
        "hypothesis test", None, QuestionType.Any, None, None, None, None, 0, 20
    )
    print(len(qs_same_course), len(qs_historical))
    questions = await question_imitate.verify(qs_historical, 614639, "hypothesis test", None)
    print(len(questions))
    for question in questions:
        print(question)


@pytest.mark.skip
async def test_question_generate(question_generate: QuestionGenerateService):
    questions = await question_generate.generate(
        course_id=614639,
        exam_kp="hypothesis test",
        context="two kinds of errors in hypothesis test",
        question_type=QuestionType.Any,
        major=None,
        course_name=None,
        num=4,
    )
    for question in questions:
        print(question)


@pytest.mark.skip
async def test_question_rewrite(question_rewrite: QuestionRewriteService):
    question = await question_rewrite.rewrite(
        rewrite_prompt="change t-distribution to Poisson distribution",
        old_question="Which is symmetric, normal distribution or t-distribution?",
        exam_kp="normal distribution",
        context="compare to another distribution",
        question_type=QuestionType.Any,
    )
    print(question)
