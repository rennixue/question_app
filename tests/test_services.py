import pytest

from question_app.models import QuestionSource, QuestionType, StreamBlock
from question_app.services import (
    AgentService,
    MysqlService,
    OllamaService,
    QdrantService,
    QuestionGenerateService,
    QuestionImitateService,
    QuestionRewriteService,
    QuestionSearchService,
)


@pytest.mark.skip
async def test_select_order_kps(mysql: MysqlService):
    order_id = -1
    processed, files, files_with_types, order_kps = await mysql.select_order_kps(order_id, 5, 2)
    print(processed)
    for it in files:
        print(it)
    for it in files_with_types:
        print(it)
    for it in order_kps:
        print(it)


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
    # aiterator = question_search.find_questions(
    #     exam_kp="t-distribution",
    #     context="compare with normal distribution",
    #     question_type=QuestionType.MultipleChoice,
    #     major_name="Business Analytics",
    #     course_name="Statistical Methods for Business",
    #     course_code="MSCI212",
    #     university="兰卡斯特大学(Lancaster University)",
    #     limit_same_course=2,
    #     limit_same_university=2,
    #     limit_historical=2,
    # )
    aiterator = question_search.find_questions(
        exam_kp="distribution",
        context=None,
        question_type=QuestionType.Any,
        major_name="Statistics",
        course_name=None,
        course_code="MSCI212",
        university="兰卡斯特大学(Lancaster University)",
        limit_same_course=20,
        limit_same_university=20,
        limit_historical=20,
    )
    qs_same_course = await anext(aiterator)
    qs_same_university = await anext(aiterator)
    qs_historical = await anext(aiterator)
    assert len(qs_same_course) > 0
    assert len(qs_same_university) > 0
    assert len(qs_historical) > 0
    for it in qs_same_course:
        assert it.source == QuestionSource.SameCourse
        print(it.meta_info)
    for it in qs_same_university:
        assert it.source == QuestionSource.SameUniversity
        print(it.meta_info)
    for it in qs_historical:
        assert it.source == QuestionSource.Historical
        print(it.meta_info)


@pytest.mark.skip
async def test_question_verify(question_search: QuestionSearchService, question_imitate: QuestionImitateService):
    aiterator = question_search.find_questions("hypothesis test", None, QuestionType.Any, None, None, None, None, 0, 20)
    _ = await anext(aiterator)
    _ = await anext(aiterator)
    qs_historical = await anext(aiterator)
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


@pytest.mark.skip
async def test_agent_analyze_description(agent: AgentService):
    exam_kp = "Poisson distribution"
    context = "compare with t-distribution"
    output = await agent.analyze_description(exam_kp=exam_kp, context=context)
    print(output)


@pytest.mark.skip
def test_stream_block_serialize():
    obj = StreamBlock(q_src=QuestionSource.SameCourse, status="finish", count=1, time=0.123)
    print(obj.model_dump_json())
