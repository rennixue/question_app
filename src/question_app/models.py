from enum import Enum
from typing import Annotated, Any, Literal, Self, assert_never
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_serializer, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ElasticsearchSettings(BaseModel):
    url: str
    api_key: str


class OllamaSettings(BaseModel):
    url: str


class OpenaiSettings(BaseModel):
    base_url: str
    api_key: str
    chat_model: str
    reason_model: str


class QdrantSettings(BaseModel):
    url: str
    api_key: str


class SqlalchemySettings(BaseModel):
    url: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="_", env_nested_max_split=1)
    callback_base_url: HttpUrl
    skip_callback: bool = False
    elasticsearch: ElasticsearchSettings
    ollama: OllamaSettings
    openai: OpenaiSettings
    qdrant: QdrantSettings
    sqlalchemy: SqlalchemySettings


class QuestionSource(Enum):
    SameOrder = "same_order"
    SameCourse = "same_course"
    SameUniversity = "same_university"
    Historical = "historical"
    Imitated = "imitated"
    Generated = "generated"
    Rewritten = "rewritten"

    def to_int(self) -> int:
        match self:
            case QuestionSource.SameOrder:
                return 1
            case QuestionSource.SameCourse:
                return 2
            case QuestionSource.Historical:
                return 4
            case QuestionSource.SameUniversity:
                return 5
            case _:
                return 3


class QuestionType(Enum):
    Any = "any"
    Calculation = "calculation"
    MultipleChoice = "multiple choice"
    Open = "open"

    @classmethod
    def from_value(cls, val: str | None) -> "QuestionType":
        if val is None:
            return QuestionType.Open
        match val:
            case "calculation":
                return QuestionType.Calculation
            case "multiple choice":
                return QuestionType.MultipleChoice
            case "open":
                return QuestionType.Open
            case _:
                return QuestionType.Open

    def to_int(self) -> int:
        match self:
            case QuestionType.Any:
                raise ValueError("QuestionType.Any cannot be converted to an integer")
            case QuestionType.Calculation:
                return 4
            case QuestionType.MultipleChoice:
                return 1
            case QuestionType.Open:
                return 2
            case _:
                assert_never(self)

    def to_natural_language(self) -> str:
        match self:
            case QuestionType.Any:
                return "a question of any type"
            case QuestionType.Calculation:
                return "a calculation question"
            case QuestionType.MultipleChoice:
                return "a multiple choice question"
            case QuestionType.Open:
                return "an open question"
            case _:
                assert_never(self)

    @classmethod
    def from_elasticsearch_keyword(cls, val: str | None) -> "QuestionType":
        if val is None:
            return QuestionType.Open
        match val:
            case "calculation":
                return QuestionType.Calculation
            case "mcq":
                return QuestionType.MultipleChoice
            case "open":
                return QuestionType.Open
            case _:
                return QuestionType.Open

    def to_elasticsearch_keyword(self) -> str | None:
        match self:
            case QuestionType.Any:
                return None
            case QuestionType.Calculation:
                return "calculation"
            case QuestionType.MultipleChoice:
                return "mcq"
            case QuestionType.Open:
                return "open"
            case _:
                assert_never(self)


class Question(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    source: QuestionSource
    type: QuestionType
    meta_info: str | None = None
    batch_no: int = 1


class KeyPoint(BaseModel):
    name: str
    explanation: str
    relevance: str


class QuestionFormInputsReq(BaseModel):
    exam_kp: Annotated[str, Field(min_length=1)]
    context: str | None
    question_type: QuestionType

    @field_validator("question_type", mode="before")
    def validate_question_type(cls, val: object) -> QuestionType:
        if not isinstance(val, int):
            raise TypeError("question_type must be an integer")
        match val:
            case 0:
                return QuestionType.Any
            case 1:
                return QuestionType.MultipleChoice
            case 2:
                return QuestionType.Open
            case 4:
                return QuestionType.Calculation
            case _:
                raise ValueError("the value of question_type is invalid")


class QuestionGenerateReq(QuestionFormInputsReq):
    task_id: int
    course_id: int
    major_name: str | None = None
    course_name: str | None = None
    course_code: str | None = None
    university_name: str | None = None


class QuestionRewriteReq(QuestionFormInputsReq):
    rewritten_from: int
    rewritten_from_no: str  # do not use UUID
    rewritten_prompt: Annotated[str, Field(min_length=1)]
    question: Annotated[str, Field(min_length=1)]


class ExtractedFile(BaseModel):
    file_name: Annotated[str, Field(min_length=1)]
    kps: Annotated[list[str], Field(min_length=1)]


class CourseMaterialType(Enum):
    # required by frontend direct use
    LectureNote = "Lecture Notes"
    Other = "Other"
    Reading = "Reading"
    PastPaper = "Past paper"
    Syllabus = "Syllabus"
    TutorialQuestion = "Tutorial question"

    @classmethod
    def from_string(cls, s: str | None) -> "CourseMaterialType":
        if s is None or s == "":
            return cls.Other
        match s:
            case "lecture_note" | "lecture notes":
                return cls.LectureNote
            case "other":
                return cls.Other
            case "past_paper" | "past paper" | "exam paper":
                return cls.PastPaper
            case "reading" | "readings" | "reading materials":
                return cls.Reading
            case "syllabus" | "unit guide":
                return cls.Syllabus
            case "tutorial_question" | "tutorial questions":
                return cls.TutorialQuestion
            case _:
                return cls.Other

    def to_int(self) -> int:
        match self:
            case CourseMaterialType.LectureNote:
                return 1
            case CourseMaterialType.Syllabus:
                return 2
            case CourseMaterialType.PastPaper:
                return 3
            case CourseMaterialType.TutorialQuestion:
                return 4
            case CourseMaterialType.Reading:
                return 5
            case CourseMaterialType.Other:
                return 6
            case _:
                assert_never()


class ExtractedFileWithType(BaseModel):
    file_name: Annotated[str, Field(min_length=1)]
    file_type: CourseMaterialType
    kps: list[str]


class KeyPointNameAndFreq(BaseModel):
    name: Annotated[str, Field(min_length=1)]
    freq: Annotated[int, Field(gt=0)]


class AnalyzeDescriptionOutput(BaseModel):
    key_concepts: str = ""
    requirement: str = ""
    referential_question: str = ""
    other_info: str = ""


class AnalyzeQueryOutput(BaseModel):
    primary_term: Annotated[str, Field(min_length=3)]
    secondary_terms: Annotated[list[str], Field(default_factory=lambda: [])]


class StreamBlock(BaseModel):
    done: bool = False
    q_src: QuestionSource | None = None
    status: Literal["start", "progress", "finish", "checkpoint"] | None = None
    count: int | None = None
    time: float | None = None
    questions: list[Question] | None = None

    @model_validator(mode="after")
    def check(self) -> Self:
        if self.done:
            assert self.q_src is None
            assert self.status is None
            assert self.count is not None
            assert self.time is not None
            assert self.questions is None
        else:
            assert self.q_src is not None
            assert self.status is not None
            match self.status:
                case "start":
                    assert self.count is None
                    assert self.time is None
                    assert self.questions is None
                case "progress":
                    assert self.time is None
                    assert self.questions is not None
                case "finish" | "checkpoint":
                    assert self.time is not None
                    assert self.count is not None
        return self

    @model_serializer()
    def serialize_model(self) -> dict[str, Any]:
        return {
            "done": self.done,
            "genType": self.q_src.to_int() if self.q_src is not None else None,
            "status": self.status,
            "count": self.count,
            "time": round(self.time, 2) if self.time is not None else None,
            "questions": [
                {
                    "questionNo": it.id.hex,
                    "questionType": it.type.to_int(),
                    "genType": it.source.to_int(),
                    "batchNo": it.batch_no,
                    "genQuestion": it.content,
                    "genQuestionInfo": it.meta_info,
                }
                for it in self.questions
            ]
            if self.questions is not None
            else None,
        }


class QuestionSection(BaseModel):
    q_src: QuestionSource
    batch_no: int
    count: int
    elapsed: float

    @model_serializer()
    def serialize_model(self) -> dict[str, Any]:
        return {
            "genType": self.q_src.to_int(),
            "batchNo": self.batch_no,
            "count": self.count,
            "elapsed": round(self.elapsed, 2),
        }
