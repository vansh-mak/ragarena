import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Float, Text, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class EvalScore(Base):
    __tablename__ = "eval_scores"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False)
    pipeline_id: Mapped[str] = mapped_column(String, nullable=False)
    query_id: Mapped[str] = mapped_column(String, nullable=False)
    faithfulness: Mapped[float] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float] = mapped_column(Float, nullable=True)
    judge_correctness: Mapped[float] = mapped_column(Float, nullable=True)
    judge_completeness: Mapped[float] = mapped_column(Float, nullable=True)
    judge_groundedness: Mapped[float] = mapped_column(Float, nullable=True)
    latency_score: Mapped[float] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=True)
    composite_score: Mapped[float] = mapped_column(Float, nullable=True)
    insight: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
