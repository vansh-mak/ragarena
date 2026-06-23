import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Integer, Float, Text, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PipelineResult(Base):
    __tablename__ = "pipeline_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False)
    pipeline_id: Mapped[str] = mapped_column(String, nullable=False)
    query_id: Mapped[str] = mapped_column(String, nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=True)
    context_chunks: Mapped[dict] = mapped_column(JSONB, nullable=True)
    retrieval_ms: Mapped[float] = mapped_column(Float, nullable=True)
    generation_ms: Mapped[float] = mapped_column(Float, nullable=True)
    token_input: Mapped[int] = mapped_column(Integer, nullable=True)
    token_output: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
