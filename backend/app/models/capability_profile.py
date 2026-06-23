import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Float, Text, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Column

from app.database import Base


class CapabilityProfile(Base):
    __tablename__ = "capability_profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False)
    pipeline_id: Mapped[str] = mapped_column(String, nullable=False)
    dim_factual: Mapped[float] = mapped_column(Float, nullable=True)
    dim_multihop: Mapped[float] = mapped_column(Float, nullable=True)
    dim_summarisation: Mapped[float] = mapped_column(Float, nullable=True)
    dim_ambiguous: Mapped[float] = mapped_column(Float, nullable=True)
    dim_keyword: Mapped[float] = mapped_column(Float, nullable=True)
    best_for = Column(ARRAY(Text), nullable=True)
    avoid_when = Column(ARRAY(Text), nullable=True)
    sweet_spot: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
