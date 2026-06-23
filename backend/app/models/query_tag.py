import uuid

from sqlalchemy import String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Column

from app.database import Base


class QueryTag(Base):
    __tablename__ = "query_tags"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False)
    query_id: Mapped[str] = mapped_column(String, nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    tags = Column(ARRAY(Text), nullable=True)
