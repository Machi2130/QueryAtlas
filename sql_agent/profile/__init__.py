"""What the agent knows about a connected database, built the same way for every database.

    ProfileService.open(...)        build or reuse the profile at connect
    ContextAssembler(svc).assemble  question -> tables + joins + annotated columns
"""

from .context import ContextAssembler, PromptContext
from .inference import infer_joins
from .models import ColumnKind, ColumnProfile, DatabaseProfile, Relationship, TableProfile, TableRole
from .router import DatabaseRouter, PrefixVocabulary, RouteDecision, database_vocabulary
from .service import ProfileService
from .vocab import names_anything, question_words

__all__ = ["ColumnKind", "ColumnProfile", "ContextAssembler", "DatabaseProfile", "DatabaseRouter",
           "PrefixVocabulary", "ProfileService", "PromptContext", "RouteDecision", "database_vocabulary",
           "Relationship", "TableProfile", "TableRole", "infer_joins", "names_anything", "question_words"]
