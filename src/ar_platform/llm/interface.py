"""The LLM contract — reserved for genuinely generative/agentic capabilities.

Deterministic work (dunning templates, severity rules, escalation thresholds)
deliberately does NOT live behind this interface; see
``ar_platform.tools.templates`` for the deterministic default. A backend here
must add something rules cannot: free-text drafting today, and (in the agentic
layer) reasoning over unstructured customer replies and negotiation within
business-rule bounds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ar_platform.dialogue import (
    NegotiationCase,
    NegotiationProposal,
    ReplyClassification,
)
from ar_platform.tools.templates import CollectionContext, EmailDraft


class LLMClient(ABC):
    """A real language-model backend (e.g. Claude).

    Beyond drafting, this is the reasoning surface of the agentic layer:
    understanding free-text customer replies and proposing negotiated terms.
    Whatever a backend proposes for a negotiation is still bounded by the
    deterministic ``NegotiationPolicy`` before it takes effect.
    """

    name: str = "abstract"

    @abstractmethod
    def draft_collection_email(self, ctx: CollectionContext) -> EmailDraft:
        """Draft a personalized collection email (replaces the fixed template)."""

    @abstractmethod
    def classify_reply(self, text: str) -> ReplyClassification:
        """Read a free-text customer reply: intent + any extracted terms."""

    @abstractmethod
    def propose_negotiation(self, case: NegotiationCase) -> NegotiationProposal:
        """Propose accept/counter/reject/escalate for a negotiation case.

        The proposal is advisory: the caller re-validates it against the
        deterministic policy, so an LLM can never grant beyond authority.
        """
