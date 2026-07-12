"""Simulated email tool.

A real deployment would call an SMTP/SES/SendGrid API here. For a reproducible
simulation we instead persist each message to a per-run "outbox" directory and
return a structured record, so the collection correspondence is inspectable
without sending anything to real inboxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ar_platform.config import OUTBOX_DIR


@dataclass
class SentEmail:
    to: str
    subject: str
    body: str
    invoice_id: str
    sim_date: str
    sent_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class EmailTool:
    """Records outbound collection emails to a simulated outbox."""

    name = "email"

    def __init__(self, outbox_dir=OUTBOX_DIR):
        self.outbox_dir = outbox_dir
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.sent: list[SentEmail] = []

    def send(
        self,
        to: str,
        subject: str,
        body: str,
        invoice_id: str,
        sim_date: str,
    ) -> SentEmail:
        email = SentEmail(
            to=to,
            subject=subject,
            body=body,
            invoice_id=invoice_id,
            sim_date=sim_date,
        )
        self.sent.append(email)

        fname = f"{sim_date}_{invoice_id}_{len(self.sent):04d}.txt"
        path = self.outbox_dir / fname
        path.write_text(
            f"To: {to}\nSubject: {subject}\nDate: {email.sent_at}\n"
            f"Invoice: {invoice_id}\n\n{body}\n",
            encoding="utf-8",
        )
        return email
