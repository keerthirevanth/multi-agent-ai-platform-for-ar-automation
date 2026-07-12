"""Simulated ERP/CRM update tool.

Stands in for writing back to SAP / NetSuite / Salesforce. Agents use it to
mutate the authoritative ledger state (invoice status, payments) through a
single, auditable surface rather than issuing raw SQL. Every mutation returns a
short description string suitable for the audit log.
"""

from __future__ import annotations

from ar_platform.data.store import Store
from ar_platform.models import Invoice, InvoiceStatus, Payment


class ERPTool:
    """Write-back interface over the ledger store."""

    name = "erp"

    def __init__(self, store: Store):
        self.store = store

    def update_invoice_status(self, invoice: Invoice, status: InvoiceStatus) -> str:
        old = invoice.status
        invoice.status = status
        self.store.upsert_invoice(invoice)
        self.store.commit()
        return f"status {old.value} -> {status.value}"

    def apply_payment(self, invoice: Invoice, payment: Payment) -> str:
        invoice.paid_amount = round(invoice.paid_amount + payment.amount, 2)
        if invoice.paid_amount >= invoice.amount - 0.01:
            invoice.paid_amount = invoice.amount
            invoice.status = InvoiceStatus.PAID
        elif invoice.paid_amount > 0:
            invoice.status = InvoiceStatus.PARTIAL
        self.store.add_payment(payment)
        self.store.upsert_invoice(invoice)
        self.store.commit()
        return f"applied ${payment.amount:,.2f}; status -> {invoice.status.value}"

    def mark_disputed(self, invoice: Invoice) -> str:
        return self.update_invoice_status(invoice, InvoiceStatus.DISPUTED)
