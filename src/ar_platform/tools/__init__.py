"""Tools that agents call to affect the world.

Agents don't touch the database or send messages directly — they act through
these tool objects. This keeps side effects centralized, auditable, and easy to
mock in tests, mirroring how a real agentic system exposes capabilities via
tool-calling.
"""
