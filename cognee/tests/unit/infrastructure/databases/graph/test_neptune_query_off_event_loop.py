"""Neptune client calls must not run on the event loop.

langchain_aws' NeptuneAnalyticsGraph.query is a synchronous boto3 call. The
adapters awaited nothing around it, so every Neptune query blocked the whole
event loop (all other requests and tasks) for the length of the network round
trip. They now run it with asyncio.to_thread.
"""

import asyncio

import pytest

from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB
from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
    NeptuneAnalyticsAdapter,
)


class _RecordingClient:
    """Fake langchain client that records whether it was called on the loop thread."""

    def __init__(self):
        self.called_on_loop = []

    def query(self, query, params=None):
        try:
            asyncio.get_running_loop()
            self.called_on_loop.append(True)
        except RuntimeError:
            self.called_on_loop.append(False)
        return []


def _adapter(cls):
    adapter = cls.__new__(cls)
    adapter._client = _RecordingClient()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cls, call",
    [
        (NeptuneGraphDB, lambda a: a.query("RETURN 1")),
        (NeptuneAnalyticsAdapter, lambda a: a.retrieve("collection", ["id"])),
        (NeptuneAnalyticsAdapter, lambda a: a.prune()),
        (NeptuneAnalyticsAdapter, lambda a: a.is_empty()),
    ],
    ids=["graph.query", "analytics.retrieve", "analytics.prune", "analytics.is_empty"],
)
async def test_client_query_runs_off_the_event_loop(cls, call):
    adapter = _adapter(cls)

    await call(adapter)

    assert adapter._client.called_on_loop == [False]
