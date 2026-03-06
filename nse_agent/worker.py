"""Simple RQ worker scaffold for `nse_agent`.

Usage (local):
  pip install rq redis
  redis-server &
  rq worker

This file is a small scaffold; integrate task functions from
`nse_agent.utils.bse_parser` or other modules as needed.
"""
import os
from redis import Redis
from rq import Queue

redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_conn = Redis.from_url(redis_url)
q = Queue('default', connection=redis_conn)

def sample_task(ticker: str):
    # Placeholder: call real parsing code here
    print(f"Processing task for {ticker}")
    return {"ticker": ticker}

if __name__ == '__main__':
    print("RQ worker entrypoint: use `rq worker` to run. Sample enqueued task:")
    job = q.enqueue(sample_task, 'HDFCBANK')
    print(f"Enqueued job: {job.id}")
