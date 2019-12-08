import rq
from rq.job import Job, JobStatus
from rq.registry import (
    StartedJobRegistry,
    FinishedJobRegistry,
    FailedJobRegistry,
)
from flask import (
    Flask,
    jsonify,
    request,
    make_response
)
from redis import Redis
from stocks import (
    StockDatabase,
)

queue_name = "trading-tasks"
db = StockDatabase('84WDI082Z0HOREL6')
tasks = []
app = Flask(__name__)
redis = Redis.from_url("redis://localhost:6379")
# ----------------- QUEUES -----------------
queue = rq.Queue(queue_name, connection=redis)
started_jobs = StartedJobRegistry(queue_name, connection=redis)
failed_jobs = FailedJobRegistry(queue_name, connection=redis)
finished_jobs = FinishedJobRegistry(queue_name, connection=redis)


# ----------------- TRADE -----------------

@app.route('/api/trading/trade', methods=["POST"])
def trade():
    if request.is_json:
        content = request.get_json()
        user = content.get("user_token", None)
        if user is not None:
            job = queue.enqueue("tasks.trade", args=(user,), job_timeout=12000, result_ttl=86400)
            job.save_meta()
            json = {
                "success": True,
                "task": {
                    "id": job.id, "result": job.return_value,
                    "created_at": job.created_at, "started_at": job.started_at,
                    "ended_at": job.ended_at, "status": job.get_status(),
                    "meta": job.meta
                }
            }
            return make_response(jsonify(json), 200)
        else:
            return make_response(jsonify({"error": "Must include user's bearer token"}), 400)
    return make_response(jsonify(None), 400)


# ----------------- STOCKS -----------------


@app.route('/api/trading/stocks/suggest/', methods=["POST"])
def suggest_stock():
    """
    Suggest a stock within a price range
    :return:
    """
    MAX_SYMBOLS = 5
    if request.is_json:
        content = request.get_json()
        symbols = content.get("symbols", None)
        if symbols is not None:
            if len(symbols) > MAX_SYMBOLS:
                return make_response(jsonify({"error": f"Too many symbols (max {MAX_SYMBOLS})"}), 400)
            # Check if symbols exist
            for symbol in symbols:
                found = False
                for stock in db.stocks:
                    if stock.symbol == symbol:
                        found = True
                        break
                if not found:
                    return make_response(jsonify({"error": f"Unknown symbol '{symbol}'"}), 400)
            job = queue.enqueue("tasks.suggest", args=(symbols,), job_timeout=1200, result_ttl=86400)
            job.meta["symbols"] = symbols
            job.save_meta()
            json = {
                "success": True,
                "task": {
                    "id": job.id, "result": job.return_value,
                    "created_at": job.created_at, "started_at": job.started_at,
                    "ended_at": job.ended_at, "status": job.get_status(),
                    "meta": job.meta
                }
            }
            return make_response(jsonify(json), 200)
    return make_response(jsonify(None), 400)


@app.route('/api/trading/stocks/search/', methods=["GET"])
def search_stock():
    """
    Describes a specific stock
    Enables search by name or symbol
    :return:
    """
    symbol = request.args.get("symbol", "<>")
    name = request.args.get("name", "<>").lower()
    found = []
    for stock in db.stocks:
        if stock.symbol == symbol or name in stock.name.lower():
            found.append(stock)
    if found is not None:
        return make_response(jsonify([stock.to_json() for stock in found]), 200)
    return make_response(jsonify(None), 404)


@app.route('/api/trading/stocks/', methods=["GET"])
def stocks():
    """
    Retrieves stats on all the available stocks
    :return:
    """
    return make_response(jsonify([stock.to_json() for stock in db.stocks]), 200)


@app.route('/api/trading/stocks/analyze/<to_analyze>', methods=["GET"])
def analyze_stock(to_analyze):
    to_analyze = to_analyze.upper()
    for stock in db.stocks:
        if stock.symbol == to_analyze:
            filename = db.generate_stock_history(stock, time_series="day", output_size="full", interval="60min")
            job = queue.enqueue("tasks.predict", args=(stock, filename), job_timeout=1200, result_ttl=86400)
            job.meta["stock"] = stock.symbol
            job.save_meta()
            json = {
                "success": True,
                "task": {"id": job.id, "result": job.return_value,
                         "created_at": job.created_at, "started_at": job.started_at,
                         "ended_at": job.ended_at, "status": job.get_status(),
                         "meta": job.meta
                }
            }
            return make_response(jsonify(json), 200)
    return make_response(jsonify({"success": False, "message": f'Stock {to_analyze} not found in our database'}), 404)


# ----------------- TASKS -----------------
@app.route('/api/trading/tasks/search/', methods=["GET"])
def search_tasks():
    """
    Retrieves all the tasks based on status
    :return:
    """
    status = request.args.get("status", "").lower()
    if status != "" and status not in [enum_val.lower() for enum_val in JobStatus.__dict__.keys()]:
        return make_response({"success": False, "message": f'Unknown status {status}'})
    jobs = []
    json_jobs = []
    # RUNNING JOBS
    if status == "" or (status != "finished" and status != "failed"):
        jobs.extend([queue.fetch_job(id) for id in started_jobs.get_job_ids()])
    # FINISHED JOBS
    if status == "" or status == "finished":
        jobs.extend(Job.fetch_many(finished_jobs.get_job_ids(), connection=redis))
    # FAILED JOBS
    if status == "" or status == "failed":
        jobs.extend([queue.fetch_job(id) for id in failed_jobs.get_job_ids()])
    # QUEUED JOBS
    if status == "" or status == "queued":
        jobs.extend(queue.get_jobs())
    # Change them to human readable format
    for job in jobs:
        json_jobs.append({"id": job.id, "result": job.return_value,
                          "created_at": job.created_at, "started_at": job.started_at,
                          "ended_at": job.ended_at, "status": job.get_status(),
                          "meta": job.meta})
    return make_response(jsonify(json_jobs), 200)


@app.route('/api/trading/tasks/<task_id>', methods=["PUT"])
def update_task(task_id):
    """
    Relaunch a failed task
    :return:
    """
    j = queue.fetch_job(task_id)
    if j is not None:
        # Only failed task can be updated
        if j.get_status() != "failed":
            return make_response('', 400)
        failed_jobs.requeue(j)
        return make_response('', 204)
    return make_response('', 404)


@app.route('/api/trading/tasks/<task_id>', methods=["GET"])
def get_task(task_id):
    """
    Retrieves task with uuid
    :return:
    """
    job = queue.fetch_job(task_id)
    if job is not None:
        return make_response(jsonify({"id": job.id, "result": job.return_value,
                                      "created_at": job.created_at, "started_at": job.started_at,
                                      "ended_at": job.ended_at, "status": job.get_status()}), 200)
    return make_response(jsonify(None), 404)


@app.route('/api/trading/tasks/<task_id>', methods=["DELETE"])
def delete_task(task_id):
    """f
    Deletes a task based on uuid
    :return:
    """
    j = queue.fetch_job(task_id)
    if j is not None:
        j.delete()
        return make_response('', 204)
    return make_response('', 404)


if __name__ == "__main__":
    app.run(port=8000, debug=True)
