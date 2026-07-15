from api.agents.agents import coordinator_agent
from api.agents.graph import State

from langsmith import Client
from time import sleep

ACC_THRESHOLD = 0.9
SLEEP_TIME = 10

ls_client = Client()

def next_agent_evaluator(run, example):
    next_agent_match = run.outputs["coordinator_agent"]["next_agent"] == example.outputs["next_agent"]
    final_answer_match = run.outputs["coordinator_agent"]["final_answer"] == example.outputs["coordinator_final_answer"]

    return next_agent_match and final_answer_match


results = ls_client.evaluate(
    lambda x: coordinator_agent(State(messages=x["messages"])),
    data="coordinator_eval_dataset",
    # num_repetitions=2,
    max_concurrency=5,
    evaluators=[
        next_agent_evaluator
    ],
    experiment_prefix="coordinator-eval-dataset"
)

print(f"Sleeping for {SLEEP_TIME} seconds...")
sleep(SLEEP_TIME)

results_resp = ls_client.read_project(
    project_name=results.experiment_name,
    include_stats=True
)

feedback_stats_exist = (
    results_resp.feedback_stats is not None 
    and results_resp.feedback_stats.get("next_agent_evaluator") is not None
)

while not feedback_stats_exist:
    print("Waiting for evaluation feedback stats...")
    sleep(5)
    results_resp = ls_client.read_project(
        project_name=results.experiment_name,
        include_stats=True
    )
    feedback_stats_exist = (
        results_resp.feedback_stats is not None 
        and results_resp.feedback_stats.get("next_agent_evaluator") is not None
    )

evaluator_stats = results_resp.feedback_stats.get("next_agent_evaluator", {})
avg_metric = evaluator_stats.get("avg")
errors = evaluator_stats.get("errors")

print(f"Evaluation complete.")
if avg_metric is not None:
    print(f"Average accuracy (next_agent_evaluator): {avg_metric:.2%}")
else:
    print("Average accuracy (next_agent_evaluator): N/A")
print(f"Errors: {errors}")

if avg_metric is not None:
    assert avg_metric >= ACC_THRESHOLD, f"Accuracy {avg_metric:.2%} is below threshold {ACC_THRESHOLD:.2%}"
else:
    raise RuntimeError("Evaluation feedback stats did not yield a valid average accuracy score.")

