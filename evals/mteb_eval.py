import mteb
# Removed task_type list as we fetch tasks by type directly

def evaluate_fn(model, log_dir, epoch, encode_kwargs):
    # Fetch all tasks of type InstructionRetrieval
    all_tasks = mteb.get_tasks(
        task_types=["InstructionRetrieval"],
        languages=["eng"],
    )

    # Initialize MTEB with all fetched tasks
    evaluation = mteb.MTEB(tasks=all_tasks)

    # Run evaluation for all tasks
    results = evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=f"{log_dir}/epoch_{epoch}",
        encode_kwargs=encode_kwargs,
        overwrite_results=True
    )

    res = {}
    total_score = 0.0
    total_count = 0

    # Process results for each task
    for r in results:
        # Ensure 'test' split results and scores exist
        if "test" in r.scores and r.scores["test"]:
            # Assuming the primary metric is the first one listed
            score = r.scores["test"][0].get("main_score")
            if score is not None:
                res[r.task_name] = score
                total_score += score
                total_count += 1
            else:
                print(f"Warning: 'main_score' not found for task {r.task_name} in 'test' split.")
        else:
            print(f"Warning: 'test' split results not found or empty for task {r.task_name}.")


    # Compute overall average score for InstructionRetrieval tasks
    overall_avg = total_score / total_count if total_count > 0 else 0.0
    res["InstructionRetrieval_average"] = overall_avg # Use a more specific key

    # Removed the per-task-type averaging logic as we only focus on one type

    return res