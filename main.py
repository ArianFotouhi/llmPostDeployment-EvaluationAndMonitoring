from data_ingestor import DatasetIngestor
from llm_evaluator import LLMRunEvaluator

if __name__ == "__main__":
    # qa_examples = [
    #     {"inputs": {"question": "What is the capital of Germany?"}, "outputs": {"answer": "Berlin"}},
    #     {"inputs": {"question": "Who wrote Hamlet?"}, "outputs": {"answer": "William Shakespeare"}},
    #     {"inputs": {"question": "Speed of light in vacuum?"}, "outputs": {"answer": "299,792,458 m/s"}}
    # ]
    dataset_name = "Trivia QA"
    project_name = "pr-large-ladybug-30"

    # Step 1: Ingest data (reset dataset and add examples)
    # ingestor = DatasetIngestor(dataset_name, qa_examples)
    # ingestor.reset_and_ingest()

    # Step 2: Evaluate LLM on dataset
    evaluator = LLMRunEvaluator(dataset_name, project_name)
    evaluator.run_evaluation()
