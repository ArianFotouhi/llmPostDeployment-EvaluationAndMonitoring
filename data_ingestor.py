from langsmith import Client

class DatasetIngestor:
    def __init__(self, dataset_name, qa_examples, description="Factual QA"):
        self.client = Client()
        self.dataset_name = dataset_name
        self.qa_examples = qa_examples
        self.description = description

    def reset_and_ingest(self):
        # Delete dataset if exists
        try:
            dataset = self.client.read_dataset(dataset_name=self.dataset_name)
            self.client.delete_dataset(dataset_id=dataset.id)
            print(f"Deleted existing dataset: {self.dataset_name}")
        except Exception:
            pass
        # Create new dataset
        dataset = self.client.create_dataset(dataset_name=self.dataset_name, description=self.description)
        print(f"Created dataset: {self.dataset_name}")
        # Add examples
        for ex in self.qa_examples:
            self.client.create_example(inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=dataset.id)
        print(f"Added {len(self.qa_examples)} examples.")
