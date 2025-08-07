from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith.utils import LangSmithConflictError

class LLMRunEvaluator:
    def __init__(self, dataset_name, project_name, llm=None):
        self.client = Client()
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.chain = self._build_chain()

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question accurately and clearly."),
            ("human", "{question}")
        ])
        return prompt | self.llm | StrOutputParser()

    def run_evaluation(self):
        # Create or reuse project session
        try:
            project = self.client.create_project(
                project_name=self.project_name,
                description="QA evaluation project",
                upsert=True
            )
            print(f"Project session ready: {project.name}")
        except LangSmithConflictError:
            print("Reusing existing project session.")
            project = self.client.read_project(project_name=self.project_name)
            print(f"Project session ready: {project.name}")

        evaluation = RunEvalConfig(
            evaluators=["qa"],
            reference_key="answer",
            prediction_key="output",
            eval_llm=self.llm
        )

        results = run_on_dataset(
            client=self.client,
            dataset_name=self.dataset_name,
            llm_or_chain_factory=lambda: self.chain,
            evaluation=evaluation,
            verbose=True
        )
        print("✅ Evaluation complete! Check your results at your LangSmith dashboard.")

        # --- Enrich: Print analytics and failed runs summary ---
        try:
            runs = list(self.client.list_runs(project_name=self.project_name, execution_order=1))
            total = len(runs)
            failed = [r for r in runs if r.error]
            print(f"\nLangSmith Analytics Summary:")
            print(f"Total runs: {total}")
            print(f"Failed runs: {len(failed)}")
            if failed:
                print("\nFailed run details:")
                for r in failed:
                    print(f"- Run ID: {r.id}, Error: {r.error}")
            print("\nSample results:")
            for r in runs[:3]:
                print(f"Input: {r.inputs}")
                print(f"Output: {r.outputs}")
                print(f"Reference: {r.reference_outputs if hasattr(r, 'reference_outputs') else 'N/A'}")
                print(f"Score: {r.evaluation_results if hasattr(r, 'evaluation_results') else 'N/A'}\n")
        except Exception as e:
            print(f"Could not fetch analytics: {e}")
