# main.py

import os
import pandas as pd
import openai
from config import openai_api_key, arizeai_api_key, arizeai_proj_name, arizeai_spid
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.evals import QAEvaluator, OpenAIModel, run_evals
from vectorStore.vector_store import PineconeVectorStore


class RAGPipeline:
    def __init__(self, namespace: str = "policy-knowledge"):
        # Set API keys
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize Arize tracer
        tracer_provider = register(
            space_id=arizeai_spid,
            api_key=arizeai_api_key,
            project_name=arizeai_proj_name
        )

        # Instrument OpenAI
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        # Set up components
        self.client = openai.OpenAI()
        self.vector_store = PineconeVectorStore(namespace=namespace)
        self.eval_model = OpenAIModel(model="gpt-4")
        self.qa_evaluator = QAEvaluator(self.eval_model)

    def generate_prompt(self, context: str, question: str) -> str:
        return f"Use the following context to answer the question:\n{context}\n\nQuestion: {question}"

    def process_prompt(self, prompt: str) -> dict:
        # RAG search
        rag_results = self.vector_store.search(query=prompt, top_k=1)
        reference = rag_results[0][0] if rag_results else "No reference found."
        context = reference

        # RAG prompt
        rag_prompt = self.generate_prompt(context, prompt)

        # OpenAI response
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": rag_prompt}],
            max_tokens=100
        )
        output = response.choices[0].message.content.strip()

        return {
            "prompt": prompt,
            "input": rag_prompt,
            "output": output,
            "reference": reference
        }

    def run_batch(self, prompts: list[str]) -> list[dict]:
        records = [self.process_prompt(p) for p in prompts]
        df = pd.DataFrame(records)

        # Evaluate using Phoenix
        qa_eval_dfs = run_evals(df, evaluators=[self.qa_evaluator], provide_explanation=True)
        eval_df = qa_eval_dfs[0]

        # Merge and return results
        result_df = df.join(eval_df[["label", "score", "explanation"]])
        return result_df.to_dict(orient="records")

    def print_results(self, results: list[dict]):
        for i, r in enumerate(results, 1):
            print(f"\n--- Prompt {i} ---")
            print(f"Prompt: {r['prompt']}")
            print(f"Reference: {r['reference']}")
            print(f"Model Output: {r['output']}")
            print(f"Label: {r['label']}")
            print(f"Score: {r['score']}")
            print(f"Explanation: {r['explanation']}")


if __name__ == "__main__":
    prompts = [
        "Are late payments penalized?",
        "When are customers billed?",
        "What is required for admin account login?",
        "How long do I have to request a refund?",
        "Is user data encrypted?"
    ]

    pipeline = RAGPipeline()
    results = pipeline.run_batch(prompts)
    pipeline.print_results(results)
