# Resume Bullets

- Refactored a NetLogo cloud scheduling simulator from free-form LLM text
  completion into a structured tool-calling agent with Pydantic validation and
  explicit fallback/rejection semantics.
- Implemented a Planner-Scheduler-Critic scheduling pipeline with trace logging
  for latency, tool calls, fallback rate, and decision outcomes.
- Built a memory-augmented scheduling prototype using working memory and
  episodic retrieval over historical scheduling decisions.
- Created benchmark and SFT data-generation tooling to support an agent data
  flywheel from traces to supervised fine-tuning samples.
