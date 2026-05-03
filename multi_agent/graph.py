def render_mermaid() -> str:
    """Return the multi-agent state machine as Mermaid text."""
    return """flowchart LR
    START([START]) --> Planner[Planner]
    Planner --> Scheduler[Scheduler]
    Scheduler --> Critic[Critic]
    Critic -->|approve| END([END])
    Critic -->|revise and revise_count < 2| Scheduler
    Critic -->|fallback| END
"""
