def test_agent_phase1_schemas_import_without_scheduler_dependencies():
    import agent_phase1.schemas as schemas

    assert schemas.ServerSnapshot(server_id=1, cpu_free_pct=10, ram_free_pct=20, net_free_pct=30).server_id == 1
