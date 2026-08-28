from pathlib import Path


def patch_server():
    filepath = "appworld-env/lib/python3.12/site-packages/appworld/serve/environment.py"
    if not Path(filepath).exists():
        print(f"Error: {filepath} not found. Please make sure appworld-env is set up.")
        return False

    with open(filepath) as f:
        content = f.read()

    if "execute_with_bookmark" in content:
        print("Server is already patched.")
        return True

    execute_marker = '@app.post("/execute")'
    idx = content.find(execute_marker)
    if idx == -1:
        print("Error: Could not find @app.post('/execute') route in environment.py")
        return False

    end_marker = 'return {"output": output}'
    end_idx = content.find(end_marker, idx)
    if end_idx == -1:
        print("Error: Could not find the end of execute function in environment.py")
        return False

    end_of_func = end_idx + len(end_marker)

    new_endpoint = """

@app.post("/execute_with_bookmark")
async def execute_with_bookmark(task_id: str = Body(...), code: str = Body(...)) -> dict[str, Any]:
    maybe_raise_exception(task_id)
    n_requests_before = len(world.requester.request_tracker.requests)
    output = world.execute(code)
    new_requests = world.requester.request_tracker.requests[n_requests_before:]
    is_bookmark = False
    if "Execution failed." not in output:
        is_bookmark = any(
            req.get("method", "").upper() in ("POST", "PUT", "DELETE", "PATCH")
            for req in new_requests
        )
    return {"output": {"output": output, "is_bookmark": is_bookmark}}"""

    patched_content = content[:end_of_func] + new_endpoint + content[end_of_func:]

    with open(filepath, "w") as f:
        f.write(patched_content)

    print("Successfully patched AppWorld server environment.py!")
    return True


if __name__ == "__main__":
    patch_server()
