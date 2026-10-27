import json

from src.llm import generate
from src.prompt import MINDMAP_PROMPTS
from src.config import JSON_GENERATION_RETRIES


def generate_mindmap_data(condensed_text: str) -> dict:
    """
    Generate the raw {"nodes": [...], "edges": [...]} structure
    from the model, retrying once if the output isn't valid JSON.

    Returns the parsed dict on success. Raises ValueError if every
    attempt fails to produce valid JSON.
    """

    prompt = MINDMAP_PROMPTS.format(
        text=condensed_text,
    )

    attempts = JSON_GENERATION_RETRIES + 1
    last_error: Exception | None = None
    last_raw = ""

    for attempt in range(1, attempts + 1):

        raw_result = generate(
            prompt=prompt,
            max_new_tokens=700,
            temperature=0.2,
            progress=lambda text: "Building the mindmap…",
        )

        last_raw = raw_result

        try:
            data = json.loads(raw_result)

            if not isinstance(data, dict):
                raise ValueError("Mindmap JSON must be an object.")

            return data

        except (json.JSONDecodeError, ValueError) as error:

            last_error = error

            if attempt < attempts:
                print(
                    f"\nMindmap generation attempt {attempt} failed "
                    f"({error}). Retrying...\n"
                )

    raise ValueError(
        f"Model did not return valid mindmap JSON after {attempts} "
        f"attempt(s): {last_error}\n\nLast raw output:\n{last_raw}"
    )


def render_tree(data: dict) -> str:
    """
    Pure function: turn a {"nodes": [...], "edges": [...]} dict
    into an ASCII tree string. Kept separate from generation so
    callers (e.g. a UI) can re-render the same data without
    hitting the model again.
    """

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Build parent -> children relationships
    children = {}

    for edge in edges:
        if len(edge) != 2:
            continue

        parent, child = edge

        if parent not in children:
            children[parent] = []

        children[parent].append(child)

    # Find root nodes
    all_children = {
        child
        for edge in edges
        if len(edge) == 2
        for child in [edge[1]]
    }

    roots = [
        node
        for node in nodes
        if node not in all_children
    ]

    # Fallback if no root is found
    if not roots and nodes:
        roots = [nodes[0]]

    lines = []

    visited = set()

    def add_branch(node, prefix="", is_last=True):

        if node in visited:
            return

        visited.add(node)

        connector = "└── " if is_last else "├── "

        lines.append(
            prefix + connector + node
        )

        node_children = children.get(node, [])

        for index, child in enumerate(node_children):

            last = index == len(node_children) - 1

            if is_last:
                new_prefix = prefix + "    "
            else:
                new_prefix = prefix + "│   "

            add_branch(
                child,
                new_prefix,
                last,
            )

    # Display each root
    for index, root in enumerate(roots):

        if index == 0:
            lines.append(root)
        else:
            lines.append("")
            lines.append(root)

        root_children = children.get(root, [])

        for child_index, child in enumerate(root_children):

            add_branch(
                child,
                "",
                child_index == len(root_children) - 1,
            )

    return "\n".join(lines)


def generate_mindmap(
    condensed_text: str,
) -> str:
    """
    Generate and display a study mindmap as a tree.
    Falls back to a clear error message if the model never
    produced parseable JSON, instead of crashing the session.
    """

    try:
        data = generate_mindmap_data(condensed_text)

        return render_tree(data)

    except ValueError as error:

        # generate_mindmap_data already retried once; surface the
        # failure clearly instead of crashing the whole session.
        return (
            "Mindmap generation failed after retrying: "
            f"{error}"
        )

    except (TypeError, AttributeError) as error:

        # Malformed but technically valid JSON (e.g. wrong shapes).
        return (
            f"Mindmap data had an unexpected shape: {error}"
        )
