import networkx as nx
import polars as pl

SOURCE_NODE_COL = "SOURCE_ID"
TARGET_NODE_COL = "TARGET_ID"
REQUIRED_SKILL_LINK_COLUMNS = {
    SOURCE_NODE_COL,
    TARGET_NODE_COL,
}


def get_reachable_node_pairs(skill_links: pl.DataFrame) -> set[tuple[str, str]]:
    missing_cols = REQUIRED_SKILL_LINK_COLUMNS - set(skill_links.columns)
    if missing_cols:
        raise ValueError(f"Missing required skill link columns: {sorted(missing_cols)}")

    edges = (
        skill_links.select(
            pl.col(SOURCE_NODE_COL).cast(pl.Utf8),
            pl.col(TARGET_NODE_COL).cast(pl.Utf8),
        )
        .drop_nulls([SOURCE_NODE_COL, TARGET_NODE_COL])
        .iter_rows()
    )
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    return {
        (source_node, target_node)
        for source_node in graph.nodes
        for target_node in nx.descendants(graph, source_node)
    }
