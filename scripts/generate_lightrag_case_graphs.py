from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx

from rageval.config import ModelConfig
from rageval.rag.lightrag_rag import LightRAGSystem
from rageval.token_tracking import TokenCounter
from rageval.utils import ensure_parent_dir


DEFAULT_CASES = (
    "LeCaRDv2_557",
    "UltraLongFinance_MGM_Resorts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build LightRAG knowledge graphs for representative documents and "
            "export thesis-friendly graph visualizations."
        )
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        default=None,
        help="Specific doc_id(s) to visualize. Defaults to two representative cases.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./dataset/processed"),
        help="Root directory containing processed JSONL datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./docs/thesis/assets/lightrag_case_graphs"),
        help="Directory for exported graph images and summaries.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("./tmp/lightrag_case_graphs"),
        help="Temporary LightRAG workspace root.",
    )
    parser.add_argument(
        "--max-display-nodes",
        type=int,
        default=24,
        help="Maximum number of nodes shown in the final figure.",
    )
    parser.add_argument(
        "--max-display-labels",
        type=int,
        default=24,
        help="Maximum number of node labels shown in the final figure.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="gpt-4o-mini",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_documents_by_id(dataset_root: Path, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
    wanted = set(doc_ids)
    found: dict[str, dict[str, Any]] = {}
    for path in dataset_root.rglob("*.jsonl"):
        for row in iter_jsonl(path):
            doc_id = row.get("doc_id")
            if doc_id in wanted:
                found[doc_id] = row
                if len(found) == len(wanted):
                    return found
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"Missing doc_id(s) in processed datasets: {sorted(missing)}")
    return found


def pick_display_subgraph(graph: nx.Graph, max_nodes: int) -> nx.Graph:
    if graph.number_of_nodes() <= max_nodes:
        return graph.copy()

    degrees = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    selected: set[str] = set()
    seeds = [node for node, _ in degrees[: min(5, len(degrees))]]

    for seed in seeds:
        selected.add(seed)
        neighbors = sorted(graph.neighbors(seed), key=lambda n: graph.degree[n], reverse=True)
        for neighbor in neighbors:
            selected.add(neighbor)
            if len(selected) >= max_nodes:
                break
        if len(selected) >= max_nodes:
            break

    if len(selected) < max_nodes:
        for node, _ in degrees:
            selected.add(node)
            if len(selected) >= max_nodes:
                break

    return graph.subgraph(selected).copy()


def graph_summary(graph: nx.Graph) -> dict[str, Any]:
    degree_items = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    top_nodes = [
        {
            "node": str(node),
            "degree": int(degree),
        }
        for node, degree in degree_items[:10]
    ]
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "density": round(nx.density(graph), 6) if graph.number_of_nodes() > 1 else 0.0,
        "connected_components": nx.number_connected_components(graph),
        "average_degree": round(
            (sum(deg for _, deg in degree_items) / graph.number_of_nodes()), 4
        )
        if graph.number_of_nodes()
        else 0.0,
        "top_nodes": top_nodes,
    }


def build_cjk_font() -> font_manager.FontProperties | None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        file_path = Path(path)
        if file_path.exists():
            try:
                font_manager.fontManager.addfont(str(file_path))
            except Exception:
                pass
            return font_manager.FontProperties(fname=str(file_path))
    return None


def build_caption(row: dict[str, Any], summary: dict[str, Any]) -> str:
    return (
        f"{row['doc_id']} | {row['dataset_name']} | "
        f"{row.get('doc_length_tokens', 0)} tokens | "
        f"{summary['node_count']} nodes / {summary['edge_count']} edges"
    )


def save_graph_figure(
    graph: nx.Graph,
    row: dict[str, Any],
    summary: dict[str, Any],
    output_path: Path,
    max_display_labels: int,
) -> None:
    ensure_parent_dir(output_path)
    font_prop = build_cjk_font()
    if font_prop:
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, seed=42, k=1.2 / math.sqrt(max(graph.number_of_nodes(), 2)))
    degrees = dict(graph.degree())
    node_sizes = [420 + 120 * degrees[node] for node in graph.nodes()]

    nx.draw_networkx_edges(graph, pos, alpha=0.22, edge_color="#7f8c8d", width=0.8)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="#2f6f8f",
        alpha=0.9,
        node_size=node_sizes,
        linewidths=0.8,
        edgecolors="#173042",
    )

    label_candidates = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    label_nodes = [node for node, _ in label_candidates[:max_display_labels]]
    labels = {node: str(node)[:30] for node in label_nodes if node in graph.nodes()}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=8,
        font_family=font_prop.get_name() if font_prop else "sans-serif",
        font_color="#111111",
    )

    plt.title(build_caption(row, summary), fontsize=14, fontproperties=font_prop)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


async def build_case_graph(
    row: dict[str, Any],
    output_dir: Path,
    workspace_root: Path,
    model_config: ModelConfig,
    max_display_nodes: int,
    max_display_labels: int,
) -> dict[str, Any]:
    token_counter = TokenCounter(model_config.tokenizer_model)
    system = LightRAGSystem(
        model_config=model_config,
        token_counter=token_counter,
        workspace_root=workspace_root,
    )
    await system.initialize(row["doc_id"], row["document"])

    if system.rag is not None:
        await system.rag.finalize_storages()

    graph_path = Path(system.workspace_dir) / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"LightRAG graph file not found: {graph_path}")

    preserved_graph_path = output_dir / f"{row['doc_id']}.graphml"
    ensure_parent_dir(preserved_graph_path)
    shutil.copy2(graph_path, preserved_graph_path)

    full_graph = nx.read_graphml(graph_path)
    display_graph = pick_display_subgraph(full_graph, max_nodes=max_display_nodes)
    summary = graph_summary(full_graph)
    summary["display_node_count"] = display_graph.number_of_nodes()
    summary["display_edge_count"] = display_graph.number_of_edges()
    summary["display_label_count"] = min(display_graph.number_of_nodes(), max_display_labels)
    summary["doc_id"] = row["doc_id"]
    summary["dataset_name"] = row["dataset_name"]
    summary["doc_length_tokens"] = row.get("doc_length_tokens", 0)
    summary["qa_pair_count"] = len(row.get("qa_pairs") or [])

    png_path = output_dir / f"{row['doc_id']}.png"
    save_graph_figure(
        display_graph,
        row,
        summary,
        png_path,
        max_display_labels=max_display_labels,
    )

    summary_path = output_dir / f"{row['doc_id']}.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    await system.close()
    return {
        "graphml_path": str(preserved_graph_path),
        "png_path": str(png_path),
        "summary_path": str(summary_path),
        **summary,
    }


async def main_async() -> None:
    args = parse_args()
    doc_ids = args.doc_id or list(DEFAULT_CASES)
    docs = load_documents_by_id(args.dataset_root, doc_ids)
    model_config = ModelConfig.from_env(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        tokenizer_model=args.tokenizer_model,
    )
    model_config.validate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workspace_root.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        row = docs[doc_id]
        print(f"[build] {doc_id} ({row['dataset_name']}, {row.get('doc_length_tokens')} tokens)")
        report.append(
            await build_case_graph(
                row=row,
                output_dir=args.output_dir,
                workspace_root=args.workspace_root,
                model_config=model_config,
                max_display_nodes=args.max_display_nodes,
                max_display_labels=args.max_display_labels,
            )
        )

    report_path = args.output_dir / "case_graph_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] report={report_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
