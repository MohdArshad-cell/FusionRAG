import os
import sys
import asyncio
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm

from app.dependencies import (
    get_retriever,
    get_reranker,
    get_llm,
    get_embedder,
    get_semantic_cache,
    get_crag_gate,
    get_groundedness_verifier,
    get_parent_store,
    get_query_processor,
    warmup_services,
)
from app.core.fusion import reciprocal_rank_fusion
from app.core.context_builder import ContextBuilder

console = Console()


async def run_interactive_cli():
    console.print(Panel.fit(
        "[bold cyan]⚡ FusionRAG Interactive Terminal Console[/bold cyan]\n"
        "[dim]Hybrid Search: Dense Vector + BM25 + RRF + Cross-Encoder + CRAG Guardrails[/dim]",
        border_style="cyan"
    ))

    with console.status("[bold green]Warming up models and initializing stores..."):
        warmup_services()
        retriever = get_retriever()
        reranker = get_reranker()
        llm = get_llm()
        embedder = get_embedder()
        cache = get_semantic_cache()
        crag_gate = get_crag_gate()
        groundedness = get_groundedness_verifier()
        parent_store = get_parent_store()
        query_proc = get_query_processor()

    console.print("[green]✓ Engine ready! Type [bold yellow]exit[/bold yellow] or [bold yellow]quit[/bold yellow] to leave.[/green]\n")

    while True:
        try:
            query = Prompt.ask("[bold blue]Ask FusionRAG[/bold blue]")
            if not query.strip():
                continue
            if query.strip().lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break

            start_t = time.perf_counter()

            # 1. Semantic Cache check
            query_emb = embedder.embed_query(query)
            cached_result = cache.lookup(query_emb)
            if cached_result:
                cached_data, sim, orig_q = cached_result
                console.print(f"[bold green]⚡ [CACHE HIT][/bold green] [dim](Cosine similarity: {sim:.3f} with '{orig_q}')[/dim]")
                console.print(Panel(Markdown(cached_data["answer"]), title="[bold green]Answer (Cached)[/bold green]", border_style="green"))
                continue

            # 2. Parallel Dual Retrieval
            with console.status("[cyan]Retrieving from ChromaDB & BM25 in parallel..."):
                d_res, b_res, timings = await retriever.retrieve_parallel(query, dense_k=20, bm25_k=20)

            # 3. Reciprocal Rank Fusion
            fused = reciprocal_rank_fusion([d_res, b_res], k=60, top_n=10)

            # 4. Cross-Encoder Re-Ranking
            with console.status("[magenta]Re-ranking top 10 candidates with Cross-Encoder..."):
                reranked = reranker.rerank(query, fused, top_n=5)

            # 5. CRAG Confidence Gate
            crag_eval = crag_gate.evaluate_candidates(reranked)
            confidence = crag_eval["decision"]

            # Display retrieval breakdown table
            table = Table(title="Top Re-Ranked Candidate Chunks", border_style="dim")
            table.add_column("Rank", justify="center", style="bold")
            table.add_column("Source File", style="cyan")
            table.add_column("Cross-Encoder Score", justify="right", style="green")
            table.add_column("Retriever Origin", style="magenta")
            table.add_column("Snippet Preview", style="dim")

            for sc in reranked[:3]:
                table.add_row(
                    str(sc.rank),
                    sc.chunk.source_file or "Document",
                    f"{sc.score:.4f}",
                    sc.retriever,
                    sc.chunk.text[:90].replace("\n", " ") + "..."
                )
            console.print(table)

            if confidence == "INSUFFICIENT":
                console.print(f"[yellow]⚠️ CRAG Gate: {confidence} - Knowledge not found in index.[/yellow]")
                answer = "Based on the verified knowledge base, no relevant documents were found to answer this question."
                grounded_info = {"groundedness_score": 1.0, "status": "SAFE_ABSTENTION"}
            else:
                # 6. Context assembly & Generation
                context_chunks = parent_store.resolve_to_parent_chunks(reranked)
                context_str = ContextBuilder.build_context_string(context_chunks, apply_lost_in_middle=True)

                with console.status("[blue]Generating answer with LLM..."):
                    answer = await llm.generate(query, context_str)

                grounded_info = groundedness.verify(answer, context_str)

                # Store in cache
                cache.put(query, query_emb, {"answer": answer, "query": query})

            total_elapsed = round((time.perf_counter() - start_t) * 1000, 1)

            # Print Answer Box
            console.print(Panel(
                Markdown(answer),
                title=f"[bold cyan]Answer[/bold cyan] [dim]({total_elapsed}ms)[/dim]",
                border_style="cyan"
            ))

            # Quality metrics footer
            g_score = grounded_info["groundedness_score"]
            g_color = "green" if g_score >= 0.70 else ("yellow" if g_score >= 0.45 else "red")
            console.print(
                f"[dim]Metrics:[/dim] CRAG Confidence: [bold]{confidence}[/bold] | "
                f"Groundedness: [{g_color}]{g_score * 100:.0f}% ({grounded_info['status']})[/{g_color}] | "
                f"Retrieval: {timings['parallel_retrieval_ms']}ms\n"
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    asyncio.run(run_interactive_cli())
