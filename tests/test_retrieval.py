from src.chunking import generate_chunk_id
from src.retrieve import QueryRouter, reciprocal_rank_fusion
from src.schemas import Chunk


def test_generate_chunk_id_determinism():
    id1 = generate_chunk_id("table", 4, 1, "Part I > Item 1", "Operations", "Total net sales 82,959")
    id2 = generate_chunk_id("table", 4, 1, "Part I > Item 1", "Operations", "Total net sales 82,959")
    id3 = generate_chunk_id("text", 4, 1, "Part I > Item 1", "Operations", "Total net sales 82,959")

    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 16


def test_reciprocal_rank_fusion():
    dense_res = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]
    sparse_res = [("c2", 15.0), ("c1", 10.0), ("c4", 5.0)]

    fused = reciprocal_rank_fusion(dense_res, sparse_res, rrf_k=60)
    assert "c1" in fused
    assert "c2" in fused
    assert fused["c2"]["rrf_score"] > fused["c3"]["rrf_score"]
    assert fused["c1"]["dense_rank"] == 1
    assert fused["c1"]["sparse_rank"] == 2


def test_query_router_boost():
    router = QueryRouter(table_boost=1.5, text_boost=1.2, figure_boost=2.0)

    table_chunk = Chunk(
        id="c1", type="table", page=4, section_path="Operations", title="Net Sales",
        units="millions", text="table text", embed_text="table text"
    )
    text_chunk = Chunk(
        id="c2", type="text", page=17, section_path="MD&A", title="",
        units="", text="text content", embed_text="text content"
    )
    figure_chunk = Chunk(
        id="c3", type="figure", page=1, section_path="Cover", title="Logo",
        units="", text="figure text", embed_text="figure text"
    )

    # Table query
    b1 = router.apply_boost("What were total net sales in millions?", table_chunk, 1.0)
    assert b1 == 1.5

    # Text query
    b2 = router.apply_boost("Why did gross margin fall in Q3?", text_chunk, 1.0)
    assert b2 == 1.2

    # Figure query
    b3 = router.apply_boost("What image is on the cover page logo?", figure_chunk, 1.0)
    assert b3 == 2.0
