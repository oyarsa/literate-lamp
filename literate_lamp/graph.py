#!/usr/bin/env python3
import graphviz


def main() -> None:
    graph = graphviz.Digraph(
        graph_attr={'splines': 'false'}
    )

    graph.edge("SimpleBase", "AttentiveReader", label="+Seq2Seq\n"
                                                      "+Bilinear Attn.")
    graph.edge("AttentiveReader", "ZeroTriAN", label="+Self Attn.\n"
                                                     "+Sequence Attn.")
    graph.edge("ZeroTriAN", "TriAN",
               label="+Word Overlap\n+Relation Emb.",
               xlabel="+POS Emb.\n+NER Emb.\n")

    graph.edge("SimpleBERT", "HierarchicalBERT", label="+LSTM\n+Attn.",
               xlabel="-Pooler")
    graph.edge("HierarchicalBERT", "RelationalBERT",
               xlabel="+Rel. Emb.")
    graph.edge("HierarchicalBERT", "BERT + RTM",
               label="+Rel. Transf.")

    graph.edge("SimpleXL", "AdvancedXL", label="+LSTM\n+Attn.",
               xlabel="-Pooler")
    graph.edge("AdvancedXL", "RelationalXL", xlabel="+Rel. Emb.")
    graph.edge("AdvancedXL", "XL + RTM", label="+Rel. Transf.")

    graph.render('models', view=True)



if __name__ == '__main__':
    main()
