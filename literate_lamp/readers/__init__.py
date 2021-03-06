"""
This module implements a reader that takes the pre-processed Json files as
input, as well as specific readers that process that input in particular ways.

Different models will generally need slightly different inputs, and this is
where they're processed.
"""
from readers.base_reader import BaseReader
from readers.extended_xlnet_reader import ExtendedXLNetReader
from readers.full_trian_reader import FullTrianReader
from readers.sentence_reader import SentenceReader
from readers.simple_mc_script_reader import SimpleMcScriptReader
from readers.simple_bert_reader import SimpleBertReader
from readers.simple_xlnet_reader import SimpleXLNetReader
from readers.simple_trian_reader import SimpleTrianReader
from readers.relation_bert_reader import RelationBertReader
from readers.relation_xlnet_reader import RelationXLNetReader
