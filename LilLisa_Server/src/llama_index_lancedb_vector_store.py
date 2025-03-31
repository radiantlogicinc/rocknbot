"""LanceDB vector store with modifications."""

import logging
import os
import warnings
from typing import Any, List, Optional

import lancedb.remote.table  # type: ignore
import lancedb.rerankers
import numpy as np
import pandas as pd
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_TEXT_KEY,
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.vector_stores.lancedb.util import sql_operator_mapper

import lancedb

_logger = logging.getLogger(__name__)


def _to_lance_filter(standard_filters: MetadataFilters, metadata_keys: list) -> Any:
    """Translate standard metadata filters to Lance specific spec."""
    filters = []
    for filter in standard_filters.filters:
        key = filter.key
        if filter.key in metadata_keys:
            key = f"metadata.{filter.key}"
        if filter.operator == FilterOperator.TEXT_MATCH or filter.operator == FilterOperator.NE:
            filters.append(key + sql_operator_mapper[filter.operator] + f"'%{filter.value}%'")
        if isinstance(filter.value, list):
            val = ",".join(filter.value)
            filters.append(key + sql_operator_mapper[filter.operator] + f"({val})")
        elif isinstance(filter.value, int):
            filters.append(key + sql_operator_mapper[filter.operator] + f"{filter.value}")
        else:
            filters.append(key + sql_operator_mapper[filter.operator] + f"'{filter.value!s}'")
    if standard_filters.condition == FilterCondition.OR:
        return " OR ".join(filters)
    else:
        return " AND ".join(filters)


def _to_llama_similarities(results: pd.DataFrame) -> List[float]:
    keys = results.keys()
    normalized_similarities: np.ndarray
    if "score" in keys:
        normalized_similarities = np.exp(results["score"] - np.max(results["score"]))
    elif "_distance" in keys:
        normalized_similarities = np.exp(-results["_distance"])
    else:
        normalized_similarities = np.linspace(1, 0, len(results))
    return normalized_similarities.tolist()


class LanceDBVectorStore(BasePydanticVectorStore):
    """
    The LanceDB Vector Store.

    Stores text and embeddings in LanceDB. The vector store will open an existing
    LanceDB dataset or create the dataset if it does not exist.
    """

    stores_text: bool = True
    flat_metadata: bool = True
    uri: Optional[str] = "/tmp/lancedb"
    vector_column_name: Optional[str] = "vector"
    nprobes: Optional[int] = 20
    refine_factor: Optional[int] = None
    text_key: Optional[str] = DEFAULT_TEXT_KEY
    doc_id_key: Optional[str] = DEFAULT_DOC_ID_KEY
    api_key: Optional[str] = None
    region: Optional[str] = None
    mode: Optional[str] = "overwrite"
    query_type: Optional[str] = "hybrid"
    overfetch_factor: Optional[int] = 1

    _table_name: Optional[str] = PrivateAttr()
    _connection: Any = PrivateAttr()
    _table: Any = PrivateAttr()
    _metadata_keys: Any = PrivateAttr()
    _fts_index: Any = PrivateAttr()
    _reranker: Any = PrivateAttr()

    def __init__(
        self,
        uri: Optional[str] = "/tmp/lancedb",
        table_name: Optional[str] = "vectors",
        vector_column_name: str = "vector",
        nprobes: int = 20,
        refine_factor: Optional[int] = None,
        text_key: str = DEFAULT_TEXT_KEY,
        doc_id_key: str = DEFAULT_DOC_ID_KEY,
        connection: Optional[Any] = None,
        table: Optional[Any] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        mode: str = "overwrite",
        query_type: str = "hybrid",
        reranker: Optional[Any] = None,
        overfetch_factor: int = 1,
        **kwargs: Any,
    ) -> None:
        # Initialize the Pydantic model.
        super().__init__(
            uri=uri,
            table_name=table_name,
            vector_column_name=vector_column_name,
            nprobes=nprobes,
            refine_factor=refine_factor,
            text_key=text_key,
            doc_id_key=doc_id_key,
            mode=mode,
            query_type=query_type,
            overfetch_factor=overfetch_factor,
            api_key=api_key,
            region=region,
            **kwargs,
        )

        # Set private attributes using object.__setattr__
        object.__setattr__(self, "_table_name", table_name)
        object.__setattr__(self, "_metadata_keys", None)
        object.__setattr__(self, "_fts_index", None)

        if isinstance(reranker, lancedb.rerankers.Reranker):
            object.__setattr__(self, "_reranker", reranker)
        elif reranker is None:
            object.__setattr__(self, "_reranker", None)
        else:
            raise ValueError("`reranker` has to be a lancedb.rerankers.Reranker object.")

        if isinstance(connection, lancedb.db.LanceDBConnection):
            object.__setattr__(self, "_connection", connection)
        elif isinstance(connection, str):
            raise ValueError("`connection` has to be a lancedb.db.LanceDBConnection object.")
        else:
            if api_key is None and os.getenv("LANCE_API_KEY") is None:
                if uri.startswith("db://"):
                    raise ValueError("API key is required for LanceDB cloud.")
                else:
                    object.__setattr__(self, "_connection", lancedb.connect(uri))
            else:
                if "db://" not in uri:
                    object.__setattr__(self, "_connection", lancedb.connect(uri))
                    warnings.warn("api key provided with local uri. The data will be stored locally")
                object.__setattr__(
                    self,
                    "_connection",
                    lancedb.connect(uri, api_key=api_key or os.getenv("LANCE_API_KEY"), region=region),
                )

        if table is not None:
            try:
                assert isinstance(table, (lancedb.db.LanceTable, lancedb.remote.table.RemoteTable))
                object.__setattr__(self, "_table", table)
                object.__setattr__(self, "_table_name", table.name if hasattr(table, "name") else "remote_table")
            except AssertionError:
                raise ValueError(
                    "`table` has to be a lancedb.db.LanceTable or lancedb.remote.table.RemoteTable object."
                )
        else:
            if self._table_exists():
                object.__setattr__(self, "_table", self._connection.open_table(table_name))
            else:
                object.__setattr__(self, "_table", None)

    @property
    def client(self) -> None:
        """Get client."""
        return self._connection

    @classmethod
    def from_table(cls, table: Any, query_type: Optional[str] = "hybrid") -> "LanceDBVectorStore":
        """Create instance from table."""
        try:
            if not isinstance(table, (lancedb.db.LanceTable, lancedb.remote.table.RemoteTable)):
                raise Exception("argument is not lancedb table instance")
            return cls(table=table, query_type=query_type)
        except Exception:
            print("ldb version", lancedb.__version__)
            raise

    def _add_reranker(self, reranker: lancedb.rerankers.Reranker) -> None:
        """Add a reranker to an existing vector store."""
        if reranker is None:
            raise ValueError("`reranker` has to be a lancedb.rerankers.Reranker object.")
        object.__setattr__(self, "_reranker", reranker)

    def _table_exists(self, tbl_name: Optional[str] = None) -> bool:
        return (tbl_name or self._table_name) in self._connection.table_names()

    def create_index(
        self,
        scalar: Optional[bool] = False,
        col_name: Optional[str] = None,
        num_partitions: Optional[int] = 256,
        num_sub_vectors: Optional[int] = 96,
        index_cache_size: Optional[int] = None,
        metric: Optional[str] = "L2",
    ) -> None:
        """
        Create a scalar (for non-vector columns) or a vector index on a table.
        """
        if scalar is None:
            self._table.create_index(
                metric=metric,
                vector_column_name=self.vector_column_name,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                index_cache_size=index_cache_size,
            )
        else:
            if col_name is None:
                raise ValueError("Column name is required for scalar index creation.")
            self._table.create_scalar_index(col_name)

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        if not nodes:
            _logger.debug("No nodes to add. Skipping the database operation.")
            return []
        data = []
        ids = []
        for node in nodes:
            metadata = node_to_metadata_dict(node, remove_text=False, flat_metadata=self.flat_metadata)
            if not self._metadata_keys:
                self._metadata_keys = list(metadata.keys())
            append_data = {
                "id": node.node_id,
                self.doc_id_key: node.ref_doc_id,
                self.vector_column_name: node.get_embedding(),
                self.text_key: node.get_content(metadata_mode=MetadataMode.NONE),
                "metadata": metadata,
            }
            data.append(append_data)
            ids.append(node.node_id)
        if self._table is None:
            self._table = self._connection.create_table(self._table_name, data, mode=self.mode)
        else:
            if self.api_key is None:
                self._table.add(data, mode="append")
            else:
                self._table.add(data)
        self._fts_index = None  # reset FTS index
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using the ref_doc_id.
        """
        self._table.delete(f'{self.doc_id_key} = "' + ref_doc_id + '"')

    def delete_nodes(self, node_ids: List[str], **delete_kwargs: Any) -> None:
        """
        Delete nodes using a list of node_ids.
        """
        self._table.delete('id in ("' + '","'.join(node_ids) + '")')

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Get nodes from the vector store.
        """
        if isinstance(self._table, lancedb.remote.table.RemoteTable):
            raise ValueError("get_nodes is not supported for LanceDB cloud yet.")
        if filters is not None:
            if "where" in kwargs:
                raise ValueError("Cannot specify filter via both query and kwargs.")
            where = _to_lance_filter(filters, self._metadata_keys)
        else:
            where = kwargs.pop("where", None)
        if node_ids is not None:
            where = 'id in ("' + '","'.join(node_ids) + '")'
        results = self._table.search().where(where).to_pandas()
        nodes = []
        for _, item in results.iterrows():
            try:
                node = metadata_dict_to_node(item.metadata)
                node.embedding = list(item[self.vector_column_name])
            except Exception:
                _logger.debug("Failed to parse Node metadata, falling back to legacy logic.")
                if item.metadata:
                    metadata, node_info, _relation = legacy_metadata_dict_to_node(
                        item.metadata, text_key=self.text_key
                    )
                else:
                    metadata, node_info = {}, {}
                node = TextNode(
                    text=item[self.text_key] or "",
                    id_=item.id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=item[self.doc_id_key]),
                    },
                )
            nodes.append(node)
        return nodes

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes."""
        if query.filters is not None:
            if "where" in kwargs:
                raise ValueError("Cannot specify filter via both query and kwargs.")
            where = _to_lance_filter(query.filters, self._metadata_keys)
        else:
            where = kwargs.pop("where", None)

        query_type = kwargs.pop("query_type", self.query_type)
        _logger.info(f"query_type: {query_type}")

        # Handle different query types separately.
        if query_type == "vector":
            _query = query.query_embedding
            lance_query = self._table.search(query=_query, vector_column_name=self.vector_column_name)
            if hasattr(lance_query, "metric"):
                lance_query = lance_query.metric("cosine")
            lance_query = lance_query.limit(query.similarity_top_k * self.overfetch_factor).where(
                where, prefilter=True
            )
            if hasattr(lance_query, "nprobes"):
                lance_query.nprobes(self.nprobes)
            results = lance_query.to_pandas()

        elif query_type == "fts":
            _query = query.query_str
            lance_query = self._table.search(query=_query, vector_column_name=self.vector_column_name)
            if hasattr(lance_query, "metric"):
                lance_query = lance_query.metric("cosine")
            lance_query = lance_query.limit(query.similarity_top_k * self.overfetch_factor).where(
                where, prefilter=True
            )
            results = lance_query.to_pandas()

        elif query_type == "hybrid":
            # Create FTS index if not already created.
            if not isinstance(self._table, lancedb.db.LanceTable):
                raise ValueError("FTS index creation not supported for LanceDB Cloud.")
            if self._fts_index is None:
                self._fts_index = self._table.create_fts_index(self.text_key, replace=True)

            # Execute vector search.
            vector_query = self._table.search(query=query.query_embedding, vector_column_name=self.vector_column_name)
            if hasattr(vector_query, "metric"):
                vector_query = vector_query.metric("cosine")
            vector_query = vector_query.limit(query.similarity_top_k * self.overfetch_factor).where(
                where, prefilter=True
            )
            if hasattr(vector_query, "nprobes"):
                vector_query.nprobes(self.nprobes)
            vector_results = vector_query.to_pandas()

            # Execute full-text search.
            fts_query = self._table.search(query=query.query_str, vector_column_name=self.vector_column_name)
            fts_query = fts_query.limit(query.similarity_top_k * self.overfetch_factor).where(where, prefilter=True)
            fts_results = fts_query.to_pandas()

            # Combine the results using pd.concat and drop duplicate IDs.
            combined_results = pd.concat([vector_results, fts_results], ignore_index=True)
            combined_results = combined_results.drop_duplicates(subset="id")
            results = combined_results

        else:
            raise ValueError(f"Invalid query type: {query_type}")

        if len(results) == 0:
            raise Warning("Query results are empty.")

        nodes = []
        for _, item in results.iterrows():
            try:
                node = metadata_dict_to_node(item.metadata)
                node.embedding = list(item[self.vector_column_name])
            except Exception:
                _logger.debug("Failed to parse Node metadata, falling back to legacy logic.")
                if item.metadata:
                    metadata, node_info, _relation = legacy_metadata_dict_to_node(
                        item.metadata, text_key=self.text_key
                    )
                else:
                    metadata, node_info = {}, {}
                node = TextNode(
                    text=item[self.text_key] or "",
                    id_=item.id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(node_id=item[self.doc_id_key]),
                    },
                )
            nodes.append(node)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=_to_llama_similarities(results),
            ids=results["id"].tolist(),
        )
