# ChunkingStrategy Enum Implementation Summary

## Overview
This document summarizes the implementation of a proper enum-based chunking strategy system that replaces the previous environment variable-based approach.

## Changes Made

### 1. ChunkingStrategy Enum
- **Location**: `/src/main.py` (after imports)
- **Purpose**: Defines available chunking strategies
- **Values**:
  - `ChunkingStrategy.TRADITIONAL` = "traditional"
  - `ChunkingStrategy.CONTEXTUAL` = "contextual"

### 2. Global Variables Updated
- **Removed**: `CHUNKING_STRATEGY` (integer-based environment variable)
- **Added**: `CURRENT_CHUNKING_STRATEGY = ChunkingStrategy.TRADITIONAL` (enum-based, defaults to traditional)

### 3. Function Updates

#### `configure_embedding_model(chunking_strategy: ChunkingStrategy)`
- **Input**: Now accepts `ChunkingStrategy` enum instead of integer
- **Logic**: 
  - `ChunkingStrategy.TRADITIONAL` → OpenAI `text-embedding-3-large`
  - `ChunkingStrategy.CONTEXTUAL` → Voyage `voyage-context-3`

#### `refresh_embedding_model()`
- **Updated**: Uses `CURRENT_CHUNKING_STRATEGY` instead of `CHUNKING_STRATEGY`

#### `set_chunking_strategy(strategy: ChunkingStrategy)`
- **New function**: Sets global chunking strategy and updates embedding model
- **Purpose**: Central point for switching between strategies

### 4. Startup Configuration
- **Removed**: Environment variable loading for `CHUNKING_STRATEGY`
- **Default**: Application starts with `ChunkingStrategy.TRADITIONAL`
- **Embedding**: Automatically configured with OpenAI `text-embedding-3-large`

### 5. Endpoint Changes

#### Original `/rebuild_docs/`
- **Updated**: Now uses traditional chunking by default
- **Purpose**: Backward compatibility

#### New `/rebuild_docs_traditional/`
- **Purpose**: Explicitly rebuild with traditional OpenAI chunking
- **Behavior**: 
  - Immediately switches query embedding to OpenAI `text-embedding-3-large`
  - Rebuilds document store with traditional chunking
  - **Rebuilds QA pairs** with traditional chunking
  - Documents, QA pairs, and query embeddings all use same model

#### New `/rebuild_docs_contextual/`
- **Purpose**: Explicitly rebuild with contextual Voyage chunking
- **Behavior**:
  - Immediately switches query embedding to Voyage `voyage-context-3`
  - Rebuilds document store with contextual chunking
  - **Rebuilds QA pairs** with contextual chunking
  - Documents, QA pairs, and query embeddings all use same model

### 6. Background Task Updates
- **`_run_rebuild_docs_task_traditional()`**: Uses `set_chunking_strategy(ChunkingStrategy.TRADITIONAL)`
- **`_run_rebuild_docs_task_contextual()`**: Uses `set_chunking_strategy(ChunkingStrategy.CONTEXTUAL)`
- **`_run_rebuild_docs_task()`**: Simplified to call traditional chunking

## Flow Description

### Initial Startup
1. Server starts with `CURRENT_CHUNKING_STRATEGY = ChunkingStrategy.TRADITIONAL`
2. Embedding model configured with OpenAI `text-embedding-3-large`
3. **Complete rebuild** executed: both documentation and QA pairs built using traditional chunking with OpenAI embeddings
4. Query processing uses OpenAI `text-embedding-3-large` for matching

### Switching to Traditional Chunking
1. Call `/rebuild_docs_traditional/` endpoint
2. **Immediate**: Query embedding model switches to OpenAI `text-embedding-3-large`
3. **Background**: 
   - Document store rebuilt with traditional chunking using OpenAI embeddings
   - **QA pairs rebuilt** with traditional chunking using OpenAI embeddings
4. **Result**: Documents, QA pairs, and queries all use OpenAI `text-embedding-3-large` (consistent embedding space)

### Switching to Contextual Chunking
1. Call `/rebuild_docs_contextual/` endpoint
2. **Immediate**: Query embedding model switches to Voyage `voyage-context-3`
3. **Background**: 
   - Document store rebuilt with contextual chunking using Voyage embeddings
   - **QA pairs rebuilt** with contextual chunking using Voyage embeddings
4. **Result**: Documents, QA pairs, and queries all use Voyage `voyage-context-3` (consistent embedding space)

## Key Benefits

1. **No Environment Variables**: System is self-contained, no external configuration needed
2. **Type Safety**: Enum prevents invalid chunking strategy values
3. **Immediate Query Updates**: Query embedding model switches immediately when endpoint is called
4. **Consistent Embeddings**: Documents, QA pairs, and query embeddings always use the same model
5. **No Embedding Mismatches**: Both document store and QA pairs are rebuilt together
6. **Clear API**: Separate endpoints for each chunking strategy
7. **Backward Compatibility**: Original `/rebuild_docs/` still works

## Usage Examples

### Switch to Traditional Chunking
```bash
curl -X POST "http://localhost:8000/rebuild_docs_traditional/" \
     -H "Content-Type: application/json" \
     -d '{"encrypted_key": "your_jwt_token"}'
```

### Switch to Contextual Chunking
```bash
curl -X POST "http://localhost:8000/rebuild_docs_contextual/" \
     -H "Content-Type: application/json" \
     -d '{"encrypted_key": "your_jwt_token"}'
```

## Important Fix: Embedding Consistency

### Problem Identified
The initial implementation had a critical issue where switching chunking strategies would only rebuild the document store but not the QA pairs. This would result in:
- Document embeddings in one vector space (e.g., Voyage)
- QA pairs embeddings in a different vector space (e.g., OpenAI)
- Query embeddings matching the current strategy but incompatible with mismatched stored embeddings
- **Result**: Embedding mismatch errors during query processing

### Solution Implemented
Created wrapper functions that ensure **both** document store and QA pairs are rebuilt together:
- `_run_complete_rebuild_traditional()`: Rebuilds docs + QA pairs with OpenAI embeddings
- `_run_complete_rebuild_contextual()`: Rebuilds docs + QA pairs with Voyage embeddings

### Updated Endpoints
All rebuild endpoints now ensure complete consistency:
- `/rebuild_docs/` → `_run_complete_rebuild_traditional()`
- `/rebuild_docs_traditional/` → `_run_complete_rebuild_traditional()`
- `/rebuild_docs_contextual/` → `_run_complete_rebuild_contextual()`

This guarantees that documents, QA pairs, and query embeddings are always in the same vector space.

## Environment Variable Cleanup

The following environment variable is no longer used and can be removed:
- `CHUNKING_STRATEGY=1` or `CHUNKING_STRATEGY=2` in `lillisa_server.env`

The system will ignore this variable if present and use the enum-based approach instead.
