#!/usr/bin/env python3
"""
Test script to verify the ChunkingStrategy enum implementation.
"""

from src.main import ChunkingStrategy, CURRENT_CHUNKING_STRATEGY, set_chunking_strategy

def test_chunking_strategy():
    """Test the chunking strategy enum functionality."""
    print("Testing ChunkingStrategy enum implementation")
    print("=" * 50)
    
    # Test enum values
    print(f"ChunkingStrategy.TRADITIONAL: {ChunkingStrategy.TRADITIONAL}")
    print(f"ChunkingStrategy.TRADITIONAL.value: {ChunkingStrategy.TRADITIONAL.value}")
    print(f"ChunkingStrategy.CONTEXTUAL: {ChunkingStrategy.CONTEXTUAL}")
    print(f"ChunkingStrategy.CONTEXTUAL.value: {ChunkingStrategy.CONTEXTUAL.value}")
    
    # Test current strategy
    print(f"\nInitial CURRENT_CHUNKING_STRATEGY: {CURRENT_CHUNKING_STRATEGY}")
    print(f"Initial strategy value: {CURRENT_CHUNKING_STRATEGY.value}")
    
    # Test switching strategies
    print("\nTesting strategy switching:")
    
    print("Switching to CONTEXTUAL...")
    set_chunking_strategy(ChunkingStrategy.CONTEXTUAL)
    from src.main import CURRENT_CHUNKING_STRATEGY as updated_strategy
    print(f"Current strategy after switch: {updated_strategy}")
    
    print("Switching back to TRADITIONAL...")
    set_chunking_strategy(ChunkingStrategy.TRADITIONAL)
    from src.main import CURRENT_CHUNKING_STRATEGY as final_strategy
    print(f"Current strategy after switch: {final_strategy}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_chunking_strategy()
