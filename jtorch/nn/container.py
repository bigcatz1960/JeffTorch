"""
container.py
JeffTorch — Lightweight PyTorch‑shaped neural network library

This module implements simple container classes for composing multiple
JeffTorch modules into larger structures. The design mirrors the role
of torch.nn containers such as Sequential, providing a clean and
predictable way to build neural network architectures from smaller
components.

Containers serve two main purposes:
    • they define an ordered execution pipeline for submodules
    • they automatically register contained modules for parameter
      collection, optimization, and introspection

Included containers:
    • Sequential — executes modules in the order they are added

These classes inherit from Module and integrate seamlessly with the
rest of the JeffTorch framework. They are intentionally minimal and
transparent, making it easy to understand how module composition works
under the hood and how complex models can be built from simple parts.

This file is part of the JeffTorch learning framework — a clear,
readable environment for exploring neural network structure and
experimenting with new architectural ideas.
"""
