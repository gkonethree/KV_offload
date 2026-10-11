# SPDX-License-Identifier: Apache-2.0
"""Retired helpers.

``initialize_for_runner`` (scheduler-side orchestrator builder) and
``prepare_sparse_pages`` (lag-prefetch patch helper) had no remaining
callers: the orchestrator is built worker-side via set_runner, and the pq
path fetches at selection time. Kept as an explanatory stub module to
avoid breaking imports.
"""
