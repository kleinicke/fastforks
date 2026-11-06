"""Modules shared by several model families (backbones, layers), each with its own LICENSE and meta.toml.

Shared code ships in the core wheel, so it must be permissively licensed. Families declare what they import from here
in their card's `uses`; a change here re-runs the goldens of exactly those families.
"""
