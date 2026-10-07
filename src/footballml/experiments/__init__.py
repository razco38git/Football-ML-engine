"""Measurement harnesses that are not part of the weekly job.

A pipeline stage produces something the site serves. These answer a question
once and write down the answer, which is why they live apart from
``pipelines``: nothing here runs on a schedule, and nothing the site shows
depends on them.
"""
