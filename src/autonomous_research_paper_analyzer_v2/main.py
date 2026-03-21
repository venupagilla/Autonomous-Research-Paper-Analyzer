#!/usr/bin/env python
# import sys
import warnings

from datetime import datetime

from autonomous_research_paper_analyzer_v2.crew import AutonomousResearchPaperAnalyzerV2

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run_analysis(
    topic: str,
    max_papers: int = 5,
    analysis_papers: int = 5,
    report_words: int = 800,
):
    """Run the crew with user inputs and return the raw crew output object."""
    current_year = datetime.now().year
    inputs = {
        'topic': topic,
        'max_papers': str(max_papers),
        'analysis_papers': str(analysis_papers),
        'report_words': str(report_words),
        'current_year': str(current_year),
        'recent_start_year': str(current_year - 3),
    }
    return AutonomousResearchPaperAnalyzerV2().crew().kickoff(inputs=inputs)

def run():
    """
    Run the crew.
    """
    try:
        run_analysis(topic='Kidney Tumor detection using machine learning', max_papers=5)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

